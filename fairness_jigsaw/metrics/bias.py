"""Reusable fairness bias metrics for Jigsaw-style datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy, ks_2samp, mannwhitneyu, wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


@dataclass
class FairnessBiasEvaluator:
    """End-to-end fairness metrics evaluator with class-based API."""

    DEFAULT_IDENTITY_COLUMNS = (
        "male",
        "female",
        "transgender",
        "other_gender",
        "heterosexual",
        "homosexual_gay_or_lesbian",
        "bisexual",
        "other_sexual_orientation",
        "christian",
        "jewish",
        "muslim",
        "hindu",
        "buddhist",
        "atheist",
        "other_religion",
        "black",
        "white",
        "asian",
        "latino",
        "other_race_or_ethnicity",
        "physical_disability",
        "intellectual_or_learning_disability",
        "psychiatric_or_mental_illness",
        "other_disability",
    )

    identity_cols: Sequence[str] = field(default_factory=lambda: FairnessBiasEvaluator.DEFAULT_IDENTITY_COLUMNS)
    toxicity_threshold: float = 0.5
    identity_threshold: float = 0.5
    min_auc_subgroup_size: int = 10
    min_shift_subgroup_size: int = 20
    pinned_p: int = -5
    bin_suffix: str = "_bin"
    _dp_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _auc_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _stat_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _dist_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _partial_corr_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _bias_table: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _bias_table_normalized: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
        try:
            if pd.Series(y_true).nunique() < 2:
                return np.nan
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return np.nan

    def add_binary_identity_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with binary identity columns added."""
        self._check_required_columns(df, self.identity_cols)
        out = df.copy()
        for col in self.identity_cols:
            out[f"{col}{self.bin_suffix}"] = (out[col].fillna(0) >= self.identity_threshold).astype(int)
        return out

    def compute_demographic_parity(self, df: pd.DataFrame, label_col: str = "toxic") -> pd.DataFrame:
        """Compute demographic parity gap per identity subgroup."""
        self._check_required_columns(
            df,
            [label_col, *[f"{c}{self.bin_suffix}" for c in self.identity_cols]],
        )
        overall_rate = df[label_col].mean()
        records: list[dict[str, float | str | int]] = []
        for col in self.identity_cols:
            subgroup = df[df[f"{col}{self.bin_suffix}"] == 1]
            if subgroup.empty:
                continue
            rate = subgroup[label_col].mean()
            se = np.sqrt((rate * (1.0 - rate)) / len(subgroup))
            gap = rate - overall_rate
            records.append(
                {
                    "identity": col,
                    "n_subgroup": len(subgroup),
                    "toxic_rate": rate,
                    "dp_gap": gap,
                    "ci_low": gap - 1.96 * se,
                    "ci_high": gap + 1.96 * se,
                }
            )
        return pd.DataFrame(records).sort_values("dp_gap", ascending=False).reset_index(drop=True)

    def compute_auc_metrics(
        self,
        df: pd.DataFrame,
        score_col: str = "target",
        label_col: str = "toxic",
    ) -> pd.DataFrame:
        """Compute Subgroup/BPSN/BNSP AUC and pinned AUC for each identity."""
        self._check_required_columns(
            df,
            [score_col, label_col, *[f"{c}{self.bin_suffix}" for c in self.identity_cols]],
        )
        records = []
        for col in self.identity_cols:
            sub_mask = df[f"{col}{self.bin_suffix}"] == 1
            subgroup = df[sub_mask]
            background = df[~sub_mask]
            if len(subgroup) < self.min_auc_subgroup_size:
                continue
            sg_auc = self._safe_auc(subgroup[label_col], subgroup[score_col])
            bpsn = pd.concat(
                [background[background[label_col] == 1], subgroup[subgroup[label_col] == 0]],
                axis=0,
            )
            bnsp = pd.concat(
                [subgroup[subgroup[label_col] == 1], background[background[label_col] == 0]],
                axis=0,
            )
            bpsn_auc = self._safe_auc(bpsn[label_col], bpsn[score_col])
            bnsp_auc = self._safe_auc(bnsp[label_col], bnsp[score_col])

            values = np.array([sg_auc, bpsn_auc, bnsp_auc], dtype=float)
            pinned_auc = np.nan if np.any(np.isnan(values)) else float(((values**self.pinned_p).mean()) ** (1.0 / self.pinned_p))
            records.append(
                {
                    "identity": col,
                    "n_subgroup": len(subgroup),
                    "subgroup_auc": sg_auc,
                    "bpsn_auc": bpsn_auc,
                    "bnsp_auc": bnsp_auc,
                    "pinned_auc": pinned_auc,
                }
            )
        return pd.DataFrame(records).sort_values("pinned_auc", ascending=True).reset_index(drop=True)

    def compute_statistical_shift_metrics(self, df: pd.DataFrame, score_col: str = "target") -> pd.DataFrame:
        """Compute mean difference, KS statistic, and significance tests."""
        self._check_required_columns(df, [score_col, *[f"{c}{self.bin_suffix}" for c in self.identity_cols]])
        records = []
        for col in self.identity_cols:
            subgroup = df.loc[df[f"{col}{self.bin_suffix}"] == 1, score_col].dropna().to_numpy()
            background = df.loc[df[f"{col}{self.bin_suffix}"] == 0, score_col].dropna().to_numpy()
            if len(subgroup) < self.min_shift_subgroup_size:
                continue
            ks_stat, ks_p = ks_2samp(subgroup, background)
            _, mw_p = mannwhitneyu(subgroup, background, alternative="two-sided")
            records.append(
                {
                    "identity": col,
                    "n_subgroup": len(subgroup),
                    "mean_subgroup": subgroup.mean(),
                    "mean_background": background.mean(),
                    "mean_diff": subgroup.mean() - background.mean(),
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                    "mw_p": mw_p,
                }
            )

        out = pd.DataFrame(records).sort_values("mean_diff", ascending=False).reset_index(drop=True)
        if not out.empty:
            threshold = 0.05 / len(out)
            out["ks_sig"] = out["ks_p"] < threshold
            out["mw_sig"] = out["mw_p"] < threshold
        return out

    def compute_distribution_metrics(
        self,
        df: pd.DataFrame,
        score_col: str = "target",
        bins: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute Wasserstein-1 and KL divergence against background distribution."""
        self._check_required_columns(df, [score_col, *[f"{c}{self.bin_suffix}" for c in self.identity_cols]])
        bins = np.linspace(0, 1, 51) if bins is None else bins
        eps = 1e-10
        records = []
        for col in self.identity_cols:
            subgroup = df.loc[df[f"{col}{self.bin_suffix}"] == 1, score_col].dropna().to_numpy()
            background = df.loc[df[f"{col}{self.bin_suffix}"] == 0, score_col].dropna().to_numpy()
            if len(subgroup) < self.min_shift_subgroup_size:
                continue
            w1 = float(wasserstein_distance(subgroup, background))
            h_sub, _ = np.histogram(subgroup, bins=bins, density=True)
            h_bg, _ = np.histogram(background, bins=bins, density=True)
            kl = float(entropy(h_sub + eps, h_bg + eps))
            records.append({"identity": col, "wasserstein_1": w1, "kl_divergence": kl})
        return pd.DataFrame(records).sort_values("wasserstein_1", ascending=False).reset_index(drop=True)

    def compute_partial_correlations(self, df: pd.DataFrame, score_col: str = "target") -> pd.DataFrame:
        """Compute partial correlation of each identity with score controlling others."""
        self._check_required_columns(df, [score_col, *[f"{c}{self.bin_suffix}" for c in self.identity_cols]])
        matrix = df[[f"{c}{self.bin_suffix}" for c in self.identity_cols]].to_numpy(dtype=float)
        y = df[score_col].to_numpy(dtype=float)
        records = []
        for i, col in enumerate(self.identity_cols):
            ctrl_idx = [j for j in range(len(self.identity_cols)) if j != i]
            x_ctrl = matrix[:, ctrl_idx]
            x = matrix[:, i]
            beta_y = np.linalg.lstsq(x_ctrl, y, rcond=None)[0]
            resid_y = y - x_ctrl @ beta_y
            beta_x = np.linalg.lstsq(x_ctrl, x, rcond=None)[0]
            resid_x = x - x_ctrl @ beta_x
            r, p = stats.pearsonr(resid_x, resid_y)
            records.append({"identity": col, "partial_r": r, "p_value": p})
        return pd.DataFrame(records).sort_values("partial_r", ascending=False).reset_index(drop=True)

    def build_bias_metrics_table(
        self,
        fields: Sequence[str] | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Build the merged fairness table from cached evaluate() outputs.

        Parameters
        ----------
        fields:
            Optional list of metric field names to keep. Supported values:
            `dp_gap`, `subgroup_auc`, `bpsn_auc`, `bnsp_auc`, `pinned_auc`,
            `ks_stat`, `mean_diff`, `wasserstein_1`, `kl_divergence`, `partial_r`.
            If omitted, all fields are included.
        normalize:
            If True, apply min-max normalization per selected metric column.
        """
        if any(x is None for x in [self._dp_df, self._auc_df, self._stat_df, self._dist_df, self._partial_corr_df]):
            raise ValueError("No cached metric tables found. Call evaluate() first.")

        source_map: dict[str, tuple[pd.DataFrame, str]] = {
            "dp_gap": (self._dp_df, "dp_gap"),
            "subgroup_auc": (self._auc_df, "subgroup_auc"),
            "bpsn_auc": (self._auc_df, "bpsn_auc"),
            "bnsp_auc": (self._auc_df, "bnsp_auc"),
            "pinned_auc": (self._auc_df, "pinned_auc"),
            "ks_stat": (self._stat_df, "ks_stat"),
            "mean_diff": (self._stat_df, "mean_diff"),
            "wasserstein_1": (self._dist_df, "wasserstein_1"),
            "kl_divergence": (self._dist_df, "kl_divergence"),
            "partial_r": (self._partial_corr_df, "partial_r"),
        }
        selected_fields = list(source_map.keys()) if fields is None else list(fields)
        invalid_fields = [field for field in selected_fields if field not in source_map]
        if invalid_fields:
            raise ValueError(f"Unsupported fields: {invalid_fields}")
        if not selected_fields:
            raise ValueError("fields must contain at least one metric.")

        merged = self._dp_df[["identity"]].copy()
        for field in selected_fields:
            src_df, src_col = source_map[field]
            merged = merged.merge(src_df[["identity", src_col]], on="identity", how="left")

        for col in ["subgroup_auc", "bpsn_auc", "bnsp_auc", "pinned_auc"]:
            if col in merged.columns:
                merged[col] = 1.0 - merged[col]
        for col in ["dp_gap", "mean_diff", "partial_r"]:
            if col in merged.columns:
                merged[col] = merged[col].abs()

        rename_map = {
            "dp_gap": "DP Gap",
            "subgroup_auc": "1-Subgroup AUC",
            "bpsn_auc": "1-BPSN AUC",
            "bnsp_auc": "1-BNSP AUC",
            "pinned_auc": "1-Pinned AUC",
            "ks_stat": "KS Statistic",
            "mean_diff": "|Mean Diff|",
            "wasserstein_1": "Wasserstein-1",
            "kl_divergence": "KL Divergence",
            "partial_r": "|Partial r|",
        }
        table = merged.set_index("identity").rename(columns=rename_map)
        if not normalize:
            table = table.sort_index()
            self._bias_table = table
            return table

        scaler = MinMaxScaler()
        table_norm = pd.DataFrame(
            scaler.fit_transform(table),
            index=table.index,
            columns=table.columns,
        )
        table_norm["mean_bias"] = table_norm.mean(axis=1)
        table_norm = table_norm.sort_values("mean_bias", ascending=False)
        table_norm = table_norm.drop(columns=["mean_bias"])
        self._bias_table_normalized = table_norm
        return table_norm

    def evaluate(
        self,
        df: pd.DataFrame,
        score_col: str = "target",
        label_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Run the full fairness evaluation pipeline."""
        required_cols = [score_col, *self.identity_cols]
        if label_col is not None:
            required_cols.append(label_col)
        self._check_required_columns(df, required_cols)
        self._bias_table = None
        self._bias_table_normalized = None

        work = self.add_binary_identity_columns(df)
        work["toxic_pred"] = (work[score_col] >= self.toxicity_threshold).astype(int)
        work["toxic_true"] = work["toxic_pred"] if label_col is None else work[label_col].astype(int)

        dp_df = self.compute_demographic_parity(work, label_col="toxic_pred")
        auc_df = self.compute_auc_metrics(work, score_col=score_col, label_col="toxic_true")
        stat_df = self.compute_statistical_shift_metrics(work, score_col=score_col)
        dist_df = self.compute_distribution_metrics(work, score_col=score_col)
        pc_df = self.compute_partial_correlations(work, score_col=score_col)

        self._dp_df = dp_df
        self._auc_df = auc_df
        self._stat_df = stat_df
        self._dist_df = dist_df
        self._partial_corr_df = pc_df

        bias_table = self.build_bias_metrics_table(normalize=False)
        return bias_table, {"dp": dp_df, "auc": auc_df, "stat": stat_df, "dist": dist_df, "partial_corr": pc_df}

    def plot_bias_heatmap(
        self,
        bias_table: pd.DataFrame | None = None,
        *,
        normalize: bool = True,
        sort_by_mean: bool = True,
        figsize: tuple[float, float] = (14, 10),
        cmap: str = "Reds",
        annot: bool = True,
        fmt: str = ".2f",
        title: str = "Fairness Bias Heatmap",
        cbar_label: str = "Bias score (higher = more biased)",
        save_path: str | None = None,
    ) -> tuple[object, object]:
        """Plot a fairness bias heatmap from a metrics table."""
        if bias_table is None:
            bias_table = (
                self._bias_table_normalized
                if self._bias_table_normalized is not None
                else self._bias_table
            )
        if bias_table is None:
            raise ValueError("No cached bias table found. Call evaluate() and build_bias_metrics_table() first.")
        if bias_table.empty:
            raise ValueError("bias_table is empty; nothing to plot.")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "plot_bias_heatmap requires matplotlib and seaborn. Install them first."
            ) from exc

        plot_df = bias_table.copy()
        if "identity" in plot_df.columns:
            plot_df = plot_df.set_index("identity")
        plot_df = plot_df.select_dtypes(include=[np.number])
        if plot_df.empty:
            raise ValueError("bias_table has no numeric columns to plot.")

        if normalize:
            scaler = MinMaxScaler()
            plot_df = pd.DataFrame(
                scaler.fit_transform(plot_df),
                index=plot_df.index,
                columns=plot_df.columns,
            )
        if sort_by_mean:
            plot_df = plot_df.assign(_mean_bias=plot_df.mean(axis=1)).sort_values("_mean_bias", ascending=False)
            plot_df = plot_df.drop(columns="_mean_bias")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            plot_df,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": cbar_label},
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Identity subgroup")
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, ax


DEFAULT_IDENTITY_COLUMNS = list(FairnessBiasEvaluator.DEFAULT_IDENTITY_COLUMNS)

