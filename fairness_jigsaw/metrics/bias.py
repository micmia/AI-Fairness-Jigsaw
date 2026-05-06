"""Reusable fairness bias metrics for Jigsaw-style datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy, ks_2samp, mannwhitneyu, wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

DEFAULT_IDENTITY_COLUMNS = [
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
]


def _check_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_binary_identity_columns(
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    threshold: float = 0.5,
    suffix: str = "_bin",
) -> pd.DataFrame:
    """Return a copy with binary identity columns added.

    Identity values are thresholded with `>= threshold`; NaN values are treated as 0.
    """
    _check_required_columns(df, identity_cols)
    out = df.copy()
    for col in identity_cols:
        out[f"{col}{suffix}"] = (out[col].fillna(0) >= threshold).astype(int)
    return out


def compute_demographic_parity(
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    label_col: str = "toxic",
    bin_suffix: str = "_bin",
) -> pd.DataFrame:
    """Compute demographic parity gap per identity subgroup."""
    _check_required_columns(df, [label_col, *[f"{c}{bin_suffix}" for c in identity_cols]])
    overall_rate = df[label_col].mean()
    records = []
    for col in identity_cols:
        mask = df[f"{col}{bin_suffix}"] == 1
        subgroup = df[mask]
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


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        if pd.Series(y_true).nunique() < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def compute_auc_metrics(
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    score_col: str = "target",
    label_col: str = "toxic",
    bin_suffix: str = "_bin",
    min_subgroup_size: int = 10,
    pinned_p: int = -5,
) -> pd.DataFrame:
    """Compute Subgroup/BPSN/BNSP AUC and pinned AUC for each identity."""
    _check_required_columns(df, [score_col, label_col, *[f"{c}{bin_suffix}" for c in identity_cols]])
    records = []
    for col in identity_cols:
        sub_mask = df[f"{col}{bin_suffix}"] == 1
        subgroup = df[sub_mask]
        background = df[~sub_mask]
        if len(subgroup) < min_subgroup_size:
            continue

        sg_auc = _safe_auc(subgroup[label_col], subgroup[score_col])
        bpsn = pd.concat(
            [background[background[label_col] == 1], subgroup[subgroup[label_col] == 0]],
            axis=0,
        )
        bnsp = pd.concat(
            [subgroup[subgroup[label_col] == 1], background[background[label_col] == 0]],
            axis=0,
        )
        bpsn_auc = _safe_auc(bpsn[label_col], bpsn[score_col])
        bnsp_auc = _safe_auc(bnsp[label_col], bnsp[score_col])

        values = np.array([sg_auc, bpsn_auc, bnsp_auc], dtype=float)
        if np.any(np.isnan(values)):
            pinned_auc = np.nan
        else:
            pinned_auc = float(((values**pinned_p).mean()) ** (1.0 / pinned_p))

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


def compute_statistical_shift_metrics(
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    score_col: str = "target",
    bin_suffix: str = "_bin",
    min_subgroup_size: int = 20,
) -> pd.DataFrame:
    """Compute mean difference, KS statistic, and significance tests."""
    _check_required_columns(df, [score_col, *[f"{c}{bin_suffix}" for c in identity_cols]])
    records = []
    for col in identity_cols:
        subgroup = df.loc[df[f"{col}{bin_suffix}"] == 1, score_col].dropna().to_numpy()
        background = df.loc[df[f"{col}{bin_suffix}"] == 0, score_col].dropna().to_numpy()
        if len(subgroup) < min_subgroup_size:
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
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    score_col: str = "target",
    bin_suffix: str = "_bin",
    min_subgroup_size: int = 20,
    bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute Wasserstein-1 and KL divergence against background distribution."""
    _check_required_columns(df, [score_col, *[f"{c}{bin_suffix}" for c in identity_cols]])
    bins = np.linspace(0, 1, 51) if bins is None else bins
    eps = 1e-10
    records = []
    for col in identity_cols:
        subgroup = df.loc[df[f"{col}{bin_suffix}"] == 1, score_col].dropna().to_numpy()
        background = df.loc[df[f"{col}{bin_suffix}"] == 0, score_col].dropna().to_numpy()
        if len(subgroup) < min_subgroup_size:
            continue
        w1 = float(wasserstein_distance(subgroup, background))
        h_sub, _ = np.histogram(subgroup, bins=bins, density=True)
        h_bg, _ = np.histogram(background, bins=bins, density=True)
        kl = float(entropy(h_sub + eps, h_bg + eps))
        records.append({"identity": col, "wasserstein_1": w1, "kl_divergence": kl})
    return pd.DataFrame(records).sort_values("wasserstein_1", ascending=False).reset_index(drop=True)


def compute_partial_correlations(
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    score_col: str = "target",
    bin_suffix: str = "_bin",
) -> pd.DataFrame:
    """Compute partial correlation of each identity with score controlling others."""
    _check_required_columns(df, [score_col, *[f"{c}{bin_suffix}" for c in identity_cols]])
    matrix = df[[f"{c}{bin_suffix}" for c in identity_cols]].to_numpy(dtype=float)
    y = df[score_col].to_numpy(dtype=float)
    records = []
    for i, col in enumerate(identity_cols):
        ctrl_idx = [j for j in range(len(identity_cols)) if j != i]
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
    dp_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    partial_corr_df: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """Merge multiple fairness metrics into one table.

    If `normalize=True`, output a min-max normalized table where larger values mean
    stronger bias across metrics.
    """
    merged = dp_df[["identity", "dp_gap"]].merge(
        auc_df[["identity", "subgroup_auc", "bpsn_auc", "bnsp_auc", "pinned_auc"]],
        on="identity",
        how="left",
    )
    merged = merged.merge(stat_df[["identity", "ks_stat", "mean_diff"]], on="identity", how="left")
    merged = merged.merge(dist_df[["identity", "wasserstein_1", "kl_divergence"]], on="identity", how="left")
    merged = merged.merge(partial_corr_df[["identity", "partial_r"]], on="identity", how="left")

    for col in ["subgroup_auc", "bpsn_auc", "bnsp_auc", "pinned_auc"]:
        merged[col] = 1.0 - merged[col]
    for col in ["dp_gap", "mean_diff", "partial_r"]:
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
        return table.sort_index()

    scaler = MinMaxScaler()
    table_norm = pd.DataFrame(
        scaler.fit_transform(table),
        index=table.index,
        columns=table.columns,
    )
    table_norm["mean_bias"] = table_norm.mean(axis=1)
    table_norm = table_norm.sort_values("mean_bias", ascending=False)
    return table_norm.drop(columns=["mean_bias"])


def evaluate_fairness_bias(
    df: pd.DataFrame,
    identity_cols: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    score_col: str = "target",
    toxicity_threshold: float = 0.5,
    identity_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the full bias evaluation pipeline and return final plus component tables."""
    _check_required_columns(df, [score_col, *identity_cols])
    work = add_binary_identity_columns(df, identity_cols=identity_cols, threshold=identity_threshold)
    work["toxic"] = (work[score_col] >= toxicity_threshold).astype(int)

    dp_df = compute_demographic_parity(work, identity_cols=identity_cols, label_col="toxic")
    auc_df = compute_auc_metrics(work, identity_cols=identity_cols, score_col=score_col, label_col="toxic")
    stat_df = compute_statistical_shift_metrics(work, identity_cols=identity_cols, score_col=score_col)
    dist_df = compute_distribution_metrics(work, identity_cols=identity_cols, score_col=score_col)
    pc_df = compute_partial_correlations(work, identity_cols=identity_cols, score_col=score_col)

    bias_table = build_bias_metrics_table(dp_df, auc_df, stat_df, dist_df, pc_df, normalize=False)
    return bias_table, {
        "dp": dp_df,
        "auc": auc_df,
        "stat": stat_df,
        "dist": dist_df,
        "partial_corr": pc_df,
    }

