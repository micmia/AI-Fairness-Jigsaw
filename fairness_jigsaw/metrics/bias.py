"""Model-level fairness bias metrics for Jigsaw-style toxicity classifiers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_IDENTITY_COLUMNS: tuple[str, ...] = (
    "male", "female", "transgender", "other_gender",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
)

_DEFAULT_NEUTRAL_TERM = "person"


@dataclass
class ModelBiasEvaluator:
    """Model-level fairness metrics evaluator.

    Computes four complementary views of model bias per identity subgroup:

    | Method              | What it measures                                      |
    |---------------------|-------------------------------------------------------|
    | AUC variants        | Discrimination ability for/against each subgroup      |
    | Subgroup FPR        | Over-flagging of non-toxic comments per group         |
    | ECE                 | Calibration quality overall and per group             |
    | Counterfactual gap  | Score change from swapping one identity term for another |
    """

    identity_cols: Sequence[str] = field(default_factory=lambda: list(DEFAULT_IDENTITY_COLUMNS))
    toxicity_threshold: float = 0.5
    identity_threshold: float = 0.5
    min_subgroup_size: int = 10
    pinned_p: int = -5        # power-mean exponent for pinned AUC; more negative → penalises min more
    n_ece_bins: int = 10

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
        if missing := [c for c in cols if c not in df.columns]:
            raise ValueError(f"Missing columns: {missing}")

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return np.nan
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return np.nan

    def _bin_cols(self) -> list[str]:
        return [f"{c}_bin" for c in self.identity_cols]

    def _ece(self, scores: np.ndarray, labels: np.ndarray) -> float:
        boundaries = np.linspace(0.0, 1.0, self.n_ece_bins + 1)
        n = len(scores)
        ece = 0.0
        for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            in_bin = (scores >= lo) & (scores <= hi if i == self.n_ece_bins - 1 else scores < hi)
            if not in_bin.any():
                continue
            ece += (in_bin.sum() / n) * abs(scores[in_bin].mean() - labels[in_bin].mean())
        return float(ece)

    # ── preprocessing ─────────────────────────────────────────────────────────

    def add_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with binarised identity columns (*_bin) appended."""
        self._require_cols(df, self.identity_cols)
        out = df.copy()
        for col in self.identity_cols:
            out[f"{col}_bin"] = (out[col].fillna(0) >= self.identity_threshold).astype(int)
        return out

    # ── AUC variants ──────────────────────────────────────────────────────────

    def compute_auc_metrics(
        self,
        df: pd.DataFrame,
        score_col: str = "score",
        label_col: str = "toxic",
    ) -> pd.DataFrame:
        """Subgroup, BPSN, BNSP, and pinned AUC per identity subgroup.

        - Subgroup AUC: AUC restricted to the identity subgroup.
        - BPSN AUC: background-positive vs subgroup-negative.
          Low value → model over-predicts toxicity for the subgroup.
        - BNSP AUC: subgroup-positive vs background-negative.
          Low value → model under-predicts toxicity for the subgroup.
        - Pinned AUC: power mean of the three (p=pinned_p), emphasising the worst.
        """
        self._require_cols(df, [score_col, label_col, *self._bin_cols()])
        records = []
        for col in self.identity_cols:
            mask = df[f"{col}_bin"] == 1
            sub, bg = df[mask], df[~mask]
            if len(sub) < self.min_subgroup_size:
                continue

            sg = self._safe_auc(sub[label_col].to_numpy(), sub[score_col].to_numpy())
            bpsn = pd.concat([bg[bg[label_col] == 1], sub[sub[label_col] == 0]])
            bnsp = pd.concat([sub[sub[label_col] == 1], bg[bg[label_col] == 0]])
            bpsn_auc = self._safe_auc(bpsn[label_col].to_numpy(), bpsn[score_col].to_numpy())
            bnsp_auc = self._safe_auc(bnsp[label_col].to_numpy(), bnsp[score_col].to_numpy())

            vals = np.array([sg, bpsn_auc, bnsp_auc])
            valid = not np.isnan(vals).any() and (vals > 0).all()
            pinned = float((vals ** self.pinned_p).mean() ** (1.0 / self.pinned_p)) if valid else np.nan

            records.append({
                "identity": col, "n": len(sub),
                "subgroup_auc": sg, "bpsn_auc": bpsn_auc,
                "bnsp_auc": bnsp_auc, "pinned_auc": pinned,
            })

        return pd.DataFrame(records).sort_values("pinned_auc", ascending=True).reset_index(drop=True)

    # ── Subgroup FPR ──────────────────────────────────────────────────────────

    def compute_subgroup_fpr(
        self,
        df: pd.DataFrame,
        score_col: str = "score",
        label_col: str = "toxic",
    ) -> pd.DataFrame:
        """False positive rate per identity subgroup at toxicity_threshold.

        FPR = FP / (FP + TN): fraction of truly non-toxic comments in the subgroup
        that the model flags as toxic. fpr_gap = subgroup FPR − background FPR;
        a positive gap indicates the model over-triggers on that group.
        """
        self._require_cols(df, [score_col, label_col, *self._bin_cols()])
        pred = (df[score_col] >= self.toxicity_threshold).astype(int)
        records = []
        for col in self.identity_cols:
            mask = df[f"{col}_bin"] == 1
            sub_neg = mask & (df[label_col] == 0)
            if sub_neg.sum() < self.min_subgroup_size:
                continue
            fpr = float(pred[sub_neg].mean())
            bg_neg = ~mask & (df[label_col] == 0)
            bg_fpr = float(pred[bg_neg].mean()) if bg_neg.sum() >= self.min_subgroup_size else np.nan
            records.append({
                "identity": col,
                "n_negatives": int(sub_neg.sum()),
                "fpr": fpr,
                "bg_fpr": bg_fpr,
                "fpr_gap": fpr - bg_fpr if not np.isnan(bg_fpr) else np.nan,
            })

        return pd.DataFrame(records).sort_values("fpr_gap", ascending=False).reset_index(drop=True)

    # ── ECE ───────────────────────────────────────────────────────────────────

    def compute_ece(
        self,
        df: pd.DataFrame,
        score_col: str = "score",
        label_col: str = "toxic",
    ) -> pd.DataFrame:
        """Expected calibration error overall and per identity subgroup.

        Bins predicted probabilities into n_ece_bins equal-width buckets and
        computes the weighted mean of |mean_confidence − fraction_positive| per bin.
        The first row is always the overall ECE across all examples.
        """
        self._require_cols(df, [score_col, label_col, *self._bin_cols()])
        scores = df[score_col].to_numpy(dtype=float)
        labels = df[label_col].to_numpy(dtype=float)

        records = [{"identity": "overall", "n": len(df), "ece": self._ece(scores, labels)}]
        for col in self.identity_cols:
            mask = df[f"{col}_bin"].to_numpy() == 1
            if mask.sum() < self.min_subgroup_size:
                continue
            records.append({"identity": col, "n": int(mask.sum()), "ece": self._ece(scores[mask], labels[mask])})

        return pd.DataFrame(records).sort_values("ece", ascending=False).reset_index(drop=True)

    # ── Counterfactual gap ────────────────────────────────────────────────────

    def compute_counterfactual_gap(
        self,
        df: pd.DataFrame,
        text_col: str,
        score_col: str,
        predict_fn: Callable[[list[str]], np.ndarray],
        neutral_term: str | None = _DEFAULT_NEUTRAL_TERM,
        swap_pairs: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Mean absolute score change from whole-word identity-term substitution.

        Two comparison modes, each producing a row per (term_a, term_b) match:

        - **neutral** (default, type="neutral"): replaces each identity term with
          neutral_term (e.g. "person"). Measures the *absolute* effect of mentioning
          an identity — unaffected by whether the reference term is itself biased.
        - **swap** (type="swap"): replaces term_a with term_b. Measures *relative*
          difference, but is misleading when both terms are biased in the same
          direction (gaps cancel out). Use only as a complement to neutral mode.

        Each swap pair is tested bidirectionally (A→B and B→A).
        Rows with fewer than min_subgroup_size matches are skipped.

        Parameters
        ----------
        neutral_term:
            Replacement word for neutral substitution. Set to None to skip.
        swap_pairs:
            Explicit (term_a, term_b) pairs for relative comparison. Optional.
        """
        self._require_cols(df, [text_col, score_col])

        # Build (term_a, term_b, type) triples
        triples: list[tuple[str, str, str]] = []
        if neutral_term is not None:
            triples += [(col, neutral_term, "neutral") for col in self.identity_cols]
        if swap_pairs is not None:
            triples += [(a, b, "swap") for a, b in swap_pairs]
            triples += [(b, a, "swap") for a, b in swap_pairs]

        records = []
        for term_a, term_b, kind in triples:
            pattern = re.compile(rf"\b{re.escape(term_a)}\b", re.IGNORECASE)
            mask = df[text_col].str.contains(pattern, na=False)
            subset = df[mask]
            if len(subset) < self.min_subgroup_size:
                continue
            cf_texts = [pattern.sub(term_b, t) for t in subset[text_col]]
            cf_scores = np.asarray(predict_fn(cf_texts), dtype=float)
            orig_scores = subset[score_col].to_numpy(dtype=float)
            gap = np.abs(cf_scores - orig_scores)
            records.append({
                "type": kind, "term_a": term_a, "term_b": term_b,
                "n_pairs": len(subset),
                "mean_gap": float(gap.mean()),
                "max_gap": float(gap.max()),
            })

        return (
            pd.DataFrame(records)
            .sort_values(["type", "mean_gap"], ascending=[True, False])
            .reset_index(drop=True)
        )

    # ── full pipeline ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        df: pd.DataFrame,
        score_col: str = "score",
        label_col: str = "toxic",
        text_col: str | None = None,
        predict_fn: Callable[[list[str]], np.ndarray] | None = None,
        neutral_term: str | None = _DEFAULT_NEUTRAL_TERM,
        swap_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run the full model-bias evaluation pipeline.

        Returns a dict with keys 'auc', 'fpr', 'ece', and — when both text_col
        and predict_fn are supplied — 'cf_gap'.
        """
        work = self.add_binary_columns(df)
        results: dict[str, pd.DataFrame] = {
            "auc": self.compute_auc_metrics(work, score_col=score_col, label_col=label_col),
            "fpr": self.compute_subgroup_fpr(work, score_col=score_col, label_col=label_col),
            "ece": self.compute_ece(work, score_col=score_col, label_col=label_col),
        }
        if text_col is not None and predict_fn is not None:
            results["cf_gap"] = self.compute_counterfactual_gap(
                work, text_col=text_col, score_col=score_col,
                predict_fn=predict_fn, neutral_term=neutral_term, swap_pairs=swap_pairs,
            )
        return results
