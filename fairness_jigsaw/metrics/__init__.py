"""Fairness and bias metrics utilities."""

from .bias import (
    DEFAULT_IDENTITY_COLUMNS,
    add_binary_identity_columns,
    build_bias_metrics_table,
    compute_auc_metrics,
    compute_demographic_parity,
    compute_distribution_metrics,
    compute_partial_correlations,
    compute_statistical_shift_metrics,
    evaluate_fairness_bias,
)

__all__ = [
    "DEFAULT_IDENTITY_COLUMNS",
    "add_binary_identity_columns",
    "compute_demographic_parity",
    "compute_auc_metrics",
    "compute_statistical_shift_metrics",
    "compute_distribution_metrics",
    "compute_partial_correlations",
    "build_bias_metrics_table",
    "evaluate_fairness_bias",
]

