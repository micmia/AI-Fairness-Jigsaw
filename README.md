# AI-Fairness-Jigsaw

Toxicity classification on the Jigsaw Unintended Bias dataset with fairness
mitigation, SHAP interpretability, and robustness analysis.

- **Consolidated notebook:** [`notebooks/10-final.ipynb`](notebooks/10-final.ipynb)
- **Full report (FR):** [`reports/rapport_projet_jigsaw.pdf`](reports/rapport_projet_jigsaw.pdf)

A GRU baseline is compared against three mitigation strategies — **reweighting**,
**adversarial (GRL)**, and a **combined** approach — across 18 identity subgroups,
plus robustness under character noise, word dropout, and truncation.

> **Note on scope.** Beyond the primary **GRU track**, we also implemented and
> evaluated a **BERT track**.
> Given the additional complexity of analysing BERT and the fact that the model
> itself is not the focus of this study, the BERT results are kept in this
> repository as supporting material but are **not included in the written
> report** — the report only covers the GRU pipeline.

## Quickstart

Requires Python 3.14+. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab    # then open notebooks/10-final.ipynb
```

Place the [Jigsaw Unintended Bias](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
`train.csv` under `data/`. Splits are frozen via `data/split_ids.json`
(80/20 stratified, `random_state=42`).

## Layout

```
fairness_jigsaw/
    metrics/
        bias.py                            # ModelBiasEvaluator: AUC variants, FPR gap, ECE,
                                           #   counterfactual identity-swap gap

notebooks/
    # Dataset audit
    00_eda_fairness.ipynb                  # Dataset-level bias: identity prevalence, label
                                           #   bias, residualisation (no model)

    # Primary track — GRU
    01_baseline_gru.ipynb                  # ToxicityGRU baseline
    04_mitigation_gru_preprocessing.ipynb  # Reweighting (pre-processing)
    05_mitigation_gru_inprocessing.ipynb   # Adversarial + gradient reversal (in-processing)
    06_mitigation_gru_combined.ipynb       # Reweighting + adversarial
    07_full_pipeline.ipynb                 # End-to-end GRU: EDA → train → fairness → SHAP → robustness
    10-final.ipynb                         # Consolidated results, figures, final analysis

    # Secondary track — BERT
    02_baseline_bert.ipynb                 # BERT baseline
    03_mitigation_bert.ipynb               # Reweighting + FRL on BERT (initial)
    03_mitigation_bert_variante.ipynb      # In-processing mitigation on BERT (refined)
    08_xai_shap_bert_variante.ipynb        # SHAP interpretability on the BERT variant
    09_robustness_bert_variante.ipynb      # Robustness analysis on the BERT variant

data/                                      # Raw CSVs, split_ids.json, EDA figures (gitignored)
reports/                                   # LaTeX source + compiled PDF
```

