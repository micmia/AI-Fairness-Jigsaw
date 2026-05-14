# AI-Fairness-Jigsaw

Toxicity classification on the Jigsaw Unintended Bias dataset with fairness
mitigation, SHAP interpretability, and robustness analysis.

- **Consolidated notebook:** [`notebooks/10-final.ipynb`](notebooks/10-final.ipynb)
- **Full report (FR):** `[reports/rapport_projet_jigsaw.pdf](reports/rapport_projet_jigsaw.pdf)`

A GRU baseline is compared against three mitigation strategies — **reweighting**,
**adversarial (GRL)**, and a **combined** approach — across 18 identity subgroups,
plus robustness under character noise, word dropout, and truncation.

## Quickstart

Requires Python 3.14+. Dependencies are managed with `[uv](https://docs.astral.sh/uv/)`.

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
    00_eda_fairness.ipynb                  # Identity prevalence, label bias, residualisation
    01_baseline_gru.ipynb                  # ToxicityGRU baseline
    02_baseline_bert.ipynb                 # BERT baseline (deprecated)
    03_mitigation_bert.ipynb               # Reweighting and FRL on BERT (deprecated)
    04_mitigation_gru_preprocessing.ipynb  # Reweighting only
    05_mitigation_gru_inprocessing.ipynb   # Adversarial + gradient reversal
    06_mitigation_gru_combined.ipynb       # Reweighting + adversarial
    07_full_pipeline.ipynb                 # End-to-end run
    10-final.ipynb                         # Consolidated results, figures, robustness

data/                                      # Raw CSVs, split_ids.json, EDA figures (gitignored)
reports/                                   # LaTeX source + compiled PDF
```

