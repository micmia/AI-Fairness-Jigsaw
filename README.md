# AI-Fairness-Jigsaw

Toxicity classification on the Jigsaw Unintended Bias dataset, with fairness mitigation, interpretability (SHAP), and robustness analysis.

We compare a GRU baseline against three mitigation strategies — reweighting, adversarial (gradient reversal), and a combined approach — across 18 identity subgroups, then study how each variant degrades under input perturbations.

## Quickstart

Requires Python 3.14+. Dependencies are managed with `[uv](https://docs.astral.sh/uv/)`.

```bash
git clone https://github.com/<you>/AI-Fairness-Jigsaw.git
cd AI-Fairness-Jigsaw
uv sync
uv run jupyter lab
```

Download the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
dataset and place `train.csv` (plus the other CSVs you want) under `data/`.

## Repository layout

```
fairness_jigsaw/           # Reusable Python package
    metrics/bias.py        # ModelBiasEvaluator: AUC variants, FPR, ECE, counterfactual gap

notebooks/
    00_eda_fairness.ipynb              # EDA: identity prevalence, label bias, residualisation
    01_baseline_gru.ipynb              # GRU baseline
    02_baseline_bert.ipynb             # BERT baseline (deprecated)
    03_mitigation_bert.ipynb           # Reweighting / GRL on BERT (deprecated)
    04_mitigation_gru_preprocessing.ipynb   # Reweighting (preprocessing)
    05_mitigation_gru_inprocessing.ipynb    # Adversarial + GRL (in-processing)
    06_mitigation_gru_combined.ipynb        # Reweighting + adversarial
    07_full_pipeline.ipynb             # End-to-end run
    final.ipynb                        # Consolidated results, figures, robustness

models/                    # Trained checkpoints (gru/, bert/, per-strategy subdirs)
data/                      # Raw CSVs, split_ids.json
reports/                   # LaTeX source and compiled PDF
```

## Methods


| Strategy    | Where it acts        | Idea                                                                               |
| ----------- | -------------------- | ---------------------------------------------------------------------------------- |
| Baseline    | —                    | ToxicityGRU, plain cross-entropy.                                                  |
| Reweighting | Preprocessing (loss) | Per-example weights inverse to per-identity toxicity rate, clipped to `[0.1, 10]`. |
| Adversarial | In-processing        | FairGRU + identity discriminator with gradient reversal (GRL).                     |
| Combined    | Pre + in-processing  | Reweighting and adversarial together.                                              |


## Metrics

`fairness_jigsaw.metrics.bias.ModelBiasEvaluator` computes per-subgroup:

- **AUC variants** — subgroup AUC, BPSN AUC, BNSP AUC, pinned AUC
- **FPR gap** — over-flagging of non-toxic comments per identity
- **ECE** — calibration error, overall and per subgroup
- **Counterfactual gap** — score shift from swapping identity tokens

Plus interpretability (`SHAP`, identity vs. non-identity token attribution) and robustness (AUC under character noise, word dropout, sequence truncation).

## Reproducing the results

The shortest path is to run `notebooks/final.ipynb` end-to-end after `uv sync`. Splits are frozen via `data/split_ids.json` (80/20 stratified, `random_state=42`), so the train / test partition is identical across runs.

## Report

The full write-up (French) lives in `reports/rapport_projet_jigsaw.pdf`.

