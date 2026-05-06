# Roadmap & Plan d'Implémentation
## Mini-projet IA Responsable — Jigsaw Unintended Bias in Toxicity Classification

---

## 1. Cadrage

### Question de recherche
> *Comment équilibrer performance prédictive, équité algorithmique et robustesse dans un classifieur de toxicité textuelle, et quels sont les compromis observables entre ces trois axes ?*

### Cas d'étude
- **Jeu de données** : Jigsaw Unintended Bias in Toxicity Classification (~1,8 M commentaires Civil Comments).
- **Tâche** : classification binaire `toxic` (seuil `target ≥ 0.5`).
- **Attribut sensible** : mention d'identité démographique (24 colonnes : genre, orientation sexuelle, religion, race/ethnie, handicap).

### Choix méthodologiques (à confirmer)

| Axe | Choix retenu | Justification |
|---|---|---|
| **Modèle** | BERT (`bert-base-uncased`) fine-tuné, tête sigmoïde 1 logit, perte BCE | Capture le contexte bidirectionnel ; meilleur compromis perf./VRAM que les CNN/Bi-LSTM ; standard en littérature |
| **Métriques perf.** | Overall ROC-AUC, ECE | Insensibles au seuil de classification ; ECE quantifie la calibration |
| **Métriques équité** | Subgroup AUC, BPSN AUC, BNSP AUC, Pinned AUC, Jigsaw final score (moyenne généralisée *p* = −5) | Standard de la compétition Kaggle, capture asymétries FP/FN par sous-groupe |
| **Mitigation équité** | **In-processing** : (a) BCE pondérée (pénalités BPSN/BNSP) + (b) tête multi-tâches identités | Pas de re-sampling qui dénature les distributions ; force la séparation latente toxicité ⊥ identité |
| **Interprétabilité** | SHAP (token-level) + agrégation globale | Attributions parcimoniaires au niveau token ; idéal pour auditer le poids des termes identitaires |
| **Robustesse** | (a) bruit caractère contrôlé, (b) attaques sémantiques TextFooler via TextAttack | Couvre bruit naturel **et** attaques d'évasion ; mesure ASR + dégradation AUC + dégradation ECE |
| **Reproductibilité** | seed fixe, splits stratifiés persistés, configs YAML, artefacts versionnés | Conformité aux checklists ACL/NeurIPS |

---

## 2. Existant déjà disponible (`projet/`)

| Composant | Emplacement | État |
|---|---|---|
| EDA équité au niveau **données** | `AI-Fairness-Jigsaw/notebooks/fairness_metrics.ipynb` | ✅ Audit DP gap, KS, Wasserstein, KL, partial corr — **niveau données uniquement** (le score `target` est utilisé comme prédicteur, ce qui force AUC=1) |
| Pipeline `src/` complet | `mini_projet_jigsaw/src/` | ✅ Modules `data`, `training` (BCE + multi-task), `models` (BERT multi-tâches), `metrics` (Subgroup/BPSN/BNSP, ECE), `xai` (SHAP, IG), `robustness` (noise, TextAttack) |
| Scripts d'orchestration | `mini_projet_jigsaw/scripts/` | ✅ `prepare_splits.py`, `train_baseline.py`, `train_fair.py`, `eval.py`, `run_xai.py`, `run_robustness.py`, `run_ig.py` |
| Configs YAML | `mini_projet_jigsaw/configs/` | ✅ `base.yaml`, `dev_small.yaml`, `smoke.yaml`, `smoke_fair_*.yaml` |
| Notebook baseline | `mini_projet_jigsaw/notebooks/colab_baseline.ipynb` | ⚠️ Minimal (juste `!python …`), pas pédagogique |
| État de l'art | `Etat_art.md` | ✅ Rédigé (introduction + SOTA fairness/XAI/robustesse) |

**Constat** : on dispose déjà de quasiment tout le code. Le travail restant est principalement (a) un notebook baseline pédagogique et exécutable bout-en-bout, (b) la reproduction expérimentale, (c) l'analyse comparative et le rapport.

---

## 3. Roadmap (jalons J = jour calendaire)

| Phase | Tâches | Livrables | Durée |
|---|---|---|---|
| **P0 — Cadrage** ✅ | Choix dataset, modèle, méthodes ; prise en main de l'existant | Ce document | — |
| **P1 — Baseline BERT** | Notebook bout-en-bout, run de référence, sauvegarde artefacts (modèle + prédictions test + métriques) | `02_baseline_bert.ipynb`, `reports/checkpoints/baseline_bert/`, `reports/metrics/baseline_metrics.json`, `reports/metrics/test_predictions.parquet` | **J +1 → J +2** |
| **P2 — Mitigation équité (in-processing)** | (a) BCE pondérée BPSN/BNSP, (b) multi-tâches identités, (c) éventuelle 3ème variante (e.g. adversarial), comparaison avec baseline sur les mêmes splits | `03_fairness_inprocessing.ipynb`, `reports/checkpoints/fair_*/`, `reports/metrics/fair_*_metrics.json` | **J +3 → J +5** |
| **P3 — Interprétabilité (SHAP)** | (a) explications locales (waterfall) sur ~30 commentaires *avant/après* mitigation incluant des cas BPSN, (b) agrégation globale (top tokens absolus + tokens identitaires), (c) audit qualitatif du poids des termes identitaires | `04_xai_shap.ipynb`, `reports/figures/shap_local_*.html`, `reports/figures/shap_global_topk.png`, `reports/metrics/identity_token_attribution.json` | **J +6 → J +8** |
| **P4 — Robustesse** | (a) bruit caractère sur l'ensemble test (typo, casse aléatoire), (b) attaque TextFooler sur ~500 ex. toxiques, (c) calcul ASR, ΔAUC, ΔECE *baseline vs fair* | `05_robustness.ipynb`, `reports/metrics/robustness_*.json`, `reports/figures/asr_vs_perturbation.png` | **J +9 → J +10** |
| **P5 — Analyse comparative & rapport** | (a) tableau récapitulatif perf./équité/robustesse (3 axes × 2 modèles), (b) discussion compromis (ex. coût AUC du fair, coût équité de la robustesse), (c) limites et travaux futurs | `06_synthesis.ipynb`, `rapport.pdf` | **J +11 → J +13** |

---

## 4. Plan d'implémentation détaillé

### Phase 1 — Baseline BERT (priorité immédiate)

**Architecture**
- Encodeur : `bert-base-uncased` (12 couches, 768 dim, ~110 M paramètres).
- Tête : `Linear(768, 1)` sur `pooler_output`.
- Activation : sigmoïde au moment de l'inférence, BCEWithLogitsLoss en entraînement (stabilité numérique).

**Pré-traitement**
- Tronquage à `max_length = 128` (la majorité des commentaires Civil Comments < 100 tokens).
- Padding `max_length` (compatible avec un `DataCollator` simple).

**Splits stratifiés**
- Clé de stratification 4-classes : `2 * y + any_identity`.
- Ratio 80 / 10 / 10 (train / val / test).
- Seed fixe (`1337`). IDs sauvegardés dans `reports/splits/split_ids.json`.

**Hyperparamètres** (alignés sur l'état de l'art Jigsaw)

| Paramètre | Valeur |
|---|---|
| Optimiseur | AdamW |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.06` |
| Batch size (train) | 32 (adapter au VRAM) |
| Gradient accumulation | 1 |
| FP16 | ✅ si CUDA |
| Epochs | 2 |
| `eval_steps` / `save_steps` | 1000 |
| Best model | `metric_for_best_model="roc_auc"` |

**Mode itération rapide**
- Variable `N_ROWS` paramétrable (par défaut 300 k pour itération en quelques minutes ; mettre `None` pour les 1,8 M complets ≈ plusieurs heures sur RTX 4080).

**Métriques persistées**
- `overall_auc`, `ece` (15 bins).
- Pour chaque identité avec ≥ 500 occurrences en test : `subgroup_auc`, `bpsn_auc`, `bnsp_auc`, `n_subgroup`.
- Score Jigsaw final (moyenne arithmétique de l'overall AUC et des 3 moyennes généralisées avec *p* = −5).

**Artefacts**
- `reports/checkpoints/baseline_bert/` (modèle + tokenizer)
- `reports/metrics/baseline_metrics.json`
- `reports/metrics/test_predictions.parquet` (id, texte, label, prob, logit) — réutilisé par P3 et P4
- `reports/figures/baseline_*.png`

### Phase 2 — Équité (in-processing)

**Variante A : BCE pondérée par groupes (BPSN/BNSP)**

Pour chaque exemple, calculer un poids `w_i` :
- 1.0 par défaut.
- 1.0 + `λ_FP` si exemple non-toxique mentionnant une identité (groupe BPSN à risque de FP).
- 1.0 + `λ_FN` si exemple toxique sans mention d'identité (groupe BNSP à risque de FN — mais ici on cherche surtout à réhausser la détection des toxiques généraux pour rééquilibrer le signal).

Hyperparamètres : `λ_FP = λ_FN = 0.25` (suivant Etat_art.md).

Implémentation : déjà disponible dans `mini_projet_jigsaw/src/training/bce_trainer.py` via la clé `loss_weight`.

**Variante B : Multi-tâches identités**

Tête auxiliaire : `Linear(768, n_identities)` qui prédit la présence des 24 identités (BCE par identité).
- Loss totale : `L_tox + α · L_identity`, `α = 0.1`.
- L'idée : forcer l'encodeur à **représenter** explicitement l'identité — donc à séparer le signal toxique de l'attribut sensible, plutôt que de les fusionner.

Implémentation : `BertForToxicityAndIdentity` + `MultiTaskBCETrainer` (déjà dans `mini_projet_jigsaw/src/`).

**Variante C (optionnelle) : Adversarial debiasing**

Discriminateur entraîné à prédire l'identité depuis l'embedding ; perte adversaire ajoutée pour confondre le discriminateur. À considérer seulement si P2-A et P2-B montrent des limites.

**Évaluation**
- Mêmes métriques que la baseline, calculées sur **exactement les mêmes splits**.
- Tableau comparatif : Δoverall_auc, Δjigsaw_score, Δsubgroup_auc moyen, Δbpsn_auc min (le pire cas).

### Phase 3 — Interprétabilité (SHAP)

**Échantillonnage**
- 30 commentaires test stratifiés par (toxicité × mention d'identité), pour comparer cartes de saillance baseline vs fair.
- Inclure obligatoirement quelques cas BPSN (non-toxiques avec identité, à risque de FP) et BNSP (toxiques sans identité).

**Local**
- `shap.Explainer` avec `shap.maskers.Text(tokenizer)`, `max_evals = 400`.
- Export HTML par exemple (`shap.plots.text`) pour le rapport.

**Global**
- Agrégation : pour chaque token, moyenne |SHAP| sur l'échantillon. Top-30.
- Métrique d'audit : ratio (importance moyenne des tokens identitaires) / (importance moyenne globale). Compare baseline vs fair.

**Critère de succès attendu** : sur le modèle fair, les tokens identitaires (`muslim`, `gay`, `black`…) doivent voir leur poids absolu chuter par rapport au baseline.

### Phase 4 — Robustesse

**Perturbations contrôlées (bruit naturel)**
- `char_typo_noise(p=0.05)` : substitution / suppression caractère stochastique.
- `random_case(p=0.2)` : altération aléatoire de casse.
- `remove_whitespace` : compression sans espaces (cas extrême).
- Calculer pour chaque type : ΔAUC, ΔECE, fraction de prédictions changées (`flip rate`).

**Attaques sémantiques (adversarial)**
- TextAttack — recette `TextFooler` (substitution synonymes guidée par le modèle).
- Échantillon : 500 commentaires toxiques mal classés mais avec haute confiance.
- Mesurer : Attack Success Rate (ASR), nombre moyen de mots modifiés, similarité USE post-attaque.

**Comparatif**
- Baseline vs fair model : la mitigation équité réduit-elle aussi (ou aggrave-t-elle) la vulnérabilité aux attaques ?
- C'est ici qu'apparaît un compromis classique : la **robustesse à l'identité** (équité) peut nuire à la **robustesse adversaire** ou à la calibration.

### Phase 5 — Synthèse

**Tableau central** (à inclure dans le rapport)

| Modèle | Overall AUC | ECE | Jigsaw score | min Subgroup AUC | min BPSN AUC | ASR TextFooler | ΔAUC sous bruit |
|---|---|---|---|---|---|---|---|
| Baseline | … | … | … | … | … | … | … |
| Fair (weighted) | … | … | … | … | … | … | … |
| Fair (multitask) | … | … | … | … | … | … | … |

**Discussion attendue**
- Compromis perf./équité : le coût d'AUC global du fair model.
- Compromis équité/robustesse : la mitigation transfère-t-elle les vulnérabilités ?
- Apport de l'XAI : confirme-t-on visuellement que les biais identitaires sont atténués ?
- Limites : couverture des identités (avec moins de 500 ex. de test elles sont écartées), généralisation hors-domaine.

---

## 5. Reproductibilité

- **Seeds** : tout est piloté par `SEED = 1337` (numpy, torch, hf, dataloaders).
- **Configs** : YAML versionnés ; un run = un fichier de config.
- **Splits** : IDs sauvegardés en JSON et réutilisés à l'identique pour tous les modèles.
- **Hardware** : RTX 4080 (12 GB VRAM) ⇒ batch 32, max_len 128, fp16.
- **Versions** : `requirements.txt` pinné. Hash git du commit dans chaque `metrics.json`.
- **Artefacts** : `reports/` arborescent par étape ; rien d'éphémère ne sort de cette arborescence.

---

## 6. Risques et contournements

| Risque | Contournement |
|---|---|
| Temps d'entraînement (1,8 M × 2 epochs ≈ 2-3 h sur RTX 4080) | Itérer sur sous-échantillon (`N_ROWS = 300_000`), full run en J +1 nuit |
| Mémoire SHAP sur tout le test set | Échantillonner 30-200 ex. seulement, suffit pour l'audit |
| TextFooler très lent (~10 s / commentaire) | Limiter à 500 ex. ; utiliser `attack.attack_dataset(...)` parallélisé |
| Identités rares (`other_disability` n=5) | Filtrer à `n ≥ 500` test (déjà prévu par `MIN_TEST_PER_IDENTITY`) |
| Divergence sklearn AUC quand subgroup pure | Retourner `nan` (déjà géré dans `_safe_auc`) |

---

## 7. Prochaine action immédiate

→ **Exécuter `notebooks/02_baseline_bert.ipynb`** (livré ci-joint) avec `N_ROWS = 300_000` pour valider le pipeline en ~20 min, puis relancer en `N_ROWS = None` pour le run de référence officiel.
