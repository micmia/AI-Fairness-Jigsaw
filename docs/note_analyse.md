# Note d'analyse expérimentale
**Classification de toxicité Jigsaw — Compromis performance / équité / robustesse**

*Mini-projet IA Responsable · Mai 2026 · BERT fine-tuned sur Civil Comments*

## 1. Protocole

**Données** : *Jigsaw Unintended Bias* (1 804 874 commentaires, 24 identités annotées). Étiquette binaire `1[target ≥ 0.5]` (taux toxique ≈ 8 %). Splits **80/10/10** stratifiés par `2·label + any_identity`, seed 1337, IDs persistés pour garantir une comparabilité stricte entre modèles.

**Modèles** : `bert-base-uncased` (max_len = 128) fine-tuné 2 epochs avec AdamW (lr = 2·10⁻⁵, batch = 32, fp16, warmup = 0.06) en trois variantes :
- **Baseline** : tête `Linear(768,1)` + BCEWithLogitsLoss.
- **Fair-A (in-processing pondérée)** : `w_i = 1 + 0.25·1[BPSN] + 0.25·1[BNSP]`.
- **Fair-B (in-processing multi-tâches)** : tête auxiliaire `Linear(768,24)` prédisant les identités (cibles continues `[0,1]`) ; `L = L_tox + 0.1·L_id`.

**Métriques** : Overall ROC-AUC, ECE (15 bins) ; par identité (≥ 500 ex. test, soit 8 retenues) Subgroup / BPSN / BNSP AUC + moyenne généralisée *p* = −5 (score Jigsaw) ; SHAP token-level sur 30 commentaires stratifiés (12 BPSN, 8 toxique×identité, 6 BNSP, 4 neutres) avec **IdRatio** = part globale de `|SHAP|` portée par les 143 termes du lexique identitaire. Robustesse (notebook livré) : bruit caractère (`typo`, `case`, `no-whitespace`) + attaque TextFooler.

## 2. Résultats clés (entraînement sur le corpus complet)

| Modèle | Overall AUC | Jigsaw | ECE | min BPSN | min Subgroup | IdRatio |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9644 | 0.9235 | 0.0156 | 0.8350 *(black)* | 0.8430 | 0.0632 |
| Fair-A | 0.9649 (+0.0005) | **0.9260** (+0.0024) | 0.0165 (+0.0009) | **0.8467** (+0.0117) | 0.8457 (+0.0027) | **0.0529** (−0.0104) |
| Fair-B | 0.9648 (+0.0004) | 0.9253 (+0.0018) | **0.0146** (−0.0010) | 0.8367 (+0.0017) | **0.8475** (+0.0046) | 0.0603 (−0.0029) |

**Constats** : (i) aucun coût AUC global mesurable ; (ii) Fair-A réduit drastiquement les faux positifs identitaires (+0.0117 sur l'identité la plus discriminée, `black`) et fait chuter de 16 % le poids des tokens identitaires dans la décision ; (iii) Fair-B améliore la calibration et la séparabilité intra-sous-groupe sans dégrader aucune métrique.

## 3. Compromis observé

**Performance vs équité** : à cette échelle, le compromis classique disparaît. La pondération redéploie la capacité du modèle vers les pires sous-groupes sans dégrader la classification générale. Le coût se déplace.

**Équité vs calibration** : la pénalisation BPSN de Fair-A **rend le modèle plus prudent** sur tout contexte identitaire — bénéfique contre les faux positifs (+0.0117 min BPSN) mais coûteux contre les faux négatifs identitaires (min BNSP −0.0048) et pour la calibration (ECE +0.0009). Fair-B régularise par la tâche auxiliaire et **améliore l'ECE** (−0.0010) mais le gain BPSN est dix fois plus modeste. On ne peut donc pas avoir simultanément **le plus gros gain BPSN** *et* **la meilleure calibration**.

**Équité vs robustesse (hypothèse à valider en P4)** : Fair-A « débranche » 16 % de l'attribution aux tokens identitaires (SHAP). Or ces tokens sont des ancres orthographiquement stables. En s'en détachant, le modèle se reporte sur des tokens contextuels plus malléables et **pourrait devenir plus vulnérable** à TextFooler (substitutions de synonymes). Fair-B conserve l'attribution identitaire (Δ IdRatio modeste) mais régularise globalement l'encodeur ; on attend une meilleure stabilité sous bruit caractère. La P4 exécutée arbitrera entre ces deux hypothèses.

## 4. Limites

- **Couverture identitaire** : seules 8 des 24 identités passent le seuil `n_test ≥ 500` ; les sous-groupes rares et les croisements **intersectionnels** (femme noire, musulman homosexuel…) ne sont pas mesurés.
- **Étiquette binarisée** : la fraction d'annotateurs `target ∈ [0,1]` est tronquée à 0.5, détruisant l'incertitude aléatoire ; une régression avec perte focal aurait préservé ce signal.
- **Hyperparamètres de mitigation non explorés** : `λ = 0.25` et `α = 0.1` sont les valeurs de la littérature sans recherche en grille (`λ ∈ {0.1, 0.5, 1.0}` éclairerait l'effet marginal).
- **Audit SHAP sans IC** : `max_evals = 200` × 30 exemples — la baisse d'IdRatio (−0.0104) n'a pas d'intervalle de confiance bootstrap, donc qualitative.
- **Robustesse partielle** : TextFooler ne couvre qu'une famille d'attaques (synonymes contraints USE). Les attaques par homoglyphes Unicode, paraphrase neuronale, ou injection de contexte ne sont pas testées.
- **ECE globale uniquement** : un modèle peut être globalement calibré mais sous-confiant sur les minorités ; une **Group-conditional ECE** serait un complément naturel.

---
*Code & artefacts : `AI-Fairness-Jigsaw/` — `memory.md`, `docs/ROADMAP.md`, `reports/metrics/*.json`.*
