# Verdict — Modèle de scoring Pyrenex Crédit v2

> Document destiné à Sophie Léger (Lead Data, Pyrenex Crédit).
> 1 page max.

## Contexte

L'objectif était de produire une version v2 du scoring crédit capable de mieux détecter les défauts que la baseline Pyrenex-risk-v1, tout en conservant une performance globale raisonnable.

## Démarche

Nous avons entraîné et comparé plusieurs configurations de `RandomForestClassifier` sur `data/lending_club_train.csv` avec un split stratifié `test_size=0.2`, `stratify=y`, `random_state=42`. Trois séries d'expériences ont été documentées : modèle par défaut, modèle balanced, puis cinq variantes `balanced_*` évaluées sur test interne et une sélection finale validée sur le holdout.

## Verdict chiffré

| Métrique | Baseline 2017 (Pyrenex-risk-v1) | Modèle retenu (v2) | Variation |
|---|---|---|---|
| Accuracy | 0.8492 | 0.7137 | -0.1355 |
| F1 macro | 0.5018 | 0.6162 | +0.1144 |
| F1 défaut | 0.1269* | 0.4227 | +0.2958 |
| ROC-AUC | 0.7296 | 0.7338 | +0.0042 |
| Recall défaut | ~0.0462 | 0.5703 | +0.5241 |

*Baseline F1 défaut estimée sur la matrice `[[2021, 11], [351, 17]]`.

**Configuration retenue** : `balanced_recall` — `n_estimators=250`, `max_depth=12`, `min_samples_leaf=5`, `class_weight='balanced_subsample'`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1`.

## Trade-off explicité au métier

Le modèle retenu augmente fortement la détection des défauts : le recall défaut passe de ~4,6 % à 57,0 %. Le prix à payer est une baisse d'accuracy globale (-13,5 points) et une précision défaut modérée (33,6 %), ce qui signifie plus d'alertes erronées, mais une couverture bien supérieure des prêts à risque.

## Précautions avant mise en production

- Vérifier que le schéma d'entrée en production est identique au schéma d'entraînement (`feature_columns` dans `pyrenex_risk_v2.json`).
- Recalibrer le seuil de décision avec l'équipe métier ; 0.5 est le point de départ, mais un seuil plus bas peut augmenter encore le recall si le business l'autorise.
- Mettre en place un monitoring de dérive des données et des métriques après déploiement.
- Auditer le risque de biais sur les variables sensibles (FICO, revenu, statut de résidence) avant passage en production.

## Recommandation

✅ **Remplacer Pyrenex-risk-v1 par Pyrenex-risk-v2** pour cette version, car elle améliore nettement la F1 macro et le rappel des défauts tout en conservant une discrimination au moins équivalente.

---

*Signé : Romain Busuttil boosté à l'IA, le 2026-06-02*