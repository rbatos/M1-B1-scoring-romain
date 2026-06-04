# Expériences — M1-B1 Pyrenex Crédit (Lending Club)

> Trace tes runs au fur et à mesure. Format imposé : un bloc par run, avec
> date, modèle, hyperparams, métriques **test interne uniquement**, verdict.
> Commit à chaque run final (pas à chaque essai jetable).
>
> ⚠️ **Règle d'or — comparabilité.** Le holdout **n'apparaît jamais** dans les
> blocs `exp_NNN`. Il sort **une seule fois**, pour le modèle retenu, dans
> la section finale en bas de fichier. Cf. mini-cours 04.

---

## exp_001 — RF default

- **Date** : 2026-06-02
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 d2da093bee40024b196e73a0d2d763193782f947e3d60552a3d7bbad0bd944e3)
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : `n_estimators=100`, `random_state=42`, `n_jobs=-1`
- **Pré-traitement** : `Numeric` → imputation médiane + StandardScaler ; `Categorical` → imputation modalité la plus fréquente + OneHotEncoder (handle_unknown='ignore')
- **Métriques (test interne)** :
  - Accuracy : 0.8194 => 81,94 % des prêts ont été correctement classés, toutes classes confondues. C’est un indicateur global, mais ici il est trompeur car la classe « Fully Paid » est majoritaire. Un modèle peut obtenir une accuracy élevée tout en ratant presque tous les défauts.
  - F1 macro : 0.5131 => Moyenne non pondérée du F1 pour les deux classes (Fully Paid et Charged Off). Cela donne le même poids aux deux classes, donc c’est plus adapté quand les classes sont déséquilibrées. Une valeur de 0.5131 indique une performance modérée globalement, meilleure qu’un simple accuracy, mais encore loin d’un bon modèle.
  - F1 défaut : 0.1269 => F1 calculé uniquement pour la classe positive (Charged Off). C’est la moyenne harmonique de la précision et du rappel sur les défauts. Ici, 0.1269 est très faible, ce qui signifie que le modèle est mauvais pour prédire correctement les défauts.
  - Precision défaut : 0.5727 => Parmi les prêts que le modèle prédit comme défauts, 57,27 % sont effectivement des défauts. C’est un niveau moyen : quand le modèle dit “Charged Off”, il a un peu plus de chance d’avoir raison qu’un pur hasard, mais ce n’est pas excellent.
  - Recall défaut : 0.0713 => Seuls 7,13 % des défauts réels sont détectés par le modèle. Autrement dit, le modèle passe à côté de 92,87 % des prêts réellement en défaut. C’est l’indicateur critique dans un scoring crédit : ici il est très bas, donc le modèle manque presque tous les cas de défaut.
  - ROC-AUC : 0.7170 => Mesure la capacité du modèle à classer correctement un défaut plus haut qu’un non-défaut indépendamment du seuil. 0.7170 signifie que dans environ 71,7 % des paires défaut/non-défaut aléatoires, le modèle donne une probabilité plus élevée au défaut. C’est acceptable, mais pas très fort.
  - Matrice de confusion : [[3870, 47], [820, 63]] => Interprétation en terme de vrais/faux :
      Vrai négatifs (TN) = 3870 : prêts correctement prédits Fully Paid
      Faux positifs (FP) = 47 : prêts prédits Charged Off alors qu’ils étaient Fully Paid
      Faux négatifs (FN) = 820 : prêts prédits Fully Paid alors qu’ils étaient Charged Off
      Vrais positifs (TP) = 63 : prêts correctement prédits Charged Off
- **Verdict** : le modèle par défaut a une accuracy correcte, mais il manque beaucoup de défauts. Il reste surtout utile si l’on veut minimiser les faux positifs, pas si l’on veut détecter les défauts à tout prix.

---

## exp_002 — RF balanced

- **Date** : 2026-06-02
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 d2da093bee40024b196e73a0d2d763193782f947e3d60552a3d7bbad0bd944e3)
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : `n_estimators=200`, `max_depth=10`, `min_samples_leaf=10`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`
- **Pré-traitement** : identique à exp_001
- **Métriques (test interne)** :
  - Accuracy : 0.6956 => 69,56 % des prêts sont correctement classés ici. C’est nettement inférieur à exp_001 (0.8194), car le modèle balanced accepte plus d’erreurs globales pour mieux détecter les défauts.
  - F1 macro : 0.6123 => Moyenne équilibrée entre les classes. C’est bien mieux que exp_001 (0.5131), ce qui montre que ce modèle gère mieux à la fois les prêts remboursés et les défauts, au lieu de privilégier uniquement la classe majoritaire.
  - F1 défaut : 0.4326 => C’est la véritable amélioration clé par rapport à exp_001 (0.1269). Le modèle balanced est beaucoup plus efficace pour prédire correctement les défauts, même si la détection reste perfectible.
  - Precision défaut : 0.3292 => Parmi les prêts prédits comme défauts, seulement 32,92 % le sont réellement. C’est moins bon que exp_001 (0.5727), donc ce modèle génère plus de faux positifs. En clair : il avertit plus souvent à tort qu’un prêt va faire défaut.
  - Recall défaut : 0.6308 => 63,08 % des défauts réels sont détectés. C’est l’amélioration majeure par rapport à exp_001 (0.0713). Le modèle balanced passe de presque tous les défauts manqués à en détecter une majorité.
  - ROC-AUC : 0.7442 => La capacité de discrimination globale est meilleure que dans exp_001 (0.7170). Cela confirme que le modèle balanced classe mieux les candidats par risque, indépendamment du seuil.
  - Matrice de confusion : [[2782, 1135], [326, 557]] => 
      TN = 2782 : prêts remboursés correctement identifiés
      FP = 1135 : prêts remboursés mal classés comme défauts
      FN = 326 : défauts manqués
      TP = 557 : défauts correctement détectés
- **Verdict** : le modèle balanced sacrifie de l'accuracy pour une bien meilleure détection des défauts. Il est plus adapté à un objectif de surveillance du risque, car il trouve beaucoup plus de défauts réels.

## comparaison exp_002 vs exp_001

- `exp_002` a une accuracy plus faible (0.6956 vs 0.8194) parce qu'il accepte davantage d'erreurs globales pour améliorer la détection des défauts.
- La F1 macro monte de 0.5131 à 0.6123, ce qui signifie que `exp_002` équilibre mieux les performances sur les deux classes et ne se contente plus de bien prédire seulement la classe majoritaire.
- La F1 défaut passe de 0.1269 à 0.4326, soit une amélioration massive du comportement sur la classe `Charged Off`.
- En revanche, la précision défaut baisse (0.3292 vs 0.5727), ce qui indique que `exp_002` génère beaucoup plus de faux positifs.
- Le recall défaut explose de 0.0713 à 0.6308 : `exp_002` détecte la majorité des défauts réels, alors que `exp_001` en manque presque tous.
- Le ROC-AUC s'améliore légèrement (0.7442 vs 0.7170), ce qui confirme une meilleure discrimination générale entre défauts et remboursements.
- La matrice de confusion montre ce compromis : `exp_002` trouve 557 défauts réels (vs 63) mais déclenche 1135 faux positifs (vs 47).

> En résumé : `exp_001` est conservateur et minimise les alertes erronées, tandis que `exp_002` est plus utile si l'objectif est de ne pas rater les défauts.

---

## comparaison avec la baseline V1 2017

- **Baseline V1 2017** : accuracy 0.8492, F1 macro 0.5018, ROC-AUC 0.7296, matrice de confusion [[2021, 11], [351, 17]].
- **Différences clés** :
  - V2_default a une accuracy plus faible (0.8194) mais une F1 macro légèrement meilleure (0.5131) ; il reste cependant trop conservateur sur le défaut.
  - V2_balanced améliore nettement la F1 macro (0.6123) et le ROC-AUC (0.7442) par rapport à V1, au prix d'une baisse d'accuracy.
  - La V2_balanced est la version la plus cohérente si l'on priorise la détection des défauts dans un contexte de scoring crédit.

---

## exp_003 — Comparaison des variantes RF balanced générées automatiquement

- **Date** : 2026-06-02
- **Modèle** : RandomForestClassifier (sklearn 1.5.1)
- **Dataset** : lending_club_train.csv (sha256 d2da093bee40024b196e73a0d2d763193782f947e3d60552a3d7bbad0bd944e3)
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Pré-traitement** : identique à exp_001 et exp_002
- **Nouvelle démarche** : plusieurs configurations ont été générées à partir d’une boucle de plages pertinentes (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`, `max_features`) dans `src/train.py`.
- **Mesure du coût** : chaque configuration enregistre désormais `duration_seconds` dans les métadonnées et le tableau comparatif, ce qui permet de juger de l’efficacité d’entraînement en plus de la qualité.

- **Description des hyperparamètres** :
  - `n_estimators` : nombre d’arbres dans la forêt. Plus d’arbres stabilise le modèle, mais coûte du temps de calcul.
  - `max_depth` : profondeur maximale des arbres. Une valeur basse régularise le modèle, une valeur élevée permet de capturer des relations plus complexes.
  - `min_samples_leaf` : nombre minimal d’échantillons dans une feuille. Un nombre élevé réduit le surapprentissage en lissant les décisions.
  - `class_weight` : rééquilibrage des classes minoritaires. `balanced` ajuste globalement les poids, `balanced_subsample` le fait sur chaque bootstrap.
  - `max_features` : nombre de variables candidates à chaque split. `sqrt` et `log2` favorisent la diversité des arbres et limitent la corrélation entre eux.

### Configurations testées
- `balanced_recall` : `n_estimators=250`, `max_depth=12`, `min_samples_leaf=5`, `class_weight='balanced_subsample'`, `max_features='sqrt'`. Plus de profondeur et `balanced_subsample` pour améliorer la détection des défauts tout en limitant l’overfitting.
- `balanced_precise` : `n_estimators=200`, `max_depth=12`, `min_samples_leaf=10`, `class_weight='balanced'`, `max_features='sqrt'`. Plus de régularisation pour limiter les faux positifs et stabiliser les prédictions.
- `balanced_shallow` : `n_estimators=200`, `max_depth=8`, `min_samples_leaf=10`, `class_weight='balanced_subsample'`, `max_features='sqrt'`. Arbres peu profonds pour maximiser le rappel défaut et réduire le surapprentissage.
- `balanced_wide` : `n_estimators=300`, `max_depth=None`, `min_samples_leaf=5`, `class_weight='balanced'`, `max_features='sqrt'`. Plus d’arbres et profondeur non limitée pour améliorer l’accuracy au prix d’un rappel plus faible.
- `balanced_deep` : `n_estimators=200`, `max_depth=20`, `min_samples_leaf=5`, `class_weight='balanced_subsample'`, `max_features='sqrt'`. Arbres plus profonds avec rééquilibrage par sous-échantillonnage pour capturer des interactions complexes.

### Résultats comparés
| Config | Accuracy | F1 macro | F1 défaut | Precision défaut | Recall défaut | ROC-AUC | Duration (s) |
|---|---|---|---|---|---|---|---|
| `balanced_recall` | 0.7233 | 0.6256 | 0.4344 | 0.3481 | 0.5776 | 0.7428 | 5.8862 |
| `balanced_precise` | 0.7131 | 0.6207 | 0.4336 | 0.3404 | 0.5968 | 0.7431 | 3.0107 |
| `balanced_shallow` | 0.6846 | 0.6101 | 0.4397 | 0.3266 | 0.6727 | 0.7444 | 3.0928 |
| `balanced_wide` | 0.7821 | 0.6255 | 0.3833 | 0.3998 | 0.3681 | 0.7375 | 5.4421 |
| `balanced_deep` | 0.7729 | 0.6202 | 0.3793 | 0.3814 | 0.3771 | 0.7355 | 4.3853 |

### Analyse comparative
- `balanced_shallow` a le meilleur recall défaut (0.6727), ce qui signifie qu’il détecte la majorité des défauts réels. En revanche, il a la pire accuracy (0.6846) et la plus faible précision défaut, ce qui veut dire beaucoup de faux positifs.
- `balanced_wide` et `balanced_deep` améliorent l’accuracy, mais réduisent fortement la capacité à détecter les défauts : leur recall défaut est inférieur à 0.38. Ces variantes sont donc moins adaptées à un scoring crédit orienté détection de défaut.
- `balanced_precise` propose un compromis intéressant : recall 0.5968, précision défaut 0.3404 et F1 macro 0.6207. Elle est plus prudente que `balanced_shallow` tout en conservant une bonne capacité de détection.
- `balanced_recall` est la meilleure configuration sur la base de la F1 macro (0.6256), ce qui en fait le modèle le plus équilibré entre les classes. Il présente également une amélioration de la précision défaut par rapport à `balanced` (0.3481 vs 0.3292) sans perdre trop en rappel. Sa durée d'entraînement reste raisonnable par rapport aux variantes plus lourdes.

### Résumé global des configurations
- Les variantes à forte profondeur (`balanced_wide`, `balanced_deep`) améliorent l’accuracy, mais leur faible recall défaut les rend moins pertinentes pour un scoring de défaut.
- Les variantes très régulières (`balanced_precise`) offrent une meilleure stabilité avec moins de faux positifs, mais elles conservent un rappel légèrement inférieur à `balanced_shallow`.
- La variante `balanced_shallow` maximise le rappel défaut, ce qui peut être utile si l’objectif métier est de ne manquer aucun défaut, au prix d’un plus grand nombre de fausses alertes.
- `balanced_recall` fournit le meilleur compromis global : F1 macro la plus élevée, bonne amélioration de la précision défaut, et un rappel défaut solide sans dégrader excessivement l’accuracy.

### Choix retenu
- **Retenu** : `balanced_recall`
- **Raison** : c’est la variante qui maximise la F1 macro, donc le compromis global entre rappel et précision sur les deux classes. Elle conserve un rappel défaut solide (0.5776) tout en évitant le nombre de faux positifs excessif observé sur `balanced_shallow`.
- **Remarque** : si l’objectif métier change et que l’on accepte davantage de faux positifs pour ne manquer presque aucun défaut, `balanced_shallow` reste une option valable.

---

## 🏁 Évaluation finale sur holdout (modèle retenu)

> **À remplir une seule fois**, à la tâche 5 du brief, **après** avoir choisi
> ton modèle retenu parmi les `exp_NNN` ci-dessus. Le holdout n'est consulté
> qu'ici.
(cp models/pyrenex_risk_v2_balanced_recall.joblib models/pyrenex_risk_v2.joblib + cp models/pyrenex_risk_v2_balanced_recall.joblib models/pyrenex_risk_v2.joblib + cp models/pyrenex_risk_v2_balanced_recall.joblib models/pyrenex_risk_v2.joblib)

- **Date** : 2026-06-02
- **Expérience retenue** : exp_003 (`balanced_recall`)
- **Modèle persisté** : `models/pyrenex_risk_v2.joblib`
- **Données holdout** : `data/lending_club_holdout.csv` (sha256 …, n=6000)
- **Métriques** :
  - Accuracy : 0.7137
  - F1 macro : 0.6162
  - F1 défaut : 0.4227
  - Precision défaut : 0.3358
  - Recall défaut : 0.5703
  - ROC-AUC : 0.7338
- **Matrice de confusion** :

|  | Pred Fully Paid | Pred Charged Off |
|---|---|---|
| **Vrai Fully Paid** | 3653 | 1244 |
| **Vrai Charged Off** | 474 | 629 |

- **Comparaison baseline V1 2017** :
  - Baseline : accuracy 0.8492, F1 macro 0.5018, ROC-AUC 0.7296, recall défaut ~0.0462.
  - Notre modèle holdout a une accuracy plus faible (-0.1355) car il privilégie la détection des défauts.
  - Il dépasse nettement la baseline sur F1 macro (+0.1144), ce qui montre une meilleure performance équilibrée entre les classes.
  - ROC-AUC est légèrement meilleure (+0.0042), ce qui confirme une meilleure discrimination globale.
  - Le recall défaut est très supérieur (+0.5241), ce qui signifie que le modèle capture une majorité des défauts réels, alors que la baseline en détecte presque aucun.
  - Conclusion : la version retenue est plus adaptée à un objectif de scoring crédit orienté détection des défauts, même si elle perd de l’accuracy globale.
