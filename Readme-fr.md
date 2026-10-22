# 💳 Détection de la fraude par carte bancaire

## 📌 À propos du projet

La fraude par carte bancaire représente un défi majeur pour les institutions financières et les systèmes de paiement électronique. Elle peut entraîner des pertes financières importantes et affecter la confiance des clients.

Ce projet consiste à développer un modèle de Machine Learning capable d'identifier les transactions potentiellement frauduleuses à partir de données de transactions bancaires.

Il couvre l'analyse exploratoire des données, le prétraitement, la gestion du déséquilibre des classes, l'entraînement et l'évaluation de plusieurs modèles de classification, ainsi que le développement d'un tableau de bord interactif avec Streamlit.

## 🎯 Objectifs du projet

- Analyser les transactions bancaires afin d'identifier les caractéristiques associées à la fraude.
- Traiter le déséquilibre entre les transactions légitimes et frauduleuses à l'aide de la technique SMOTE.
- Entraîner et comparer plusieurs algorithmes de Machine Learning.
- Évaluer les performances des modèles à l'aide de métriques adaptées à la détection de fraude.
- Optimiser le seuil de décision afin d'équilibrer la détection des fraudes et les fausses alertes.
- Développer une application interactive permettant d'analyser des transactions et de visualiser les prédictions.

## 📊 Jeu de données

Le projet utilise le jeu de données européen sur la fraude par carte bancaire, disponible sur Kaggle.

**Source :** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Le jeu de données comprend **284 807 transactions**, dont **492 transactions frauduleuses**.

La variable cible `Class` distingue deux catégories :

- `0` : transaction légitime ;
- `1` : transaction frauduleuse.

Les variables `V1` à `V28` correspondent à des caractéristiques numériques anonymisées. Le jeu de données contient également les variables `Time` et `Amount`.

## ⚙️ Méthodologie

### 1. Analyse exploratoire des données

- Étude de la structure et de la qualité des données.
- Analyse des valeurs manquantes et des doublons.
- Visualisation de la répartition des transactions légitimes et frauduleuses.
- Analyse des montants et des caractéristiques des transactions.

### 2. Prétraitement des données

- Séparation des variables explicatives et de la variable cible.
- Division des données en ensembles d'entraînement et de test.
- Standardisation des variables `Time` et `Amount`.

### 3. Gestion du déséquilibre des classes

La technique **SMOTE (Synthetic Minority Over-sampling Technique)** est appliquée uniquement aux données d'entraînement afin d'améliorer l'apprentissage de la classe minoritaire.

### 4. Entraînement des modèles

Trois algorithmes sont étudiés :

- Régression logistique (*Logistic Regression*) ;
- Forêt aléatoire (*Random Forest*) ;
- XGBoost.

### 5. Évaluation et optimisation

Les modèles sont évalués à l'aide des métriques suivantes :

- Précision (*Precision*) ;
- Rappel (*Recall*) ;
- Score F1 (*F1-score*) ;
- Aire sous la courbe ROC (*ROC-AUC*) ;
- Aire moyenne sous la courbe précision-rappel (*Average Precision*).

Le seuil de classification du modèle Random Forest est ensuite ajusté pour étudier le compromis entre les fraudes détectées et les fausses alertes.

## 📈 Résultats du modèle

Le modèle Random Forest a été retenu pour le tableau de bord interactif. Au seuil de classification de `0.75`, les résultats rapportés sont les suivants :

| Indicateur | Résultat |
|---|---:|
| Précision | 95,00 % |
| Rappel | 77,55 % |
| Score F1 | 85,39 % |
| Fraudes correctement détectées | 76 |
| Fausses alertes | 4 |
| Fraudes non détectées | 22 |
| Transactions légitimes correctement classées | 56 860 |

Ces résultats indiquent que le modèle a correctement identifié 76 transactions frauduleuses, mais en a manqué 22 au seuil choisi.

**Limite méthodologique :** le seuil a été sélectionné après examen des résultats sur l'ensemble de test. Ces performances doivent donc être considérées comme préliminaires. Pour une évaluation plus rigoureuse, le seuil doit être déterminé sur un ensemble de validation, puis évalué une seule fois sur un ensemble de test indépendant.

## 🖥️ Tableau de bord Streamlit

L'application est conçue pour permettre :

- Le chargement de fichiers CSV contenant des transactions.
- La prédiction de la probabilité de fraude.
- Le réglage du seuil de détection.
- L'affichage des indicateurs clés et des résultats de classification.
- La visualisation de la distribution des probabilités de fraude.
- Le téléchargement des résultats au format CSV.
- L'évaluation des prédictions lorsque les véritables étiquettes sont disponibles.

## 🛠️ Technologies utilisées

- **Langage :** Python
- **Analyse de données :** Pandas, NumPy
- **Machine Learning :** Scikit-learn, XGBoost
- **Gestion du déséquilibre :** Imbalanced-learn (SMOTE)
- **Visualisation :** Matplotlib
- **Application interactive :** Streamlit
- **Sauvegarde du modèle :** Joblib
- **Environnement de développement :** Google Colab
- **Versionnement du code :** GitHub

## 📁 Structure du projet

```text
credit-card-fraud-detection/
├── app.py
├── requirements.txt
├── README.md
├── README_FR.md
└── fraud_detection_artifacts/
    ├── random_forest_fraud_model.pkl
    ├── fraud_scaler.pkl
    └── fraud_model_config.pkl
```

## 🚀 Installation et utilisation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'application

```bash
streamlit run app.py
```

Les trois fichiers du modèle doivent être présents dans le dossier `fraud_detection_artifacts/`, conformément aux chemins définis dans le code.

## 🔮 Perspectives d'amélioration

- Évaluer les performances sur un ensemble de test totalement indépendant.
- Optimiser les hyperparamètres des modèles.
- Intégrer des méthodes d'explicabilité telles que SHAP.
- Étudier la dérive des données et l'évolution des comportements frauduleux.
- Déployer le tableau de bord en ligne pour faciliter les démonstrations interactives.

## ⚠️ Limites et avertissement

Ce projet est réalisé à des fins pédagogiques et de constitution d'un portfolio en Data Science.

Une probabilité élevée de fraude ne constitue pas une preuve de fraude. Le modèle n'est pas un système de détection validé pour une utilisation en production et ne doit pas être utilisé seul pour prendre des décisions financières réelles.

## 👤 Auteur

Projet réalisé par NYMBUSSEDE LENY NAPALA dans le cadre du développement de compétences en Data Science, Machine Learning et analyse des risques financiers.
