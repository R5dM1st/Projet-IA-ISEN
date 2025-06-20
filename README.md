# Partie Intelligence Artificielle (IA) – Projet Année 3

## Description du sujet

Le sujet du projet consiste à exploiter des données AIS (Automatic Identification System) issues du trafic maritime afin de développer des modèles d’intelligence artificielle permettant d’analyser et de prédire les comportements des navires. L’objectif est double : d’une part, comprendre et modéliser les schémas de navigation grâce à des techniques d’apprentissage non-supervisé ; d’autre part, mettre en œuvre des méthodes supervisées pour prédire le type de navire ainsi que sa trajectoire future.

Ce projet couvre l’ensemble du cycle d’un projet IA : préparation et prétraitement des données, choix des variables pertinentes, expérimentation avec différents algorithmes, évaluation des performances à l’aide de métriques adaptées, et intégration finale dans des scripts réutilisables. La finalité est de fournir une analyse approfondie et des outils prédictifs pour la visualisation et l’aide à la décision dans le domaine maritime.

---

## Table des matières

- [Description du sujet](#description-du-sujet)
- [Objectifs IA](#objectifs-ia)
- [Fonctionnalités IA](#fonctionnalit%C3%A9s-ia)
- [Outils utilisés](#outils-utilises)
- [Organisation de la partie IA](#organisation-de-la-partie-ia)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Auteurs](#auteurs)

---

## Objectifs IA

- Comprendre le cycle d’un projet IA
- Évaluer et comparer différents modèles
- Mettre en œuvre des techniques d’apprentissage supervisé et non-supervisé

## Sujet

Modélisation et prédiction de comportements de navigation maritime à partir de données AIS.

---

## Fonctionnalités IA

### 1. Préparation des données
- Nettoyage, sélection et transformation des variables issues des données AIS

### 2. Apprentissage non-supervisé (Clustering)
- Objectif : regrouper les navires selon leurs comportements de navigation
- Étapes :
    - Sélection des variables pertinentes
    - Choix et justification du modèle de clustering (ex : KMeans)
    - Évaluation (Silhouette, Davies-Bouldin, etc.)
    - Visualisation des clusters (ex : Plotly)
    - Création d’un script réutilisable

### 3. Apprentissage supervisé (Classification)
- Objectif : prédire le type de navire à partir de ses caractéristiques
- Étapes :
    - Prétraitement + encodage des données
    - Sélection de la cible et des variables explicatives
    - Choix du modèle de classification (ex : RandomForest, SVM…)
    - Évaluation (accuracy, f1-score…)
    - Optimisation des hyperparamètres (GridSearchCV)
    - Script final

### 4. Prédiction de trajectoire
- Objectif : prédire la position future (LAT, LON) des navires à différents horizons temporels (5, 10, 15 min)
- Variables d’entrée : vitesse, cap, heading, type, etc.
- Sortie : position future
- Problématique : séries temporelles

### 5. Évaluation & scripts CLI
- Utilisation de métriques adaptées pour chaque tâche
- Production de scripts en ligne de commande pour chaque besoin

---

## Outils utilisés

- **Langage** : Python
- **Librairies** : pandas, scikit-learn, plotly, etc...
---

## Organisation de la partie IA

- Scripts organisés par besoin :
    - `Besoin_Client_1/`
      -clustering_script.py
    - `Besoin_Client_2/`
      -classification_script.py
    - `Besoin_Client_3/`
      -prediction_trajectoire_script.py
- Chaque dossier contient : scripts, modèles (.pkl), README et rapport explicatif

---

## Installation

1. Cloner le dépôt :
   ```sh
   git clone https://github.com/R5dM1st/Projet-IA-ISEN.git
   ```
2. Aller dans le dossier du projet :
   ```sh
   cd Projet-IA-ISEN
   ```
---

## Utilisation

Chaque fonctionnalité IA possède un script dédié (voir la structure du projet).

Exemples d’exécution en ligne de commande :

- Clustering (non-supervisé) :
  ```sh
  python Besoin_Client_1/clustering_script.py --input data/ais.csv
  ```
- Classification (type de navire) :
  ```sh
  python Besoin_Client_2/classification_script.py --input data/ais.csv
  ```
- Prédiction de trajectoire :
  ```sh
  python Besoin_Client_3/prediction_script.py --input data/ais.csv
  ```

Les options disponibles sont détaillées dans chaque dossier.

---

## Structure du projet

```
/
├── Besoin_Client_1/
│   ├── clustering_script.py
│   ├── model.pkl
│   ├── README.md
├── Besoin_Client_2/
│   ├── classification_script.py
│   ├── CV_full.py
│   ├── CV_RF_SVM_XGB_LGBM_regroupe.py
│   ├── label_encoder_r.pkl
│   ├── scaler_r.pkl
│   ├── test_Regroupe.py
│   ├── random_forest_model_r.pkl
│   ├── vessel-clean-final.csv
├── Besoin_Client_3/
│   ├── label_encoder.pkl
│   ├── model.pkl
│   ├── prediction_trajectoire.py
│   ├── scaler.pkl
│   ├── vessel-clean-final.csv
└── README_IA.md
```

---

## Auteurs

- [Emile Duplais](https://github.com/R5dM1st)
- [Matteo D'ettore](https://github.com/matteodettore)
- [Alex LETOUZE](https://github.com/Alex-LTZ)
