# Breast Cancer Prediction - README

Ce projet présente une démarche complète pour la prédiction du cancer du sein à partir de données cliniques anonymisées.

## Structure du projet
- `model.ipynb` : Notebook principal contenant toutes les étapes du workflow data science.
- `client.py` : Application Streamlit pour l'aide au diagnostic.
- `Breast_Cancer.csv` : Jeu de données utilisé pour l'entraînement et l'évaluation.
- `cancer_detection_pipeline_rf.pkl` : Modèle RandomForest optimisé et sauvegardé.
- `cancer_features_names.pkl` : Liste des colonnes utilisées par le modèle.

## Workflow du notebook
1. **Import des librairies** : Chargement des outils nécessaires (pandas, scikit-learn, matplotlib, SHAP, etc.).
2. **Provenance et exploration des données** : Description du dataset, affichage des premières lignes, analyse des types et des valeurs manquantes.
3. **Prétraitement** :
   - Encodage des variables catégorielles
   - Imputation des valeurs manquantes
   - Normalisation des variables
   - Vérification de l’équilibre des classes
4. **Séparation des données** : Split en train/dev/test pour éviter la fuite de données.
5. **Comparaison automatisée des modèles** : Validation croisée (cross-validation) sur RandomForest, XGBoost, SVM, KNN, Logistic Regression.
6. **Optimisation** : GridSearchCV pour optimiser les hyperparamètres du meilleur modèle (RandomForest).
7. **Pipeline** : Construction d’une pipeline scikit-learn (imputer, scaler, modèle).
8. **Évaluation** : Calcul du F1-score sur le jeu de test.
9. **Interprétabilité** : Analyse de l’importance des variables et explications locales avec SHAP.
10. **Sauvegarde** : Export du modèle et des colonnes pour déploiement.

## Application Streamlit
- Saisie des données cliniques via formulaire
- Prédiction du risque et recommandations médicales
- Visualisation de l’importance des variables

## Reproductibilité
- Toutes les étapes sont documentées et automatisées
- Les modèles et pipelines sont sauvegardés pour une utilisation future

## À propos
Ce projet est conforme aux standards internationaux pour l’analyse prédictive en santé. Il peut être adapté à d’autres jeux de données ou modèles selon les besoins.
