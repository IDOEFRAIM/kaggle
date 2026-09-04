import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

# On garde imblearn pour la fonction SMOTE uniquement
from imblearn.over_sampling import SMOTE

try:
    import xgboost as xgb
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

def prepare_balanced_data(X, y, preprocessor):
    """
    Applique le préprocesseur puis SMOTE manuellement.
    Nécessaire car le pipeline Sklearn ne supporte pas SMOTE en interne.
    """
    X_processed = preprocessor.fit_transform(X)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_processed, y)
    return X_res, y_res, preprocessor

def build_final_xgb(random_state=42):
    """Pipeline Sklearn pur (sans SMOTE à l'intérieur)"""
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost/SHAP non installés")
    
    model = xgb.XGBClassifier(
        eval_metric='logloss', 
        random_state=random_state, 
        tree_method='hist'
    )
    return Pipeline([('clf', model)])

def tune_pcos_v2(X_res, y_res, n_iter=60):
    """Optimisation sur les données déjà équilibrées"""
    pipe = build_final_xgb()
    
    param_dist = {
        'clf__n_estimators': randint(100, 1000),
        'clf__max_depth': randint(3, 9),
        'clf__learning_rate': uniform(0.01, 0.2),
        'clf__subsample': uniform(0.7, 0.3),
        'clf__colsample_bytree': uniform(0.7, 0.3),
        'clf__gamma': uniform(0, 2),          # Aide à la généralisation
        'clf__reg_lambda': uniform(1, 10),    # L2 pour éviter les poids extrêmes
        'clf__min_child_weight': randint(1, 6) # Empêche de créer des feuilles trop petites
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    search.fit(X_res, y_res)
    return search.best_estimator_, search.best_score_

def get_impact_factors(best_model, X_res, feature_names):
    """Analyse SHAP pour extraire les biomarqueurs du SOPK"""
    clf = best_model.named_steps['clf']
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_res)
    
    # Affichage des facteurs les plus importants




if __name__ == "__main__":
    try:
        df = pd.read_csv('PCOS_infertility.csv')
    except FileNotFoundError:
        print("Erreur : Fichier PCOS_infertility.csv non trouvé.")
        exit()

    # 1. NETTOYAGE CRUCIAL DES DONNÉES TEXTES DANS LES COLONNES NUMÉRIQUES
    # On force la conversion en numérique, les erreurs ('a', ' ', etc.) deviennent NaN
    for col in df.columns:
        if col not in ['Patient File No.', 'PCOS (Y/N)']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Suppression des colonnes inutiles
    if 'Patient File No.' in df.columns:
        df = df.drop(['Patient File No.', 'Unnamed: 44'], axis=1, errors='ignore')

    # 2. Séparation Features / Cible (après avoir nettoyé les NaN créés par to_numeric)
    # On remplit les NaN par la médiane ici aussi pour être sûr avant SMOTE
    df = df.fillna(df.median()) 
    
    X = df.drop('PCOS (Y/N)', axis=1)
    y = df['PCOS (Y/N)']

    # 3. Définition du préprocesseur
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    num_features = X.columns
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features)
    ])

    print("--- Équilibrage des données avec SMOTE ---")
    # Maintenant fit_transform ne plantera plus car tout est numérique (float/int)
    X_res, y_res, fitted_pre = prepare_balanced_data(X, y, preprocessor)
    
    feature_names = num_features.tolist()
    print(f"Ancienne taille : {len(y)} | Nouvelle taille (SMOTE) : {len(y_res)}")


    # 3. Optimisation
    print("\n--- Recherche des meilleurs paramètres (RandomizedSearch) ---")
    best_model, best_score = tune_pcos_v2(X_res, y_res, n_iter=20)
    print(f"Meilleur score ROC_AUC : {best_score:.4f}")

    # 4. Sauvegarde du Pipeline COMPLET (Scaler + Modèle)
    # C'est ce qui permettra à Streamlit de comprendre les données brutes
    final_pipeline = Pipeline([
        ('pre', fitted_pre),
        ('clf', best_model.named_steps['clf'])
    ])
    
    joblib.dump(final_pipeline, 'pcos_model_final.joblib')
    print("Succès : Pipeline complet (Preprocess + XGBoost) sauvegardé.")
    # 5. ANALYSE DES FACTEURS (L'instant de vérité)
    print("\n--- Génération de l'analyse SHAP ---")
    # Note: On utilise X_res qui est déjà transformé par le preprocessor
    get_impact_factors(best_model, X_res, feature_names)