import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Gestion de l'import SHAP pour éviter le crash lié à NumPy 2.4
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Suppression des alertes de versioning pour l'utilisateur
warnings.filterwarnings('ignore')

# --- CONFIGURATION PAGE ---
st.set_page_config(
    page_title="PCOS Diagnostic & Impact AI", 
    page_icon="🌸", 
    layout="wide"
)

@st.cache_resource
def load_all():
    """Charge le pipeline complet (Préprocesseur + Modèle XGBoost)"""
    try:
        # On s'attend à charger un Pipeline sklearn
        model = joblib.load('pcos_model_final.joblib')
        return model
    except Exception as e:
        st.error(f"Erreur de chargement du fichier modèle : {e}")
        return None

pipeline = load_all()

# --- INTERFACE ---
st.title("🌸 Assistant Diagnostic SOPK & Analyse d'Impact")
st.markdown("""
Cet outil évalue le risque de **Syndrome des Ovaires Polykystiques** et utilise l'analyse **SHAP** pour expliquer quels facteurs (biométrie, hormones, échographie) influencent le score.
""")

if not SHAP_AVAILABLE:
    st.sidebar.error("⚠️ SHAP/Numba nécessite NumPy < 2.3. L'analyse d'impact est désactivée.")

if pipeline:
    # Formulaire de saisie
    with st.form("main_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Biométrie")
            age = st.number_input("Âge (ans)", 15, 50, 25)
            weight = st.number_input("Poids (kg)", 30.0, 150.0, 65.0)
            height = st.number_input("Taille (cm)", 100.0, 220.0, 165.0)
            
        with col2:
            st.subheader("🩸 Cycles & Peau")
            cycle_len = st.number_input("Durée du cycle (jours)", 2, 60, 28)
            hair = st.selectbox("Pilosité excessive (Hirsutisme) ?", ["Non", "Oui"])
            skin = st.selectbox("Acné / Peau foncée (Acanthosis) ?", ["Non", "Oui"])
            
        with col3:
            st.subheader("🏥 Échographie")
            f_left = st.number_input("Follicules (Ovaire Gauche)", 0, 30, 6)
            f_right = st.number_input("Follicules (Ovaire Droit)", 0, 30, 6)
        
        st.markdown("---")
        submitted = st.form_submit_button("Lancer l'Analyse Prédictive")

    if submitted:
        # 1. Préparation des données brutes et calculs dérivés
        bmi = weight / ((height/100)**2)
        
        # Mapping conforme aux noms de colonnes du dataset original
        raw_data = {
            'Age (yrs)': age,
            'Weight (kg)': weight,
            'Height(cm)': height,
            'BMI': bmi,
            'Cycle length(days)': cycle_len,
            'Hair growth(Y/N)': 1 if hair == "Oui" else 0,
            'Skin darkening (Y/N)': 1 if skin == "Oui" else 0,
            'Follicle No. (L)': f_left,
            'Follicle No. (R)': f_right,
            'Cycle(R/I)': 4 if cycle_len > 35 else 2
        }

        input_df = pd.DataFrame([raw_data])
        
        try:
            # --- ÉTAPE A : Extraction des composants du Pipeline ---
            preprocessor = pipeline.named_steps['pre']
            clf = pipeline.named_steps['clf']
            
            # --- ÉTAPE B : Alignement des colonnes ---
            # On s'assure que toutes les colonnes attendues par le préprocesseur sont là
            expected_cols = preprocessor.feature_names_in_
            input_df = input_df.reindex(columns=expected_cols, fill_value=0)
            
            # Transformation (Imputation + Scaling)
            X_transformed = preprocessor.transform(input_df)
            
            # --- ÉTAPE C : Prédiction ---
            prob = clf.predict_proba(X_transformed)[0][1]
            
            # Affichage des résultats
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.subheader("🎯 Résultat de l'Analyse")
                if prob > 0.5:
                    st.error(f"### Risque Élevé : {prob:.1%}")
                else:
                    st.success(f"### Risque Faible : {prob:.1%}")
                
                st.write("**Interprétation :**")
                st.caption("""
                Ce score est basé sur la corrélation de vos données avec des milliers de cas cliniques. 
                Une probabilité élevée indique que vos symptômes correspondent au profil type du SOPK.
                """)

            # --- ÉTAPE D : SHAP (Analyse des facteurs) ---
            with res_col2:
                st.subheader("🔍 Facteurs d'Influence")
                
                if SHAP_AVAILABLE:
                    st.write("Impact des variables sur votre diagnostic :")
                    
                    # Calcul des valeurs SHAP pour l'explication locale
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_transformed)
                    
                    # Récupération des noms de colonnes transformées
                    feature_names_out = preprocessor.get_feature_names_out()
                    # Nettoyage cosmétique des noms (ex: 'num__Age' -> 'Age')
                    clean_names = [n.split('__')[-1] for n in feature_names_out]
                    
                    # Génération du graphique
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.bar_plot(
                        shap_values[0], 
                        feature_names=clean_names, 
                        max_display=10, 
                        show=False
                    )
                    plt.title("Importance locale des biomarqueurs")
                    st.pyplot(fig)
                    
                    st.info("""
                    **Lecture :** Les barres vers la **droite** augmentent le risque. 
                    Les barres vers la **gauche** le diminuent.
                    """)
                else:
                    st.warning("L'analyse SHAP n'a pas pu être générée. Vérifiez les versions de NumPy et Numba.")

        except Exception as e:
            st.error(f"Une erreur est survenue lors du traitement : {e}")
            st.info("Astuce : Vérifiez que vous avez sauvegardé le Pipeline entier et non juste le modèle XGBoost.")

# --- PIED DE PAGE ---
st.markdown("---")
st.caption("⚠️ **Avertissement Médical :** Cet outil est fourni à titre indicatif pour faciliter le dialogue avec votre médecin. Il ne remplace en aucun cas un diagnostic clinique, une échographie pelvienne ou un bilan hormonal réalisé par un professionnel de santé.")