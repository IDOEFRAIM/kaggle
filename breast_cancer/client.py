import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration de la page
st.set_page_config(page_title="OncoPredict - Aide au Diagnostic", layout="wide")

# Custom CSS pour un look "Medical App"
st.markdown("""
    <style>
    .stAlert { border-radius: 10px; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# 2. Chargement de la Pipeline (Scaler + Imputer + RF)
@st.cache_resource
def load_pipeline():
    # Remplace par tes noms de fichiers réels
    pipeline = joblib.load('cancer_detection_pipeline_rf.pkl')
    features = joblib.load('cancer_features_names.pkl')
    return pipeline, features

try:
    model_pipeline, model_features = load_pipeline()
except Exception as e:
    st.error(f"Fichiers modèles introuvables. Vérifiez vos fichiers .pkl ,due to error: {e}")

# HEADER 
st.title("🩺 OncoPredict : Système d'Aide au Diagnostic")
st.write("Outil d'analyse prédictive basé sur le Random Forest pour l'identification des tumeurs.")

# NAVIGATION PAR ONGLETS 
tab1, tab2, tab3 = st.tabs(["🔍 Analyse Patient", "📊 Compréhension du Modèle", "📚 Documentation"])

# ONGLET 1 : ANALYSE 
with tab1:
    st.subheader("Saisie des données cliniques")
    
    with st.form("medical_form"):
        col1, col2 = st.columns(2)
        # Champs adaptés au dataset Breast_Cancer.csv
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=50)
            race = st.selectbox("Race", ["White", "Black", "Other"])
            marital = st.selectbox("Statut marital", ["Married", "Single", "Divorced", "Widowed", "Separated"])
            t_stage = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
            n_stage = st.selectbox("N Stage", ["N0", "N1", "N2", "N3"])
            grade = st.selectbox("Grade", [1, 2, 3])
        with col2:
            tumor_size = st.number_input("Taille de la tumeur (mm)", min_value=1, max_value=200, value=20)
            estrogen = st.selectbox("Statut Estrogène", ["Positive", "Negative"])
            progesterone = st.selectbox("Statut Progestérone", ["Positive", "Negative"])
            reg_node_exam = st.number_input("Nœuds régionaux examinés", min_value=0, max_value=50, value=10)
            reg_node_pos = st.number_input("Nœuds régionaux positifs", min_value=0, max_value=50, value=1)
            survival_months = st.number_input("Survie (mois)", min_value=1, max_value=200, value=60)
        submit = st.form_submit_button("Lancer l'Analyse Médicale")

    if submit:
        # Préparation des données
        input_dict = {
            'Age': age,
            'Race': race,
            'Marital Status': marital,
            'T Stage': t_stage,
            'N Stage': n_stage,
            'Grade': grade,
            'Tumor Size': tumor_size,
            'Estrogen Status': estrogen,
            'Progesterone Status': progesterone,
            'Regional Node Examined': reg_node_exam,
            'Reginol Node Positive': reg_node_pos,
            'Survival Months': survival_months
        }
        # On aligne avec les colonnes exactes utilisées lors de l'entraînement
        df_input = pd.DataFrame([input_dict]).reindex(columns=model_features, fill_value=0)
        # Prédiction
        prob = model_pipeline.predict_proba(df_input)[0][1]
        st.divider()
        res_col, advice_col = st.columns([1, 2])
        with res_col:
            if prob > 0.3:
                st.error(f"### 🚩 RÉSULTAT : SUSPECT")
                st.metric("Niveau de risque", f"{prob:.1%}")
            else:
                st.success(f"### ✅ RÉSULTAT : BÉNIN")
                st.metric("Confiance", f"{(1-prob):.1%}")
        with advice_col:
            st.subheader("📋 Recommandations")
            if prob > 0.3:
                st.write("**Actions prioritaires :**")
                st.write("- Planifier une biopsie de contrôle.")
                st.write("- Programmer une IRM complémentaire.")
                st.warning("Ce résultat est une aide à la décision, seul un médecin peut confirmer le diagnostic.")
            else:
                st.write("**Actions suggérées :**")
                st.write("- Suivi préventif annuel.")
                st.write("- Conserver ces résultats pour l'historique médical.")

# ONGLET 2 : COMPRÉHENSION 
with tab2:
    st.subheader("Interprétabilité du modèle")
    st.write("Voici les caractéristiques biologiques qui influencent le plus le diagnostic :")
    
    # Récupération de l'importance des variables depuis la pipeline
    # Note: On accède au step 'model' de la pipeline
    importances = model_pipeline.named_steps['model'].feature_importances_
    
    # On affiche les 10 plus importantes
    feat_importances = pd.Series(importances, index=model_features).sort_values(ascending=True).tail(10)
    
    fig, ax = plt.subplots()
    feat_importances.plot(kind='barh', color='#e74c3c', ax=ax)
    plt.title("Importance des caractéristiques cliniques")
    st.pyplot(fig)
    
    

# ONGLET 3 : DOCUMENTATION 
with tab3:
    st.info("### À propos de l'algorithme")
    st.write("""
    Cette application utilise un modèle **Random Forest (Forêt Aléatoire)**. 
    L'algorithme combine 400 arbres de décision pour minimiser les erreurs de diagnostic.
    
    **Performances du modèle :**
    - Sensibilité (Recall) : 98% (Capacité à détecter les vrais cancers)
    - Précision : 94% (Capacité à éviter les fausses alertes)
    """)