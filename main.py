import streamlit as st
import xgboost as xgb
import numpy as np

# Charger le modèle XGBoost
model_path = "calcul_duree_theorique.json"
xgb_model = xgb.Booster()
xgb_model.load_model(model_path)

# Configuration de l'application Streamlit
st.title("Estimation de la charge de travail d'un dossier en fonction de ses caractéristiques")

# Entrées utilisateur
nombre_de_salaries = st.number_input("Nombre de salariés", min_value=0, value=0, step=1)
openpaye_ou_silae = st.selectbox("Logiciel utilisé (1 = Openpaye, 2 = Silae)", [1, 2])
declarations_sociales = st.selectbox("Fréquence des déclarations sociales (1 = Mensuelle, 2 = Trimestrielle)", [1, 2])
forme_juridique = st.selectbox("Forme juridique (1 = SAS, 2 = SASU, 3 = EURL, 4 = SARL)", [1, 2, 3, 4])
regime = st.selectbox("Régime (1 = Normal, 2 = Simplifié)", [1, 2])
CA = st.number_input("Chiffre d'affaires", min_value=0, value=0, step=1000)
immobilisations = st.number_input("Nombre d'immobilisations", min_value=0, value=0, step=1)
ecritures_comptables = st.number_input("Nombre d'écritures comptables", min_value=0, value=0, step=10)
secteur = st.selectbox("Secteur (1 = Commerce, 2 = Artisanat, 3 = Libéral, 4 = Industrie, 5 = Immobilier)", [1, 2, 3, 4, 5])
tva = st.selectbox("TVA (1 = Mensuelle, 2 = Trimestrielle)", [1, 2])
erp = st.selectbox("ERP utilisé (1 = Teogest, 2 = Odoo)", [1, 2])
compta = st.selectbox("Comptabilité (1 = Interne, 2 = Externe)", [1, 2])

# Préparation des données pour la prédiction
data_to_predict = np.array([[
    nombre_de_salaries,
    openpaye_ou_silae,
    declarations_sociales,
    forme_juridique,
    regime,
    CA,
    immobilisations,
    ecritures_comptables,
    secteur,
    tva,
    erp,
    compta
]])

# Réaliser une prédiction si l'utilisateur clique sur le bouton
if st.button("Calcul du nombre d'heures théorique"):
    dmatrix = xgb.DMatrix(data_to_predict)
    predicted_value = xgb_model.predict(dmatrix)
    st.subheader(f"En fonction des caractéristiques renseignées, le dossier devrait représenter une charge de travail de {predicted_value[0]:.2f} heures en moyenne par mois")

st.write("---")
