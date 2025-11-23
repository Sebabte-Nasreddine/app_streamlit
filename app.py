import streamlit as st
import pandas as pd
import pickle

# --- 1. Charger le modèle ---
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Prédiction de la maladie cardiaque (CHD)")

# Champs de saisie pour les variabls
st.sidebar.header("Paramètres de l'utilisateur")

# Exemple : variables numériques
age = st.sidebar.number_input("Âge", min_value=0, max_value=120, value=50)
sbp = st.sidebar.number_input("SBP", value=120)
tobacco = st.sidebar.number_input("Tobacco", value=5.0)
ldl = st.sidebar.number_input("LDL", value=100.0)
adiposity = st.sidebar.number_input("Adiposity", value=25.0)
famhist = st.sidebar.selectbox("Historique familial (famhist)", ["present", "absent"])
typea = st.sidebar.number_input("Type-A score", value=50)
obesity = st.sidebar.number_input("Obesity", value=25.0)
alcohol = st.sidebar.number_input("Alcohol", value=10.0)
ageo = st.sidebar.number_input("Ageo", value=50)


input_data = pd.DataFrame({
    'age': [age],
    'sbp': [sbp],
    'tobacco': [tobacco],
    'ldl': [ldl],
    'adiposity': [adiposity],
    'famhist': [famhist],
    'typea': [typea],
    'obesity': [obesity],
    'alcohol': [alcohol],
    'ageo': [ageo]
})

st.subheader("Données saisies")
st.write(input_data)

# Prédiction 
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][1]  

st.subheader("Résultat de la prédiction")
st.write(f"Prédiction CHD : {'Présent' if prediction == 1 else 'Absent'}")
st.write(f"Probabilité associée : {prediction_proba:.2f}")
