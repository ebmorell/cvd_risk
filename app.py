import os
import gdown
import cloudpickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# 🔗 DESCARGAR MODELO Y VARIABLES DESDE GOOGLE DRIVE
# ----------------------------

# Sustituye con tus IDs reales de Google Drive (¡IMPORTANTE!)
model_url = "https://drive.google.com/uc?id=15EkxZs9U8Xjro3dTTLxmozE-LaWT0vOI"
features_url = "https://drive.google.com/uc?id=1YrDDcQfaloaEDuKnmWPy8OmMRkHPzfcK"

if not os.path.exists("rsf_model.pkl"):
    gdown.download(model_url, "rsf_model.pkl", quiet=False)

if not os.path.exists("model_features.pkl"):
    gdown.download(features_url, "model_features.pkl", quiet=False)

# ----------------------------
# 🧠 CARGA DEL MODELO Y FEATURES CON CLOUDPICKLE
# ----------------------------

with open("rsf_model.pkl", "rb") as f:
    rsf = cloudpickle.load(f)

with open("model_features.pkl", "rb") as f:
    model_features = cloudpickle.load(f)

# ----------------------------
# 🧠 INTERFAZ STREAMLIT
# ----------------------------

st.title("🧠 Predicción de Riesgo Cardiovascular a 5 Años en Pacientes con VIH")

def prepare_input():
    input_dict = {}

    # Variables numéricas
    input_dict["Age"] = st.number_input("Edad", min_value=0, max_value=100, value=45)
    input_dict["CD4_Nadir"] = st.number_input("CD4 nadir", min_value=0, value=350)
    input_dict["CD8_Nadir"] = st.number_input("CD8 nadir", min_value=0, value=1000)
    input_dict["CD4_CD8_Ratio"] = st.number_input("CD4/CD8 Ratio", min_value=0.0, value=0.5)
    input_dict["Cholesterol"] = st.number_input("Colesterol total (mg/dL)", value=180.0)
    input_dict["HDL"] = st.number_input("HDL (mg/dL)", value=50.0)
    input_dict["Triglycerides"] = st.number_input("Triglicéridos (mg/dL)", value=150.0)
    input_dict["Non_HDL_Cholesterol"] = st.number_input("Colesterol no HDL (mg/dL)", value=130.0)
    input_dict["Triglyceride_HDL_Ratio"] = st.number_input("Relación TG/HDL", value=3.0)

    # Variables categóricas codificadas como dummies
    cat_vars = {
        "Sex": ["Man", "Woman"],
        "Transmission_mode": ["Homo/Bisexual", "Injecting Drug User", "Heterosexual", "Other or Unknown"],
        "Origin": ["Spain", "Not Spain"],
        "Education_Level": ["No studies", "Primary", "Secondary/High School", "University", "Other/Unknown"],
        "AIDS": ["No", "Yes"],
        "Viral_Load": ["< 100.000 copies/ml", "≥ 100.000 copies/ml"],
        "ART": ["2NRTI+1NNRTI", "2NRTI+1IP", "2NRTI+1II", "Other"],
        "Hepatitis_C": ["Negative", "Positive"],
        "Anticore_HBV": ["Negative", "Positive"],
        "HBP": ["No", "Yes"],
        "Smoking": ["No Smoking", "Current Smoking", "Past Smoking"],
        "Diabetes": ["No", "Yes"]
    }

    for var, options in cat_vars.items():
        selected = st.selectbox(var.replace("_", " "), options)
        for opt in options[1:]:  # crear columnas dummies omitida la referencia
            col_name = f"{var}_{opt}"
            input_dict[col_name] = 1 if selected == opt else 0

    return pd.DataFrame([input_dict])

# Obtener inputs del usuario
input_df = prepare_input()

# Añadir columnas faltantes con 0
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reordenar columnas
input_df = input_df[model_features]

# ----------------------------
# 🔮 PREDICCIÓN Y VISUALIZACIÓN
# ----------------------------

if st.button("Calcular riesgo"):
    surv_fn = rsf.predict_survival_function(input_df)[0]
    risk_5y = 1 - surv_fn(5)

    st.subheader(f"🔎 Riesgo estimado de evento cardiovascular a 5 años: {risk_5y:.2%}")

    st.subheader("📈 Curva de Supervivencia Estimada")
    fig, ax = plt.subplots()
    ax.plot(surv_fn.x, surv_fn.y, label="Supervivencia estimada")
    ax.axvline(x=5, color='red', linestyle='--', label="5 años")
    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel("Probabilidad de no tener evento")
    ax.set_title("Curva de Supervivencia")
    ax.legend()
    st.pyplot(fig)

