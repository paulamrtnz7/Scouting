import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Función para cargar el modelo
@st.cache_resource
def load_model():
    with open('modelo_random_forest_TFM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def mostrar_prediccion():
    st.title("Predicción de Jugadoras con Random Forest")
    st.write("""
    En esta página puedes predecir la **posición de una jugadora** utilizando sus estadísticas.

    Para hacerlo, introduce las métricas y presiona "Predecir".
    """)

    model = load_model()

    # Entradas del usuario con claves únicas
    st.subheader("Selecciona las métricas de la jugadora")

    goles = st.number_input("Goles", min_value=0, value=5, key="goles_input")
    asistencias = st.number_input("Asistencias", min_value=0, value=2, key="asistencias_input")
    goles_esperados = st.number_input("Goles Esperados (xG)", min_value=0.0, value=0.3, key="goles_esperados_input")
    asistencias_esperadas = st.number_input("Asistencias Esperadas (xA)", min_value=0.0, value=0.2, key="asistencias_esperadas_input")
    goles_por_90 = st.number_input("Goles por 90 minutos", min_value=0.0, value=0.3, key="goles_por_90_input")
    asistencias_por_90 = st.number_input("Asistencias por 90 minutos", min_value=0.0, value=0.2, key="asistencias_por_90_input")
    goles_esperados_por_90 = st.number_input("Goles Esperados por 90 minutos", min_value=0.0, value=0.3, key="goles_esperados_por_90_input")
    asistencias_esperadas_por_90 = st.number_input("Asistencias Esperadas por 90 minutos", min_value=0.0, value=0.2, key="asistencias_esperadas_por_90_input")
    pases_completados_percent = st.number_input("Pases Completados (%)", min_value=0.0, value=80.0, key="pases_completados_percent_input")
    regates_completados_percent = st.number_input("Regates Completados (%)", min_value=0.0, value=60.0, key="regates_completados_percent_input")
    tackles_ganados_por_90 = st.number_input("Entradas Ganadas por 90 minutos", min_value=0.0, value=2.0, key="tackles_ganados_por_90_input")
    intercepciones_por_90 = st.number_input("Intercepciones por 90 minutos", min_value=0.0, value=1.0, key="intercepciones_por_90_input")
    acciones_contribuyendo_a_gol_por_90 = st.number_input("Acciones contribuyendo a gol por 90 minutos", min_value=0.0, value=0.1, key="acciones_contribuyendo_a_gol_por_90_input")
    acciones_contribuyendo_a_disparo_por_90 = st.number_input("Acciones contribuyendo a disparo por 90 minutos", min_value=0.0, value=0.1, key="acciones_contribuyendo_a_disparo_por_90_input")
    duelos_aereos_ganados_percent = st.number_input("Duelos Aéreos Ganados (%)", min_value=0.0, value=50.0, key="duelos_aereos_ganados_percent_input")
    pases_completados_por_90 = st.number_input("Pases Completados por 90 minutos", min_value=0.0, value=80.0, key="pases_completados_por_90_input")
    pases_clave_por_90 = st.number_input("Pases Clave por 90 minutos", min_value=0.0, value=0.2, key="pases_clave_por_90_input")

    # Crear array para predicción
    features = np.array([[
        goles, asistencias, goles_esperados, asistencias_esperadas,
        goles_por_90, asistencias_por_90, goles_esperados_por_90, asistencias_esperadas_por_90,
        pases_completados_percent, regates_completados_percent, tackles_ganados_por_90, intercepciones_por_90,
        acciones_contribuyendo_a_gol_por_90, acciones_contribuyendo_a_disparo_por_90,
        duelos_aereos_ganados_percent, pases_completados_por_90, pases_clave_por_90,
        asistencias_esperadas_por_90  # Duplicada si está en los datos originales
    ]])

    # Botón de predicción
    if st.button("Predecir"):
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)

        st.success(f"Posición predicha: **{prediction[0]}**")

        # Probabilidades por clase
        positions = model.classes_
        prob_df = pd.DataFrame(prediction_prob, columns=positions)

        st.write("Probabilidades por posición:")
        st.dataframe(prob_df)
