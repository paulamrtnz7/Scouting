import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle

# Cargar el DataFrame desde el archivo de estadísticas
@st.cache_data
def cargar_datos():
    return pd.read_csv('data/estadisticas_ligas.csv')  # Ajusta el path si es necesario

df = cargar_datos()

# Página de comparación
def mostrar_comparacion():
    st.title("Comparación de Jugadoras")
    st.write("""
        Aquí podrás comparar las estadísticas de dos o más jugadoras en función de las métricas clave. 
        Selecciona las jugadoras que deseas comparar y explora sus datos de rendimiento.
    """)

    # Filtro de selección de jugadoras
    jugadoras = df['Player'].unique()
    seleccionadas = st.multiselect("Selecciona dos o más jugadoras", options=jugadoras)

    # Métricas disponibles para comparar
    metricas = [
        'Goles', 'Asistencias', 'Goles_esperados', 'Asistencias_esperadas',
        'Goles_por_90', 'Asistencias_por_90', 'Goles_esperados_por_90',
        'Asistencias_esperadas_por_90', 'Pases_completados%', 'Regates_completados%',
        'Tackles_ganados_por_90', 'Intercepciones_por_90',
        'Acciones_contribuyendo_a_disparo_por_90', 'Acciones_contribuyendo_a_gol_por_90',
        'Duelos_aereos_ganados%', 'Pases_completados_por_90', 'Pases_clave_por_90',
        'Eficiencia_goles', 'Eficiencia_asistencias', 'Participación_ofensiva',
        'Contribución_defensiva'
    ]

    selected_metrics = st.multiselect("Selecciona métricas a comparar", metricas, default=metricas)

    if len(seleccionadas) >= 2:
        comparacion = df[df['Player'].isin(seleccionadas)]

        st.write(f"Comparación de: {', '.join(seleccionadas)}")
        st.dataframe(comparacion[['Player', 'Age', 'Pos', 'Squad'] + selected_metrics])

        # Gráfico Radar
        st.subheader("Gráfico Radar de Comparación")
        fig = go.Figure()

        for _, row in comparacion.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metrica] for metrica in selected_metrics],
                theta=selected_metrics,
                fill='toself',
                name=row['Player']
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(df[selected_metrics].max()) * 1.1])
            ),
            showlegend=True,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    elif len(seleccionadas) == 1:
        st.warning("Selecciona al menos dos jugadoras para comparar.")
    else:
        st.info("Selecciona jugadoras para comenzar la comparación.")

# Carga del modelo ML (si lo necesitas más adelante)
@st.cache_resource
def load_model():
    with open('modelo_random_forest_TFM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
