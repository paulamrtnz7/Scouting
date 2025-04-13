import streamlit as st
import pandas as pd

def aplicar_filtros_sidebar(df):
    st.sidebar.title("Filtros")

    # Filtro de posición
    posiciones = df['Pos'].unique()
    posicion_filtro = st.sidebar.selectbox("Selecciona una posición", ["Todas las posiciones"] + list(posiciones))

    # Filtro de liga
    ligas = df['Competition'].unique()
    liga_filtro = st.sidebar.selectbox("Selecciona una liga", ["Todas las ligas"] + list(ligas))

    # Filtro de texto: jugador o equipo
    busqueda = st.sidebar.text_input("Buscar por nombre de jugador o equipo")

    # Filtro de edad
    edad_min = int(df['Age'].min())
    edad_max = int(df['Age'].max())
    edad_filtro = st.sidebar.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))

    # Filtro de goles
    goles_min = int(df['Goles'].min())
    goles_max = int(df['Goles'].max())
    goles_filtro = st.sidebar.slider("Rango de goles", goles_min, goles_max, (goles_min, goles_max))

    # Selección de métricas
    st.sidebar.subheader("Selecciona las métricas a mostrar:")
    metricas_disponibles = [
        'Goles',
        'Asistencias',
        'Goles_esperados',
        'Asistencias_esperadas',
        'Goles_por_90',
        'Asistencias_por_90',
        'Goles_esperados_por_90',
        'Asistencias_esperadas_por_90',
        'Pases_completados%',
        'Regates_completados%',
        'Tackles_ganados_por_90',
        'Intercepciones_por_90',
        'Acciones_contribuyendo_a_disparo_por_90',
        'Acciones_contribuyendo_a_gol_por_90',
        'Duelos_aereos_ganados%',
        'Pases_completados_por_90',
        'Pases_clave_por_90',
        'Eficiencia_goles',
        'Eficiencia_asistencias',
        'Participación_ofensiva',
        'Contribución_defensiva'
    ]
    metricas_seleccionadas = st.sidebar.multiselect("Métricas", metricas_disponibles, default=metricas_disponibles)

    # Devolver valores seleccionados
    return {
        'posicion': posicion_filtro,
        'liga': liga_filtro,
        'busqueda': busqueda,
        'edad': edad_filtro,
        'goles': goles_filtro,
        'metricas': metricas_seleccionadas
    }


def filtrar_dataframe(df, filtros):
    # Aplicar filtro de posición
    if filtros['posicion'] != "Todas las posiciones":
        df = df[df['Pos'] == filtros['posicion']]

    # Aplicar filtro de liga
    if filtros['liga'] != "Todas las ligas":
        df = df[df['Competition'] == filtros['liga']]

    # Filtro de texto (jugador o equipo)
    if filtros['busqueda']:
        df = df[df['Player'].str.contains(filtros['busqueda'], case=False, na=False) |
                df['Squad'].str.contains(filtros['busqueda'], case=False, na=False)]

    # Filtro de edad
    df = df[(df['Age'] >= filtros['edad'][0]) & (df['Age'] <= filtros['edad'][1])]

    # Filtro de goles
    df = df[(df['Goles'] >= filtros['goles'][0]) & (df['Goles'] <= filtros['goles'][1])]

    return df
