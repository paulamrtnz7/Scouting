import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# =====================
# CONFIGURACIÓN DE LOGIN
# =====================
USUARIO_VALIDO = 'admin'
CONTRASENA_VALIDA = 'proyectopaula'

def login():
    st.title('Inicio de Sesión')
    usuario = st.text_input('Usuario', '')
    contrasena = st.text_input('Contraseña', '', type='password')
    if st.button('Iniciar sesión'):
        if usuario == USUARIO_VALIDO and contrasena == CONTRASENA_VALIDA:
            st.session_state.logged_in = True
            st.success('¡Inicio de sesión exitoso!')
        else:
            st.session_state.logged_in = False
            st.error('Usuario o contraseña incorrectos')

# =====================
# ESTADO DE SESIÓN
# =====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()  # Detener ejecución hasta que se loguee correctamente

# =====================
# CARGA DE DATOS Y MODELO
# =====================
file_path = "/content/drive/MyDrive/MASTER PYTHON/TFM/FBREF_players_2223.csv"
df = pd.read_csv(file_path, sep=';')

loaded_model = joblib.load('/content/modelo_random_forest_TFM.pkl')

df['Pos'] = df['Pos'].astype('category')
df['Pos'] = df['Pos'].apply(lambda x: x[:2] if len(x) > 2 else x)

# Renombrar columnas
df.rename(columns={
    'Gls': 'Goles',
    'Ast': 'Asistencias',
    'xG': 'Goles_esperados',
    'xAG': 'Asistencias_esperadas',
    'Gls/90': 'Goles_por_90',
    'Ast/90': 'Asistencias_por_90',
    'xG/90': 'Goles_esperados_por_90',
    'xAG/90': 'Goles_asistidos_esperados_por_90',
    'Passes%': 'Pases_completados%',
    'Dribbles%': 'Regates_completados%',
    'TklW/90': 'Tackles_ganados_por_90',
    'Int/90': 'Intercepciones_por_90',
    'SCA/90': 'Acciones_contribuyendo_a_disparo_por_90',
    'GCA/90': 'Acciones_contribuyendo_a_gol_por_90',
    'Aerial%': 'Duelos_aereos_ganados%',
    'PassesCompleted/90': 'Pases_completados_por_90',
    'KP/90': 'Pases_clave_por_90',
    'xA/90': 'Asistencias_esperadas_por_90'
}, inplace=True)

# Métricas derivadas
df['Eficiencia_goles'] = df['Goles'] / df['Goles_esperados']
df['Eficiencia_asistencias'] = df['Asistencias'] / df['Asistencias_esperadas']
df['Participación_ofensiva'] = df['Goles'] + df['Asistencias']
df['Contribución_defensiva'] = df['Tackles_ganados_por_90'] + df['Intercepciones_por_90']

# =====================
# INTERFAZ STREAMLIT
# =====================
st.sidebar.title("Menú")
menu = st.sidebar.selectbox(
    "Selecciona una página",
    ("Inicio", "Estadísticas", "Comparar Jugadores", "Predicción", "Acerca de")
)

# Página de inicio
if menu == "Inicio":
  st.title("Bienvenido al Scouting de Jugadores de Fútbol")
  st.write("""
    ¡Bienvenido a la plataforma de scouting de jugadores de fútbol!

    Aquí podrás explorar y comparar las estadísticas de jugadores de diversas ligas de fútbol. El objetivo de esta aplicación es permitirte analizar el rendimiento de los jugadores, comparar diferentes jugadores y encontrar nuevos talentos.

    **¿Qué puedes hacer aquí?**

    - Explorar las **estadísticas individuales** de los jugadores de varias ligas.
    - **Comparar jugadores** entre sí en función de sus estadísticas y métricas.
    - Conocer el **propósito** y el contexto del proyecto en la sección "Acerca de".

    Para empezar, puedes dirigirte a la página de **Estadísticas** o **Comparar Jugadores** en el menú.
  """)

# Página de estadísticas
elif menu == "Estadísticas":
  st.title("Estadísticas de Jugadores")
  st.write("""
    En esta sección podrás explorar las estadísticas de los jugadores de fútbol de diversas ligas. Utiliza los filtros para personalizar los datos que quieres ver.
  """)

  # Filtro Desplegable (Selectbox): Para seleccionar la posición
  posiciones = df['Pos'].unique()
  posicion_filtro = st.sidebar.selectbox("Selecciona una posición", ["Todas las posiciones"] + list(posiciones))

  # Filtro Desplegable (Selectbox): Para seleccionar la liga o el equipo
  ligas = df['Competition'].unique()
  liga_filtro = st.sidebar.selectbox("Selecciona una liga", ["Todas las ligas"] + list(ligas))

  # Filtro de Texto: Para buscar por jugador o equipo
  busqueda = st.sidebar.text_input("Buscar por nombre de jugador o equipo")

  # Filtro de Checkboxes: Para activar o desactivar el filtro por métricas
  st.sidebar.subheader("Selecciona las métricas que quieres ver:")
  metricas = [
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

  # Checkboxes para seleccionar métricas
  selected_metrics = st.sidebar.multiselect('Selecciona las métricas a filtrar', metricas, default=metricas)  # Por defecto, selecciona todas las métricas

  # Slider para rango de edad
  edad_min = int(df['Age'].min())
  edad_max = int(df['Age'].max())
  edad_filtro = st.sidebar.slider("Selecciona el rango de edad", edad_min, edad_max, (edad_min, edad_max))

  # Slider para rango de goles
  goles_min = int(df['Goles'].min())
  goles_max = int(df['Goles'].max())
  goles_filtro = st.sidebar.slider("Selecciona el rango de goles", goles_min, goles_max, (goles_min, goles_max))

  # Aplicar los filtros
  df_filtrado = df[
      (df['Pos'].isin([posicion_filtro] if posicion_filtro != "Todas las posiciones" else posiciones)) &
      (df['Competition'].isin([liga_filtro] if liga_filtro != "Todas las ligas" else ligas)) &
      (df['Squad'].str.contains(busqueda, case=False) if busqueda else df['Squad'])  # Buscando por nombre de equipo o jugador
  ]

  # Aplicar los filtros de edad y goles con los sliders
  df_filtrado = df_filtrado[
      (df_filtrado['Age'] >= edad_filtro[0]) & (df_filtrado['Age'] <= edad_filtro[1]) &
      (df_filtrado['Goles'] >= goles_filtro[0]) & (df_filtrado['Goles'] <= goles_filtro[1])
  ]

  # Mostrar los datos filtrados
  st.write("Jugadores encontrados:", df_filtrado.shape[0])

  if selected_metrics:
    st.dataframe(df_filtrado[['Player', 'Age', 'Pos', 'Squad', 'Competition', 'Nation'] + selected_metrics])
  else:
    st.warning("Selecciona al menos una métrica para mostrar.")

  # Gráficos
  if not df_filtrado.empty:
    st.subheader("Visualizaciones")

    # 1. Top goleadores
    st.markdown("#### Jugadores con más goles")
    top_goleadoras = df_filtrado.sort_values('Goles', ascending=False).head(10)
    fig1 = px.bar(top_goleadoras, x='Player', y='Goles', color='Squad', text='Goles')
    st.plotly_chart(fig1)

    # 2. Goles vs Asistencias
    st.markdown("#### Relación entre Goles y Asistencias")
    fig2 = px.scatter(df_filtrado, x='Goles', y='Asistencias', color='Pos', hover_data=['Player', 'Squad'])
    st.plotly_chart(fig2)

    # 3. Boxplot de goles por posición
    st.markdown("#### Distribución de goles por posición")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_filtrado, x='Pos', y='Goles', palette='pastel', ax=ax3)
    ax3.set_title("Boxplot de Goles por Posición")
    st.pyplot(fig3)

    # 4. Distribución por posición
    st.markdown("#### Distribución de jugadores por posición")
    distrib_pos = df_filtrado['Pos'].value_counts().reset_index()
    distrib_pos.columns = ['Posición', 'Cantidad']
    fig4 = px.bar(distrib_pos, x='Posición', y='Cantidad', title='Distribución de jugadores por posición')
    st.plotly_chart(fig4)

  # Botón para exportar a PDF
  if st.button("Exportar a PDF"):
    # Generar PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.cell(200, 10, txt="Estadísticas de Jugadores de Fútbol", ln=True, align='C')

    # Insertar tabla
    if not df_filtrado.empty:
      table_data = df_filtrado[['Player', 'Age', 'Pos', 'Squad', 'Competition', 'Nation'] + selected_metrics].values.tolist()
      for row in table_data:
        pdf.cell(200, 10, txt=" | ".join(str(x) for x in row), ln=True)

    # Guardar el PDF
    pdf.output("/tmp/estadisticas_jugadores.pdf")
    st.success("PDF generado exitosamente. Haz clic en el siguiente enlace para descargarlo:")
    st.download_button("Descargar PDF", "/tmp/estadisticas_jugadores.pdf", file_name="estadisticas_jugadores.pdf")

# Página de comparación de jugadores
elif menu == "Comparar Jugadores":
  st.title("Comparación de Jugadores")
  st.write("""
    Aquí podrás comparar las estadísticas de dos o más jugadores en función de las métricas clave. Selecciona los jugadores que deseas comparar y explora sus datos de rendimiento.
  """)

  # Filtro de selección de jugadores
  jugadores = df['Player'].unique()
  seleccionados = st.multiselect("Selecciona dos o más jugadoras", options=jugadores)

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

  if len(seleccionados) >= 2:
    comparacion = df[df['Player'].isin(seleccionados)]

    st.write(f"Comparación de: {', '.join(seleccionados)}")
    st.dataframe(comparacion[['Player', 'Age', 'Pos', 'Squad'] + selected_metrics])

    # Gráfico de radar
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

  elif len(seleccionados) == 1:
    st.warning("Selecciona al menos dos jugadores para comparar.")
  else:
    st.info("Selecciona jugadores para comenzar la comparación.")

# Cargar el modelo entrenado (asegúrate de tener el archivo .pkl)
import joblib

@st.cache
def load_model():
  model = joblib.load('modelo_random_forest_TFM.pkl')
  st.write(f"Tipo del modelo cargado: {type(model)}")  # Verifica el tipo del modelo
  return model


# Página de predicción
if menu == "Predicción":
  st.title("Predicción de Jugadores con Random Forest")
  st.write("""
    En esta página puedes predecir la **posición de un jugador** utilizando sus estadísticas.

    Para hacerlo, selecciona las métricas del jugador y presiona "Predecir".
  """)

  # Cargar el modelo
  model = load_model()

  # Selección de estadísticas de un jugador
  st.subheader("Selecciona las métricas del jugador")

  # Filtros y entrada de estadísticas con claves únicas
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

  # Crear el array de características para la predicción
  features = np.array([[goles, asistencias, goles_esperados, asistencias_esperadas,
                        goles_por_90, asistencias_por_90, goles_esperados_por_90, asistencias_esperadas_por_90,
                        pases_completados_percent, regates_completados_percent, tackles_ganados_por_90, intercepciones_por_90,
                        acciones_contribuyendo_a_gol_por_90, acciones_contribuyendo_a_disparo_por_90, duelos_aereos_ganados_percent,
                        pases_completados_por_90, pases_clave_por_90, asistencias_esperadas_por_90]])

  # Predecir con el modelo
  if st.button("Predecir"):
    # Realizar la predicción
    prediction = model.predict(features)
    prediction_prob = model.predict_proba(features)

    # Mostrar la predicción
    st.write(f"Posición predicha: {prediction[0]}")
    st.write(f"Probabilidades: {prediction_prob}")

    # Mostrar probabilidades
    positions = model.classes_  # Las clases (posiciones) del modelo
    prob_df = pd.DataFrame(prediction_prob, columns=positions)
    st.write("Probabilidades por posición:")
    st.dataframe(prob_df)

# Página "Acerca de"
elif menu == "Acerca de":
  st.title("Acerca de este Proyecto")
  st.write("""
    **Objetivo del Proyecto**

    Este proyecto está diseñado para ayudar a los scouts de fútbol a evaluar y comparar jugadores de diversas ligas, con el fin de encontrar talentos potenciales. Utilizando estadísticas de rendimiento como goles, asistencias, duelos, intercepciones y más, la plataforma permite realizar análisis detallados.

    **¿Cómo funciona?**

    - Los usuarios pueden explorar estadísticas individuales de jugadores.
    - Pueden comparar estadísticas de diferentes jugadores para evaluar su rendimiento.
    - La aplicación utiliza datos de ligas de fútbol de Europa.

    **Tecnologías utilizadas**
    - Python (Streamlit para la interfaz)
    - Pandas (para el análisis de datos)
    - Gráficos interactivos con Plotly
    - Machine Learning (para el análisis predictivo de jugadores)
  """)
