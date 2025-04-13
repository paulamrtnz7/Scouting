import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF

# Cargar los datos (asegúrate de ajustar la ruta al archivo CSV)
df = pd.read_csv('ruta/a/tu/archivo.csv')

# Página de estadísticas
def mostrar_estadisticas():
    st.title("Estadísticas de Jugadores")
    st.write("""
        En esta sección podrás explorar las estadísticas de los jugadores de fútbol de diversas ligas.
        Utiliza los filtros para personalizar los datos que quieres ver.
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
        'Goles', 'Asistencias', 'Goles_esperados', 'Asistencias_esperadas', 
        'Goles_por_90', 'Asistencias_por_90', 'Goles_esperados_por_90', 
        'Asistencias_esperadas_por_90', 'Pases_completados%', 'Regates_completados%',
        'Tackles_ganados_por_90', 'Intercepciones_por_90', 
        'Acciones_contribuyendo_a_disparo_por_90', 'Acciones_contribuyendo_a_gol_por_90',
        'Duelos_aereos_ganados%', 'Pases_completados_por_90', 
        'Pases_clave_por_90', 'Eficiencia_goles', 'Eficiencia_asistencias',
        'Participación_ofensiva', 'Contribución_defensiva'
    ]

    # Checkboxes para seleccionar métricas
    selected_metrics = st.sidebar.multiselect(
        'Selecciona las métricas a filtrar',
        metricas,
        default=metricas  # Por defecto, selecciona todas las métricas
    )

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
        (df['Squad'].str.contains(busqueda, case=False) if busqueda else df['Squad'])
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

if __name__ == "__main__":
    mostrar_estadisticas()
