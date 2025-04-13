
# Scouting de Jugadores de Fútbol 

Este proyecto es una herramienta de **scouting** de jugadores de fútbol, que permite evaluar y comparar el rendimiento de jugadores de diferentes equipos y ligas. El sistema utiliza métricas clave y un modelo de clasificación para facilitar la detección de talentos y comparaciones entre posibles fichajes. Está basado en un archivo de datos con estadísticas por jornada de jugadores de equipos de las ligas española, francesa, italiana, alemana, inglesa, portuguesa y holandesa. 

## Características
- Filtros interactivos para seleccionar **posición**, **edad**, **club**, y **país**.
- **Visualizaciones interactivas** de las métricas clave de los jugadores.
- **Modelo de clasificación** para predecir posiciones de los jugadores utilizando **Random Forest**.
- Opción para **exportar los resultados en un archivo PDF**.
- **Exportación de datos** en formato CSV.

## Requisitos

El proyecto está construido utilizando Python y las siguientes librerías:

- **streamlit**: Para la interfaz web.
- **pandas**: Para manipulación y análisis de datos.
- **numpy**: Para cálculos numéricos.
- **matplotlib** y **plotly**: Para las visualizaciones.
- **scikit-learn**: Para los modelos de machine learning (Random Forest y XGBoost).
- **fpdf**: Para la exportación de los datos a PDF.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/paulamrtnz7/Scouting-jugadores-futbol.git
   cd Scouting-jugadores-futbol
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows usa venv\Scriptsctivate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Coloca los archivos CSV de estadísticas de los jugadores en la carpeta correspondiente del proyecto.
2. Corre la aplicación de Streamlit:
   ```bash
   streamlit run app.py
   ```

   Esto abrirá la aplicación en tu navegador predeterminado.

## Funcionalidad

La aplicación permite:

- **Filtros de selección**: Puedes filtrar los jugadores por **posición**, **edad**, **club**, y **país**.
- **Gráficos interactivos**: Los gráficos de barras y circulares muestran las métricas de los jugadores por distintas categorías.
- **Exportar a PDF**: Los datos seleccionados pueden ser exportados a un archivo PDF con un resumen de las estadísticas.

## Modelos de Machine Learning

El proyecto incluye un modelo de **clasificación** para predecir la posición de los jugadores basado en **Random Forest**. 

## Contribuciones

Si deseas contribuir al proyecto:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-caracteristica`).
3. Realiza tus cambios y haz un commit (`git commit -am 'Añadir nueva característica'`).
4. Sube tus cambios (`git push origin feature/nueva-caracteristica`).
5. Abre un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Si tienes alguna pregunta, no dudes en contactarme a través de [mi perfil en GitHub](https://github.com/paulamrtnz7).
