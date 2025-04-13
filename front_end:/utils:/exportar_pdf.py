from fpdf import FPDF

def exportar_pdf(dataframe, columnas, path_salida="/tmp/estadisticas_jugadores.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título
    pdf.cell(200, 10, txt="Estadísticas de Jugadoras de Fútbol", ln=True, align='C')

    # Insertar tabla si hay datos
    if not dataframe.empty:
        table_data = dataframe[columnas].values.tolist()
        for row in table_data:
            fila = " | ".join(str(x) for x in row)
            pdf.cell(200, 10, txt=fila, ln=True)

    # Guardar el PDF
    pdf.output(path_salida)
    return path_salida
