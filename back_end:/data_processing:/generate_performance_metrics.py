def generate_performance_metrics(df):
    # Eficiencia de goles (Goles / Goles Esperados)
    df['Eficiencia_goles'] = df['Goles'] / df['Goles_esperados']

    # Eficiencia de asistencias (Asistencias / Asistencias Esperadas)
    df['Eficiencia_asistencias'] = df['Asistencias'] / df['Asistencias_esperadas']

    # Participación ofensiva total (Goles + Asistencias)
    df['Participación_ofensiva'] = df['Goles'] + df['Asistencias']

    # Contribución defensiva (Tackles ganados + Intercepciones)
    df['Contribución_defensiva'] = df['Tackles_ganados_por_90'] + df['Intercepciones_por_90']

    return df
