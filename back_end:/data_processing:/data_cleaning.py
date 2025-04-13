def clean_data(df):
    # Asegurar que 'Pos' sea categórica
    df['Pos'] = df['Pos'].astype('category')

    # Limpiar las posiciones de los jugadores (quedarse solo con las dos primeras letras si tiene más de dos)
    df['Pos'] = df['Pos'].apply(lambda x: x[:2] if len(x) > 2 else x)

    # Renombrar columnas para hacerlas más manejables
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

    return df
