import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution_goles(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Goles'], kde=True)
    plt.title('Distribución de Goles')
    plt.xlabel('Goles')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_distribution_asistencias(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Asistencias'], kde=True)
    plt.title('Distribución de Asistencias')
    plt.xlabel('Asistencias')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_goles_por_posicion(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Pos', y='Goles', data=df)
    plt.title('Distribución de Goles por Posición')
    plt.xlabel('Posición')
    plt.ylabel('Goles')
    plt.show()

def plot_xg_vs_goles(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Goles_esperados', y='Goles', data=df)
    plt.title('Relación entre Goles y Goles Esperados')
    plt.xlabel('Goles Esperados (xG)')
    plt.ylabel('Goles Anotados')
    plt.show()

def plot_distribution_edad(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True)
    plt.title('Distribución de Edad')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.show()
