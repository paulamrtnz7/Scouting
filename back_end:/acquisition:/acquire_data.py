# Montamos Drive con:
from google.colab import drive
drive.mount('/content/drive')

# Librerías necesarias
import pandas as pd

# Cargar archivo
def load_data(file_path):
    file_path = "/content/drive/MyDrive/MASTER PYTHON/TFM/FBREF_players_2223.csv"
    df = pd.read_csv(file_path)
    return df
