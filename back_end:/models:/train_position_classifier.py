from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def train_position_classifier(df):
    # Características (X) y Etiqueta (y)
    X = df[['Age', 'Goles', 'Asistencias', 'Goles_esperados', 'Asistencias_esperadas',
            'Goles_por_90', 'Asistencias_por_90', 'Goles_esperados_por_90', 'Asistencias_esperadas_por_90',
            'Pases_completados%', 'Regates_completados%', 'Tackles_ganados_por_90', 'Intercepciones_por_90',
            'Acciones_contribuyendo_a_gol_por_90', 'Acciones_contribuyendo_a_disparo_por_90', 'Duelos_aereos_ganados%',
            'Pases_completados_por_90', 'Pases_clave_por_90',
            'Eficiencia_goles', 'Eficiencia_asistencias',
            'Participación_ofensiva', 'Contribución_defensiva']]

    y = df['Pos']  # Etiqueta

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modelo
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Predicciones
    y_pred = rf_model.predict(X_test)

    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Matriz de confusión (gráfico)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
    plt.title('Matriz de Confusión - Random Forest')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

    # Importancia de características
    feature_importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_df = pd.DataFrame({'Característica': feature_names, 'Importancia': feature_importances})
    feature_df = feature_df.sort_values(by='Importancia', ascending=False)

    # Gráfico de importancia
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Característica', data=feature_df)
    plt.title('Importancia de Características - Random Forest')
    plt.show()

    # Guardar modelo
    joblib.dump(rf_model, 'modelo_random_forest_TFM.pkl')
    joblib.dump(scaler, 'scaler_random_forest_TFM.pkl')

    return rf_model
