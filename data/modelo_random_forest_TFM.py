import joblib
joblib.dump(rf_model, 'modelo_random_forest_TFM.pkl')

#Para cargarlo más adelante
loaded_model = joblib.load('data/modelo_random_forest_TFM.pkl')
