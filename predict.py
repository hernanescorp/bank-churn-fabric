import pickle
import pandas as pd
import numpy as np
import os

# Crear carpeta de salida si no existe
os.makedirs('test_predictions/output_predictions', exist_ok=True)

# 1. Cargar modelo
print("📦 Cargando modelo...")
with open("models/lgbm_sm/model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Modelo cargado")

# 2. Generar datos simulados (100 registros con 18 features)
print("📊 Generando datos simulados...")
np.random.seed(42)  # Para reproducibilidad
X = pd.DataFrame({
    'CreditScore': np.random.randint(300, 900, 100),
    'Geography_France': np.random.randint(0, 2, 100),
    'Geography_Germany': np.random.randint(0, 2, 100),
    'Geography_Spain': np.random.randint(0, 2, 100),
    'Gender_Female': np.random.randint(0, 2, 100),
    'Gender_Male': np.random.randint(0, 2, 100),
    'Age': np.random.randint(18, 100, 100),
    'Tenure': np.random.randint(0, 10, 100),
    'Balance': np.random.uniform(0, 250000, 100),
    'NumOfProducts': np.random.randint(1, 4, 100),
    'HasCrCard': np.random.randint(0, 2, 100),
    'IsActiveMember': np.random.randint(0, 2, 100),
    'EstimatedSalary': np.random.randint(10000, 200000, 100),
    'NewTenure': np.random.randint(0, 10, 100),
    'NewCreditsScore': np.random.randint(0, 100, 100),
    'NewAgeScore': np.random.randint(0, 100, 100),
    'NewBalanceScore': np.random.randint(0, 100, 100),
    'NewEstSalaryScore': np.random.randint(0, 100, 100),
})

print(f"✅ Datos generados: {X.shape[0]} registros, {X.shape[1]} features")

# 3. Hacer predicciones
print("🤖 Haciendo predicciones...")
predictions = model.predict(X)
print("✅ Predicciones completadas")

# 4. Crear resultado con datos + predicción
results = pd.concat([X, pd.DataFrame({'Prediction': predictions})], axis=1)

# 5. Guardar CSV
output_path = 'test_predictions/output_predictions/predictions.csv'
results.to_csv(output_path, index=False)
print(f"✅ Predicciones guardadas en: {output_path}")

# 6. Mostrar resumen
print(f"\n📈 RESUMEN:")
print(f"   - Total registros: {len(predictions)}")
print(f"   - Churn (1): {sum(predictions)} ({100*sum(predictions)/len(predictions):.1f}%)")
print(f"   - No Churn (0): {len(predictions)-sum(predictions)} ({100*(len(predictions)-sum(predictions))/len(predictions):.1f}%)")
print(f"\n💾 Archivo guardado: {output_path}")
