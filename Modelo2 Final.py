from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Cargar datos
df = pd.read_csv(r'C:\Users\anton\OneDrive - Universidad de los andes\Antonia Streubel\ANDES\7. Semestre\Analítica Computacional\Proyecto 2\df_limpio.csv')

# Separar variables
X_num = df[['FAMI_TIENELAVADORA_Si', 'FAMI_TIENEAUTOMOVIL_Si', 'FAMI_TIENECOMPUTADOR_Si', 'COLE_AREA_UBICACION_URBANO']]
X_cat = df[[
    'ESTU_COD_RESIDE_MCPIO','FAMI_PERSONASHOGAR','ESTU_COD_MCPIO_PRESENTACION',
    'ESTU_TIPODOCUMENTO','FAMI_CUARTOSHOGAR','FAMI_EDUCACIONMADRE',
    'FAMI_ESTRATOVIVIENDA'
]]
X = pd.concat([X_num, X_cat], axis=1)
y = df['ALTO_RENDIMIENTO_MATE']

# Preprocesador
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), X_num.columns.tolist()),
    ("cat", OneHotEncoder(handle_unknown='ignore'), X_cat.columns.tolist())
])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Transformar los datos
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Crear modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Guardar modelo y preprocesador
model.save("modelo2.keras")
with open("preprocesador.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("\n✅ Modelo y preprocesador guardados.")

from sklearn.metrics import classification_report, roc_auc_score

y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")

