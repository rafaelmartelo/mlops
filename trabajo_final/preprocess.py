import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str)
parser.add_argument('--output-train', type=str)
parser.add_argument('--output-test', type=str)
args = parser.parse_args()

print("📥 Leyendo archivo desde:", args.input_data)
df = pd.read_csv(os.path.join(args.input_data, "diabetic_data.csv"))

# Reemplazar "?" con NaN
df.replace("?", np.nan, inplace=True)

# Variable objetivo
y = df["readmitted"].copy()
X = df.drop(columns=["readmitted"])

# Columnas categóricas y numéricas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

# Convertir a categorías
for col in cat_cols:
    X[col] = X[col].astype("category")

# Codificar categorías
X[cat_cols] = X[cat_cols].apply(lambda x: x.cat.codes)

# Escalar numéricas
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Sobremuestreo manual
print("⚖️ Aplicando sobremuestreo con resample()...")
df_resampled = pd.concat([X, y], axis=1)
majority = df_resampled[df_resampled["readmitted"] == "NO"]
minority = df_resampled[df_resampled["readmitted"] != "NO"]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_balanced = pd.concat([majority, minority_upsampled])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Dividir nuevamente
X_bal = df_balanced.drop(columns=["readmitted"])
y_bal = df_balanced["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Guardar
print("💾 Guardando sets...")
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv(os.path.join(args.output_train, "train.csv"), index=False)
test.to_csv(os.path.join(args.output_test, "test.csv"), index=False)

print("✅ Preprocesamiento terminado.")
