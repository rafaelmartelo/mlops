import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str)
parser.add_argument('--output-train', type=str)
parser.add_argument('--output-test', type=str)
args = parser.parse_args()

print("📥 Leyendo archivo desde:", args.input_data)
df = pd.read_csv(os.path.join(args.input_data, "diabetic_data.csv"))

# Reemplazar valores "?" por NaN
df.replace("?", np.nan, inplace=True)

# Variable objetivo
y = df["readmitted"].copy()
X = df.drop(columns=["readmitted"])

# Identificar columnas categóricas y numéricas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

print("🔣 Columnas categóricas:", cat_cols)
print("🔢 Columnas numéricas:", num_cols)

# Convertir columnas categóricas a tipo category
for col in cat_cols:
    X[col] = X[col].astype("category")

# Codificar variables categóricas como códigos numéricos
X[cat_cols] = X[cat_cols].apply(lambda x: x.cat.codes)

# Escalar variables numéricas
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Oversampling con RandomOverSampler
print("⚖️ Aplicando RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# División train/test
print("🧪 Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Combinar y guardar
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

print("💾 Guardando train en:", args.output_train)
train.to_csv(os.path.join(args.output_train, "train.csv"), index=False)

print("💾 Guardando test en:", args.output_test)
test.to_csv(os.path.join(args.output_test, "test.csv"), index=False)

print("✅ Preprocesamiento completado.")
