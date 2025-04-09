import subprocess
import sys

# Instalar imbalanced-learn si no está instalada
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])

import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
parser.add_argument("--output-train", type=str, default="/opt/ml/processing/train")
parser.add_argument("--output-test", type=str, default="/opt/ml/processing/test")
args = parser.parse_args()

# Cargar datos
input_path = os.path.join(args.input_data, "diabetic_data.csv")
print(f"📥 Leyendo archivo desde: {input_path}")
df = pd.read_csv(input_path)

# Eliminar columnas innecesarias o que no aportan al modelo
df = df.drop(columns=["encounter_id", "patient_nbr"])

# Eliminar filas con valores desconocidos en la variable objetivo
df = df[df["readmitted"] != "NA"]

# Simplificar la variable objetivo a clasificación binaria
df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

# Eliminar columnas constantes
df = df.loc[:, df.nunique() > 1]

# Separar variables predictoras y objetivo
X = df.drop(columns=["readmitted"])
y = df["readmitted"]

# Mostrar tipos de datos
print("\n📊 Tipos de columnas en X:")
print(X.dtypes)

# Detectar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\n🔢 Columnas numéricas:", numeric_features)
print("🔣 Columnas categóricas:", categorical_features)

# Pipelines para preprocesamiento
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Aplicar preprocesamiento
print("\n⚙️ Procesando datos...")
X_processed = preprocessor.fit_transform(X)

# Aplicar SMOTE para balancear clases
print("\n⚖️ Aplicando SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Guardar archivos en formato CSV
os.makedirs(args.output_train, exist_ok=True)
os.makedirs(args.output_test, exist_ok=True)

train_output = os.path.join(args.output_train, "train.csv")
test_output = os.path.join(args.output_test, "test.csv")

train_df = pd.DataFrame(X_train)
train_df["target"] = y_train
train_df.to_csv(train_output, index=False)

test_df = pd.DataFrame(X_test)
test_df["target"] = y_test
test_df.to_csv(test_output, index=False)

print(f"\n✅ Archivos guardados:\n - {train_output}\n - {test_output}")
print("✅ Preprocesamiento completado.")
