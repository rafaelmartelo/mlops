# preprocess.py

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import argparse
import os

# Parseo de argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, required=True)
parser.add_argument('--output-train', type=str, required=True)
parser.add_argument('--output-test', type=str, required=True)
args = parser.parse_args()

# Cargar dataset
df = pd.read_csv(os.path.join(args.input_data, "diabetic_data.csv"))
df.replace("?", pd.NA, inplace=True)
df["readmitted_30"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

cols_to_drop = ['encounter_id', 'patient_nbr', 'readmitted', 'weight', 'payer_code', 'medical_specialty']
df.drop(columns=cols_to_drop, inplace=True)
df.dropna(subset=["race", "gender"], inplace=True)

X = df.drop(columns=["readmitted_30"])
y = df["readmitted_30"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include="number").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Guardar archivos
pd.DataFrame(X_train).to_csv(os.path.join(args.output_train, "X_train.csv"), index=False, header=False)
pd.DataFrame(y_train).to_csv(os.path.join(args.output_train, "y_train.csv"), index=False, header=False)
pd.DataFrame(X_test).to_csv(os.path.join(args.output_test, "X_test.csv"), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(args.output_test, "y_test.csv"), index=False, header=False)
