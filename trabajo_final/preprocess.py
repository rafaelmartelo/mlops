import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-train", type=str, required=True)
    parser.add_argument("--output-test", type=str, required=True)
    args = parser.parse_args()

    print(f"📥 Leyendo archivo desde: {args.input_data}")
    df = pd.read_csv(os.path.join(args.input_data, "diabetic_data.csv"))

    print("🧹 Preprocesando datos...")
    df.replace("?", np.nan, inplace=True)
    df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)

    df = df.dropna()

    target = "readmitted"
    df = df[df[target] != "NO"]
    df[target] = df[target].apply(lambda x: 1 if x == "<30" else 0)

    X = df.drop(columns=[target])
    y = df[target]

    print("🔣 Codificando variables categóricas...")
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    scaler = StandardScaler()

    X_cat = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_num = pd.DataFrame(scaler.fit_transform(X[numerical_cols]))

    X_processed = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    print(f"📐 Shape antes del resampling: {X_processed.shape}")

    print("⚖️ Aplicando pipeline: RandomUnderSampler + SMOTE...")
    sampler_pipeline = Pipeline([
        ("under", RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42))
    ])

    X_resampled, y_resampled = sampler_pipeline.fit_resample(X_processed, y)
    print(f"📏 Shape después del resampling: {X_resampled.shape}")

    print("✂️ Dividiendo en train y test...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print("💾 Guardando datos procesados...")
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)

    X_train["target"] = y_train
    X_test["target"] = y_test

    X_train.to_csv(os.path.join(args.output_train, "train.csv"), index=False)
    X_test.to_csv(os.path.join(args.output_test, "test.csv"), index=False)

    print("✅ ¡Preprocesamiento finalizado!")
