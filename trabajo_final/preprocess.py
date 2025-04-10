import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-train", type=str, default="/opt/ml/processing/train")
    parser.add_argument("--output-test", type=str, default="/opt/ml/processing/test")
    args = parser.parse_args()

    # Cargar datos
    input_path = os.path.join(args.input_data, "diabetic_data.csv")
    df = pd.read_csv(input_path)

    # Convertir columna objetivo a binaria
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df.rename(columns={'readmitted': 'readmitted<30'}, inplace=True)

    # Filtrar columnas
    filtro = ['num_lab_procedures', 'num_medications', 'time_in_hospital',
              'number_inpatient', 'age', 'number_diagnoses', 'readmitted<30']
    df_filtrado = df[filtro]

    # One-hot encoding de 'age'
    age_dummies = pd.get_dummies(df_filtrado['age'], prefix='age', dtype=int)
    df_filtrado = pd.concat([df_filtrado.drop(columns=['age']), age_dummies], axis=1)

    # Muestra del 20% estratificada
    df_sampled, _ = train_test_split(
        df_filtrado,
        train_size=0.20,
        stratify=df_filtrado['readmitted<30'],
        random_state=42
    )

    # Separar X e y
    X = df_sampled.drop(columns=['readmitted<30'])
    y = df_sampled['readmitted<30']

    # Separar en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Crear carpetas de salida
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)

    # Guardar datasets
    X_train.to_csv(os.path.join(args.output_train, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(args.output_train, "y_train.csv"), index=False)

    X_test.to_csv(os.path.join(args.output_test, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(args.output_test, "y_test.csv"), index=False)

    print("Preprocesamiento finalizado:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
