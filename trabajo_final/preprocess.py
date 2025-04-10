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
    input_path = os.path.join(args.input_data, "diabetes.csv")
    df = pd.read_csv(input_path)

    # Convertir columna objetivo a binaria
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    df.rename(columns={'readmitted': 'readmitted<30'}, inplace=True)

    # Filtrar columnas
    filtro = ['num_lab_procedures', 'num_medications', 'time_in_hospital', 'number_inpatient', 'age', 'number_diagnoses', 'readmitted<30']
    df_filtrado = df[filtro]

    # One-hot encoding de 'age'
    age_dummies = pd.get_dummies(df_filtrado['age'], prefix='age', dtype=int)
    df_filtrado = pd.concat([df_filtrado.drop(columns=['age']), age_dummies], axis=1)

    # Tomar muestra del 20% manteniendo proporciones de la clase (estratificado)
    df_sampled, _ = train_test_split(
        df_filtrado,
        train_size=0.20,
        stratify=df_filtrado['readmitted<30'],
        random_state=42
    )

    # Dividir en entrenamiento y prueba (80/20)
    train_df, test_df = train_test_split(
        df_sampled,
        test_size=0.2,
        stratify=df_sampled['readmitted<30'],
        random_state=42
    )

    # Guardar archivos
    os.makedirs(args.output_train, exist_ok=True)
    os.makedirs(args.output_test, exist_ok=True)

    train_path = os.path.join(args.output_train, "train.csv")
    test_path = os.path.join(args.output_test, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Preprocesamiento terminado. Shapes:")
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)
