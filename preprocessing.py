import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    # Eliminar G1 y G2 si están
    for col in ['G1', 'G2']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Separar variables categóricas
    cat_cols = df.select_dtypes(include='object').columns

    # Codificación one-hot
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Separar X e y
    X = df.drop(columns=['G3'])
    y = df['G3']

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # División
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    print("Leyendo datos...")
    df = pd.read_csv(os.path.join(input_path, "student-por.csv"), sep=';')
    print("Columnas disponibles:", df.columns.tolist())

    print("Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocess(df)

    os.makedirs(output_path, exist_ok=True)

    print("Guardando archivos procesados...")
    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)

    print("✅ Preprocesamiento completo.")
