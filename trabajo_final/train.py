# train.py
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def find_csv_file(path):
    """Devuelve el primer archivo CSV encontrado en un directorio"""
    for file in os.listdir(path):
        if file.endswith(".csv"):
            return os.path.join(path, file)
    raise FileNotFoundError(f"No se encontró ningún archivo CSV en {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    args = parser.parse_args()

    # Paths
    train_dir = "/opt/ml/input/data/train"
    test_dir = "/opt/ml/input/data/test"
    model_dir = "/opt/ml/model"

    train_path = find_csv_file(train_dir)
    test_path = find_csv_file(test_dir)

    # Leer datos
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Validaciones básicas
    if "readmitted" not in train_data.columns:
        raise ValueError("La columna 'readmitted' no se encuentra en los datos de entrenamiento.")

    X_train = train_data.drop("readmitted", axis=1)
    y_train = train_data["readmitted"]
    X_test = test_data.drop("readmitted", axis=1)
    y_test = test_data["readmitted"]

    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluar
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    # Guardar modelo
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
