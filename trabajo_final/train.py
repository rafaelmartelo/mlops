# train.py
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import traceback

def find_csv_file(path):
    for file in os.listdir(path):
        if file.endswith(".csv"):
            return os.path.join(path, file)
    raise FileNotFoundError(f"No se encontró ningún archivo CSV en {path}")

if __name__ == "__main__":
    try:
        print("==== Iniciando entrenamiento ====")

        parser = argparse.ArgumentParser()
        parser.add_argument("--n-estimators", type=int, default=100)
        parser.add_argument("--max-depth", type=int, default=10)
        args = parser.parse_args()

        print("Parámetros recibidos:", args)

        # Paths esperados por SageMaker
        train_dir = "/opt/ml/input/data/train"
        test_dir = "/opt/ml/input/data/test"
        model_dir = "/opt/ml/model"

        print(f"Buscando archivos en {train_dir} y {test_dir}...")

        train_path = find_csv_file(train_dir)
        test_path = find_csv_file(test_dir)

        print("Archivo de entrenamiento:", train_path)
        print("Archivo de prueba:", test_path)

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        print("Columnas de train:", train_data.columns)
        print("Columnas de test:", test_data.columns)

        if "readmitted" not in train_data.columns:
            raise ValueError("La columna 'readmitted' no está en los datos de entrenamiento")

        X_train = train_data.drop("readmitted", axis=1)
        y_train = train_data["readmitted"]
        X_test = test_data.drop("readmitted", axis=1)
        y_test = test_data["readmitted"]

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Precisión del modelo: {acc}")

        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        print("Modelo guardado exitosamente.")

    except Exception as e:
        print("❌ Error en el entrenamiento:")
        traceback.print_exc()
        raise e
