# train.py
import os
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hiperparámetros
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)

    args = parser.parse_args()

    # Directorios de entrada/salida
    train_dir = "/opt/ml/input/data/train"
    test_dir = "/opt/ml/input/data/test"
    model_dir = "/opt/ml/model"
    output_dir = "/opt/ml/output"

    # Cargar datos
    X_train = pd.read_csv(os.path.join(train_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(train_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(test_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(test_dir, "y_test.csv")).values.ravel()

    # Entrenar modelo con hiperparámetros
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluar y guardar métrica en archivo para SageMaker HPO
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # SageMaker espera que se escriban los resultados en un archivo
    with open(os.path.join(output_dir, "validation_accuracy.txt"), "w") as f:
        f.write(f"{acc}\n")

    # Guardar modelo
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
