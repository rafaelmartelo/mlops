# train.py
import os
import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hiperparámetros relevantes para regresión logística
    parser.add_argument("--C", type=float, default=1.0)  # Inverso de la regularización
    parser.add_argument("--max_iter", type=int, default=100)

    args = parser.parse_args()

    if args.C <= 0:
        raise ValueError("C must be > 0")
    if args.max_iter <= 0:
        raise ValueError("max_iter must be > 0")

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

    # Entrenar modelo
    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    # Evaluar
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Mostrar métrica para HPO
    print(f"validation:accuracy={acc}")

    # Guardar modelo
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
