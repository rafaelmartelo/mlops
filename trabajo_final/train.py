# train.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # Directorios predefinidos por SageMaker
    train_dir = "/opt/ml/input/data/train"
    test_dir = "/opt/ml/input/data/test"
    model_dir = "/opt/ml/model"

    # Cargar datos
    X_train = pd.read_csv(os.path.join(train_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(train_dir, "y_train.csv")).values.ravel()

    X_test = pd.read_csv(os.path.join(test_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(test_dir, "y_test.csv")).values.ravel()

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Guardar modelo
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    print("Entrenamiento completado y modelo guardado.")
