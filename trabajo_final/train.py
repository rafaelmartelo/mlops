# train.py
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    args = parser.parse_args()

    train_path = os.path.join("/opt/ml/input/data/train", "train.csv")
    test_path = os.path.join("/opt/ml/input/data/test", "test.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

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
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    joblib.dump(model, os.path.join("/opt/ml/model", "model.joblib"))
