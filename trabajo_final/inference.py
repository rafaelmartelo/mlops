import joblib
import os
import pandas as pd
import io

# Cargar el modelo desde el directorio donde SageMaker lo deja
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

# Convertir el input en DataFrame. SageMaker por defecto envía texto CSV.
def input_fn(input_data, content_type):
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(input_data), header=None)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Predecir con el modelo cargado
def predict_fn(input_data, model):
    return model.predict(input_data)

# Devolver la predicción en formato aceptado por SageMaker
def output_fn(prediction, accept):
    if accept == "application/json":
        return {"predictions": prediction.tolist()}
    elif accept == "text/csv":
        return ",".join(str(x) for x in prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
