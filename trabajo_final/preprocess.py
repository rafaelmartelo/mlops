import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess(df):
    # Reemplazar valores faltantes codificados como '?'
    df.replace('?', np.nan, inplace=True)

    # Eliminar columnas con más del 20% de valores faltantes
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.2].index
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Transformar variable objetivo
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # Separar variables
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    
    # Separar columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encoding para categóricas
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(X[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)

    # Escalado MinMax para numéricas
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[num_cols])
    X_num_df = pd.DataFrame(X_num, columns=num_cols)

    # Concatenar de nuevo
    X_processed = pd.concat([X_num_df.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Leyendo datos...")
    input_path = "/opt/ml/processing/input/diabetic_data.csv"
    output_train_path = "/opt/ml/processing/train"
    output_test_path = "/opt/ml/processing/test"

    df = pd.read_csv(input_path)

    print("Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocess(df)

    print("Guardando datos preprocesados...")
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    X_train.to_csv(f"{output_train_path}/X_train.csv", index=False)
    y_train.to_csv(f"{output_train_path}/y_train.csv", index=False)
    X_test.to_csv(f"{output_test_path}/X_test.csv", index=False)
    y_test.to_csv(f"{output_test_path}/y_test.csv", index=False)

    print("Finalizado.")
