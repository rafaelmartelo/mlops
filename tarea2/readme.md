# Análisis del Rendimiento Estudiantil en Educación Secundaria

Este proyecto tiene como objetivo analizar y preprocesar un conjunto de datos sobre el rendimiento estudiantil en la asignatura de Lengua Portuguesa, con el fin de construir modelos de machine learning que predigan la calificación final de los estudiantes (atributo `G3`).

## 🎯 Objetivo del análisis

El análisis busca predecir la calificación final (`G3`) de los estudiantes a partir de variables demográficas, sociales y académicas. Se hace énfasis en modelos que **no utilicen las calificaciones de los primeros dos períodos (`G1` y `G2`)**, dado que estos atributos están altamente correlacionados con `G3`, pero no estarían disponibles al momento de realizar una predicción temprana.

## 🧪 Script `preprocessing.py`

Este script realiza las siguientes tareas:

1. **Carga de datos:** Lee un archivo CSV delimitado por `;` desde un bucket de Amazon S3.
2. **Limpieza y transformación:**
   - Elimina las columnas `G1` y `G2`.
   - Codifica variables categóricas usando `OneHotEncoding`.
   - Escala variables numéricas usando `StandardScaler`.
3. **Separación de datos:** Divide el conjunto de datos en conjuntos de entrenamiento y prueba.
4. **Almacenamiento de resultados:** Guarda los archivos procesados en el bucket de S3 para que puedan ser usados en pasos posteriores (entrenamiento, evaluación, etc.).

## 📦 Integración con SageMaker

Este script se ejecuta dentro de un **Processing Job** de Amazon SageMaker, el cual crea un entorno temporal basado en una imagen de procesamiento (`sklearn`) y ejecuta el script con datos almacenados en S3.

El notebook de SageMaker:

- Descarga el script desde GitHub.
- Define rutas de entrada y salida en S3.
- Lanza un `SKLearnProcessor` para ejecutar `preprocessing.py`.
- Espera que el procesamiento termine y deja listos los datos para entrenamiento.

## 🚀 Cómo ejecutar el Notebook en SageMaker

1. Abre un Notebook en SageMaker Studio o desde una instancia de notebook.
2. Asegúrate de tener una IAM Role con permisos de acceso a S3 y SageMaker.
3. Crea los buckets necesarios (si no existen) o usa uno ya creado (por ejemplo: `sagemakertarea2`).
4. Carga el dataset `student-por.csv` al bucket en una carpeta `input/`.
5. Asegúrate de tener el script `preprocessing.py` disponible:
   ```python
   !wget https://raw.githubusercontent.com/rafaelmartelo/mlops/main/preprocessing.py
