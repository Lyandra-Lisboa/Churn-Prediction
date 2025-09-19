"""
Avaliação do modelo K-Modes
"""
import os
import joblib
import logging
import pandas as pd
from sklearn.metrics import silhouette_score
from src.data.extractors import load_data
from src.data.validators import validate_dataframe

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/kmodes_churn.pkl"

def evaluate_model():
    """
    Avalia o modelo K-Modes usando métricas de clusterização.
    """
    logging.info("Carregando dados...")
    df = load_data()
    validate_dataframe(df)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Treine primeiro.")

    model = joblib.load(MODEL_PATH)

    logging.info("Gerando previsões para avaliação...")
    preds = model.predict(df)

    # Avaliação por silhouette (métrica de qualidade de clusters)
    score = silhouette_score(df, preds, metric="hamming")
    logging.info(f"Silhouette Score: {score:.4f} (quanto mais próximo de 1, melhor)")

    # Distribuição dos clusters
    dist = pd.Series(preds).value_counts()
    logging.info(f"Distribuição dos clusters:\n{dist}")

    return score, dist


if __name__ == "__main__":
    evaluate_model()
