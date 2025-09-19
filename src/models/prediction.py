"""
Inferência com modelo K-Modes em novos clientes
"""
import os
import joblib
import logging
import pandas as pd
from src.data.validators import validate_dataframe

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/kmodes_churn.pkl"

def predict_new(data: pd.DataFrame):
    """
    Faz inferência em novos dados usando o modelo treinado.

    Args:
        data (pd.DataFrame): Dados dos clientes (já pré-processados).
    Returns:
        list: Cluster atribuído a cada cliente.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Treine primeiro.")

    validate_dataframe(data)

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(data)

    logging.info(f"Predições realizadas para {len(data)} clientes.")
    return predictions


if __name__ == "__main__":
    # Exemplo com dados fictícios
    novos_clientes = pd.DataFrame([
        {"sexo": "M", "estado_civil": "Solteiro", "plano": "Básico"},
        {"sexo": "F", "estado_civil": "Casado", "plano": "Premium"}
    ])

    clusters = predict_new(novos_clientes)
    print("Clusters atribuídos:", clusters)
