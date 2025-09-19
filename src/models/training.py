"""
Treinamento do modelo K-Modes para predição de churn
"""
import os
import joblib
import logging
import argparse
import pandas as pd
from kmodes.kmodes import KModes
from src.data.extractors import load_data
from src.data.validators import validate_dataframe

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "kmodes_churn.pkl")

def train_kmodes(n_clusters=3, init="Huang", max_iter=100, random_state=42):
    """
    Treina o modelo K-Modes para detectar perfis de clientes.
    """
    logging.info("Carregando dados...")
    df = load_data()

    logging.info("Validando dados...")
    validate_dataframe(df)

    logging.info(f"Treinando modelo K-Modes com {n_clusters} clusters...")
    model = KModes(
        n_clusters=n_clusters,
        init=init,
        n_init=5,
        max_iter=max_iter,
        verbose=1,
        random_state=random_state
    )

    clusters = model.fit_predict(df)
    df["cluster"] = clusters

    # Salvar modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Modelo salvo em {MODEL_PATH}")

    logging.info(f"Distribuição dos clusters:\n{df['cluster'].value_counts()}")

    return df, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo K-Modes para churn")
    parser.add_argument("--clusters", type=int, default=3, help="Número de clusters")
    parser.add_argument("--init", type=str, default="Huang", choices=["Huang", "Cao"], help="Método de inicialização")
    parser.add_argument("--max_iter", type=int, default=100, help="Máximo de iterações")
    args = parser.parse_args()

    train_kmodes(n_clusters=args.clusters, init=args.init, max_iter=args.max_iter)
