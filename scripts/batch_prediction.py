"""
batch_prediction.py
--------------------
Script para realizar predições em lote com o modelo K-Modes.
Processa um conjunto de clientes e gera as probabilidades de churn.
"""

import os
import sys
import joblib
import logging
import pandas as pd
from pathlib import Path

# Ajustar caminho para importar módulos do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.inference import ChurnPredictor
from src.data.extractors import load_batch_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def main(input_file: str, output_file: str):
    logging.info("🚀 Iniciando predição em lote...")

    # 1. Carregar modelo
    model_path = Path("artifacts/models/kmodes_model.pkl")
    if not model_path.exists():
        logging.error("❌ Modelo não encontrado. Treine antes de rodar predições.")
        return

    model = joblib.load(model_path)
    predictor = ChurnPredictor(model)

    # 2. Carregar dados
    df = load_batch_data(input_file)
    logging.info(f"✅ Dados de entrada carregados ({df.shape[0]} registros)")

    # 3. Fazer predições
    predictions = predictor.predict(df)

    # 4. Salvar resultados
    df["churn_prediction"] = predictions
    df.to_csv(output_file, index=False)
    logging.info(f"💾 Resultados salvos em {output_file}")


if __name__ == "__main__":
    # Exemplo de execução:
    # python scripts/batch_prediction.py input.csv output.csv
    import argparse

    parser = argparse.ArgumentParser(description="Batch Prediction")
    parser.add_argument("input_file", type=str, help="Arquivo CSV de entrada")
    parser.add_argument("output_file", type=str, help="Arquivo CSV de saída")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
