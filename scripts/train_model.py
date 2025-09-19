"""
train_model.py
----------------
Script responsável por treinar o modelo de churn utilizando o algoritmo K-Modes.
Inclui:
- Carregamento de dados
- Pré-processamento
- Treinamento
- Salvamento do modelo
- Registro de métricas
"""

import os
import sys
import logging
import joblib
import pandas as pd
from pathlib import Path

# Ajustar caminho para importar módulos do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.extractors import load_customer_data
from src.data.validators import validate_data
from src.models.trainer import KModesTrainer
from src.models.evaluator import ModelEvaluator

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def main():
    logging.info("🚀 Iniciando processo de treinamento do modelo K-Modes...")

    # 1. Carregar dados
    df = load_customer_data()
    logging.info(f"✅ Dados carregados com {df.shape[0]} linhas e {df.shape[1]} colunas")

    # 2. Validar dados
    if not validate_data(df):
        logging.error("❌ Dados inválidos para treinamento. Abortando.")
        return

    # 3. Separar features
    X = df.drop(columns=["churn"])  # assumindo que 'churn' é a variável alvo

    # 4. Treinar modelo
    trainer = KModesTrainer(n_clusters=2, init="Huang", n_init=5)
    model = trainer.train(X)

    # 5. Avaliar modelo
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X)
    logging.info(f"📊 Métricas do modelo: {metrics}")

    # 6. Salvar modelo treinado
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "kmodes_model.pkl"
    joblib.dump(model, model_path)
    logging.info(f"💾 Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
