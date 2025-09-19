"""
deploy_model.py
----------------
Script para disponibilizar o modelo K-Modes via API REST usando Flask.
Endpoints:
- /predict : Recebe JSON com dados de clientes e retorna predição de churn
"""

import os
import sys
import joblib
import logging
from flask import Flask, request, jsonify
from pathlib import Path

# Ajustar caminho para importar módulos do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.inference import ChurnPredictor

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Flask app
app = Flask(__name__)

# Carregar modelo
model_path = Path("artifacts/models/kmodes_model.pkl")
if not model_path.exists():
    raise FileNotFoundError("❌ Modelo não encontrado. Treine antes de rodar o deploy.")

model = joblib.load(model_path)
predictor = ChurnPredictor(model)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recebe JSON com dados de clientes e retorna predição de churn.
    Exemplo de entrada:
    {
        "clientes": [
            {"idade": 35, "plano": "premium", "tempo_contrato": 12},
            {"idade": 28, "plano": "basic", "tempo_contrato": 3}
        ]
    }
    """
    try:
        data = request.json.get("clientes", [])
        if not data:
            return jsonify({"error": "Nenhum cliente fornecido"}), 400

        predictions = predictor.predict_json(data)
        return jsonify({"predictions": predictions})

    except Exception as e:
        logging.error(f"Erro na predição: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.info("🚀 API de predição iniciada em http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
