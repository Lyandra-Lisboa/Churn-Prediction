"""
deploy_model.py - Ligga Edition
----------------
API REST para disponibilizar o modelo K-Modes da Ligga.
Fornece endpoints para predições individuais, em lote e consultas específicas.

Endpoints:
- /health           : Health check da API
- /predict          : Predição individual com dados JSON
- /predict/batch    : Predição em lote
- /predict/contract : Predição por contrato específico
- /predict/cpf      : Predição por CPF/CNPJ
- /model/info       : Informações do modelo
- /docs             : Documentação da API

Funcionalidades:
- Integração completa com pipeline da Ligga
- Retorno de insights de negócio
- Documentação automática
- Tratamento robusto de erros
- Logs detalhados
- Múltiplos formatos de entrada
"""

import os
import sys
import joblib
import logging
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Importar módulos específicos da Ligga
from extractors import LiggaDataExtractor
from preprocessing import preprocess_ligga_data
from transformers import LiggaCategoricalEncoder
from batch_prediction import LiggaBatchPredictor

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/api_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)

logger = logging.getLogger(__name__)

class LiggaAPI:
    """
    Classe principal da API da Ligga
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Permitir CORS para requests de frontend
        
        # Componentes do modelo
        self.model = None
        self.transformer = None
        self.metadata = None
        self.batch_predictor = None
        self.extractor = LiggaDataExtractor()
        
        # Personas info
        self.personas_info = {
            0: {"nome": "Recente Negativo", "risco": "ALTO", "acao": "Atenção especial, suporte financeiro"},
            1: {"nome": "Recente Positivo", "risco": "BAIXO", "acao": "Manter engajamento, oferecer upgrades"},
            2: {"nome": "Alto Valor Não Fidelizado", "risco": "MÉDIO-ALTO", "acao": "URGENTE: Oferecer fidelização"},
            3: {"nome": "Alto Valor Fidelizado", "risco": "BAIXO", "acao": "Manter satisfação, atendimento premium"},
            4: {"nome": "Padrão Capital", "risco": "MÉDIO", "acao": "Monitorar, ofertas segmentadas"},
            5: {"nome": "Baixo Valor Negativo", "risco": "ALTO", "acao": "Avaliar viabilidade, suporte intensivo"},
            6: {"nome": "Baixo Valor Positivo", "risco": "BAIXO", "acao": "Manter relacionamento, upgrades graduais"},
            7: {"nome": "Interior Não Fidelizado", "risco": "MÉDIO", "acao": "Oferecer fidelização regional"},
            8: {"nome": "Interior Fidelizado", "risco": "BAIXO", "acao": "Manter benefícios, expansão"},
            9: {"nome": "Baixa Velocidade", "risco": "MÉDIO", "acao": "Ofertas de upgrade, modernização"}
        }
        
        # Estatísticas da API
        self.stats = {
            'requests_total': 0,
            'predictions_total': 0,
            'errors_total': 0,
            'start_time': datetime.now()
        }
        
        # Carregar modelo
        self._load_model()
        
        # Configurar rotas
        self._setup_routes()
    
    def _load_model(self):
        """Carrega modelo e componentes"""
        logger.info("📦 Carregando modelo da Ligga...")
        
        try:
            models_dir = Path("artifacts/models")
            
            # Carregar modelo
            model_path = models_dir / "latest_kmodes_ligga.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info(f"✅ Modelo carregado: {model_path}")
            
            # Carregar transformer
            transformer_path = models_dir / "latest_transformer_ligga.pkl"
            if not transformer_path.exists():
                logger.warning(f"⚠️  Transformer não encontrado: {transformer_path}")
            else:
                self.transformer = joblib.load(transformer_path)
                logger.info(f"✅ Transformer carregado: {transformer_path}")
            
            # Carregar metadata
            metadata_path = models_dir / "latest_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Metadata carregada: {metadata_path}")
            
            # Inicializar batch predictor
            self.batch_predictor = LiggaBatchPredictor()
            logger.info("✅ Batch predictor inicializado")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def _setup_routes(self):
        """Configura todas as rotas da API"""
        
        @self.app.before_request
        def before_request():
            self.stats['requests_total'] += 1
        
        @self.app.errorhandler(Exception)
        def handle_error(error):
            self.stats['errors_total'] += 1
            logger.error(f"Erro na API: {error}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                'error': str(error),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check da API"""
            uptime = datetime.now() - self.stats['start_time']
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': int(uptime.total_seconds()),
                'model_loaded': self.model is not None,
                'transformer_loaded': self.transformer is not None,
                'stats': self.stats
            })
        
        @self.app.route('/model/info', methods=['GET'])
        def model_info():
            """Informações do modelo"""
            info = {
                'model_type': 'K-Modes',
                'n_clusters': getattr(self.model, 'n_clusters', None),
                'personas_count': len(self.personas_info),
                'features_used': self.metadata.get('features_used', []) if self.metadata else [],
                'model_timestamp': self.metadata.get('timestamp') if self.metadata else None,
                'personas': self.personas_info
            }
            
            return jsonify(info)
        
        @self.app.route('/predict', methods=['POST'])
        def predict_single():
            """
            Predição individual com dados JSON
            
            Exemplo de entrada:
            {
                "contrato": "12345",
                "valor_contrato": 120.50,
                "regiao": "curitiba",
                "aging_meses": 24,
                "total_atendimentos": 2,
                "atendimentos_negativos": 0
            }
            """
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Dados JSON necessários'}), 400
                
                # Converter para DataFrame
                df = pd.DataFrame([data])
                
                # Processar e predizer
                result = self._process_and_predict(df)
                
                if len(result) == 0:
                    return jsonify({'error': 'Falha no processamento dos dados'}), 500
                
                # Preparar resposta
                prediction = result.iloc[0]
                response = self._format_prediction_response(prediction)
                
                self.stats['predictions_total'] += 1
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Erro na predição individual: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/batch', methods=['POST'])
        def predict_batch():
            """
            Predição em lote com array de clientes
            
            Exemplo de entrada:
            {
                "clientes": [
                    {"contrato": "12345", "valor_contrato": 120.50, ...},
                    {"contrato": "67890", "valor_contrato": 85.00, ...}
                ]
            }
            """
            try:
                data = request.get_json()
                clientes = data.get('clientes', [])
                
                if not clientes:
                    return jsonify({'error': 'Lista de clientes necessária'}), 400
                
                # Converter para DataFrame
                df = pd.DataFrame(clientes)
                
                # Processar e predizer
                results = self._process_and_predict(df)
                
                if len(results) == 0:
                    return jsonify({'error': 'Falha no processamento dos dados'}), 500
                
                # Preparar resposta
                predictions = []
                for _, row in results.iterrows():
                    predictions.append(self._format_prediction_response(row))
                
                self.stats['predictions_total'] += len(predictions)
                
                return jsonify({
                    'predictions': predictions,
                    'total_processed': len(predictions),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Erro na predição em lote: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/contract/<contract_id>', methods=['GET'])
        def predict_by_contract(contract_id):
            """Predição por contrato específico consultando o banco"""
            try:
                # Consultar dados do contrato
                results = self.batch_predictor.predict_from_database(
                    contratos=[contract_id]
                )
                
                if len(results) == 0:
                    return jsonify({'error': f'Contrato {contract_id} não encontrado'}), 404
                
                # Preparar resposta
                prediction = results.iloc[0]
                response = self._format_prediction_response(prediction, include_raw_data=True)
                
                self.stats['predictions_total'] += 1
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Erro na predição por contrato: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/cpf/<cpf_cnpj>', methods=['GET'])
        def predict_by_cpf(cpf_cnpj):
            """Predição por CPF/CNPJ consultando o banco"""
            try:
                # Consultar dados do CPF/CNPJ
                results = self.batch_predictor.predict_from_database(
                    cpf_cnpj=[cpf_cnpj]
                )
                
                if len(results) == 0:
                    return jsonify({'error': f'CPF/CNPJ {cpf_cnpj} não encontrado'}), 404
                
                # Preparar resposta (pode ter múltiplos contratos)
                predictions = []
                for _, row in results.iterrows():
                    predictions.append(self._format_prediction_response(row, include_raw_data=True))
                
                self.stats['predictions_total'] += len(predictions)
                
                return jsonify({
                    'cpf_cnpj': cpf_cnpj,
                    'total_contracts': len(predictions),
                    'predictions': predictions,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Erro na predição por CPF: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict/sample', methods=['GET'])
        def predict_sample():
            """Predição de uma amostra aleatória (para demonstração)"""
            try:
                limit = request.args.get('limit', 10, type=int)
                limit = min(limit, 100)  # Máximo 100 para não sobrecarregar
                
                # Consultar amostra
                results = self.batch_predictor.predict_from_database(limit=limit)
                
                if len(results) == 0:
                    return jsonify({'error': 'Nenhum dado encontrado'}), 404
                
                # Preparar resposta
                predictions = []
                for _, row in results.iterrows():
                    predictions.append(self._format_prediction_response(row))
                
                # Resumo estatístico
                summary = {
                    'total_processed': len(predictions),
                    'distribuicao_risco': {},
                    'distribuicao_personas': {}
                }
                
                # Calcular distribuições
                for pred in predictions:
                    risco = pred['risco_churn']
                    persona = pred['persona']
                    
                    summary['distribuicao_risco'][risco] = summary['distribuicao_risco'].get(risco, 0) + 1
                    summary['distribuicao_personas'][persona] = summary['distribuicao_personas'].get(persona, 0) + 1
                
                self.stats['predictions_total'] += len(predictions)
                
                return jsonify({
                    'predictions': predictions,
                    'summary': summary,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Erro na predição de amostra: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/docs', methods=['GET'])
        def api_docs():
            """Documentação da API"""
            docs_html = self._generate_docs_html()
            return docs_html
        
        @self.app.route('/', methods=['GET'])
        def home():
            """Página inicial da API"""
            return jsonify({
                'message': 'API de Predição de Churn - Ligga',
                'version': '1.0.0',
                'endpoints': [
                    '/health - Health check',
                    '/model/info - Informações do modelo',
                    '/predict - Predição individual (POST)',
                    '/predict/batch - Predição em lote (POST)',
                    '/predict/contract/<id> - Predição por contrato (GET)',
                    '/predict/cpf/<cpf> - Predição por CPF/CNPJ (GET)',
                    '/predict/sample - Amostra aleatória (GET)',
                    '/docs - Documentação completa'
                ],
                'timestamp': datetime.now().isoformat()
            })
    
    def _process_and_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa dados e faz predições"""
        try:
            # Pré-processar
            df_processed = preprocess_ligga_data(df)
            
            # Aplicar transformações se temos transformer
            if self.transformer:
                df_transformed = self.transformer.transform(df_processed)
            else:
                df_transformed = df_processed
            
            # Preparar features para modelo
            if self.metadata and 'features_used' in self.metadata:
                features_used = self.metadata['features_used']
            else:
                features_used = [
                    'faixa_valor', 'faixa_aging', 'regiao', 'faixa_velocidade',
                    'perfil_atendimento', 'is_churner', 'cliente_tipo',
                    'faixa_vencimento', 'tipo_produto', 'canal_produto'
                ]
            
            # Filtrar features disponíveis
            available_features = [f for f in features_used if f in df_transformed.columns]
            
            if not available_features:
                raise ValueError("Nenhuma feature necessária encontrada")
            
            X = df_transformed[available_features]
            
            # Fazer predições
            clusters = self.model.predict(X)
            
            # Adicionar predições aos dados originais
            result = df.copy()
            result['cluster_pred'] = clusters
            result['persona'] = [self.personas_info[c]['nome'] for c in clusters]
            result['risco_churn'] = [self.personas_info[c]['risco'] for c in clusters]
            result['acao_recomendada'] = [self.personas_info[c]['acao'] for c in clusters]
            
            # Adicionar timestamp
            result['data_predicao'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            raise
    
    def _format_prediction_response(self, prediction: pd.Series, include_raw_data: bool = False) -> Dict[str, Any]:
        """Formata resposta da predição"""
        response = {
            'cluster': int(prediction.get('cluster_pred', -1)),
            'persona': prediction.get('persona', 'Desconhecida'),
            'risco_churn': prediction.get('risco_churn', 'DESCONHECIDO'),
            'acao_recomendada': prediction.get('acao_recomendada', 'Analisar caso'),
            'data_predicao': prediction.get('data_predicao', datetime.now().isoformat()),
            'confianca': 'alta'  # Pode ser calculada baseada na distância ao centroide
        }
        
        # Adicionar dados identificadores se disponíveis
        for field in ['contrato', 'cliente', 'cpf_cnpj']:
            if field in prediction and pd.notna(prediction[field]):
                response[field] = prediction[field]
        
        # Adicionar insights de negócio se disponíveis
        business_fields = [
            'valor_contrato', 'aging_meses', 'regiao', 'score_engajamento',
            'potencial_upsell', 'segmento_valor', 'prioridade_acao'
        ]
        
        insights = {}
        for field in business_fields:
            if field in prediction and pd.notna(prediction[field]):
                insights[field] = prediction[field]
        
        if insights:
            response['insights'] = insights
        
        # Incluir dados brutos se solicitado
        if include_raw_data:
            raw_data = {}
            for col, val in prediction.items():
                if pd.notna(val) and col not in response:
                    # Converter tipos numpy para tipos Python JSON-serializáveis
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    elif isinstance(val, np.ndarray):
                        val = val.tolist()
                    raw_data[col] = val
            
            if raw_data:
                response['dados_completos'] = raw_data
        
        return response
    
    def _generate_docs_html(self) -> str:
        """Gera documentação HTML da API"""
        docs_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Ligga - Documentação</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { color: white; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
                .get { background-color: #28a745; }
                .post { background-color: #007bff; }
                pre { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
                h1, h2 { color: #ff6600; }
            </style>
        </head>
        <body>
            <h1>🔮 API de Predição de Churn - Ligga</h1>
            <p>API REST para predições de churn usando modelo K-Modes específico da Ligga.</p>
            
            <h2>📋 Endpoints Disponíveis</h2>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /health</h3>
                <p>Health check da API e estatísticas.</p>
                <pre>Response: {
  "status": "healthy",
  "uptime_seconds": 3600,
  "model_loaded": true,
  "stats": {...}
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /model/info</h3>
                <p>Informações do modelo e personas.</p>
                <pre>Response: {
  "model_type": "K-Modes",
  "n_clusters": 10,
  "personas": {...}
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict</h3>
                <p>Predição individual com dados JSON.</p>
                <pre>Request: {
  "contrato": "12345",
  "valor_contrato": 120.50,
  "regiao": "curitiba",
  "aging_meses": 24
}

Response: {
  "cluster": 3,
  "persona": "Alto Valor Fidelizado", 
  "risco_churn": "BAIXO",
  "acao_recomendada": "Manter satisfação"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /predict/batch</h3>
                <p>Predição em lote para múltiplos clientes.</p>
                <pre>Request: {
  "clientes": [
    {"contrato": "12345", "valor_contrato": 120.50},
    {"contrato": "67890", "valor_contrato": 85.00}
  ]
}

Response: {
  "predictions": [...],
  "total_processed": 2
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /predict/contract/{contract_id}</h3>
                <p>Predição consultando contrato específico no banco.</p>
                <pre>GET /predict/contract/12345

Response: {
  "cluster": 2,
  "persona": "Alto Valor Não Fidelizado",
  "risco_churn": "MÉDIO-ALTO", 
  "dados_completos": {...}
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /predict/cpf/{cpf_cnpj}</h3>
                <p>Predição consultando CPF/CNPJ no banco (pode retornar múltiplos contratos).</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /predict/sample?limit=10</h3>
                <p>Predição de amostra aleatória para demonstração.</p>
            </div>
            
            <h2>🎯 Personas da Ligga</h2>
            <ul>
                <li><strong>Recente Negativo</strong> - ALTO risco</li>
                <li><strong>Recente Positivo</strong> - BAIXO risco</li>
                <li><strong>Alto Valor Não Fidelizado</strong> - MÉDIO-ALTO risco</li>
                <li><strong>Alto Valor Fidelizado</strong> - BAIXO risco</li>
                <li><strong>Padrão Capital</strong> - MÉDIO risco</li>
                <li><strong>Baixo Valor Negativo</strong> - ALTO risco</li>
                <li><strong>Baixo Valor Positivo</strong> - BAIXO risco</li>
                <li><strong>Interior Não Fidelizado</strong> - MÉDIO risco</li>
                <li><strong>Interior Fidelizado</strong> - BAIXO risco</li>
                <li><strong>Baixa Velocidade</strong> - MÉDIO risco</li>
            </ul>
            
            <h2>⚠️ Códigos de Erro</h2>
            <ul>
                <li><strong>400</strong> - Dados de entrada inválidos</li>
                <li><strong>404</strong> - Contrato/CPF não encontrado</li>
                <li><strong>500</strong> - Erro interno do servidor</li>
            </ul>
        </body>
        </html>
        """
        
        return docs_template
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Executa a API"""
        logger.info(f"🚀 Iniciando API da Ligga em http://{host}:{port}")
        logger.info("📚 Documentação disponível em /docs")
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API de Predição de Churn - Ligga")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host da API')
    parser.add_argument('--port', type=int, default=5000, help='Porta da API')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    # Criar diretório de logs
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Inicializar API
        api = LiggaAPI()
        
        # Executar
        api.run(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        logger.info("⏹️  API interrompida pelo usuário")
    except Exception as e:
        logger.error(f"💥 Erro fatal na API: {e}")
        raise

if __name__ == "__main__":
    main()
