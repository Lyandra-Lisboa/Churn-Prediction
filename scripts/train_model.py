import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import List, Optional, Union
from config import settings, kmodes_config, model_config
from src.data.extractors import DataExtractor
from src.features.categorical_engineering import preprocess_data
from src.features.transformers import CategoricalTransformer
from src.clustering.cluster_predictor import ClusterPredictor

# Configuração de logs usando settings
log_config = settings.get_log_config()
logging.basicConfig(
    level=getattr(logging, log_config['level']),
    format=log_config['format'],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

class OptimizedBatchPredictor:
    """
    Classe integrada com configurações otimizadas
    """
    
    def __init__(self, model_path: Optional[str] = None, transformer_path: Optional[str] = None):
        # ✅ Usar configurações centralizadas
        self.n_clusters = kmodes_config.n_clusters_range[1]  
        self.batch_size = settings.batch_size
        self.max_workers = settings.max_workers
        
        # ✅ Usar estrutura de diretórios otimizada
        self.models_dir = Path("models/clusters")
        self.output_dir = Path("data/predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ Usar extractor integrado
        self.extractor = DataExtractor()
        
        # ✅ Personas usando config
        self.personas_info = self._load_personas_from_config()
        
        self._load_model_and_transformer(model_path, transformer_path)
    
    def _load_personas_from_config(self) -> dict:
        """Carrega personas das configurações"""
        # Pode vir de um arquivo de config ou ser definido em feature_config
        return {
            0: {"nome": "Recente Negativo", "risco": "ALTO", "acao_recomendada": "Atenção especial"},
            1: {"nome": "Recente Positivo", "risco": "BAIXO", "acao_recomendada": "Manter engajamento"},
            # ... outras personas
        }
    
    def _load_model_and_transformer(self, model_path: Optional[str], transformer_path: Optional[str]):
        """Carrega modelo usando estrutura otimizada"""
        logger.info("📦 Carregando modelo...")
        
        try:
            # ✅ Usar diretórios corretos
            if model_path is None:
                model_path = self.models_dir / "latest_kmodes.pkl"
            
            if transformer_path is None:
                transformer_path = self.models_dir / "latest_transformer.pkl"
            
            # Verificar arquivos
            for path in [model_path, transformer_path]:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            
            # Carregar
            self.model = joblib.load(model_path)
            self.transformer = joblib.load(transformer_path)
            
            logger.info("✅ Modelo e transformer carregados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def predict_from_database(self, contratos: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Predições usando extractor integrado
        """
        logger.info("🗄️ Consultando dados...")
        
        try:
            # ✅ Usar método integrado do extractor
            if contratos:
                df = self.extractor.extract_customers_data(customer_ids=contratos)
            else:
                df = self.extractor.extract_customers_data(limit=settings.batch_size)
            
            # ✅ Usar preprocessamento integrado
            df_processed = preprocess_data(df)
            
            # ✅ Usar transformer integrado
            df_transformed = self.transformer.transform(df_processed)
            
            # ✅ Usar features das configurações
            features_used = kmodes_config.categorical_columns
            available_features = [f for f in features_used if f in df_transformed.columns]
            
            if not available_features:
                raise ValueError("Nenhuma feature configurada encontrada")
            
            # Predição
            X = df_transformed[available_features]
            clusters = self.model.predict(X)
            
            # Preparar resultados
            results = df.copy()
            results['cluster_pred'] = clusters
            results['persona'] = [self.personas_info[c]['nome'] for c in clusters]
            results['risco_churn'] = [self.personas_info[c]['risco'] for c in clusters]
            results['acao_recomendada'] = [self.personas_info[c]['acao_recomendada'] for c in clusters]
            results['data_predicao'] = datetime.now().isoformat()
            
            logger.info(f"✅ Predições geradas: {len(results):,} registros")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            raise
