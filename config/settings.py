import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .database_config import PostgreSQLConfig, TableConfig

@dataclass
class Settings:
    """Configurações otimizadas do sistema de predição de churn com k-modes"""
    
    def __init__(self):
        # Configurações de banco
        self.database = PostgreSQLConfig()
        self.tables = TableConfig()
        
        # Configurações de ambiente
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Configurações de análise otimizadas
        self.min_customers_for_analysis = int(os.getenv('MIN_CUSTOMERS', '100'))
        self.n_clusters_default = int(os.getenv('DEFAULT_CLUSTERS', '7'))
        self.score_dimensions = int(os.getenv('SCORE_DIMENSIONS', '5'))
        
        # Períodos de análise (em meses)
        self.billing_analysis_months = int(os.getenv('BILLING_MONTHS', '12'))
        self.usage_analysis_months = int(os.getenv('USAGE_MONTHS', '6'))
        self.support_analysis_months = int(os.getenv('SUPPORT_MONTHS', '12'))
        
        # Thresholds de risco
        self.high_risk_threshold = float(os.getenv('HIGH_RISK_THRESHOLD', '0.7'))
        self.medium_risk_threshold = float(os.getenv('MEDIUM_RISK_THRESHOLD', '0.4'))
        
        # Configurações de processamento otimizadas
        self.batch_size = int(os.getenv('BATCH_SIZE', '10000'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '4'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        
        # Configurações de cache
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
        self.cache_ttl = int(os.getenv('CACHE_TTL', '3600'))  # 1 hora
        
        # Configurações específicas do K-modes
        self.kmodes_config = self._init_kmodes_config()
        
        # Configurações de modelos ML
        self.model_config = self._init_model_config()
        
        # Configurações de features categóricas
        self.feature_config = self._init_feature_config()
        
        # Configurações de monitoramento
        self.monitoring_enabled = os.getenv('MONITORING_ENABLED', 'True').lower() == 'true'
        self.alert_threshold_drift = float(os.getenv('ALERT_DRIFT_THRESHOLD', '0.1'))
    
    def _init_kmodes_config(self) -> Dict[str, Any]:
        """Configurações específicas do algoritmo K-modes"""
        return {
            'n_clusters_range': (
                int(os.getenv('KMODES_MIN_CLUSTERS', '3')),
                int(os.getenv('KMODES_MAX_CLUSTERS', '15'))
            ),
            'init_method': os.getenv('KMODES_INIT', 'Huang'),
            'n_init': int(os.getenv('KMODES_N_INIT', '10')),
            'max_iter': int(os.getenv('KMODES_MAX_ITER', '100')),
            'random_state': int(os.getenv('RANDOM_STATE', '42')),
            'verbose': int(os.getenv('KMODES_VERBOSE', '1')),
            
            # Métricas de qualidade
            'min_silhouette_score': float(os.getenv('MIN_SILHOUETTE', '0.3')),
            'max_unassigned_rate': float(os.getenv('MAX_UNASSIGNED', '0.05')),
            'stability_threshold': float(os.getenv('STABILITY_THRESHOLD', '0.90')),
            
            # Colunas categóricas principais para k-modes
            'categorical_columns': [
                'contract_type', 'client_type', 'business_sector', 'city',
                'product_type', 'product_channel', 'cancellation_reason',
                'churn_type', 'ticket_type', 'support_origin'
            ]
        }
    
    def _init_model_config(self) -> Dict[str, Any]:
        """Configurações dos modelos de ML especializados"""
        return {
            'algorithms': os.getenv('ML_ALGORITHMS', 'random_forest,gradient_boosting,logistic_regression').split(','),
            'cv_folds': int(os.getenv('CV_FOLDS', '5')),
            'test_size': float(os.getenv('TEST_SIZE', '0.2')),
            'random_state': int(os.getenv('RANDOM_STATE', '42')),
            
            # Métricas mínimas de performance
            'min_auc_score': float(os.getenv('MIN_AUC', '0.7')),
            'min_precision': float(os.getenv('MIN_PRECISION', '0.6')),
            'min_recall': float(os.getenv('MIN_RECALL', '0.6')),
            
            # Configurações de hiperparâmetros
            'hyperparameter_tuning': os.getenv('HYPERPARAMETER_TUNING', 'True').lower() == 'true',
            'max_evals': int(os.getenv('MAX_EVALS', '50'))
        }
    
    def _init_feature_config(self) -> Dict[str, Any]:
        """Configurações de feature engineering categórico"""
        return {
            # Categorias de valor
            'value_categories': {
                'baixo_valor': (0, 100),
                'medio_valor': (100, 300),
                'alto_valor': (300, 1000),
                'premium': (1000, float('inf'))
            },
            
            # Categorias de tenure (em meses)
            'tenure_categories': {
                'novo': 6,           # 0-6 meses
                'estabelecido': 24,  # 6-24 meses
                'veterano': 60,      # 24-60 meses
                'longo_prazo': 999   # 60+ meses
            },
            
            # Configurações de encoding
            'encoding_method': os.getenv('ENCODING_METHOD', 'target'),
            'handle_missing': os.getenv('HANDLE_MISSING', 'category'),
            'missing_category': os.getenv('MISSING_CATEGORY', 'MISSING'),
            'min_category_frequency': int(os.getenv('MIN_CATEGORY_FREQ', '50')),
            'max_categories_per_feature': int(os.getenv('MAX_CATEGORIES', '20'))
        }
    
    def is_production(self) -> bool:
        """Verifica se está em ambiente de produção"""
        return self.environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """Verifica se está em ambiente de desenvolvimento"""
        return self.environment.lower() == 'development'
    
    def get_log_config(self) -> Dict[str, Any]:
        """Configurações de logging otimizadas"""
        return {
            'level': self.log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'handlers': ['console', 'file'] if self.is_production() else ['console'],
            'filename': 'logs/churn_prediction.log' if self.is_production() else None
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Configurações de processamento de dados"""
        return {
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'parallel_processing': self.max_workers > 1
        }
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """Valida configurações principais"""
        errors = []
        
        # Validar banco
        if not self.database.test_connection():
            errors.append("Falha na conexão com banco de dados")
        
        # Validar clusters
        min_clusters, max_clusters = self.kmodes_config['n_clusters_range']
        if min_clusters >= max_clusters:
            errors.append("Range de clusters inválido")
        
        # Validar thresholds
        if not (0 <= self.medium_risk_threshold <= self.high_risk_threshold <= 1):
            errors.append("Thresholds de risco inválidos")
        
        # Validar algoritmos
        valid_algorithms = ['random_forest', 'gradient_boosting', 'logistic_regression']
        invalid_algs = [alg for alg in self.model_config['algorithms'] if alg not in valid_algorithms]
        if invalid_algs:
            errors.append(f"Algoritmos inválidos: {invalid_algs}")
        
        return len(errors) == 0, errors
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das configurações principais"""
        return {
            'environment': self.environment,
            'database': {
                'host': self.database.host,
                'database': self.database.database,
                'schema': self.database.schema
            },
            'clustering': {
                'n_clusters_range': self.kmodes_config['n_clusters_range'],
                'categorical_columns': len(self.kmodes_config['categorical_columns'])
            },
            'models': {
                'algorithms': self.model_config['algorithms'],
                'cv_folds': self.model_config['cv_folds']
            },
            'processing': {
                'batch_size': self.batch_size,
                'max_workers': self.max_workers
            }
        }

# Instância global otimizada
settings = Settings()

# Validação automática na importação
if settings.debug:
    is_valid, errors = settings.validate_config()
    if is_valid:
        print("✅ Configurações validadas com sucesso")
        print(f"🔧 Environment: {settings.environment}")
        print(f"🗄️ Database: {settings.database.host}:{settings.database.database}")
    else:
        print("❌ Erros nas configurações:")
        for error in errors:
            print(f"   - {error}")
