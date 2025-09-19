from config.database_config import PostgreSQLConfig, TableConfig

class Settings:
    def __init__(self):
        # Configurações de banco
        self.database = PostgreSQLConfig()
        self.tables = TableConfig()
        
        # Configurações de análise
        self.min_customers_for_analysis = 100
        self.n_clusters_default = 7
        self.score_dimensions = 5
        
        # Períodos de análise (em meses)
        self.billing_analysis_months = 12
        self.usage_analysis_months = 6
        self.support_analysis_months = 12
        
        # Thresholds de risco
        self.high_risk_threshold = 0.7
        self.medium_risk_threshold = 0.4

settings = Settings()
