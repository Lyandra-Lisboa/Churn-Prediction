import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class PostgreSQLConfig:
    """Configuração específica para PostgreSQL"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'postgres')
    username: str = os.getenv('DB_USER', 'postgres') 
    password: str = os.getenv('DB_PASSWORD', 'password')
    schema: str = os.getenv('DB_SCHEMA', 'public')
    
    def get_connection_string(self) -> str:
        """String de conexão PostgreSQL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass 
class TableConfig:
    """Configuração de mapeamento das tabelas existentes"""
    
    # CONFIGURE AQUI OS NOMES DAS SUAS TABELAS REAIS
    customers_table: str = "customers"  # ← ALTERE para nome da sua tabela de clientes
    billing_table: str = "billing"     # ← ALTERE para nome da sua tabela de faturas  
    usage_table: str = "usage"         # ← ALTERE para nome da sua tabela de uso
    support_table: str = "tickets"     # ← ALTERE para nome da sua tabela de suporte
    services_table: str = "services"   # ← ALTERE para nome da sua tabela de serviços
    
    # CONFIGURE AQUI OS NOMES DAS COLUNAS DAS SUAS TABELAS
    column_mapping: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.column_mapping is None:
            # CONFIGURE AQUI O MAPEAMENTO DAS SUAS COLUNAS
            self.column_mapping = {
                "customers": {
                    # "nome_no_sistema": "nome_padrao_interno"
                    "customer_id": "cliente_id",           # ID do cliente
                    "age": "idade",                        # Idade
                    "signup_date": "data_ativacao",        # Data de ativação
                    "monthly_value": "valor_mensalidade",  # Valor mensal
                    "payment_method": "meio_pagamento",    # Meio de pagamento
                    "region": "regiao_geografica",         # Região
                    "acquisition_channel": "canal_aquisicao", # Canal de aquisição
                    "residence_type": "tipo_residencia",   # Tipo de residência
                    "contracted_speed": "velocidade_contratada", # Velocidade
                    "credit_score": "score_credito"        # Score de crédito
                },
                "billing": {
                    "customer_id": "cliente_id",
                    "invoice_date": "data_fatura", 
                    "due_date": "data_vencimento",
                    "payment_date": "data_pagamento",
                    "amount": "valor_fatura",
                    "status": "status_pagamento"
                },
                "usage": {
                    "customer_id": "cliente_id",
                    "usage_date": "data_uso",
                    "data_consumed_gb": "consumo_gb",
                    "devices_connected": "dispositivos_conectados"
                },
                "support": {
                    "customer_id": "cliente_id", 
                    "ticket_date": "data_abertura",
                    "ticket_type": "tipo_ticket",
                    "status": "status_ticket"
                },
                "services": {
                    "customer_id": "cliente_id",
                    "service_name": "nome_servico",
                    "status": "status_servico"
                }
            }
