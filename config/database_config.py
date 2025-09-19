import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class PostgreSQLConfig:
    """Configuração específica para PostgreSQL"""
    host: str = os.getenv('DB_HOST', 'svlxdlkvip')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'dwdb')
    username: str = os.getenv('DB_USER', 'lyandra.lisboa') 
    password: str = os.getenv('DB_PASSWORD', 'password')
    schema: str = os.getenv('DB_SCHEMA', 'ngweb')  # Schema atualizado conforme tabelas
    
    def get_connection_string(self) -> str:
        """String de conexão PostgreSQL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass 
class TableConfig:
    """Configuração de mapeamento das tabelas existentes da Ligga"""
    
    # TABELAS DA LIGGA
    churn_table: str = "ngweb.tb_fat_ngweb_churn"
    atendimento_table: str = "ngweb.tb_fat_ngweb_registro_atendimento" 
    cancelados_table: str = "ngweb.tb_fat_ngweb_dados_contrato_cancelado"
    
    # CONFIGURAÇÃO DE COLUNAS DAS TABELAS REAIS
    column_mapping: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.column_mapping is None:
            # MAPEAMENTO BASEADO NAS TABELAS REAIS DA LIGGA
            self.column_mapping = {
                # TABELA DE CHURN - tb_fat_ngweb_churn
                "churn": {
                    "customer_id": "contrato",
                    "account": "conta",
                    "customer_name": "cliente", 
                    "customer_doc": "cpf_cnpj",
                    "contract_status": "status",
                    "contract_type": "tipo_contrato",
                    "contract_value": "valor_contrato",
                    "cancellation_date": "data_pedido_cancelamento",
                    "end_date": "data_fim",
                    "end_date_adjusted": "data_fim_ajuste",
                    "cancellation_reason": "motivo_cancelamento",
                    "churn_type": "tipo_churn",
                    "client_type": "tipo_cliente",
                    "business_sector": "ramo_atividade",
                    "city": "cidade",
                    "neighborhood": "bairro",
                    "zip_code": "cep",
                    "final_product": "produto_final",
                    "is_mandatory": "is_obrigatorio",
                    "speed": "velocidade",
                    "product_type": "tipo_produto",
                    "product_channel": "canal_produto",
                    "contract_amount": "valor_contratacao",
                    "due_day": "dia_vencimento",
                    "current_promotion": "promocao_atual",
                    "trade_name": "nome_fantasia",
                    "phone_ddd": "ddd_telefone_celular",
                    "email": "email_cliente",
                    "birth_date": "data_nascimento",
                    "creation_date": "data_criacao_contrato",
                    "installation_date": "data_instalacao",
                    "original_installation_date": "data_instalacao_original",
                    "last_update_date": "data_ultima_alteracao",
                    "origin": "origem",
                    "seller_id": "vendedor_id",
                    "seller": "vendedor",
                    "last_update": "last_update"
                },
                
                # TABELA DE ATENDIMENTO - tb_fat_ngweb_registro_atendimento
                "support": {
                    "customer_name": "cliente",
                    "customer_doc": "cpf_cnpj_cliente", 
                    "customer_id": "contrato",
                    "ticket_id": "protocolo_atendimento",
                    "ticket_type": "tipo",
                    "origin": "origem",
                    "status": "status",
                    "opening_date": "abertura",
                    "closing_date": "fechamento",
                    "description": "observacao_atendimento",
                    "attendant": "funcionario",
                    "last_update": "last_update"
                },
                
                # TABELA DE CONTRATOS CANCELADOS - tb_fat_ngweb_dados_contrato_cancelado
                "cancelled_contracts": {
                    "customer_name": "nome",
                    "customer_doc": "cpf_cnpj",
                    "strategic": "estrategico",
                    "person_type": "tipo_pessoa",
                    "contract_type": "tipo_contrato",
                    "contract_id": "contrato",
                    "contract_status": "status_contrato",
                    "request_status": "status_solicitacao",
                    "request_date": "data_solicitacao",
                    "request_channel": "canal_solicitacao",
                    "request_login": "login_solicitacao",
                    "cancellation_reason": "motivo_cancelamento",
                    "requires_documentation": "exige_documentacao",
                    "documentation_received": "documentacao_recebida",
                    "work_order": "numero_os",
                    "critical_date": "data_critica",
                    "request_cancellation_date": "data_cancelamento_solicitacao",
                    "removal_date": "data_remocao",
                    "cancellation_login": "Login que cancelou",
                    "cancellation_person": "Pessoa que cancelou",
                    "contract_cancellation_date": "data_cancelamento_contrato",
                    "execution_date": "data_execucao",
                    "last_update": "last_update"
                }
            }
