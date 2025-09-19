import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any, List
import logging
from contextlib import contextmanager
from config.database_config import PostgreSQLConfig

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """Gerenciador específico para PostgreSQL"""
    
    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Estabelece conexão com PostgreSQL"""
        try:
            connection_string = self.config.get_connection_string()
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False  # True para debug de queries
            )
            
            # Teste de conexão
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.info("✅ Conexão PostgreSQL estabelecida")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar PostgreSQL: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Executa query e retorna DataFrame"""
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params or {})
            return df
        except Exception as e:
            logger.error(f"Erro ao executar query: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Obtém informações sobre uma tabela"""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = :table_name 
        AND table_schema = :schema
        ORDER BY ordinal_position
        """
        return self.execute_query(query, {
            'table_name': table_name,
            'schema': self.config.schema
        })
    
    def check_tables_exist(self, table_names: List[str]) -> Dict[str, bool]:
        """Verifica quais tabelas existem"""
        results = {}
        for table_name in table_names:
            query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = :table_name 
                AND table_schema = :schema
            )
            """
            result = self.execute_query(query, {
                'table_name': table_name,
                'schema': self.config.schema
            })
            results[table_name] = result.iloc[0, 0]
        return results
    
    def get_row_count(self, table_name: str) -> int:
        """Conta registros em uma tabela"""
        query = f"SELECT COUNT(*) FROM {self.config.schema}.{table_name}"
        result = self.execute_query(query)
        return int(result.iloc[0, 0])

    @contextmanager
    def get_connection(self):
        """Context manager para conexão"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
