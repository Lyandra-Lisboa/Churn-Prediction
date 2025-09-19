import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from src.data.database import PostgreSQLManager
from config.settings import settings

logger = logging.getLogger(__name__)

class DataExtractor:
    """Extrator de dados das tabelas PostgreSQL existentes"""
    
    def __init__(self, db_manager: PostgreSQLManager):
        self.db = db_manager
        self.tables = settings.tables
        self.column_map = self.tables.column_mapping
        
    def validate_setup(self) -> Dict[str, Any]:
        """Valida se tabelas e colunas existem"""
        logger.info("🔍 Validando estrutura do banco de dados...")
        
        # Verificar tabelas
        required_tables = [
            self.tables.customers_table,
            self.tables.billing_table,
            self.tables.usage_table,
            self.tables.support_table,
            self.tables.services_table
        ]
        
        tables_exist = self.db.check_tables_exist(required_tables)
        missing_tables = [t for t, exists in tables_exist.items() if not exists]
        
        validation_result = {
            'tables_exist': tables_exist,
            'missing_tables': missing_tables,
            'table_info': {},
            'row_counts': {},
            'column_validation': {}
        }
        
        # Para tabelas que existem, obter informações detalhadas
        for table_name, exists in tables_exist.items():
            if exists:
                try:
                    # Informações da tabela
                    table_info = self.db.get_table_info(table_name)
                    validation_result['table_info'][table_name] = table_info
                    
                    # Contagem de registros
                    row_count = self.db.get_row_count(table_name)
                    validation_result['row_counts'][table_name] = row_count
                    
                    # Validar colunas esperadas
                    available_columns = table_info['column_name'].tolist()
                    expected_columns = list(self.column_map.get(table_name, {}).keys())
                    missing_columns = [col for col in expected_columns if col not in available_columns]
                    
                    validation_result['column_validation'][table_name] = {
                        'available_columns': available_columns,
                        'expected_columns': expected_columns,
                        'missing_columns': missing_columns
                    }
                    
                    logger.info(f"✅ Tabela {table_name}: {row_count:,} registros")
                    
                    if missing_columns:
                        logger.warning(f"⚠️  Colunas faltantes em {table_name}: {missing_columns}")
                    
                except Exception as e:
                    logger.error(f"❌ Erro ao validar tabela {table_name}: {e}")
        
        if missing_tables:
            logger.warning(f"⚠️  Tabelas não encontradas: {missing_tables}")
            logger.info("💡 Configure os nomes corretos em config/database_config.py")
        
        return validation_result
    
    def extract_customers_data(self) -> pd.DataFrame:
        """Extrai dados básicos dos clientes"""
        logger.info("📊 Extraindo dados de clientes...")
        
        # Mapear colunas da tabela de clientes
        col_map = self.column_map.get("customers", {})
        
        # Construir SELECT com mapeamento de colunas
        select_columns = []
        for db_col, standard_col in col_map.items():
            select_columns.append(f"{db_col} AS {standard_col}")
        
        # Query básica para clientes
        query = f"""
        SELECT 
            {', '.join(select_columns)},
            EXTRACT(EPOCH FROM (CURRENT_DATE - {col_map.get('signup_date', 'created_date')}))/2629746 as tempo_base_meses
        FROM {settings.database.schema}.{self.tables.customers_table}
        WHERE status = 'ACTIVE'  -- Ajuste conforme sua coluna de status
        """
        
        try:
            df = self.db.execute_query(query)
            logger.info(f"✅ Dados de clientes extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de clientes: {e}")
            logger.info("💡 Verifique o mapeamento de colunas em config/database_config.py")
            raise
    
    def extract_billing_data(self, months: int = 12) -> pd.DataFrame:
        """Extrai dados de faturamento dos últimos N meses"""
        logger.info(f"💰 Extraindo dados de faturamento ({months} meses)...")
        
        col_map = self.column_map.get("billing", {})
        
        # Construir SELECT
        select_columns = []
        for db_col, standard_col in col_map.items():
            select_columns.append(f"{db_col} AS {standard_col}")
        
        query = f"""
        SELECT 
            {', '.join(select_columns)}
        FROM {settings.database.schema}.{self.tables.billing_table}
        WHERE {col_map.get('invoice_date', 'created_date')} >= CURRENT_DATE - INTERVAL '{months} months'
        ORDER BY {col_map.get('customer_id', 'customer_id')}, {col_map.get('invoice_date', 'created_date')}
        """
        
        try:
            df = self.db.execute_query(query)
            logger.info(f"✅ Dados de faturamento extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de faturamento: {e}")
            raise
    
    def extract_usage_data(self, months: int = 6) -> pd.DataFrame:
        """Extrai dados de uso dos últimos N meses"""
        logger.info(f"📈 Extraindo dados de uso ({months} meses)...")
        
        col_map = self.column_map.get("usage", {})
        
        select_columns = []
        for db_col, standard_col in col_map.items():
            select_columns.append(f"{db_col} AS {standard_col}")
        
        query = f"""
        SELECT 
            {', '.join(select_columns)}
        FROM {settings.database.schema}.{self.tables.usage_table}
        WHERE {col_map.get('usage_date', 'created_date')} >= CURRENT_DATE - INTERVAL '{months} months'
        ORDER BY {col_map.get('customer_id', 'customer_id')}, {col_map.get('usage_date', 'created_date')}
        """
        
        try:
            df = self.db.execute_query(query)
            logger.info(f"✅ Dados de uso extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de uso: {e}")
            raise
    
    def extract_support_data(self, months: int = 12) -> pd.DataFrame:
        """Extrai dados de suporte dos últimos N meses"""
        logger.info(f"🎧 Extraindo dados de suporte ({months} meses)...")
        
        col_map = self.column_map.get("support", {})
        
        select_columns = []
        for db_col, standard_col in col_map.items():
            select_columns.append(f"{db_col} AS {standard_col}")
        
        query = f"""
        SELECT 
            {', '.join(select_columns)}
        FROM {settings.database.schema}.{self.tables.support_table}
        WHERE {col_map.get('ticket_date', 'created_date')} >= CURRENT_DATE - INTERVAL '{months} months'
        ORDER BY {col_map.get('customer_id', 'customer_id')}, {col_map.get('ticket_date', 'created_date')}
        """
        
        try:
            df = self.db.execute_query(query)
            logger.info(f"✅ Dados de suporte extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de suporte: {e}")
            raise
    
    def extract_services_data(self) -> pd.DataFrame:
        """Extrai dados de serviços adicionais"""
        logger.info("⚙️ Extraindo dados de serviços...")
        
        col_map = self.column_map.get("services", {})
        
        select_columns = []
        for db_col, standard_col in col_map.items():
            select_columns.append(f"{db_col} AS {standard_col}")
        
        query = f"""
        SELECT 
            {', '.join(select_columns)}
        FROM {settings.database.schema}.{self.tables.services_table}
        WHERE {col_map.get('status', 'status')} = 'ACTIVE'
        ORDER BY {col_map.get('customer_id', 'customer_id')}
        """
        
        try:
            df = self.db.execute_query(query)
            logger.info(f"✅ Dados de serviços extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de serviços: {e}")
            raise
    
    def extract_aggregated_customer_data(self) -> pd.DataFrame:
        """
        Extrai dados agregados de clientes combinando todas as tabelas
        Esta é a função principal que você vai usar
        """
        logger.info("🔄 Extraindo dados agregados de todos os clientes...")
        
        try:
            # 1. Dados básicos dos clientes
            customers_df = self.extract_customers_data()
            
            # 2. Agregar dados de faturamento
            billing_df = self.extract_billing_data()
            billing_agg = self._aggregate_billing_data(billing_df)
            
            # 3. Agregar dados de uso
            usage_df = self.extract_usage_data()
            usage_agg = self._aggregate_usage_data(usage_df)
            
            # 4. Agregar dados de suporte
            support_df = self.extract_support_data()
            support_agg = self._aggregate_support_data(support_df)
            
            # 5. Agregar dados de serviços
            services_df = self.extract_services_data()
            services_agg = self._aggregate_services_data(services_df)
            
            # 6. Combinar tudo
            final_df = customers_df
            
            # Merge com dados agregados
            for agg_df, suffix in [
                (billing_agg, 'billing'),
                (usage_agg, 'usage'),
                (support_agg, 'support'),
                (services_agg, 'services')
            ]:
                if not agg_df.empty:
                    final_df = final_df.merge(
                        agg_df,
                        on='cliente_id',
                        how='left',
                        suffixes=('', f'_{suffix}')
                    )
            
            # Preencher valores nulos com zeros onde apropriado
            numeric_columns = final_df.select_dtypes(include=['number']).columns
            final_df[numeric_columns] = final_df[numeric_columns].fillna(0)
            
            logger.info(f"✅ Dados agregados finalizados: {len(final_df):,} clientes, {len(final_df.columns)} colunas")
            
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados agregados: {e}")
            raise
    
    def _aggregate_billing_data(self, billing_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados de faturamento por cliente"""
        if billing_df.empty:
            return pd.DataFrame()
        
        # Calcular atrasos
        billing_df['dias_atraso'] = (
            pd.to_datetime(billing_df['data_pagamento']) - 
            pd.to_datetime(billing_df['data_vencimento'])
        ).dt.days.clip(lower=0)
        
        agg_data = billing_df.groupby('cliente_id').agg({
            'valor_fatura': ['mean', 'sum', 'count'],
            'dias_atraso': ['sum', 'mean'],
            'status_pagamento': lambda x: (x == 'ATRASADO').sum()
        }).round(2)
        
        # Flatten column names
        agg_data.columns = [
            'valor_medio_fatura', 'receita_total_12m', 'qtd_faturas_12m',
            'total_dias_atraso', 'media_dias_atraso', 'faturas_atraso_12m'
        ]
        
        return agg_data.reset_index()
    
    def _aggregate_usage_data(self, usage_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados de uso por cliente"""
        if usage_df.empty:
            return pd.DataFrame()
        
        agg_data = usage_df.groupby('cliente_id').agg({
            'consumo_gb': ['mean', 'sum', 'std'],
            'dispositivos_conectados': ['mean', 'max']
        }).round(2)
        
        agg_data.columns = [
            'consumo_gb_medio', 'consumo_gb_total', 'variabilidade_uso',
            'dispositivos_medio', 'max_dispositivos'
        ]
        
        return agg_data.reset_index()
    
    def _aggregate_support_data(self, support_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados de suporte por cliente"""
        if support_df.empty:
            return pd.DataFrame()
        
        agg_data = support_df.groupby('cliente_id').agg({
            'tipo_ticket': ['count'],
            'status_ticket': lambda x: (x == 'RESOLVIDO').sum() / len(x) if len(x) > 0 else 0
        }).round(2)
        
        agg_data.columns = ['tickets_suporte_12m', 'taxa_resolucao']
        
        return agg_data.reset_index()
    
    def _aggregate_services_data(self, services_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados de serviços por cliente"""
        if services_df.empty:
            return pd.DataFrame()
        
        agg_data = services_df.groupby('cliente_id').agg({
            'nome_servico': 'count'
        })
        
        agg_data.columns = ['servicos_adicionais']
        
        return agg_data.reset_index()
