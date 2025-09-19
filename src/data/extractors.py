import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import numpy as np
from sqlalchemy import create_engine, text
from config_ligga import PostgreSQLConfig, TableConfig

logger = logging.getLogger(__name__)

class LiggaDataExtractor:
    """Extrator de dados das tabelas específicas da Ligga"""
    
    def __init__(self):
        self.db_config = PostgreSQLConfig()
        self.table_config = TableConfig()
        self.engine = None
        
    def _get_engine(self):
        """Obtém conexão com o banco"""
        if self.engine is None:
            self.engine = create_engine(self.db_config.get_connection_string())
        return self.engine
    
    def validate_setup(self) -> Dict[str, Any]:
        """Valida se tabelas e colunas existem"""
        logger.info("🔍 Validando estrutura do banco de dados Ligga...")
        
        engine = self._get_engine()
        
        # Tabelas da Ligga
        required_tables = [
            self.table_config.churn_table,
            self.table_config.atendimento_table,
            self.table_config.cancelados_table
        ]
        
        validation_result = {
            'tables_exist': {},
            'missing_tables': [],
            'table_info': {},
            'row_counts': {},
            'column_validation': {}
        }
        
        for table_name in required_tables:
            try:
                # Verificar se tabela existe e contar registros
                query = f"SELECT COUNT(*) as count FROM {table_name}"
                result = pd.read_sql(query, engine)
                
                validation_result['tables_exist'][table_name] = True
                validation_result['row_counts'][table_name] = result['count'].iloc[0]
                
                # Obter informações das colunas
                info_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_schema = 'ngweb' 
                AND table_name = '{table_name.split('.')[-1]}'
                ORDER BY ordinal_position
                """
                
                columns_info = pd.read_sql(info_query, engine)
                validation_result['table_info'][table_name] = columns_info
                
                logger.info(f"✅ Tabela {table_name}: {result['count'].iloc[0]:,} registros")
                
            except Exception as e:
                validation_result['tables_exist'][table_name] = False
                validation_result['missing_tables'].append(table_name)
                logger.error(f"❌ Erro ao validar tabela {table_name}: {e}")
        
        if validation_result['missing_tables']:
            logger.warning(f"⚠️  Tabelas não encontradas: {validation_result['missing_tables']}")
        
        return validation_result
    
    def extract_churn_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Extrai dados da tabela de churn"""
        logger.info("📊 Extraindo dados de churn...")
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT 
            contrato,
            conta,
            cliente,
            cpf_cnpj,
            status,
            tipo_contrato,
            data_criacao_contrato,
            data_instalacao,
            data_pedido_cancelamento,
            data_fim,
            data_fim_ajuste,
            motivo_cancelamento,
            tipo_churn,
            tipo_cliente,
            ramo_atividade,
            valor_contrato,
            cidade,
            bairro,
            cep,
            produto_final,
            is_obrigatorio,
            velocidade,
            tipo_produto,
            canal_produto,
            valor_contratacao,
            dia_vencimento,
            promocao_atual,
            nome_fantasia,
            ddd_telefone_celular,
            email_cliente,
            data_nascimento,
            data_ultima_alteracao,
            origem,
            vendedor_id,
            vendedor,
            last_update
        FROM {self.table_config.churn_table}
        WHERE contrato IS NOT NULL
        ORDER BY contrato
        {limit_clause}
        """
        
        try:
            engine = self._get_engine()
            df = pd.read_sql(query, engine)
            logger.info(f"✅ Dados de churn extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de churn: {e}")
            raise
    
    def extract_atendimento_data(self, months: int = 12) -> pd.DataFrame:
        """Extrai dados de atendimento dos últimos N meses"""
        logger.info(f"🎧 Extraindo dados de atendimento ({months} meses)...")
        
        query = f"""
        SELECT 
            cliente,
            cpf_cnpj_cliente,
            contrato,
            protocolo_atendimento,
            tipo,
            origem,
            status,
            abertura,
            fechamento,
            observacao_atendimento,
            funcionario,
            last_update
        FROM {self.table_config.atendimento_table}
        WHERE contrato IS NOT NULL
        AND last_update >= CURRENT_DATE - INTERVAL '{months} months'
        ORDER BY contrato, abertura
        """
        
        try:
            engine = self._get_engine()
            df = pd.read_sql(query, engine)
            logger.info(f"✅ Dados de atendimento extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de atendimento: {e}")
            raise
    
    def extract_cancelados_data(self) -> pd.DataFrame:
        """Extrai dados de contratos cancelados"""
        logger.info("❌ Extraindo dados de contratos cancelados...")
        
        query = f"""
        SELECT 
            nome,
            cpf_cnpj,
            estrategico,
            tipo_pessoa,
            tipo_contrato,
            contrato,
            status_contrato,
            status_solicitacao,
            data_solicitacao,
            canal_solicitacao,
            login_solicitacao,
            motivo_cancelamento,
            exige_documentacao,
            documentacao_recebida,
            numero_os,
            data_critica,
            data_cancelamento_solicitacao,
            data_remocao,
            "Login que cancelou",
            "Pessoa que cancelou",
            data_cancelamento_contrato,
            data_execucao,
            last_update
        FROM {self.table_config.cancelados_table}
        WHERE contrato IS NOT NULL
        ORDER BY contrato
        """
        
        try:
            engine = self._get_engine()
            df = pd.read_sql(query, engine)
            logger.info(f"✅ Dados de cancelados extraídos: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados de cancelados: {e}")
            raise
    
    def _aggregate_atendimento_data(self, atendimento_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados de atendimento por contrato"""
        if atendimento_df.empty:
            return pd.DataFrame()
        
        logger.info("🔄 Agregando dados de atendimento...")
        
        # Converter datas
        atendimento_df['abertura'] = pd.to_datetime(atendimento_df['abertura'], errors='coerce')
        atendimento_df['fechamento'] = pd.to_datetime(atendimento_df['fechamento'], errors='coerce')
        
        # Classificar tipos de atendimento
        atendimento_df['tipo_lower'] = atendimento_df['tipo'].str.lower().fillna('')
        
        # Identificar atendimentos negativos (reclamações, problemas)
        termos_negativos = ['reclamacao', 'reclamação', 'problema', 'erro', 'falha', 'defeito', 'ruim', 'insatisfacao']
        atendimento_df['is_negativo'] = atendimento_df['tipo_lower'].apply(
            lambda x: any(termo in x for termo in termos_negativos)
        )
        
        # Identificar atendimentos resolvidos
        atendimento_df['is_resolvido'] = atendimento_df['status'].str.upper().isin(['FECHADO', 'RESOLVIDO', 'CONCLUIDO'])
        
        # Calcular tempo de resolução
        atendimento_df['tempo_resolucao_dias'] = (
            atendimento_df['fechamento'] - atendimento_df['abertura']
        ).dt.days
        
        # Agregar por contrato
        agg_data = atendimento_df.groupby('contrato').agg({
            'protocolo_atendimento': 'count',  # Total de atendimentos
            'is_negativo': 'sum',              # Atendimentos negativos
            'is_resolvido': 'sum',             # Atendimentos resolvidos
            'tempo_resolucao_dias': 'mean',    # Tempo médio de resolução
            'abertura': 'count'                # Confirmação de contagem
        }).round(2)
        
        # Renomear colunas
        agg_data.columns = [
            'total_atendimentos',
            'atendimentos_negativos', 
            'atendimentos_resolvidos',
            'tempo_medio_resolucao',
            'total_protocolos'
        ]
        
        # Calcular métricas derivadas
        agg_data['taxa_resolucao'] = (
            agg_data['atendimentos_resolvidos'] / agg_data['total_atendimentos']
        ).fillna(0).round(2)
        
        agg_data['taxa_atendimentos_negativos'] = (
            agg_data['atendimentos_negativos'] / agg_data['total_atendimentos'] 
        ).fillna(0).round(2)
        
        # Classificar perfil de atendimento
        def classificar_perfil(row):
            if row['total_atendimentos'] == 0:
                return 'sem_atendimento'
            elif row['atendimentos_negativos'] > 2:
                return 'problematico'
            elif row['total_atendimentos'] > 0:
                return 'com_atendimento'
            else:
                return 'sem_atendimento'
        
        agg_data['perfil_atendimento'] = agg_data.apply(classificar_perfil, axis=1)
        
        return agg_data.reset_index()
    
    def _aggregate_cancelados_data(self, cancelados_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados de cancelados por contrato"""
        if cancelados_df.empty:
            return pd.DataFrame()
        
        logger.info("🔄 Agregando dados de cancelados...")
        
        # Converter datas
        date_columns = [
            'data_solicitacao', 'data_cancelamento_contrato',
            'data_execucao', 'data_critica', 'data_remocao'
        ]
        
        for col in date_columns:
            if col in cancelados_df.columns:
                cancelados_df[col] = pd.to_datetime(cancelados_df[col], errors='coerce')
        
        # Classificar status de cancelamento
        cancelados_df['tem_cancelamento'] = cancelados_df['data_cancelamento_contrato'].notna()
        cancelados_df['tem_solicitacao'] = cancelados_df['data_solicitacao'].notna()
        
        # Agregar por contrato
        agg_data = cancelados_df.groupby('contrato').agg({
            'tem_cancelamento': 'max',         # Se tem cancelamento efetivo
            'tem_solicitacao': 'max',          # Se tem solicitação de cancelamento
            'motivo_cancelamento': 'first',    # Primeiro motivo registrado
            'canal_solicitacao': 'first',      # Canal da solicitação
            'status_solicitacao': 'first',     # Status da solicitação
            'data_solicitacao': 'min',         # Data da primeira solicitação
            'data_cancelamento_contrato': 'min' # Data do primeiro cancelamento
        })
        
        # Renomear colunas
        agg_data.columns = [
            'contrato_cancelado',
            'solicitou_cancelamento',
            'motivo_cancelamento',
            'canal_solicitacao', 
            'status_solicitacao',
            'data_primeira_solicitacao',
            'data_cancelamento'
        ]
        
        # Classificar status de churn
        def classificar_churn_status(row):
            if row['contrato_cancelado']:
                return 'cancelado'
            elif row['solicitou_cancelamento']:
                return 'solicitou_cancel'
            else:
                return 'ativo'
        
        agg_data['status_churn'] = agg_data.apply(classificar_churn_status, axis=1)
        
        return agg_data.reset_index()
    
    def extract_aggregated_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extrai e agrega dados de todas as tabelas da Ligga
        Esta é a função principal para usar no modelo
        """
        logger.info("🔄 Extraindo dados agregados de todas as tabelas...")
        
        try:
            # 1. Dados principais de churn (base)
            churn_df = self.extract_churn_data(limit=limit)
            
            if churn_df.empty:
                logger.warning("⚠️  Nenhum dado encontrado na tabela de churn")
                return pd.DataFrame()
            
            # 2. Dados de atendimento agregados
            atendimento_df = self.extract_atendimento_data()
            atendimento_agg = self._aggregate_atendimento_data(atendimento_df)
            
            # 3. Dados de cancelados agregados  
            cancelados_df = self.extract_cancelados_data()
            cancelados_agg = self._aggregate_cancelados_data(cancelados_df)
            
            # 4. Combinar todos os dados
            final_df = churn_df.copy()
            
            # Merge com dados de atendimento
            if not atendimento_agg.empty:
                final_df = final_df.merge(
                    atendimento_agg,
                    on='contrato',
                    how='left'
                )
                logger.info(f"✅ Merged com dados de atendimento: {len(atendimento_agg)} contratos")
            
            # Merge com dados de cancelados
            if not cancelados_agg.empty:
                final_df = final_df.merge(
                    cancelados_agg,
                    on='contrato', 
                    how='left',
                    suffixes=('', '_cancelados')
                )
                logger.info(f"✅ Merged com dados de cancelados: {len(cancelados_agg)} contratos")
            
            # 5. Preencher valores nulos com padrões apropriados
            # Dados de atendimento
            atendimento_cols = [
                'total_atendimentos', 'atendimentos_negativos', 'atendimentos_resolvidos',
                'tempo_medio_resolucao', 'taxa_resolucao', 'taxa_atendimentos_negativos'
            ]
            for col in atendimento_cols:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna(0)
            
            if 'perfil_atendimento' in final_df.columns:
                final_df['perfil_atendimento'] = final_df['perfil_atendimento'].fillna('sem_atendimento')
            
            # Dados de cancelamento
            cancel_bool_cols = ['contrato_cancelado', 'solicitou_cancelamento']
            for col in cancel_bool_cols:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna(False)
            
            if 'status_churn' in final_df.columns:
                final_df['status_churn'] = final_df['status_churn'].fillna('ativo')
            
            # 6. Criar features adicionais
            final_df = self._create_additional_features(final_df)
            
            logger.info(f"✅ Dados agregados finalizados: {len(final_df):,} contratos, {len(final_df.columns)} colunas")
            
            # Log das principais estatísticas
            self._log_data_summary(final_df)
            
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Erro ao extrair dados agregados: {e}")
            raise
    
    def _create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features adicionais para o modelo"""
        logger.info("🔧 Criando features adicionais...")
        
        # 1. Calcular aging em meses
        df['data_criacao_contrato'] = pd.to_datetime(df['data_criacao_contrato'], errors='coerce')
        hoje = pd.Timestamp.now()
        df['aging_meses'] = (hoje - df['data_criacao_contrato']).dt.days / 30.44
        df['aging_meses'] = df['aging_meses'].fillna(0).round(1)
        
        # 2. Extrair velocidade numérica
        df['velocidade_num'] = pd.to_numeric(
            df['velocidade'].astype(str).str.extract('(\d+)')[0], 
            errors='coerce'
        ).fillna(0)
        
        # 3. Classificar região (Curitiba vs Interior)
        cidades_curitiba = ['CURITIBA', 'Curitiba', 'curitiba']
        df['regiao'] = df['cidade'].apply(
            lambda x: 'curitiba' if str(x).strip() in cidades_curitiba else 'interior'
        )
        
        # 4. Classificar faixa de valor (baseado na análise da Ligga)
        df['faixa_valor'] = pd.cut(
            df['valor_contrato'].fillna(0),
            bins=[0, 105, 120, 135, 155, float('inf')],
            labels=['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto']
        ).astype(str)
        
        # 5. Classificar faixa de aging
        df['faixa_aging'] = pd.cut(
            df['aging_meses'],
            bins=[0, 10, 21, 38, 73, float('inf')],
            labels=['muito_novo', 'novo', 'medio', 'antigo', 'muito_antigo']
        ).astype(str)
        
        # 6. Classificar velocidade
        df['faixa_velocidade'] = pd.cut(
            df['velocidade_num'],
            bins=[0, 100, 300, 500, 700, float('inf')],
            labels=['baixa', 'media', 'alta', 'muito_alta', 'ultra']
        ).astype(str)
        
        # 7. Classificar dia de vencimento
        df['faixa_vencimento'] = pd.cut(
            df['dia_vencimento'].fillna(15),
            bins=[0, 10, 20, 31],
            labels=['inicio_mes', 'meio_mes', 'fim_mes']
        ).astype(str)
        
        # 8. Status de churn consolidado
        df['is_churner'] = 'ativo'
        if 'status_churn' in df.columns:
            df['is_churner'] = df['status_churn']
        elif 'contrato_cancelado' in df.columns:
            df.loc[df['contrato_cancelado'] == True, 'is_churner'] = 'cancelado'
            df.loc[df['data_pedido_cancelamento'].notna(), 'is_churner'] = 'solicitou_cancel'
        
        # 9. Tipo de cliente simplificado
        df['cliente_tipo'] = df['tipo_cliente'].fillna('indefinido').astype(str)
        
        logger.info("✅ Features adicionais criadas")
        return df
    
    def _log_data_summary(self, df: pd.DataFrame):
        """Log do resumo dos dados extraídos"""
        logger.info("\n" + "="*50)
        logger.info("📊 RESUMO DOS DADOS EXTRAÍDOS")
        logger.info("="*50)
        
        # Estatísticas gerais
        logger.info(f"Total de contratos: {len(df):,}")
        logger.info(f"Total de colunas: {len(df.columns)}")
        
        # Distribuição por região
        if 'regiao' in df.columns:
            regiao_dist = df['regiao'].value_counts()
            logger.info(f"\nDistribuição por região:")
            for regiao, count in regiao_dist.items():
                pct = (count / len(df)) * 100
                logger.info(f"  • {regiao}: {count:,} ({pct:.1f}%)")
        
        # Distribuição por status de churn
        if 'is_churner' in df.columns:
            churn_dist = df['is_churner'].value_counts()
            logger.info(f"\nDistribuição por status de churn:")
            for status, count in churn_dist.items():
                pct = (count / len(df)) * 100
                logger.info(f"  • {status}: {count:,} ({pct:.1f}%)")
        
        # Estatísticas de valor
        if 'valor_contrato' in df.columns:
            logger.info(f"\nEstatísticas de valor:")
            logger.info(f"  • Valor médio: R$ {df['valor_contrato'].mean():.2f}")
            logger.info(f"  • Valor mediano: R$ {df['valor_contrato'].median():.2f}")
            logger.info(f"  • Valor mín/máx: R$ {df['valor_contrato'].min():.2f} / R$ {df['valor_contrato'].max():.2f}")
        
        # Estatísticas de atendimento
        if 'total_atendimentos' in df.columns:
            atend_media = df['total_atendimentos'].mean()
            clientes_com_atend = (df['total_atendimentos'] > 0).sum()
            logger.info(f"\nEstatísticas de atendimento:")
            logger.info(f"  • Atendimentos médios: {atend_media:.1f}")
            logger.info(f"  • Clientes com atendimento: {clientes_com_atend:,} ({(clientes_com_atend/len(df))*100:.1f}%)")
        
        logger.info("="*50 + "\n")

def load_data(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Função principal para carregar dados da Ligga
    Esta função é chamada pelo training.py
    """
    extractor = LiggaDataExtractor()
    return extractor.extract_aggregated_data(limit=limit)

def validate_data_structure() -> Dict[str, Any]:
    """
    Função para validar a estrutura dos dados
    """
    extractor = LiggaDataExtractor()
    return extractor.validate_setup()

if __name__ == "__main__":
    # Teste do extractor
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testando extractor da Ligga...")
    
    # Validar estrutura
    print("\n1. Validando estrutura...")
    validation = validate_data_structure()
    
    # Extrair amostra
    print("\n2. Extraindo amostra de dados...")
    df = load_data(limit=100)
    
    if not df.empty:
        print(f"\n✅ Extração concluída!")
        print(f"📊 Amostra: {len(df)} registros, {len(df.columns)} colunas")
        print(f"\nColunas disponíveis:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
    else:
        print("\n❌ Nenhum dado extraído")
