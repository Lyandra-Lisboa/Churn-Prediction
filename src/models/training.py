"""
Treinamento do modelo K-Modes para predição de churn - Ligga
"""
import os
import joblib
import logging
import argparse
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from kmodes.kmodes import KModes
from config_ligga import PostgreSQLConfig, TableConfig

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "kmodes_churn_ligga.pkl")
DATA_PATH = os.path.join(MODEL_DIR, "processed_data_ligga.csv")

def load_ligga_data():
    """
    Carrega e processa dados das tabelas da Ligga
    """
    # Configurações
    db_config = PostgreSQLConfig()
    table_config = TableConfig()
    
    logging.info("Conectando ao banco de dados...")
    engine = create_engine(db_config.get_connection_string())
    
    try:
        # 1. DADOS DE CHURN (tabela principal)
        logging.info("Carregando dados de churn...")
        churn_query = f"""
        SELECT 
            contrato,
            cliente,
            cpf_cnpj,
            status,
            tipo_contrato,
            valor_contrato,
            cidade,
            bairro,
            produto_final,
            velocidade,
            tipo_produto,
            canal_produto,
            valor_contratacao,
            dia_vencimento,
            motivo_cancelamento,
            tipo_churn,
            tipo_cliente,
            data_criacao_contrato,
            data_instalacao,
            data_pedido_cancelamento,
            data_fim
        FROM {table_config.churn_table}
        WHERE contrato IS NOT NULL
        """
        
        df_churn = pd.read_sql(churn_query, engine)
        logging.info(f"Carregados {len(df_churn)} registros de churn")
        
        # 2. DADOS DE ATENDIMENTO (agregados por contrato)
        logging.info("Carregando dados de atendimento...")
        atendimento_query = f"""
        SELECT 
            contrato,
            COUNT(*) as total_atendimentos,
            COUNT(CASE WHEN tipo LIKE '%reclamacao%' OR tipo LIKE '%problema%' THEN 1 END) as atendimentos_negativos,
            COUNT(CASE WHEN status = 'Fechado' THEN 1 END) as atendimentos_resolvidos,
            MAX(CASE WHEN abertura IS NOT NULL THEN 1 ELSE 0 END) as teve_atendimento
        FROM {table_config.atendimento_table}
        WHERE contrato IS NOT NULL
        GROUP BY contrato
        """
        
        df_atendimento = pd.read_sql(atendimento_query, engine)
        logging.info(f"Carregados dados de atendimento para {len(df_atendimento)} contratos")
        
        # 3. DADOS DE CANCELAMENTO (para identificar churners)
        logging.info("Carregando dados de cancelamento...")
        cancelamento_query = f"""
        SELECT 
            contrato,
            motivo_cancelamento as motivo_cancel_detalhado,
            canal_solicitacao,
            status_solicitacao,
            CASE WHEN data_cancelamento_contrato IS NOT NULL THEN 1 ELSE 0 END as contrato_cancelado
        FROM {table_config.cancelados_table}
        WHERE contrato IS NOT NULL
        """
        
        df_cancelamento = pd.read_sql(cancelamento_query, engine)
        logging.info(f"Carregados dados de cancelamento para {len(df_cancelamento)} contratos")
        
        engine.dispose()
        
    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        raise e
    
    # MERGE DOS DADOS
    logging.info("Combinando dados das tabelas...")
    
    # Base: dados de churn
    df = df_churn.copy()
    
    # Adicionar dados de atendimento
    df = df.merge(df_atendimento, on='contrato', how='left')
    
    # Adicionar dados de cancelamento
    df = df.merge(df_cancelamento, on='contrato', how='left')
    
    # Preencher valores nulos
    df['total_atendimentos'] = df['total_atendimentos'].fillna(0)
    df['atendimentos_negativos'] = df['atendimentos_negativos'].fillna(0)
    df['atendimentos_resolvidos'] = df['atendimentos_resolvidos'].fillna(0)
    df['teve_atendimento'] = df['teve_atendimento'].fillna(0)
    df['contrato_cancelado'] = df['contrato_cancelado'].fillna(0)
    
    logging.info(f"Dataset final com {len(df)} registros e {df.shape[1]} colunas")
    
    return df

def preprocess_for_kmodes(df):
    """
    Pré-processa os dados para o algoritmo K-Modes
    """
    logging.info("Pré-processando dados para K-Modes...")
    
    # Criar features categóricas baseadas na análise da Ligga
    df_processed = df.copy()
    
    # 1. CATEGORIZAR VALOR DO CONTRATO (baseado nos percentis da Ligga)
    df_processed['faixa_valor'] = pd.cut(
        df_processed['valor_contrato'].fillna(0),
        bins=[0, 105, 120, 135, 155, float('inf')],
        labels=['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto']
    )
    
    # 2. CATEGORIZAR AGING (tempo na base)
    df_processed['data_criacao_contrato'] = pd.to_datetime(df_processed['data_criacao_contrato'], errors='coerce')
    hoje = pd.Timestamp.now()
    df_processed['aging_meses'] = (hoje - df_processed['data_criacao_contrato']).dt.days / 30
    
    df_processed['faixa_aging'] = pd.cut(
        df_processed['aging_meses'].fillna(0),
        bins=[0, 10, 21, 38, 73, float('inf')],
        labels=['muito_novo', 'novo', 'medio', 'antigo', 'muito_antigo']
    )
    
    # 3. REGIÃO (Curitiba vs Interior)
    cidades_curitiba = ['CURITIBA', 'Curitiba', 'curitiba']
    df_processed['regiao'] = df_processed['cidade'].apply(
        lambda x: 'curitiba' if x in cidades_curitiba else 'interior'
    )
    
    # 4. VELOCIDADE CATEGORIZADA
    df_processed['velocidade_num'] = pd.to_numeric(
        df_processed['velocidade'].str.extract('(\d+)')[0], 
        errors='coerce'
    )
    
    df_processed['faixa_velocidade'] = pd.cut(
        df_processed['velocidade_num'].fillna(0),
        bins=[0, 100, 300, 500, 700, float('inf')],
        labels=['baixa', 'media', 'alta', 'muito_alta', 'ultra']
    )
    
    # 5. PERFIL DE ATENDIMENTO
    df_processed['perfil_atendimento'] = 'sem_atendimento'
    df_processed.loc[df_processed['total_atendimentos'] > 0, 'perfil_atendimento'] = 'com_atendimento'
    df_processed.loc[df_processed['atendimentos_negativos'] > 2, 'perfil_atendimento'] = 'problematico'
    
    # 6. STATUS DE CHURN
    df_processed['is_churner'] = 'ativo'
    df_processed.loc[df_processed['contrato_cancelado'] == 1, 'is_churner'] = 'cancelado'
    df_processed.loc[df_processed['data_pedido_cancelamento'].notna(), 'is_churner'] = 'solicitou_cancel'
    
    # 7. TIPO DE CLIENTE SIMPLIFICADO
    df_processed['cliente_tipo'] = df_processed['tipo_cliente'].fillna('indefinido')
    
    # 8. DIA VENCIMENTO CATEGORIZADO
    df_processed['faixa_vencimento'] = pd.cut(
        df_processed['dia_vencimento'].fillna(15),
        bins=[0, 10, 20, 31],
        labels=['inicio_mes', 'meio_mes', 'fim_mes']
    )
    
    # SELECIONAR FEATURES CATEGÓRICAS PARA K-MODES
    features_kmodes = [
        'faixa_valor',
        'faixa_aging', 
        'regiao',
        'faixa_velocidade',
        'perfil_atendimento',
        'is_churner',
        'cliente_tipo',
        'faixa_vencimento',
        'tipo_produto',
        'canal_produto'
    ]
    
    # Preparar dataset final
    X = df_processed[features_kmodes].copy()
    
    # Tratar valores nulos
    for col in X.columns:
        X[col] = X[col].fillna('desconhecido').astype(str)
    
    logging.info(f"Features selecionadas: {features_kmodes}")
    logging.info(f"Shape do dataset para K-Modes: {X.shape}")
    
    return X, df_processed

def train_kmodes(n_clusters=10, init="Huang", max_iter=100, random_state=42):
    """
    Treina o modelo K-Modes para detectar perfis de clientes da Ligga.
    """
    try:
        # Carregar dados
        df_raw = load_ligga_data()
        
        # Pré-processar
        X, df_processed = preprocess_for_kmodes(df_raw)
        
        logging.info(f"Treinando modelo K-Modes com {n_clusters} clusters...")
        
        # Treinar modelo
        model = KModes(
            n_clusters=n_clusters,
            init=init,
            n_init=5,
            max_iter=max_iter,
            verbose=1,
            random_state=random_state
        )
        
        clusters = model.fit_predict(X)
        
        # Adicionar clusters ao dataset
        df_processed['cluster'] = clusters
        X['cluster'] = clusters
        
        # Salvar modelo e dados
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        df_processed.to_csv(DATA_PATH, index=False)
        
        logging.info(f"Modelo salvo em {MODEL_PATH}")
        logging.info(f"Dados processados salvos em {DATA_PATH}")
        
        # Análise dos clusters
        logging.info("=== ANÁLISE DOS CLUSTERS ===")
        logging.info(f"Distribuição dos clusters:\n{pd.Series(clusters).value_counts().sort_index()}")
        
        # Análise por cluster
        for i in range(n_clusters):
            cluster_data = df_processed[df_processed['cluster'] == i]
            logging.info(f"\n--- CLUSTER {i} ({len(cluster_data)} clientes) ---")
            
            # Características principais
            logging.info(f"Valor médio: R$ {cluster_data['valor_contrato'].mean():.2f}")
            logging.info(f"Região predominante: {cluster_data['regiao'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'}")
            logging.info(f"% Churners: {(cluster_data['is_churner'] != 'ativo').mean()*100:.1f}%")
            logging.info(f"Atendimentos médios: {cluster_data['total_atendimentos'].mean():.1f}")
        
        return df_processed, model, X
        
    except Exception as e:
        logging.error(f"Erro no treinamento: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo K-Modes para churn - Ligga")
    parser.add_argument("--clusters", type=int, default=10, help="Número de clusters (padrão: 10 como nas personas da Ligga)")
    parser.add_argument("--init", type=str, default="Huang", choices=["Huang", "Cao"], help="Método de inicialização")
    parser.add_argument("--max_iter", type=int, default=100, help="Máximo de iterações")
    
    args = parser.parse_args()
    
    logging.info("=== INICIANDO TREINAMENTO MODELO LIGGA ===")
    df_result, model, X = train_kmodes(
        n_clusters=args.clusters, 
        init=args.init, 
        max_iter=args.max_iter
    )
    logging.info("=== TREINAMENTO CONCLUÍDO ===")
