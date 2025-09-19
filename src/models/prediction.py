"""
Inferência com modelo K-Modes em novos clientes - Ligga
"""
import os
import joblib
import logging
import pandas as pd
import numpy as np
import argparse
from sqlalchemy import create_engine
from config_ligga import PostgreSQLConfig, TableConfig

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/kmodes_churn_ligga.pkl"
DATA_PATH = "models/processed_data_ligga.csv"

class LiggaPredictor:
    """
    Classe para predição de clusters de clientes Ligga
    """
    
    def __init__(self):
        self.model = None
        self.personas_info = self._load_personas_info()
        self.db_config = PostgreSQLConfig()
        self.table_config = TableConfig()
        
    def _load_personas_info(self):
        """
        Informações das personas baseadas na análise da Ligga
        """
        return {
            0: {
                "nome": "Recente Negativo",
                "descricao": "Clientes novos com problemas de pagamento",
                "risco": "ALTO",
                "acao_recomendada": "Atenção especial, oferecer suporte financeiro"
            },
            1: {
                "nome": "Recente Positivo", 
                "descricao": "Clientes novos com bom comportamento",
                "risco": "BAIXO",
                "acao_recomendada": "Manter engajamento, oferecer upgrades"
            },
            2: {
                "nome": "Alto Valor Não Fidelizado",
                "descricao": "Clientes de alto valor sem fidelização",
                "risco": "MÉDIO-ALTO",
                "acao_recomendada": "URGENTE: Oferecer fidelização com benefícios"
            },
            3: {
                "nome": "Alto Valor Fidelizado",
                "descricao": "Clientes VIP fidelizados",
                "risco": "BAIXO",
                "acao_recomendada": "Manter satisfação, atendimento premium"
            },
            4: {
                "nome": "Padrão Capital",
                "descricao": "Clientes padrão de Curitiba",
                "risco": "MÉDIO",
                "acao_recomendada": "Monitorar, ofertas segmentadas"
            },
            5: {
                "nome": "Baixo Valor Negativo",
                "descricao": "Clientes de baixo valor com problemas",
                "risco": "ALTO",
                "acao_recomendada": "Avaliar viabilidade, suporte intensivo"
            },
            6: {
                "nome": "Baixo Valor Positivo",
                "descricao": "Clientes de baixo valor estáveis",
                "risco": "BAIXO",
                "acao_recomendada": "Manter relacionamento, upgrades graduais"
            },
            7: {
                "nome": "Interior Não Fidelizado",
                "descricao": "Clientes do interior sem fidelização",
                "risco": "MÉDIO",
                "acao_recomendada": "Oferecer fidelização regional"
            },
            8: {
                "nome": "Interior Fidelizado",
                "descricao": "Clientes do interior fidelizados",
                "risco": "BAIXO",
                "acao_recomendada": "Manter benefícios, expansão de serviços"
            },
            9: {
                "nome": "Baixa Velocidade",
                "descricao": "Clientes com planos básicos antigos",
                "risco": "MÉDIO",
                "acao_recomendada": "Ofertas de upgrade, modernização"
            }
        }
    
    def load_model(self):
        """
        Carrega o modelo treinado
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute o training.py primeiro.")
        
        self.model = joblib.load(MODEL_PATH)
        logging.info("Modelo carregado com sucesso")
    
    def load_customer_data(self, contratos=None, cpf_cnpj=None, limit=None):
        """
        Carrega dados de clientes específicos ou amostra do banco
        """
        engine = create_engine(self.db_config.get_connection_string())
        
        # Construir filtros
        where_clauses = ["contrato IS NOT NULL"]
        
        if contratos:
            contratos_str = "','".join(contratos)
            where_clauses.append(f"contrato IN ('{contratos_str}')")
            
        if cpf_cnpj:
            cpf_str = "','".join(cpf_cnpj) 
            where_clauses.append(f"cpf_cnpj IN ('{cpf_str}')")
        
        where_clause = " AND ".join(where_clauses)
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        # Query principal
        query = f"""
        SELECT 
            c.contrato,
            c.cliente,
            c.cpf_cnpj,
            c.status,
            c.tipo_contrato,
            c.valor_contrato,
            c.cidade,
            c.bairro,
            c.produto_final,
            c.velocidade,
            c.tipo_produto,
            c.canal_produto,
            c.valor_contratacao,
            c.dia_vencimento,
            c.motivo_cancelamento,
            c.tipo_churn,
            c.tipo_cliente,
            c.data_criacao_contrato,
            c.data_instalacao,
            c.data_pedido_cancelamento,
            c.data_fim,
            
            -- Dados de atendimento
            COALESCE(a.total_atendimentos, 0) as total_atendimentos,
            COALESCE(a.atendimentos_negativos, 0) as atendimentos_negativos,
            COALESCE(a.atendimentos_resolvidos, 0) as atendimentos_resolvidos,
            COALESCE(a.teve_atendimento, 0) as teve_atendimento,
            
            -- Dados de cancelamento
            COALESCE(cc.contrato_cancelado, 0) as contrato_cancelado,
            cc.canal_solicitacao,
            cc.motivo_cancel_detalhado
            
        FROM {self.table_config.churn_table} c
        
        LEFT JOIN (
            SELECT 
                contrato,
                COUNT(*) as total_atendimentos,
                COUNT(CASE WHEN tipo LIKE '%reclamacao%' OR tipo LIKE '%problema%' THEN 1 END) as atendimentos_negativos,
                COUNT(CASE WHEN status = 'Fechado' THEN 1 END) as atendimentos_resolvidos,
                MAX(CASE WHEN abertura IS NOT NULL THEN 1 ELSE 0 END) as teve_atendimento
            FROM {self.table_config.atendimento_table}
            WHERE contrato IS NOT NULL
            GROUP BY contrato
        ) a ON c.contrato = a.contrato
        
        LEFT JOIN (
            SELECT 
                contrato,
                motivo_cancelamento as motivo_cancel_detalhado,
                canal_solicitacao,
                CASE WHEN data_cancelamento_contrato IS NOT NULL THEN 1 ELSE 0 END as contrato_cancelado
            FROM {self.table_config.cancelados_table}
            WHERE contrato IS NOT NULL
        ) cc ON c.contrato = cc.contrato
        
        WHERE {where_clause}
        ORDER BY c.contrato
        {limit_clause}
        """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        
        logging.info(f"Carregados {len(df)} clientes para predição")
        return df
    
    def preprocess_data(self, df):
        """
        Aplica o mesmo pré-processamento usado no treinamento
        """
        df_processed = df.copy()
        
        # 1. CATEGORIZAR VALOR DO CONTRATO
        df_processed['faixa_valor'] = pd.cut(
            df_processed['valor_contrato'].fillna(0),
            bins=[0, 105, 120, 135, 155, float('inf')],
            labels=['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto']
        )
        
        # 2. CATEGORIZAR AGING
        df_processed['data_criacao_contrato'] = pd.to_datetime(df_processed['data_criacao_contrato'], errors='coerce')
        hoje = pd.Timestamp.now()
        df_processed['aging_meses'] = (hoje - df_processed['data_criacao_contrato']).dt.days / 30
        
        df_processed['faixa_aging'] = pd.cut(
            df_processed['aging_meses'].fillna(0),
            bins=[0, 10, 21, 38, 73, float('inf')],
            labels=['muito_novo', 'novo', 'medio', 'antigo', 'muito_antigo']
        )
        
        # 3. REGIÃO
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
        
        # FEATURES PARA PREDIÇÃO
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
        
        X = df_processed[features_kmodes].copy()
        
        # Tratar valores nulos
        for col in X.columns:
            X[col] = X[col].fillna('desconhecido').astype(str)
        
        return X, df_processed
    
    def predict(self, data):
        """
        Faz predição usando o modelo carregado
        """
        if self.model is None:
            self.load_model()
        
        predictions = self.model.predict(data)
        logging.info(f"Predições realizadas para {len(data)} clientes")
        
        return predictions
    
    def predict_customers(self, contratos=None, cpf_cnpj=None, limit=100):
        """
        Pipeline completo: carrega dados, processa e prediz
        """
        # Carregar dados
        df_raw = self.load_customer_data(contratos=contratos, cpf_cnpj=cpf_cnpj, limit=limit)
        
        if len(df_raw) == 0:
            logging.warning("Nenhum cliente encontrado com os critérios especificados")
            return pd.DataFrame()
        
        # Pré-processar
        X, df_processed = self.preprocess_data(df_raw)
        
        # Predizer
        clusters = self.predict(X)
        
        # Adicionar resultados
        df_processed['cluster_pred'] = clusters
        df_processed['persona'] = [self.personas_info[c]['nome'] for c in clusters]
        df_processed['risco_churn'] = [self.personas_info[c]['risco'] for c in clusters]
        df_processed['acao_recomendada'] = [self.personas_info[c]['acao_recomendada'] for c in clusters]
        
        return df_processed
    
    def analyze_predictions(self, df_results):
        """
        Analisa os resultados das predições
        """
        if len(df_results) == 0:
            return
        
        logging.info("=== ANÁLISE DAS PREDIÇÕES ===")
        
        # Distribuição por cluster
        cluster_dist = df_results['cluster_pred'].value_counts().sort_index()
        logging.info(f"\nDistribuição por cluster:")
        for cluster, count in cluster_dist.items():
            persona = self.personas_info[cluster]
            pct = (count / len(df_results)) * 100
            logging.info(f"Cluster {cluster} - {persona['nome']}: {count} clientes ({pct:.1f}%)")
        
        # Distribuição por risco
        logging.info(f"\nDistribuição por risco de churn:")
        risco_dist = df_results['risco_churn'].value_counts()
        for risco, count in risco_dist.items():
            pct = (count / len(df_results)) * 100
            logging.info(f"{risco}: {count} clientes ({pct:.1f}%)")
        
        # Clientes de alto risco
        alto_risco = df_results[df_results['risco_churn'] == 'ALTO']
        if len(alto_risco) > 0:
            logging.info(f"\n⚠️  ATENÇÃO: {len(alto_risco)} clientes de ALTO RISCO identificados!")
            logging.info("Contratos em risco:")
            for _, row in alto_risco.head(10).iterrows():
                logging.info(f"  - {row['contrato']} ({row['cliente']}) - {row['persona']}")

def main():
    """
    Função principal para execução via linha de comando
    """
    parser = argparse.ArgumentParser(description="Predição de clusters para clientes Ligga")
    parser.add_argument("--contratos", nargs="+", help="Lista de contratos específicos")
    parser.add_argument("--cpf", nargs="+", help="Lista de CPF/CNPJ específicos")  
    parser.add_argument("--limit", type=int, default=100, help="Limite de clientes para análise")
    parser.add_argument("--output", type=str, help="Arquivo CSV para salvar resultados")
    
    args = parser.parse_args()
    
    # Inicializar preditor
    predictor = LiggaPredictor()
    
    # Fazer predições
    logging.info("Iniciando predições...")
    results = predictor.predict_customers(
        contratos=args.contratos,
        cpf_cnpj=args.cpf,
        limit=args.limit
    )
    
    if len(results) > 0:
        # Analisar resultados
        predictor.analyze_predictions(results)
        
        # Salvar se especificado
        if args.output:
            results.to_csv(args.output, index=False)
            logging.info(f"Resultados salvos em {args.output}")
        
        # Mostrar amostra
        logging.info("\n=== AMOSTRA DOS RESULTADOS ===")
        sample_cols = ['contrato', 'cliente', 'valor_contrato', 'cluster_pred', 'persona', 'risco_churn']
        print(results[sample_cols].head(10).to_string(index=False))
    
    return results

if __name__ == "__main__":
    # Exemplo de uso direto
    if len(os.sys.argv) == 1:
        # Modo exemplo - analisa amostra de 50 clientes
        logging.info("=== MODO EXEMPLO - Analisando amostra de clientes ===")
        predictor = LiggaPredictor()
        results = predictor.predict_customers(limit=50)
        
        if len(results) > 0:
            predictor.analyze_predictions(results)
            print("\nPrimeiros 10 resultados:")
            sample_cols = ['contrato', 'cliente', 'valor_contrato', 'persona', 'risco_churn']
            print(results[sample_cols].head(10).to_string(index=False))
    else:
        # Modo linha de comando
        main()
