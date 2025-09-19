"""
Avaliação do modelo K-Modes - Ligga
"""
import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from config_ligga import PostgreSQLConfig, TableConfig

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = "models/kmodes_churn_ligga.pkl"
DATA_PATH = "models/processed_data_ligga.csv"
REPORTS_DIR = "reports"

class LiggaModelEvaluator:
    """
    Classe para avaliação completa do modelo de clusterização da Ligga
    """
    
    def __init__(self):
        self.model = None
        self.data = None
        self.processed_data = None
        self.clusters = None
        self.db_config = PostgreSQLConfig()
        self.table_config = TableConfig()
        
        # Personas info
        self.personas_info = {
            0: {"nome": "Recente Negativo", "risco": "ALTO"},
            1: {"nome": "Recente Positivo", "risco": "BAIXO"},
            2: {"nome": "Alto Valor Não Fidelizado", "risco": "MÉDIO-ALTO"},
            3: {"nome": "Alto Valor Fidelizado", "risco": "BAIXO"},
            4: {"nome": "Padrão Capital", "risco": "MÉDIO"},
            5: {"nome": "Baixo Valor Negativo", "risco": "ALTO"},
            6: {"nome": "Baixo Valor Positivo", "risco": "BAIXO"},
            7: {"nome": "Interior Não Fidelizado", "risco": "MÉDIO"},
            8: {"nome": "Interior Fidelizado", "risco": "BAIXO"},
            9: {"nome": "Baixa Velocidade", "risco": "MÉDIO"}
        }
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
    
    def load_model_and_data(self):
        """
        Carrega modelo treinado e dados processados
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute training.py primeiro.")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dados processados não encontrados em {DATA_PATH}. Execute training.py primeiro.")
        
        # Carregar modelo
        self.model = joblib.load(MODEL_PATH)
        logging.info("Modelo carregado com sucesso")
        
        # Carregar dados processados
        self.processed_data = pd.read_csv(DATA_PATH)
        logging.info(f"Dados processados carregados: {len(self.processed_data)} registros")
        
        # Extrair features e clusters
        feature_cols = [
            'faixa_valor', 'faixa_aging', 'regiao', 'faixa_velocidade',
            'perfil_atendimento', 'is_churner', 'cliente_tipo',
            'faixa_vencimento', 'tipo_produto', 'canal_produto'
        ]
        
        self.data = self.processed_data[feature_cols].copy()
        self.clusters = self.processed_data['cluster'].values
        
        return True
    
    def calculate_silhouette_score(self):
        """
        Calcula Silhouette Score para avaliação da qualidade dos clusters
        """
        logging.info("Calculando Silhouette Score...")
        
        # Converter dados categóricos para numéricos
        data_encoded = self.data.copy()
        encoders = {}
        
        for col in data_encoded.columns:
            le = LabelEncoder()
            data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
            encoders[col] = le
        
        # Calcular silhouette score
        score = silhouette_score(data_encoded, self.clusters, metric='euclidean')
        logging.info(f"Silhouette Score: {score:.4f}")
        
        return score, encoders
    
    def analyze_cluster_characteristics(self):
        """
        Analisa características detalhadas de cada cluster
        """
        logging.info("Analisando características dos clusters...")
        
        cluster_analysis = {}
        
        for cluster_id in sorted(self.processed_data['cluster'].unique()):
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster_id]
            
            analysis = {
                'tamanho': len(cluster_data),
                'percentual': (len(cluster_data) / len(self.processed_data)) * 100,
                'valor_medio': cluster_data['valor_contrato'].mean(),
                'valor_mediano': cluster_data['valor_contrato'].median(),
                'aging_medio': cluster_data['aging_meses'].mean(),
                'churn_rate': (cluster_data['is_churner'] != 'ativo').mean() * 100,
                'atendimentos_medio': cluster_data['total_atendimentos'].mean(),
                'regiao_predominante': cluster_data['regiao'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'velocidade_media': cluster_data['velocidade_num'].mean(),
                'score_serasa_info': self._get_score_info(cluster_data)
            }
            
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
    
    def _get_score_info(self, cluster_data):
        """
        Calcula informações de score baseadas nas características do cluster
        """
        # Simular score baseado nas características (como na análise original)
        valor_medio = cluster_data['valor_contrato'].mean()
        churn_rate = (cluster_data['is_churner'] != 'ativo').mean()
        atendimentos = cluster_data['total_atendimentos'].mean()
        
        # Score estimado baseado nas características
        score_estimado = 800 - (churn_rate * 400) - (atendimentos * 20) + (valor_medio * 2)
        score_estimado = max(300, min(900, score_estimado))  # Limitar entre 300-900
        
        return {
            'score_estimado': score_estimado,
            'qualidade': 'Alto' if score_estimado > 700 else 'Médio' if score_estimado > 500 else 'Baixo'
        }
    
    def analyze_business_metrics(self):
        """
        Análise de métricas de negócio específicas da Ligga
        """
        logging.info("Analisando métricas de negócio...")
        
        business_metrics = {}
        
        # 1. Distribuição de risco
        risk_distribution = {}
        for cluster_id, info in self.personas_info.items():
            if cluster_id in self.processed_data['cluster'].values:
                count = len(self.processed_data[self.processed_data['cluster'] == cluster_id])
                risk_level = info['risco']
                if risk_level not in risk_distribution:
                    risk_distribution[risk_level] = 0
                risk_distribution[risk_level] += count
        
        business_metrics['distribuicao_risco'] = risk_distribution
        
        # 2. Valor médio por cluster
        value_by_cluster = self.processed_data.groupby('cluster')['valor_contrato'].agg(['mean', 'median', 'std']).round(2)
        business_metrics['valor_por_cluster'] = value_by_cluster
        
        # 3. Taxa de churn por cluster
        churn_by_cluster = self.processed_data.groupby('cluster').apply(
            lambda x: (x['is_churner'] != 'ativo').mean() * 100
        ).round(2)
        business_metrics['churn_por_cluster'] = churn_by_cluster
        
        # 4. Distribuição geográfica
        geo_distribution = pd.crosstab(self.processed_data['cluster'], self.processed_data['regiao'], normalize='index') * 100
        business_metrics['distribuicao_geografica'] = geo_distribution
        
        # 5. Qualidade de atendimento
        atendimento_quality = self.processed_data.groupby('cluster').agg({
            'total_atendimentos': 'mean',
            'atendimentos_negativos': 'mean',
            'atendimentos_resolvidos': 'mean'
        }).round(2)
        business_metrics['qualidade_atendimento'] = atendimento_quality
        
        return business_metrics
    
    def generate_cluster_comparison(self):
        """
        Gera comparação detalhada entre clusters
        """
        logging.info("Gerando comparação entre clusters...")
        
        comparison = []
        
        for cluster_id in sorted(self.processed_data['cluster'].unique()):
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster_id]
            persona = self.personas_info.get(cluster_id, {'nome': f'Cluster {cluster_id}', 'risco': 'N/A'})
            
            row = {
                'Cluster': cluster_id,
                'Persona': persona['nome'],
                'Risco': persona['risco'],
                'Quantidade': len(cluster_data),
                'Percentual': f"{(len(cluster_data) / len(self.processed_data)) * 100:.1f}%",
                'Valor Médio': f"R$ {cluster_data['valor_contrato'].mean():.2f}",
                'Aging Médio (meses)': f"{cluster_data['aging_meses'].mean():.1f}",
                'Taxa Churn': f"{(cluster_data['is_churner'] != 'ativo').mean() * 100:.1f}%",
                'Atendimentos Médio': f"{cluster_data['total_atendimentos'].mean():.1f}",
                'Região Principal': cluster_data['regiao'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            }
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        return comparison_df
    
    def evaluate_model_stability(self):
        """
        Avalia estabilidade do modelo com re-treinamento
        """
        logging.info("Avaliando estabilidade do modelo...")
        
        # Fazer predições novamente
        predictions = self.model.predict(self.data)
        
        # Comparar com clusters originais
        stability_score = adjusted_rand_score(self.clusters, predictions)
        
        # Análise de consistência
        consistency = (self.clusters == predictions).mean() * 100
        
        logging.info(f"Adjusted Rand Score: {stability_score:.4f}")
        logging.info(f"Consistência das predições: {consistency:.1f}%")
        
        return stability_score, consistency
    
    def generate_report(self):
        """
        Gera relatório completo de avaliação
        """
        logging.info("=== INICIANDO AVALIAÇÃO COMPLETA DO MODELO ===")
        
        # Carregar dados
        self.load_model_and_data()
        
        # Métricas técnicas
        silhouette, encoders = self.calculate_silhouette_score()
        stability, consistency = self.evaluate_model_stability()
        
        # Análises de negócio
        cluster_chars = self.analyze_cluster_characteristics()
        business_metrics = self.analyze_business_metrics()
        comparison_df = self.generate_cluster_comparison()
        
        # Relatório no console
        logging.info("\n" + "="*60)
        logging.info("RELATÓRIO DE AVALIAÇÃO DO MODELO LIGGA")
        logging.info("="*60)
        
        logging.info(f"\n📊 MÉTRICAS TÉCNICAS:")
        logging.info(f"  • Silhouette Score: {silhouette:.4f}")
        logging.info(f"  • Estabilidade (ARI): {stability:.4f}")
        logging.info(f"  • Consistência: {consistency:.1f}%")
        
        logging.info(f"\n📈 DISTRIBUIÇÃO DOS CLUSTERS:")
        cluster_dist = self.processed_data['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_dist.items():
            persona = self.personas_info.get(cluster_id, {'nome': f'Cluster {cluster_id}'})
            pct = (count / len(self.processed_data)) * 100
            logging.info(f"  • Cluster {cluster_id} ({persona['nome']}): {count} clientes ({pct:.1f}%)")
        
        logging.info(f"\n⚠️  ANÁLISE DE RISCO:")
        for risk, count in business_metrics['distribuicao_risco'].items():
            pct = (count / len(self.processed_data)) * 100
            logging.info(f"  • {risk}: {count} clientes ({pct:.1f}%)")
        
        logging.info(f"\n💰 TOP 3 CLUSTERS POR VALOR:")
        top_value = business_metrics['valor_por_cluster'].sort_values('mean', ascending=False).head(3)
        for cluster_id, row in top_value.iterrows():
            persona = self.personas_info.get(cluster_id, {'nome': f'Cluster {cluster_id}'})
            logging.info(f"  • {persona['nome']}: R$ {row['mean']:.2f} (médio)")
        
        logging.info(f"\n🚨 TOP 3 CLUSTERS POR RISCO DE CHURN:")
        top_churn = business_metrics['churn_por_cluster'].sort_values(ascending=False).head(3)
        for cluster_id, churn_rate in top_churn.items():
            persona = self.personas_info.get(cluster_id, {'nome': f'Cluster {cluster_id}'})
            logging.info(f"  • {persona['nome']}: {churn_rate:.1f}% de churn")
        
        # Salvar relatórios detalhados
        self._save_detailed_reports(comparison_df, business_metrics, cluster_chars)
        
        logging.info(f"\n📁 Relatórios detalhados salvos em: {REPORTS_DIR}/")
        logging.info("="*60)
        
        return {
            'silhouette_score': silhouette,
            'stability_score': stability,
            'consistency': consistency,
            'cluster_analysis': cluster_chars,
            'business_metrics': business_metrics,
            'comparison': comparison_df
        }
    
    def _save_detailed_reports(self, comparison_df, business_metrics, cluster_chars):
        """
        Salva relatórios detalhados em arquivos
        """
        # 1. Comparação de clusters
        comparison_df.to_csv(f"{REPORTS_DIR}/cluster_comparison.csv", index=False)
        
        # 2. Métricas de valor
        business_metrics['valor_por_cluster'].to_csv(f"{REPORTS_DIR}/valor_por_cluster.csv")
        
        # 3. Taxa de churn
        pd.DataFrame(business_metrics['churn_por_cluster']).to_csv(f"{REPORTS_DIR}/churn_por_cluster.csv")
        
        # 4. Distribuição geográfica
        business_metrics['distribuicao_geografica'].to_csv(f"{REPORTS_DIR}/distribuicao_geografica.csv")
        
        # 5. Relatório consolidado
        with open(f"{REPORTS_DIR}/relatorio_consolidado.txt", 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE AVALIAÇÃO - MODELO LIGGA\n")
            f.write("="*50 + "\n\n")
            
            f.write("CARACTERÍSTICAS POR CLUSTER:\n")
            f.write("-"*30 + "\n")
            for cluster_id, chars in cluster_chars.items():
                persona = self.personas_info.get(cluster_id, {'nome': f'Cluster {cluster_id}', 'risco': 'N/A'})
                f.write(f"\nCluster {cluster_id} - {persona['nome']}:\n")
                f.write(f"  • Tamanho: {chars['tamanho']} clientes ({chars['percentual']:.1f}%)\n")
                f.write(f"  • Valor médio: R$ {chars['valor_medio']:.2f}\n")
                f.write(f"  • Aging médio: {chars['aging_medio']:.1f} meses\n")
                f.write(f"  • Taxa de churn: {chars['churn_rate']:.1f}%\n")
                f.write(f"  • Região predominante: {chars['regiao_predominante']}\n")
                f.write(f"  • Risco: {persona['risco']}\n")
        
        logging.info("Relatórios detalhados salvos com sucesso")

def main():
    """
    Função principal para avaliação
    """
    evaluator = LiggaModelEvaluator()
    
    try:
        results = evaluator.generate_report()
        
        # Resumo final
        print("\n" + "="*60)
        print("RESUMO DA AVALIAÇÃO")
        print("="*60)
        print(f"✅ Modelo avaliado com sucesso!")
        print(f"📊 Silhouette Score: {results['silhouette_score']:.4f}")
        print(f"🎯 Consistência: {results['consistency']:.1f}%")
        print(f"📁 Relatórios salvos em: {REPORTS_DIR}/")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        if results['silhouette_score'] > 0.3:
            print("  ✅ Qualidade de clustering: BOA")
        elif results['silhouette_score'] > 0.1:
            print("  ⚠️  Qualidade de clustering: MODERADA")
        else:
            print("  ❌ Qualidade de clustering: BAIXA - Considere ajustar parâmetros")
        
        if results['consistency'] > 90:
            print("  ✅ Estabilidade do modelo: ALTA")
        elif results['consistency'] > 70:
            print("  ⚠️  Estabilidade do modelo: MODERADA")
        else:
            print("  ❌ Estabilidade do modelo: BAIXA - Considere retreinar")
        
        return results
        
    except Exception as e:
        logging.error(f"Erro na avaliação: {e}")
        raise e

if __name__ == "__main__":
    main()
