"""
batch_prediction.py - Ligga Edition
--------------------
Script para realizar predições em lote com o modelo K-Modes da Ligga.
Processa um conjunto de clientes e gera predições de cluster/persona
com insights de negócio e recomendações de ação.

Funcionalidades:
- Predições em lote a partir de arquivo CSV
- Consulta direta no banco de dados
- Contratos específicos via linha de comando
- Aplicação do pipeline completo de transformação
- Geração de insights e recomendações de negócio
- Exportação em múltiplos formatos
"""

import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import List, Optional, Union

# Importar módulos específicos da Ligga
from extractors import LiggaDataExtractor
from preprocessing import preprocess_ligga_data
from transformers import LiggaCategoricalEncoder
from prediction import LiggaPredictor

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

class LiggaBatchPredictor:
    """
    Classe para predições em lote usando modelos da Ligga
    """
    
    def __init__(self, model_path: Optional[str] = None, transformer_path: Optional[str] = None):
        """
        Inicializa o preditor em lote
        
        Args:
            model_path: Caminho para o modelo (se None, usa o mais recente)
            transformer_path: Caminho para o transformer (se None, usa o mais recente)
        """
        self.model = None
        self.transformer = None
        self.metadata = None
        
        # Componentes
        self.extractor = LiggaDataExtractor()
        self.predictor = LiggaPredictor()
        
        # Caminhos
        self.models_dir = Path("artifacts/models")
        self.output_dir = Path("artifacts/predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar modelo e transformer
        self._load_model_and_transformer(model_path, transformer_path)
        
        # Informações das personas
        self.personas_info = self.predictor.personas_info
    
    def _load_model_and_transformer(self, model_path: Optional[str], transformer_path: Optional[str]):
        """Carrega modelo e transformer"""
        logger.info("📦 Carregando modelo e transformer...")
        
        try:
            # Determinar caminhos
            if model_path is None:
                model_path = self.models_dir / "latest_kmodes_ligga.pkl"
            else:
                model_path = Path(model_path)
            
            if transformer_path is None:
                transformer_path = self.models_dir / "latest_transformer_ligga.pkl"
            else:
                transformer_path = Path(transformer_path)
            
            # Verificar se arquivos existem
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
            if not transformer_path.exists():
                raise FileNotFoundError(f"Transformer não encontrado: {transformer_path}")
            
            # Carregar modelo
            self.model = joblib.load(model_path)
            logger.info(f"✅ Modelo carregado: {model_path}")
            
            # Carregar transformer
            self.transformer = joblib.load(transformer_path)
            logger.info(f"✅ Transformer carregado: {transformer_path}")
            
            # Carregar metadata se disponível
            metadata_path = self.models_dir / "latest_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Metadata carregada: {metadata_path}")
            
            # Configurar predictor
            self.predictor.model = self.model
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo/transformer: {e}")
            raise
    
    def predict_from_csv(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Faz predições a partir de arquivo CSV
        
        Args:
            input_file: Arquivo CSV de entrada
            output_file: Arquivo CSV de saída
            
        Returns:
            DataFrame com predições
        """
        logger.info(f"📄 Processando arquivo CSV: {input_file}")
        
        try:
            # Carregar dados do CSV
            df = pd.read_csv(input_file)
            logger.info(f"✅ Dados carregados: {len(df):,} registros")
            
            # Processar e predizer
            results = self._process_and_predict(df, source="csv")
            
            # Salvar resultados
            results.to_csv(output_file, index=False)
            logger.info(f"💾 Resultados salvos em: {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar CSV: {e}")
            raise
    
    def predict_from_database(self, contratos: Optional[List[str]] = None, 
                            cpf_cnpj: Optional[List[str]] = None,
                            limit: Optional[int] = None,
                            output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Faz predições consultando dados diretamente do banco
        
        Args:
            contratos: Lista de contratos específicos
            cpf_cnpj: Lista de CPF/CNPJ específicos
            limit: Limite de registros
            output_file: Arquivo de saída (opcional)
            
        Returns:
            DataFrame com predições
        """
        logger.info("🗄️  Consultando dados do banco...")
        
        try:
            # Carregar dados do banco
            df = self.extractor.extract_aggregated_data(limit=limit)
            
            # Filtrar por contratos se especificado
            if contratos:
                df = df[df['contrato'].isin(contratos)]
                logger.info(f"🎯 Filtrado por contratos: {len(df):,} registros")
            
            # Filtrar por CPF/CNPJ se especificado
            if cpf_cnpj:
                df = df[df['cpf_cnpj'].isin(cpf_cnpj)]
                logger.info(f"🎯 Filtrado por CPF/CNPJ: {len(df):,} registros")
            
            if len(df) == 0:
                logger.warning("⚠️  Nenhum dado encontrado com os filtros especificados")
                return pd.DataFrame()
            
            # Processar e predizer
            results = self._process_and_predict(df, source="database")
            
            # Salvar se especificado
            if output_file:
                results.to_csv(output_file, index=False)
                logger.info(f"💾 Resultados salvos em: {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro ao consultar banco: {e}")
            raise
    
    def _process_and_predict(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Processa dados e faz predições
        
        Args:
            df: DataFrame com dados
            source: Fonte dos dados ("csv", "database")
            
        Returns:
            DataFrame com predições e insights
        """
        logger.info("🔄 Processando dados para predição...")
        
        try:
            # 1. Pré-processar dados
            logger.info("   🔧 Pré-processando...")
            df_processed = preprocess_ligga_data(df)
            
            # 2. Aplicar transformações categóricas
            logger.info("   🔄 Aplicando transformações...")
            df_transformed = self.transformer.transform(df_processed)
            
            # 3. Preparar features para o modelo
            if self.metadata and 'features_used' in self.metadata:
                features_used = self.metadata['features_used']
            else:
                # Features padrão
                features_used = [
                    'faixa_valor', 'faixa_aging', 'regiao', 'faixa_velocidade',
                    'perfil_atendimento', 'is_churner', 'cliente_tipo',
                    'faixa_vencimento', 'tipo_produto', 'canal_produto'
                ]
            
            # Filtrar apenas features disponíveis
            available_features = [f for f in features_used if f in df_transformed.columns]
            
            if not available_features:
                raise ValueError("Nenhuma feature necessária encontrada nos dados")
            
            X = df_transformed[available_features]
            
            # 4. Fazer predições
            logger.info("   🤖 Gerando predições...")
            clusters = self.model.predict(X)
            
            # 5. Preparar resultados
            results = df.copy()
            
            # Adicionar predições
            results['cluster_pred'] = clusters
            results['persona'] = [self.personas_info[c]['nome'] for c in clusters]
            results['risco_churn'] = [self.personas_info[c]['risco'] for c in clusters]
            results['acao_recomendada'] = [self.personas_info[c]['acao_recomendada'] for c in clusters]
            
            # Adicionar timestamp
            results['data_predicao'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results['fonte_dados'] = source
            
            # Adicionar informações derivadas
            results = self._add_business_insights(results)
            
            logger.info(f"✅ Predições geradas para {len(results):,} registros")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento: {e}")
            raise
    
    def _add_business_insights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona insights de negócio às predições
        """
        logger.info("   💡 Adicionando insights de negócio...")
        
        # Prioridade de ação baseada no risco
        priority_map = {
            'ALTO': 1,
            'MÉDIO-ALTO': 2, 
            'MÉDIO': 3,
            'BAIXO': 4
        }
        
        df['prioridade_acao'] = df['risco_churn'].map(priority_map).fillna(5)
        
        # Segmento de valor
        if 'valor_contrato' in df.columns:
            df['segmento_valor'] = pd.cut(
                df['valor_contrato'].fillna(0),
                bins=[0, 100, 130, 160, float('inf')],
                labels=['Básico', 'Padrão', 'Premium', 'VIP']
            )
        
        # Score de engajamento (baseado em múltiplos fatores)
        engagement_score = 50  # Base
        
        if 'aging_meses' in df.columns:
            # Clientes antigos = maior engajamento
            engagement_score += np.minimum(df['aging_meses'].fillna(0) * 2, 30)
        
        if 'total_atendimentos' in df.columns:
            # Muitos atendimentos = menor engajamento
            engagement_score -= np.minimum(df['total_atendimentos'].fillna(0) * 5, 40)
        
        if 'valor_contrato' in df.columns:
            # Maior valor = maior engajamento
            valor_norm = (df['valor_contrato'].fillna(0) - 100) / 100
            engagement_score += np.minimum(valor_norm * 20, 20)
        
        df['score_engajamento'] = np.clip(engagement_score, 0, 100).round(0).astype(int)
        
        # Categoria de engajamento
        df['categoria_engajamento'] = pd.cut(
            df['score_engajamento'],
            bins=[0, 30, 50, 70, 100],
            labels=['Baixo', 'Médio', 'Alto', 'Muito Alto']
        )
        
        # Potencial de upsell
        df['potencial_upsell'] = 'Baixo'
        
        # Clientes com bom histórico e valor médio têm potencial
        if all(col in df.columns for col in ['valor_contrato', 'score_engajamento']):
            upsell_mask = (
                (df['valor_contrato'] < 150) & 
                (df['score_engajamento'] > 60) &
                (df['risco_churn'] == 'BAIXO')
            )
            df.loc[upsell_mask, 'potencial_upsell'] = 'Alto'
            
            # Potencial médio para casos intermediários
            medium_upsell_mask = (
                (df['valor_contrato'] < 130) & 
                (df['score_engajamento'] > 40) &
                (df['risco_churn'].isin(['BAIXO', 'MÉDIO']))
            )
            df.loc[medium_upsell_mask, 'potencial_upsell'] = 'Médio'
        
        return df
    
    def generate_summary_report(self, results: pd.DataFrame, output_path: Optional[str] = None) -> dict:
        """
        Gera relatório resumido das predições
        """
        logger.info("📊 Gerando relatório resumido...")
        
        summary = {
            'total_clientes': len(results),
            'data_processamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'distribuicao_clusters': results['cluster_pred'].value_counts().to_dict(),
            'distribuicao_personas': results['persona'].value_counts().to_dict(),
            'distribuicao_risco': results['risco_churn'].value_counts().to_dict(),
            'estatisticas_valor': {
                'valor_medio': results['valor_contrato'].mean() if 'valor_contrato' in results.columns else None,
                'valor_mediano': results['valor_contrato'].median() if 'valor_contrato' in results.columns else None,
            } if 'valor_contrato' in results.columns else {},
            'clientes_alto_risco': len(results[results['risco_churn'] == 'ALTO']),
            'clientes_potencial_upsell': len(results[results['potencial_upsell'] == 'Alto']) if 'potencial_upsell' in results.columns else None
        }
        
        # Salvar relatório se especificado
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"📋 Relatório salvo em: {output_path}")
        
        return summary
    
    def export_to_multiple_formats(self, results: pd.DataFrame, base_filename: str):
        """
        Exporta resultados em múltiplos formatos
        """
        logger.info("📤 Exportando em múltiplos formatos...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV completo
        csv_path = self.output_dir / f"{base_filename}_{timestamp}.csv"
        results.to_csv(csv_path, index=False)
        
        # Excel com múltiplas abas
        excel_path = self.output_dir / f"{base_filename}_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Aba principal
            results.to_excel(writer, sheet_name='Predições', index=False)
            
            # Aba resumo por cluster
            cluster_summary = results.groupby(['cluster_pred', 'persona']).agg({
                'contrato': 'count',
                'valor_contrato': ['mean', 'median'] if 'valor_contrato' in results.columns else 'count'
            }).round(2)
            cluster_summary.to_excel(writer, sheet_name='Resumo_Clusters')
            
            # Aba alto risco
            high_risk = results[results['risco_churn'] == 'ALTO']
            if len(high_risk) > 0:
                high_risk.to_excel(writer, sheet_name='Alto_Risco', index=False)
        
        # JSON resumo
        summary = self.generate_summary_report(results)
        json_path = self.output_dir / f"{base_filename}_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Arquivos exportados:")
        logger.info(f"   📊 CSV: {csv_path}")
        logger.info(f"   📈 Excel: {excel_path}")
        logger.info(f"   📋 JSON: {json_path}")
        
        return {
            'csv': str(csv_path),
            'excel': str(excel_path),
            'json': str(json_path)
        }

def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description="Batch Prediction para modelo Ligga")
    
    # Subcommands para diferentes modos
    subparsers = parser.add_subparsers(dest='mode', help='Modo de operação')
    
    # Modo CSV
    csv_parser = subparsers.add_parser('csv', help='Predição a partir de arquivo CSV')
    csv_parser.add_argument('input_file', type=str, help='Arquivo CSV de entrada')
    csv_parser.add_argument('output_file', type=str, help='Arquivo CSV de saída')
    
    # Modo Database
    db_parser = subparsers.add_parser('database', help='Predição consultando banco de dados')
    db_parser.add_argument('--contratos', nargs='+', help='Lista de contratos específicos')
    db_parser.add_argument('--cpf', nargs='+', help='Lista de CPF/CNPJ específicos')
    db_parser.add_argument('--limit', type=int, help='Limite de registros')
    db_parser.add_argument('--output', type=str, help='Arquivo de saída')
    
    # Argumentos comuns
    for p in [csv_parser, db_parser]:
        p.add_argument('--model', type=str, help='Caminho para modelo específico')
        p.add_argument('--transformer', type=str, help='Caminho para transformer específico')
        p.add_argument('--export-multiple', action='store_true', 
                      help='Exportar em múltiplos formatos (CSV, Excel, JSON)')
        p.add_argument('--base-filename', type=str, default='ligga_predictions',
                      help='Nome base para arquivos de saída')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    # Criar diretório de logs
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Inicializar preditor
        predictor = LiggaBatchPredictor(
            model_path=getattr(args, 'model', None),
            transformer_path=getattr(args, 'transformer', None)
        )
        
        # Executar predição baseada no modo
        if args.mode == 'csv':
            logger.info(f"📄 Modo CSV: {args.input_file} → {args.output_file}")
            results = predictor.predict_from_csv(args.input_file, args.output_file)
            
        elif args.mode == 'database':
            logger.info("🗄️  Modo Database")
            results = predictor.predict_from_database(
                contratos=getattr(args, 'contratos', None),
                cpf_cnpj=getattr(args, 'cpf', None),
                limit=getattr(args, 'limit', None),
                output_file=getattr(args, 'output', None)
            )
        
        # Exportar em múltiplos formatos se solicitado
        if getattr(args, 'export_multiple', False):
            predictor.export_to_multiple_formats(results, args.base_filename)
        
        # Gerar resumo
        summary = predictor.generate_summary_report(results)
        
        # Mostrar estatísticas
        print("\n" + "="*60)
        print("📊 RESUMO DAS PREDIÇÕES")
        print("="*60)
        print(f"   • Total de clientes: {summary['total_clientes']:,}")
        print(f"   • Clientes alto risco: {summary['clientes_alto_risco']:,}")
        
        if summary.get('clientes_potencial_upsell'):
            print(f"   • Potencial upsell: {summary['clientes_potencial_upsell']:,}")
        
        print(f"\n📈 Distribuição por risco:")
        for risco, count in summary['distribuicao_risco'].items():
            pct = (count / summary['total_clientes']) * 100
            print(f"     • {risco}: {count:,} ({pct:.1f}%)")
        
        print("="*60)
        print("✅ Predições concluídas com sucesso!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️  Processo interrompido pelo usuário")
        return 1
    except Exception as e:
        logger.error(f"💥 Erro fatal: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
