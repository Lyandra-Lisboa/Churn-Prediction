"""
train_model.py - Ligga Edition
----------------
Script responsável por treinar o modelo de churn utilizando o algoritmo K-Modes.
Adaptado para trabalhar com as tabelas reais da Ligga.

Inclui:
- Carregamento de dados das tabelas da Ligga
- Pré-processamento específico da Ligga
- Transformação categórica otimizada
- Treinamento com K-Modes
- Avaliação completa do modelo
- Salvamento do modelo e artefatos
- Registro de métricas detalhadas
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from kmodes.kmodes import KModes

# Importar módulos específicos da Ligga
from extractors import LiggaDataExtractor, validate_data_structure
from preprocessing import preprocess_ligga_data
from transformers import encode_categoricals_advanced, get_categorical_summary
from evaluation import LiggaModelEvaluator

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

class LiggaModelTrainer:
    """
    Classe principal para treinamento do modelo de churn da Ligga
    """
    
    def __init__(self, n_clusters: int = 10, encoding_type: str = 'ordinal'):
        """
        Inicializa o trainer
        
        Args:
            n_clusters: Número de clusters (padrão: 10 como nas personas da Ligga)
            encoding_type: Tipo de encoding categórico ('label', 'ordinal', 'codes')
        """
        self.n_clusters = n_clusters
        self.encoding_type = encoding_type
        
        # Componentes do pipeline
        self.extractor = LiggaDataExtractor()
        self.transformer = None
        self.model = None
        self.evaluator = LiggaModelEvaluator()
        
        # Dados do pipeline
        self.raw_data = None
        self.processed_data = None
        self.encoded_data = None
        self.features_for_model = None
        
        # Resultados
        self.training_results = {}
        
        # Diretórios
        self.artifacts_dir = Path("artifacts")
        self.models_dir = self.artifacts_dir / "models"
        self.data_dir = self.artifacts_dir / "data"
        self.reports_dir = self.artifacts_dir / "reports"
        
        # Criar diretórios
        for dir_path in [self.artifacts_dir, self.models_dir, self.data_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_setup(self) -> bool:
        """
        Valida se o ambiente está configurado corretamente
        """
        logger.info("🔍 Validando configuração do ambiente...")
        
        try:
            # Validar estrutura do banco
            validation = self.extractor.validate_setup()
            
            missing_tables = validation.get('missing_tables', [])
            if missing_tables:
                logger.error(f"❌ Tabelas não encontradas: {missing_tables}")
                logger.error("💡 Verifique a configuração em config_ligga.py")
                return False
            
            # Verificar se há dados suficientes
            row_counts = validation.get('row_counts', {})
            total_records = sum(row_counts.values())
            
            if total_records < 1000:
                logger.warning(f"⚠️  Poucos dados encontrados: {total_records} registros")
                logger.warning("📊 Recomendamos pelo menos 1000 registros para treinamento")
            
            logger.info("✅ Validação concluída com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}")
            return False
    
    def load_data(self, limit: int = None) -> pd.DataFrame:
        """
        Carrega dados das tabelas da Ligga
        
        Args:
            limit: Limite de registros (para testes)
            
        Returns:
            DataFrame com dados agregados
        """
        logger.info("📊 Carregando dados das tabelas da Ligga...")
        
        try:
            self.raw_data = self.extractor.extract_aggregated_data(limit=limit)
            
            if self.raw_data.empty:
                raise ValueError("Nenhum dado foi extraído das tabelas")
            
            logger.info(f"✅ Dados carregados: {len(self.raw_data):,} registros, {len(self.raw_data.columns)} colunas")
            
            # Salvar dados brutos
            raw_data_path = self.data_dir / "raw_data.csv"
            self.raw_data.to_csv(raw_data_path, index=False)
            logger.info(f"💾 Dados brutos salvos em: {raw_data_path}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Pré-processa os dados carregados
        """
        logger.info("🔧 Pré-processando dados...")
        
        if self.raw_data is None:
            raise ValueError("Dados não foram carregados. Execute load_data() primeiro.")
        
        try:
            self.processed_data = preprocess_ligga_data(self.raw_data)
            
            logger.info(f"✅ Pré-processamento concluído: {len(self.processed_data):,} registros")
            
            # Salvar dados processados
            processed_data_path = self.data_dir / "processed_data.csv"
            self.processed_data.to_csv(processed_data_path, index=False)
            logger.info(f"💾 Dados processados salvos em: {processed_data_path}")
            
            # Gerar resumo categórico
            cat_summary = get_categorical_summary(self.processed_data)
            summary_path = self.reports_dir / "categorical_summary.csv"
            cat_summary.to_csv(summary_path, index=False)
            logger.info(f"📋 Resumo categórico salvo em: {summary_path}")
            
            return self.processed_data
            
        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento: {e}")
            raise
    
    def transform_data(self) -> pd.DataFrame:
        """
        Aplica transformações categóricas
        """
        logger.info("🔄 Aplicando transformações categóricas...")
        
        if self.processed_data is None:
            raise ValueError("Dados não foram pré-processados. Execute preprocess_data() primeiro.")
        
        try:
            # Aplicar encoding categórico avançado
            self.encoded_data, self.transformer = encode_categoricals_advanced(
                self.processed_data,
                encoding_type=self.encoding_type,
                save_mappings=True
            )
            
            logger.info(f"✅ Transformação concluída: {len(self.encoded_data.columns)} features")
            
            # Preparar features para o modelo K-Modes
            # Selecionar apenas colunas categóricas/discretas
            categorical_features = [
                'faixa_valor', 'faixa_aging', 'regiao', 'faixa_velocidade',
                'perfil_atendimento', 'is_churner', 'cliente_tipo',
                'faixa_vencimento', 'tipo_produto', 'canal_produto'
            ]
            
            # Filtrar apenas colunas que existem
            available_features = [col for col in categorical_features if col in self.encoded_data.columns]
            
            if not available_features:
                raise ValueError("Nenhuma feature categórica disponível para o modelo")
            
            self.features_for_model = self.encoded_data[available_features].copy()
            
            logger.info(f"🎯 Features selecionadas para modelo: {available_features}")
            
            # Salvar dados transformados
            encoded_data_path = self.data_dir / "encoded_data.csv"
            self.encoded_data.to_csv(encoded_data_path, index=False)
            
            features_path = self.data_dir / "model_features.csv"
            self.features_for_model.to_csv(features_path, index=False)
            
            logger.info(f"💾 Dados transformados salvos em: {encoded_data_path}")
            
            return self.features_for_model
            
        except Exception as e:
            logger.error(f"❌ Erro na transformação: {e}")
            raise
    
    def train_model(self, init: str = "Huang", n_init: int = 5, max_iter: int = 100, random_state: int = 42) -> KModes:
        """
        Treina o modelo K-Modes
        
        Args:
            init: Método de inicialização
            n_init: Número de inicializações
            max_iter: Máximo de iterações
            random_state: Seed para reprodutibilidade
            
        Returns:
            Modelo K-Modes treinado
        """
        logger.info(f"🤖 Treinando modelo K-Modes com {self.n_clusters} clusters...")
        
        if self.features_for_model is None:
            raise ValueError("Features não foram preparadas. Execute transform_data() primeiro.")
        
        try:
            # Verificar se há dados suficientes
            if len(self.features_for_model) < self.n_clusters * 10:
                logger.warning(f"⚠️  Poucos dados para {self.n_clusters} clusters")
                logger.warning(f"📊 Recomendamos pelo menos {self.n_clusters * 10} registros")
            
            # Criar e treinar modelo
            self.model = KModes(
                n_clusters=self.n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                verbose=1,
                random_state=random_state
            )
            
            # Treinar
            start_time = datetime.now()
            clusters = self.model.fit_predict(self.features_for_model)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Adicionar clusters aos dados
            self.encoded_data['cluster'] = clusters
            
            # Registrar resultados
            self.training_results = {
                'n_clusters': self.n_clusters,
                'encoding_type': self.encoding_type,
                'training_time_seconds': training_time,
                'features_used': list(self.features_for_model.columns),
                'cluster_distribution': pd.Series(clusters).value_counts().to_dict(),
                'model_params': {
                    'init': init,
                    'n_init': n_init,
                    'max_iter': max_iter,
                    'random_state': random_state
                }
            }
            
            logger.info(f"✅ Modelo treinado em {training_time:.2f} segundos")
            logger.info(f"📊 Distribuição dos clusters: {self.training_results['cluster_distribution']}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            raise
    
    def evaluate_model(self) -> dict:
        """
        Avalia o modelo treinado
        """
        logger.info("📈 Avaliando modelo treinado...")
        
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train_model() primeiro.")
        
        try:
            # Configurar evaluator
            self.evaluator.model = self.model
            self.evaluator.data = self.features_for_model
            self.evaluator.processed_data = self.encoded_data
            self.evaluator.clusters = self.encoded_data['cluster'].values
            
            # Executar avaliação completa
            evaluation_results = self.evaluator.generate_report()
            
            # Adicionar aos resultados de treinamento
            self.training_results['evaluation'] = evaluation_results
            
            logger.info("✅ Avaliação concluída")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Erro na avaliação: {e}")
            raise
    
    def save_artifacts(self) -> dict:
        """
        Salva modelo e artefatos relacionados
        """
        logger.info("💾 Salvando artefatos do modelo...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Salvar modelo principal
            model_path = self.models_dir / f"kmodes_ligga_{timestamp}.pkl"
            joblib.dump(self.model, model_path)
            
            # Salvar transformer
            transformer_path = self.models_dir / f"transformer_ligga_{timestamp}.pkl"
            joblib.dump(self.transformer, transformer_path)
            
            # Salvar dados finais com clusters
            final_data_path = self.data_dir / f"final_data_with_clusters_{timestamp}.csv"
            self.encoded_data.to_csv(final_data_path, index=False)
            
            # Salvar resultados de treinamento
            results_path = self.reports_dir / f"training_results_{timestamp}.json"
            import json
            with open(results_path, 'w') as f:
                # Converter numpy types para JSON serializável
                serializable_results = self._make_json_serializable(self.training_results)
                json.dump(serializable_results, f, indent=2)
            
            # Criar metadata do modelo
            metadata = {
                'timestamp': timestamp,
                'model_path': str(model_path),
                'transformer_path': str(transformer_path),
                'data_path': str(final_data_path),
                'results_path': str(results_path),
                'n_clusters': self.n_clusters,
                'encoding_type': self.encoding_type,
                'total_records': len(self.encoded_data),
                'features_count': len(self.features_for_model.columns),
                'features_used': list(self.features_for_model.columns)
            }
            
            metadata_path = self.models_dir / f"model_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Criar link simbólico para o modelo mais recente
            latest_model_path = self.models_dir / "latest_kmodes_ligga.pkl"
            latest_transformer_path = self.models_dir / "latest_transformer_ligga.pkl"
            latest_metadata_path = self.models_dir / "latest_metadata.json"
            
            # Remover links antigos se existirem
            for path in [latest_model_path, latest_transformer_path, latest_metadata_path]:
                if path.exists():
                    path.unlink()
            
            # Criar novos links
            latest_model_path.symlink_to(model_path.name)
            latest_transformer_path.symlink_to(transformer_path.name)
            latest_metadata_path.symlink_to(metadata_path.name)
            
            logger.info(f"✅ Artefatos salvos:")
            logger.info(f"   📦 Modelo: {model_path}")
            logger.info(f"   🔄 Transformer: {transformer_path}")
            logger.info(f"   📊 Dados: {final_data_path}")
            logger.info(f"   📋 Resultados: {results_path}")
            logger.info(f"   📝 Metadata: {metadata_path}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar artefatos: {e}")
            raise
    
    def _make_json_serializable(self, obj):
        """Converte objetos para formato JSON serializável"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def run_full_pipeline(self, limit: int = None) -> dict:
        """
        Executa o pipeline completo de treinamento
        
        Args:
            limit: Limite de registros para teste
            
        Returns:
            Metadata do modelo treinado
        """
        logger.info("🚀 Iniciando pipeline completo de treinamento...")
        
        try:
            # 1. Validar setup
            if not self.validate_setup():
                raise RuntimeError("Falha na validação do ambiente")
            
            # 2. Carregar dados
            self.load_data(limit=limit)
            
            # 3. Pré-processar
            self.preprocess_data()
            
            # 4. Transformar
            self.transform_data()
            
            # 5. Treinar modelo
            self.train_model()
            
            # 6. Avaliar modelo
            self.evaluate_model()
            
            # 7. Salvar artefatos
            metadata = self.save_artifacts()
            
            logger.info("🎉 Pipeline completo executado com sucesso!")
            
            # Resumo final
            logger.info("\n" + "="*60)
            logger.info("📊 RESUMO DO TREINAMENTO")
            logger.info("="*60)
            logger.info(f"   • Registros processados: {len(self.encoded_data):,}")
            logger.info(f"   • Features utilizadas: {len(self.features_for_model.columns)}")
            logger.info(f"   • Clusters criados: {self.n_clusters}")
            logger.info(f"   • Encoding usado: {self.encoding_type}")
            logger.info(f"   • Tempo de treinamento: {self.training_results.get('training_time_seconds', 0):.2f}s")
            
            if 'evaluation' in self.training_results:
                eval_results = self.training_results['evaluation']
                logger.info(f"   • Silhouette Score: {eval_results.get('silhouette_score', 'N/A')}")
                logger.info(f"   • Consistência: {eval_results.get('consistency', 'N/A')}%")
            
            logger.info("="*60)
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {e}")
            raise

def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description="Treinar modelo K-Modes para dados da Ligga")
    parser.add_argument("--clusters", type=int, default=10, help="Número de clusters (padrão: 10)")
    parser.add_argument("--encoding", type=str, default="ordinal", choices=["label", "ordinal", "codes"], 
                       help="Tipo de encoding categórico")
    parser.add_argument("--limit", type=int, help="Limite de registros para teste")
    parser.add_argument("--init", type=str, default="Huang", choices=["Huang", "Cao"], 
                       help="Método de inicialização")
    parser.add_argument("--max-iter", type=int, default=100, help="Máximo de iterações")
    parser.add_argument("--n-init", type=int, default=5, help="Número de inicializações")
    
    args = parser.parse_args()
    
    # Criar diretório de logs
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Inicializar trainer
        trainer = LiggaModelTrainer(
            n_clusters=args.clusters,
            encoding_type=args.encoding
        )
        
        # Executar pipeline
        if args.limit:
            logger.info(f"🧪 Modo teste: limitando a {args.limit} registros")
        
        metadata = trainer.run_full_pipeline(limit=args.limit)
        
        print("\n🎉 Treinamento concluído com sucesso!")
        print(f"📦 Modelo salvo em: {metadata['model_path']}")
        print(f"📊 Metadados disponíveis em: artifacts/models/latest_metadata.json")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️  Treinamento interrompido pelo usuário")
        return 1
    except Exception as e:
        logger.error(f"💥 Erro fatal: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
