from config import settings, kmodes_config, model_config
from src.data.extractors import DataExtractor
from src.features.categorical_engineering import preprocess_data
from src.features.transformers import CategoricalTransformer
from src.clustering.kmodes_engine import KModesEngine
from src.clustering.cluster_analyzer import ClusterAnalyzer

class OptimizedModelTrainer:
    """
    Trainer integrado com configurações otimizadas
    """
    
    def __init__(self):
        # ✅ Usar configurações centralizadas
        self.kmodes_config = kmodes_config
        self.model_config = model_config
        self.settings = settings
        
        # ✅ Componentes integrados
        self.extractor = DataExtractor()
        self.kmodes_engine = KModesEngine()
        self.analyzer = ClusterAnalyzer()
        
        # ✅ Diretórios corretos
        self.models_dir = Path("models/clusters")
        self.data_dir = Path("data/processed")
        self.reports_dir = Path("reports/cluster_profiles")
        
        # Criar diretórios
        for dir_path in [self.models_dir, self.data_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_setup(self) -> bool:
        """Valida usando configurações integradas"""
        logger.info("🔍 Validando setup...")
        
        try:
            # ✅ Usar validação das configurações
            is_valid, errors = self.settings.validate_config()
            
            if not is_valid:
                logger.error("❌ Configuração inválida:")
                for error in errors:
                    logger.error(f"   - {error}")
                return False
            
            # ✅ Testar conexão com banco
            if not self.settings.database.test_connection():
                logger.error("❌ Falha na conexão com banco")
                return False
            
            logger.info("✅ Setup validado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}")
            return False
    
    def load_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Carrega dados usando extractor integrado"""
        logger.info("📊 Carregando dados...")
        
        try:
            # ✅ Usar limite das configurações se não especificado
            if limit is None:
                limit = self.settings.batch_size
            
            # ✅ Extrair usando método integrado
            self.raw_data = self.extractor.extract_aggregated_data(limit=limit)
            
            if self.raw_data.empty:
                raise ValueError("Nenhum dado extraído")
            
            logger.info(f"✅ Dados carregados: {len(self.raw_data):,} registros")
            
            # Salvar usando estrutura correta
            self.data_dir.mkdir(exist_ok=True)
            self.raw_data.to_csv(self.data_dir / "raw_data.csv", index=False)
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def train_model(self) -> dict:
        """Treina usando configurações integradas"""
        logger.info("🤖 Treinando modelo...")
        
        try:
            # ✅ Usar parâmetros das configurações
            kmodes_params = self.kmodes_config.get_kmodes_params()
            
            # ✅ Usar engine integrada
            self.model = self.kmodes_engine.fit(
                self.features_for_model,
                **kmodes_params
            )
            
            # ✅ Analisar usando analyzer integrado
            clusters = self.model.predict(self.features_for_model)
            analysis = self.analyzer.analyze_clusters(self.features_for_model, clusters)
            
            logger.info("✅ Modelo treinado e analisado")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            raise
    
    def save_artifacts(self) -> dict:
        """Salva usando estrutura correta"""
        logger.info("💾 Salvando artefatos...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # ✅ Usar diretórios corretos
            model_path = self.models_dir / f"kmodes_{timestamp}.pkl"
            transformer_path = self.models_dir / f"transformer_{timestamp}.pkl"
            
            # Salvar
            joblib.dump(self.model, model_path)
            joblib.dump(self.transformer, transformer_path)
            
            # ✅ Criar links simbólicos
            latest_model = self.models_dir / "latest_kmodes.pkl"
            latest_transformer = self.models_dir / "latest_transformer.pkl"
            
            for path in [latest_model, latest_transformer]:
                if path.exists():
                    path.unlink()
            
            latest_model.symlink_to(model_path.name)
            latest_transformer.symlink_to(transformer_path.name)
            
            metadata = {
                'timestamp': timestamp,
                'model_path': str(model_path),
                'transformer_path': str(transformer_path),
                'config_used': {
                    'kmodes': self.kmodes_config.__dict__,
                    'model': self.model_config.__dict__
                }
            }
            
            logger.info("✅ Artefatos salvos")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar: {e}")
            raise
