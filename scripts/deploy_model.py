from flask import Flask, request, jsonify
from flask_cors import CORS
from config import settings, api_config, kmodes_config

class OptimizedAPI:
    """API integrada com configurações"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, origins=api_config.cors_origins)
        
        # ✅ Usar configurações da API
        self.config = api_config
        
        # ✅ Componentes integrados
        self.batch_predictor = OptimizedBatchPredictor()
        
        # ✅ Stats usando configurações
        self.stats = {
            'requests_total': 0,
            'predictions_total': 0,
            'errors_total': 0,
            'start_time': datetime.now(),
            'version': self.config.version
        }
        
        self._setup_routes()
    
    def run(self):
        """Executa usando configurações"""
        # ✅ Usar configurações do uvicorn/flask
        uvicorn_config = self.config.get_uvicorn_config()
        
        logger.info(f"🚀 API iniciada em http://{uvicorn_config['host']}:{uvicorn_config['port']}")
        
        self.app.run(
            host=uvicorn_config['host'],
            port=uvicorn_config['port'],
            debug=settings.debug
        )

def main_integrated_example():
    """Exemplo de uso com configurações integradas"""
    
    # ✅ Importações corretas
    from config import settings
    
    print("🔧 Configurações carregadas:")
    print(f"   Environment: {settings.environment}")
    print(f"   Database: {settings.database.host}")
    print(f"   K-modes range: {settings.kmodes_config['n_clusters_range']}")
    
    # ✅ Validar configurações
    is_valid, errors = settings.validate_config()
    if not is_valid:
        print("❌ Configuração inválida:")
        for error in errors:
            print(f"   - {error}")
        return
    
    # ✅ Usar trainer integrado
    trainer = OptimizedModelTrainer()
    
    if trainer.validate_setup():
        print("✅ Setup válido - pronto para treinar")
    else:
        print("❌ Problemas no setup")

if __name__ == "__main__":
    main_integrated_example()
