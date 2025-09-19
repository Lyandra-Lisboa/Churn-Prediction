import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import PostgreSQLManager
from src.data.extractors import DataExtractor
from src.data.validators import DataValidator
from config.settings import settings
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Script principal para configurar dados reais
    """
    print("🚀 CONFIGURAÇÃO DE DADOS REAIS")
    print("=" * 50)
    
    print("📋 CHECKLIST DE CONFIGURAÇÃO:")
    print("1. ✅ Configure .env com credenciais do PostgreSQL")
    print("2. ✅ Edite config/database_config.py com nomes das suas tabelas") 
    print("3. ✅ Mapeie suas colunas no TableConfig")
    print("4. ✅ Execute este script para validar")
    print()
    
    try:
        # 1. Conectar ao banco
        print("🔌 Conectando ao PostgreSQL...")
        db_manager = PostgreSQLManager(settings.database)
        print("✅ Conexão estabelecida!")
        
        # 2. Validar estrutura
        print("\n🔍 Validando estrutura das tabelas...")
        extractor = DataExtractor(db_manager)
        validation = extractor.validate_setup()
        
        # Mostrar resultados da validação
        print("\n📊 RESULTADOS DA VALIDAÇÃO:")
        print("-" * 40)
        
        for table_name, exists in validation['tables_exist'].items():
            if exists:
                row_count = validation['row_counts'][table_name]
                missing_cols = validation['column_validation'][table_name]['missing_columns']
                
                print(f"✅ {table_name}: {row_count:,} registros")
                
                if missing_cols:
                    print(f"   ⚠️  Colunas não encontradas: {missing_cols}")
                    print(f"   💡 Configure o mapeamento correto em database_config.py")
            else:
                print(f"❌ {table_name}: Tabela não encontrada")
        
        # 3. Testar extração de dados
        if not validation['missing_tables']:
            print("\n📊 Testando extração de dados...")
            
            try:
                # Extrair amostra de clientes
                customers_sample = extractor.extract_customers_data()
                print(f"✅ Clientes extraídos: {len(customers_sample):,}")
                print(f"   Colunas: {list(customers_sample.columns)}")
                
                # Validar qualidade
                validator = DataValidator()
                quality_results = validator.validate_dataset(customers_sample)
                
                print(f"\n📈 QUALIDADE DOS DADOS: {quality_results['quality_score']}%")
                
                if quality_results['critical_issues']:
                    print("\n🚨 Issues críticos:")
                    for issue in quality_results['critical_issues'][:3]:
                        print(f"   • {issue}")
                
                if quality_results['quality_score'] >= 75:
                    print("\n🎉 CONFIGURAÇÃO CONCLUÍDA!")
                    print("✅ Seus dados estão prontos para análise de churn")
                    print("\n📋 Próximos passos:")
                    print("1. Execute: python scripts/train_model.py")
                    print("2. Inicie a API: python src/api/main.py")
                    print("3. Acesse dashboard: streamlit run src/dashboard/app.py")
                else:
                    print("\n⚠️  ATENÇÃO: Qualidade dos dados precisa melhorar")
                    print("💡 Revise os issues encontrados antes de prosseguir")
                
            except Exception as e:
                print(f"\n❌ Erro na extração de dados: {e}")
                print("💡 Verifique o mapeamento de colunas em database_config.py")
        
        else:
            print("\n❌ Configure as tabelas antes de prosseguir")
            print("💡 Edite config/database_config.py com os nomes corretos")
    
    except Exception as e:
        print(f"\n❌ Erro na configuração: {e}")
        print("💡 Verifique:")
        print("   • Credenciais do banco no .env")
        print("   • Conectividade com PostgreSQL")
        print("   • Permissões de acesso às tabelas")

if __name__ == "__main__":
    main()
