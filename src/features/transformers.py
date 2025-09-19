import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import joblib
import os

logger = logging.getLogger(__name__)

class LiggaCategoricalEncoder:
    """
    Transformador categórico avançado para dados da Ligga
    Suporta diferentes tipos de encoding e mantém mapeamentos
    """
    
    def __init__(self, encoding_type: str = 'label', save_mappings: bool = True):
        """
        Inicializa o codificador
        
        Args:
            encoding_type: Tipo de encoding ('label', 'ordinal', 'codes')
            save_mappings: Se deve salvar mapeamentos para decodificação
        """
        self.encoding_type = encoding_type
        self.save_mappings = save_mappings
        self.encoders = {}
        self.mappings = {}
        self.original_dtypes = {}
        
    def fit_transform(self, df: pd.DataFrame, cat_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Ajusta e transforma colunas categóricas
        
        Args:
            df: DataFrame com dados
            cat_cols: Lista de colunas categóricas (se None, detecta automaticamente)
            
        Returns:
            DataFrame com colunas categóricas codificadas
        """
        logger.info(f"🔢 Iniciando encoding categórico ({self.encoding_type})...")
        
        df_encoded = df.copy()
        
        # Detectar colunas categóricas se não especificadas
        if cat_cols is None:
            cat_cols = self._detect_categorical_columns(df_encoded)
        
        # Verificar se colunas existem
        cat_cols = [col for col in cat_cols if col in df_encoded.columns]
        
        if not cat_cols:
            logger.warning("⚠️  Nenhuma coluna categórica encontrada para encoding")
            return df_encoded
        
        logger.info(f"   • Colunas a codificar: {cat_cols}")
        
        # Salvar tipos originais
        for col in cat_cols:
            self.original_dtypes[col] = df_encoded[col].dtype
        
        # Aplicar encoding por tipo
        if self.encoding_type == 'label':
            df_encoded = self._label_encoding(df_encoded, cat_cols)
        elif self.encoding_type == 'ordinal':
            df_encoded = self._ordinal_encoding(df_encoded, cat_cols)
        elif self.encoding_type == 'codes':
            df_encoded = self._codes_encoding(df_encoded, cat_cols)
        else:
            raise ValueError(f"Tipo de encoding '{self.encoding_type}' não suportado")
        
        # Salvar mapeamentos se necessário
        if self.save_mappings:
            self._save_mappings()
        
        logger.info(f"✅ Encoding concluído para {len(cat_cols)} colunas")
        return df_encoded
    
    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detecta automaticamente colunas categóricas"""
        cat_cols = []
        
        for col in df.columns:
            # Colunas explicitamente categóricas
            if df[col].dtype.name == 'category':
                cat_cols.append(col)
            # Colunas object com poucos valores únicos
            elif df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 or df[col].nunique() < 20:  # Menos que 10% ou menos que 20 valores únicos
                    cat_cols.append(col)
            # Colunas bool
            elif df[col].dtype == 'bool':
                cat_cols.append(col)
        
        return cat_cols
    
    def _label_encoding(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Aplica Label Encoding"""
        logger.info("   🏷️  Aplicando Label Encoding...")
        
        for col in cat_cols:
            # Tratar valores nulos
            df[col] = df[col].fillna('UNKNOWN')
            
            # Criar e ajustar encoder
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            
            # Salvar encoder e mapeamento
            self.encoders[col] = encoder
            
            if self.save_mappings:
                unique_values = encoder.classes_
                encoded_values = range(len(unique_values))
                self.mappings[col] = dict(zip(unique_values, encoded_values))
                
                logger.info(f"     • {col}: {len(unique_values)} categorias → 0-{len(unique_values)-1}")
        
        return df
    
    def _ordinal_encoding(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Aplica Ordinal Encoding com ordenação inteligente"""
        logger.info("   📊 Aplicando Ordinal Encoding...")
        
        # Definir ordenações específicas para Ligga
        ordinal_orders = self._get_ligga_ordinal_orders()
        
        for col in cat_cols:
            # Tratar valores nulos
            df[col] = df[col].fillna('UNKNOWN')
            
            # Verificar se temos ordem específica para esta coluna
            if col in ordinal_orders:
                categories = ordinal_orders[col]
                
                # Adicionar categorias não previstas
                unique_vals = df[col].unique()
                missing_cats = [val for val in unique_vals if val not in categories]
                if missing_cats:
                    categories = categories + missing_cats
                
                logger.info(f"     • {col}: ordem customizada ({len(categories)} categorias)")
            else:
                # Ordem alfabética por padrão
                categories = sorted(df[col].unique())
                logger.info(f"     • {col}: ordem alfabética ({len(categories)} categorias)")
            
            # Aplicar encoding ordinal
            encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
            df[[col]] = encoder.fit_transform(df[[col]])
            
            # Salvar encoder e mapeamento
            self.encoders[col] = encoder
            
            if self.save_mappings:
                self.mappings[col] = dict(zip(categories, range(len(categories))))
        
        return df
    
    def _codes_encoding(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Aplica encoding usando códigos de categoria do pandas"""
        logger.info("   🏗️  Aplicando Codes Encoding...")
        
        for col in cat_cols:
            # Converter para categórico se não for
            if df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')
            
            # Obter códigos
            codes = df[col].cat.codes
            categories = df[col].cat.categories
            
            # Aplicar códigos
            df[col] = codes
            
            # Salvar mapeamento
            if self.save_mappings:
                self.mappings[col] = dict(zip(categories, range(len(categories))))
                
                logger.info(f"     • {col}: {len(categories)} categorias → códigos 0-{len(categories)-1}")
        
        return df
    
    def _get_ligga_ordinal_orders(self) -> Dict[str, List[str]]:
        """Define ordenações específicas para colunas da Ligga"""
        return {
            'faixa_valor': ['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto'],
            'faixa_aging': ['muito_novo', 'novo', 'medio', 'antigo', 'muito_antigo'],
            'faixa_velocidade': ['baixa', 'media', 'alta', 'muito_alta', 'ultra'],
            'faixa_vencimento': ['inicio_mes', 'meio_mes', 'fim_mes'],
            'is_churner': ['ativo', 'solicitou_cancel', 'cancelado'],
            'perfil_atendimento': ['sem_atendimento', 'com_atendimento', 'problematico'],
            'regiao': ['interior', 'curitiba'],  # Interior primeiro (maior volume)
            'cliente_fidelizado': ['Não', 'Sim'],
            'tipo_meio_cobranca': ['Boleto', 'Débito Automático'],
            # SVAs - Não para Sim
            'mesh_ponto_1': ['Não', 'Sim'],
            'mesh_ponto_2': ['Não', 'Sim'],
            'mesh_ponto_3': ['Não', 'Sim'],
            'clube_ligga': ['Não', 'Sim'],
            'hbo_max': ['Não', 'Sim'],
            'paramount_plus': ['Não', 'Sim'],
            'seguro_residencial': ['Não', 'Sim']
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma novos dados usando encoders já ajustados
        """
        if not self.encoders:
            raise ValueError("Encoders não foram ajustados. Execute fit_transform primeiro.")
        
        df_encoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col in df_encoded.columns:
                if self.encoding_type in ['label']:
                    # Tratar valores não vistos
                    df_encoded[col] = df_encoded[col].fillna('UNKNOWN')
                    
                    # Para Label Encoder, tratar valores não vistos
                    known_classes = set(encoder.classes_)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in known_classes else 'UNKNOWN'
                    )
                    
                    df_encoded[col] = encoder.transform(df_encoded[col])
                    
                elif self.encoding_type == 'ordinal':
                    df_encoded[[col]] = encoder.transform(df_encoded[[col]])
                    
                elif self.encoding_type == 'codes':
                    # Para codes, converter para categórico com as mesmas categorias
                    original_categories = list(self.mappings[col].keys())
                    df_encoded[col] = pd.Categorical(df_encoded[col], categories=original_categories)
                    df_encoded[col] = df_encoded[col].cat.codes
        
        return df_encoded
    
    def inverse_transform(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Decodifica dados transformados de volta para valores originais
        """
        if not self.mappings:
            raise ValueError("Mapeamentos não disponíveis. Execute com save_mappings=True")
        
        df_decoded = df.copy()
        
        if cols is None:
            cols = list(self.mappings.keys())
        
        for col in cols:
            if col in df_decoded.columns and col in self.mappings:
                # Criar mapeamento reverso
                reverse_mapping = {v: k for k, v in self.mappings[col].items()}
                
                # Aplicar decodificação
                df_decoded[col] = df_decoded[col].map(reverse_mapping)
                
                # Restaurar tipo original se disponível
                if col in self.original_dtypes:
                    try:
                        df_decoded[col] = df_decoded[col].astype(self.original_dtypes[col])
                    except:
                        pass  # Se não conseguir converter, manter como está
        
        return df_decoded
    
    def _save_mappings(self):
        """Salva mapeamentos em arquivo"""
        mappings_dir = "models/encoders"
        os.makedirs(mappings_dir, exist_ok=True)
        
        # Salvar encoders
        for col, encoder in self.encoders.items():
            encoder_path = os.path.join(mappings_dir, f"{col}_encoder.pkl")
            joblib.dump(encoder, encoder_path)
        
        # Salvar mapeamentos
        mappings_path = os.path.join(mappings_dir, "categorical_mappings.pkl")
        joblib.dump({
            'mappings': self.mappings,
            'original_dtypes': self.original_dtypes,
            'encoding_type': self.encoding_type
        }, mappings_path)
        
        logger.info(f"   💾 Mapeamentos salvos em {mappings_dir}/")
    
    def load_mappings(self):
        """Carrega mapeamentos de arquivo"""
        mappings_dir = "models/encoders"
        mappings_path = os.path.join(mappings_dir, "categorical_mappings.pkl")
        
        if os.path.exists(mappings_path):
            data = joblib.load(mappings_path)
            self.mappings = data['mappings']
            self.original_dtypes = data['original_dtypes']
            self.encoding_type = data['encoding_type']
            
            # Carregar encoders individuais
            for col in self.mappings.keys():
                encoder_path = os.path.join(mappings_dir, f"{col}_encoder.pkl")
                if os.path.exists(encoder_path):
                    self.encoders[col] = joblib.load(encoder_path)
            
            logger.info(f"   📂 Mapeamentos carregados de {mappings_dir}/")
        else:
            logger.warning(f"   ⚠️  Arquivo de mapeamentos não encontrado: {mappings_path}")

def encode_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Função de compatibilidade com código original
    Converte colunas categóricas para códigos numéricos
    """
    logger.info("🔢 Aplicando transformação categórica (compatibilidade)...")
    
    df_encoded = df.copy()
    
    for col in cat_cols:
        if col in df_encoded.columns:
            # Converter para categórico e obter códigos
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
            
            unique_count = df_encoded[col].nunique()
            logger.info(f"   • {col}: {unique_count} categorias → códigos 0-{unique_count-1}")
    
    logger.info(f"✅ Transformação concluída para {len(cat_cols)} colunas")
    return df_encoded

def encode_categoricals_advanced(df: pd.DataFrame, 
                                cat_cols: Optional[List[str]] = None,
                                encoding_type: str = 'label',
                                save_mappings: bool = True) -> Tuple[pd.DataFrame, LiggaCategoricalEncoder]:
    """
    Versão avançada da transformação categórica para Ligga
    
    Args:
        df: DataFrame com dados
        cat_cols: Colunas categóricas (detecta automaticamente se None)
        encoding_type: Tipo de encoding ('label', 'ordinal', 'codes')
        save_mappings: Se deve salvar mapeamentos
        
    Returns:
        Tuple com DataFrame transformado e encoder para transformações futuras
    """
    encoder = LiggaCategoricalEncoder(encoding_type=encoding_type, save_mappings=save_mappings)
    df_encoded = encoder.fit_transform(df, cat_cols)
    
    return df_encoded, encoder

def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera resumo das colunas categóricas
    """
    logger.info("📋 Gerando resumo de colunas categóricas...")
    
    categorical_info = []
    
    for col in df.columns:
        if df[col].dtype.name in ['category', 'object'] or df[col].dtype == 'bool':
            info = {
                'coluna': col,
                'tipo': str(df[col].dtype),
                'valores_unicos': df[col].nunique(),
                'valores_nulos': df[col].isnull().sum(),
                'pct_nulos': (df[col].isnull().sum() / len(df)) * 100,
                'categorias_principais': list(df[col].value_counts().head(3).index),
                'categoria_mais_comum': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
            }
            categorical_info.append(info)
    
    summary_df = pd.DataFrame(categorical_info)
    
    logger.info(f"   • Encontradas {len(summary_df)} colunas categóricas")
    
    return summary_df

if __name__ == "__main__":
    # Teste dos transformers
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testando transformers categóricos da Ligga...")
    
    # Criar dados de exemplo
    data_exemplo = {
        'regiao': ['curitiba', 'interior', 'curitiba', 'interior'],
        'faixa_valor': ['baixo', 'alto', 'medio', 'muito_alto'],
        'cliente_fidelizado': ['Sim', 'Não', 'Sim', 'Não'],
        'is_churner': ['ativo', 'ativo', 'cancelado', 'solicitou_cancel'],
        'valor_numerico': [100, 200, 150, 300]
    }
    
    df_teste = pd.DataFrame(data_exemplo)
    
    print("\n1. Dados originais:")
    print(df_teste)
    
    # Teste encoding simples (compatibilidade)
    print("\n2. Testando encoding simples...")
    cat_cols = ['regiao', 'faixa_valor', 'cliente_fidelizado', 'is_churner']
    df_encoded_simple = encode_categoricals(df_teste.copy(), cat_cols)
    print(df_encoded_simple)
    
    # Teste encoding avançado
    print("\n3. Testando encoding avançado (ordinal)...")
    df_encoded_adv, encoder = encode_categoricals_advanced(
        df_teste.copy(), 
        cat_cols=cat_cols,
        encoding_type='ordinal'
    )
    print(df_encoded_adv)
    
    # Teste decodificação
    print("\n4. Testando decodificação...")
    df_decoded = encoder.inverse_transform(df_encoded_adv)
    print(df_decoded)
    
    # Resumo categórico
    print("\n5. Resumo categórico:")
    summary = get_categorical_summary(df_teste)
    print(summary.to_string(index=False))
    
    print("\n✅ Testes concluídos!")
