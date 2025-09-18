import pandas as pd

def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra idades e seleciona colunas para análise
    """
    df = df[(df['idade'] > 15) & (df['idade'] < 100)]
    col_analysis = ['idade', 'regiao', 'tipo_meio_cobranca', 'mesh_ponto_1', 
                    'mesh_ponto_2', 'mesh_ponto_3', 'clube_ligga', 'hbo_max', 
                    'paramount_plus', 'seguro_residencial', 'cliente_fidelizado',
                    'v_velocidade', 'score_serasa_final', 'renda_media',
                    'qtde_massivas', 'qtde_reparos', 'valor_circuito']
    df_analysis = df[col_analysis].copy()
    
    # Converter categorias
    cat_cols = ['regiao', 'tipo_meio_cobranca', 'mesh_ponto_1', 'mesh_ponto_2',
                'mesh_ponto_3', 'clube_ligga', 'hbo_max', 'paramount_plus',
                'seguro_residencial', 'cliente_fidelizado']
    for col in cat_cols:
        df_analysis[col] = df_analysis[col].astype('category')
    return df_analysis
