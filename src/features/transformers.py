import pandas as pd

def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Converte colunas categóricas para códigos numéricos
    """
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return df
