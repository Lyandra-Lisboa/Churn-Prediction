import pandas as pd

def validate_age(df: pd.DataFrame, min_age: int = 15, max_age: int = 100) -> pd.DataFrame:
    """Filtra clientes fora do intervalo de idade"""
    return df[(df['idade'] > min_age) & (df['idade'] < max_age)]

def validate_na(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove linhas com valores NA em colunas específicas"""
    return df.dropna(subset=columns)
