import pandas as pd

def read_excel(path: str, sheet_name: str = 0) -> pd.DataFrame:
    """Leitura de arquivos Excel"""
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df

def read_csv(path: str, sep: str = ",") -> pd.DataFrame:
    """Leitura de arquivos CSV"""
    return pd.read_csv(path, sep=sep)
