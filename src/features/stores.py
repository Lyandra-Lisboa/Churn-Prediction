import pandas as pd

def save_features(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
