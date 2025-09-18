import pandas as pd

def cluster_distribution(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    Calcula distribuição de clusters
    """
    return df.groupby(cluster_col).size().reset_index(name="count")
