import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

def evaluate_clustering(model, df: pd.DataFrame, cat_cols: list = None) -> dict:
    """
    Avalia o modelo de cluster usando silhouette score
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(['category']).columns.tolist()
    cat_idx = [df.columns.get_loc(c) for c in cat_cols]
    
    labels = model.predict(df, categorical=cat_idx)
    
    # Silhouette Score (apenas para dados numéricos)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    score = silhouette_score(df[numeric_cols], labels)
    
    return {
        "silhouette_score": score,
        "labels": labels
    }
