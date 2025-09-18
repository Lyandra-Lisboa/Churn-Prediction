import pandas as pd
import pickle

def load_model(path: str = "models/trained/kproto_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_clusters(model, df: pd.DataFrame, cat_cols: list = None) -> pd.Series:
    if cat_cols is None:
        cat_cols = df.select_dtypes(['category']).columns.tolist()
    cat_idx = [df.columns.get_loc(c) for c in cat_cols]
    
    clusters = model.predict(df, categorical=cat_idx)
    return pd.Series(clusters, index=df.index, name="cluster")
