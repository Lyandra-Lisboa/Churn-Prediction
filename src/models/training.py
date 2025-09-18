import pandas as pd
from kmodes.kprototypes import KPrototypes
import pickle

def train_kproto(df: pd.DataFrame, n_clusters: int = 10, cat_cols: list = None) -> KPrototypes:
    """
    Treina modelo KPrototypes
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(['category']).columns.tolist()
    cat_idx = [df.columns.get_loc(c) for c in cat_cols]
    
    model = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2, random_state=12)
    model.fit(df, categorical=cat_idx)
    
    # Salvar modelo
    with open("models/trained/kproto_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model
