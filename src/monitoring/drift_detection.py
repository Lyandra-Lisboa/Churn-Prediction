import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference: pd.Series, new: pd.Series, alpha: float = 0.05) -> bool:
    """
    Detecta drift entre duas distribuições usando teste KS
    """
    stat, p_value = ks_2samp(reference, new)
    return p_value < alpha
