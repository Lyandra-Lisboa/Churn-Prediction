import streamlit as st
import pandas as pd
from .charts import plot_boxplot, plot_bar

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("Dashboard de Segmentação de Clientes")

# Carregar base processada
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path, sep="|", encoding="utf-16le")

data = load_data("data/processed/Clientes_por_cluster.txt")

# Seção de gráficos
st.header("Análise por Cluster")
plot_boxplot(data, "Nome_Cluster.x", "idade", "Idade por Cluster")
plot_bar(data, "Nome_Cluster.x", "cliente_fidelizado", "Clientes Fidelizados por Cluster")
