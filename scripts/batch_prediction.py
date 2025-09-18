import pandas as pd
from src.models.prediction import load_model, predict_clusters

model = load_model()
df_new = pd.read_csv("data/processed/Base_Teste.csv")

clusters = predict_clusters(model, df_new)
df_new["cluster"] = clusters
df_new.to_csv("data/processed/Base_Teste_Clusters.csv", index=False)
print("Predição em lote concluída.")
