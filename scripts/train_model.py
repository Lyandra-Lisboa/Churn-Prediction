from src.models.training import train_kproto
from src.data.extractors import load_processed_data

# Carregar dados processados
data = load_processed_data("data/processed/Base_Analise.csv")

# Treinar modelo
model = train_kproto(data, n_clusters=10)

# Salvar modelo
import pickle
with open("models/trained/kproto_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo treinado e salvo em models/trained/kproto_model.pkl")
