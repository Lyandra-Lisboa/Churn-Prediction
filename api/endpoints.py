from fastapi import APIRouter
from .models import ClusterRequest, ClusterResponse
from src.models.prediction import load_model, predict_clusters
import pandas as pd

router = APIRouter()
model = load_model()

@router.post("/predict", response_model=ClusterResponse)
def predict_cluster(request: ClusterRequest):
    df = pd.DataFrame([request.data])
    cluster = predict_clusters(model, df)[0]
    return ClusterResponse(cluster=int(cluster))
