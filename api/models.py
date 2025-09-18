from pydantic import BaseModel

class ClusterRequest(BaseModel):
    data: dict

class ClusterResponse(BaseModel):
    cluster: int
