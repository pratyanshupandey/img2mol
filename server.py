from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from cddd.inference import InferenceModel
import numpy as np

class EmbeddingsClass(BaseModel):
    embeddings: List[List[float]]


app = FastAPI()
inf_model = InferenceModel(model_dir="../cddd/default_model/", use_gpu=True, cpu_threads=15)


@app.post("/embeddings_to_smiles/")
async def create_item(embeddings: EmbeddingsClass):
    print("Converting")
    emb = np.array(embeddings.embeddings)
    smiles = inf_model.emb_to_seq(emb)
    return {"smiles": smiles}

