import numpy as np
import requests

file = open("data/train/" + "smiles.txt", "r")
smiles_list = np.array([line.strip() for line in file.readlines()])
file.close()

emb = np.load("embeddings.npy").tolist()

host_url="http://127.0.0.1:8000"

data = requests.post(url=host_url + "/embeddings_to_smiles/", json={"embeddings": emb})
predicted_smiles = np.array(data.json()["smiles"])

correct_prediction = sum(smiles_list == predicted_smiles)

print(f"Accuracy: {print(100 * correct_prediction / len(smiles_list))} %")