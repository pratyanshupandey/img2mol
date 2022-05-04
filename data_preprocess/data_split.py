import os
import json
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from sklearn.cluster import KMeans
import numpy as np
import tqdm



def getSmiles(smiles, indices):
    selected = []
    for index in indices:
        selected.append(smiles[index])
    return selected


def writeToFile(data, filename):
    with open(filename, "w") as file:
        for smile in data:
            file.write(smile + "\n")


DIR = "./data/"
combined_data = {}
for filename in sorted(os.listdir(DIR)):
    file_path = DIR + filename
    with open(file_path, "r") as file:
        data = json.load(file)
        for smile in data:
            if smile not in combined_data:
                combined_data[smile] = data[smile]

print(len(combined_data))

smiles = list(combined_data.keys())
smiles = list(filter(lambda x: len(x) <= 50, smiles))
print(len(smiles))
smiles = list(filter(lambda x: x is not None, smiles))
np.random.shuffle(smiles)
num_smiles = 120000
smiles = smiles[:num_smiles]
vects = np.zeros((len(smiles), 166), dtype=np.int8)
collected_smiles = []
for i, smile in enumerate(tqdm.tqdm(smiles)):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        continue
    key = MACCSkeys.GenMACCSKeys(mol)
    for j in range(1, len(key)):
        vects[i][j-1] = int(key[j])
    collected_smiles.append(smile)


vects = vects[~np.all(vects == 0, axis=1)]


print(len(collected_smiles))
print(vects.shape)
# train_len = 1000000
# val_test_len = 20000
# train_smiles = collected_smiles[:train_len]
# val_smiles = collected_smiles[train_len: train_len+val_test_len]
# test_smiles = collected_smiles[train_len+val_test_len: train_len+val_test_len+val_test_len]
smiles = collected_smiles

with open("keys.npy", "wb") as file:
    np.save(file, vects)

with open("smiles.npy", "wb") as file:
    np.save(file, smiles)

n_clusters = 10
print("Starting clustering")
model = KMeans(n_clusters=n_clusters).fit(vects)
print("Clustering Done")
cluster_centers = model.cluster_centers_
labels = model.labels_
avg_pairwise_dist = np.zeros(n_clusters)
for i in range(n_clusters):
    dist = np.linalg.norm(cluster_centers-cluster_centers[i], axis=1)
    avg_pairwise_dist[i] = np.mean(dist)

cluster_nums = np.argsort(-avg_pairwise_dist)
test_cluster = cluster_nums[0]

cluster_centers = np.vstack([cluster_centers[:test_cluster],
                            cluster_centers[test_cluster+1:]])
avg_pairwise_dist = np.zeros(n_clusters-1)
for i in range(cluster_centers.shape[0]):
    dist = np.linalg.norm(cluster_centers-cluster_centers[i], axis=1)
    avg_pairwise_dist[i] = np.mean(dist)

cluster_nums = np.argsort(-avg_pairwise_dist)
cluster_nums[cluster_nums >= test_cluster] += 1
val_cluster = cluster_nums[0]

print(test_cluster, val_cluster)

test_points = np.nonzero(labels == test_cluster)[0]
val_points = np.nonzero(labels == val_cluster)[0]
train_points = np.nonzero((labels != test_cluster) & (labels != val_cluster))[0]

print(len(train_points))
print(len(test_points))
print(len(val_points))


train_smiles = getSmiles(smiles, train_points)
test_smiles = getSmiles(smiles, test_points)
val_smiles = getSmiles(smiles, val_points)

writeToFile(train_smiles, "train_smiles(50).txt")
writeToFile(val_smiles, "val_smiles(50).txt")
writeToFile(test_smiles, "test_smiles(50).txt")
