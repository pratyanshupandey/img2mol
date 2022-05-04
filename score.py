import numpy as np
import nltk
import argparse

parser = argparse.ArgumentParser(description='Get configurations')
parser.add_argument('--data_dir', default="", type=str)
parser.add_argument('--data_type', default="smile", type=str)
CONFIG = parser.parse_args()

data_dir = CONFIG.data_dir 
if data_dir[-1] != "/":
            data_dir += '/'

file = open("predicted_smiles.txt", "r")
predicted_smiles = np.array([line.strip() for line in file.readlines()])
file.close()

file = open(data_dir + "smiles.txt", "r")
if CONFIG.data_type == 'smile':
    true_smiles = np.array([line.strip() for line in file.readlines()])
elif CONFIG.data_type == 'image':
    true_smiles = np.array([line.strip().split(",")[0] for line in file.readlines()])
file.close()


exact_predictions = sum(true_smiles == predicted_smiles)
print(f"Exact match: {exact_predictions} / {len(true_smiles)} \t Accuracy: {exact_predictions / (len(true_smiles))}")

#Levenshtein edit-distance between two strings
l_dist = 0
for true_smile, pred_smile in zip(true_smiles, predicted_smiles):
    l_dist += nltk.edit_distance(true_smile, pred_smile)

print("Average Levenshtein edit-distance / string length: ", l_dist / len(true_smiles))