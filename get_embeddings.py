"""
Get the CDDD embeddings of the train set.

Run only with CDDD environment activated.
"""
from cddd.inference import InferenceModel
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Get configurations to convert data')
parser.add_argument('--model_dir', default="../cddd/default_model/", type=str)
parser.add_argument('--use_gpu', dest='gpu', action='store_true')
parser.set_defaults(gpu= False)
parser.add_argument('--device', default="0", type=str)
parser.add_argument('--cpu_threads', default=5, type=int)
parser.add_argument('--output_file', default="emb", type=str)
parser.add_argument('--input_file', default="emb", type=str)
CONFIG = None

def parse_smiles():
    file = open(CONFIG.input_file, "r")
    smiles_list = np.array([line.strip() for line in file.readlines()])
    file.close()
    return smiles_list

def create_embeddings(smiles_list):
    inf_model = InferenceModel(model_dir=CONFIG.model_dir, use_gpu=CONFIG.gpu, cpu_threads=CONFIG.cpu_threads)
    embeddings = inf_model.seq_to_emb(smiles_list)
    return embeddings

def save_embeddings(embeddings):
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    np.save(CONFIG.output_file, embeddings)


if __name__ == '__main__':
    CONFIG = parser.parse_args()
    print(f"GPU: {CONFIG.gpu}")
    save_embeddings(create_embeddings(parse_smiles()))
    # python get_embeddings.py --use_gpu --cpu_threads=20 --input_file="data/train/smiles.txt" --output_file="data/train/embeddings"