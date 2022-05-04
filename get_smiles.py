"""
Get the smiles frm CDDD embeddings.
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

def get_embeddings():
    embeddings = np.load(CONFIG.input_file)
    return embeddings

def create_smiles(embeddings):
    inf_model = InferenceModel(model_dir=CONFIG.model_dir, use_gpu=CONFIG.gpu, cpu_threads=CONFIG.cpu_threads)
    smiles = inf_model.emb_to_seq(embeddings)
    return smiles

def save_smiles(smiles):
    file = open(CONFIG.output_file, "w+")
    for smile in smiles:
        file.write(smile + "\n")
    file.close()


if __name__ == '__main__':
    CONFIG = parser.parse_args()
    save_smiles(create_smiles(get_embeddings()))