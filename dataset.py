"""
Data format is assumed to be arranged in the following manner

data/
    |
    |-- train
        |
        |-images/
        |-metadata.txt
        |-embeddings.npy
    |
    |-- val
        |
        |-images/
        |-metadata.txt
        |-embeddings.npy
    |
    |-- test
        |
        |-images/
        |-metadata.txt
        |-embeddings.npy


"""


import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from PIL import Image


class MoleculeData(Dataset):
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        if self.data_dir[-1] != "/":
            self.data_dir += '/'

        # Get all molecules from the text file
        file = open(self.data_dir + "metadata.txt", "r")
        data = np.array([line.strip().split(',') for line in file.readlines()])
        
        self.smiles_list = data[:,0]
        self.file_names_list = data[:,1]

        file.close()

        self.length = len(data)
        self.embeddings = torch.from_numpy(np.load(self.data_dir + "embeddings.npy"))
        
        self.tensor_transform = T.PILToTensor()

    def __getitem__(self, index):
        image_file = self.data_dir + 'images/' + self.file_names_list[index]
        image = self.tensor_transform(Image.open(image_file).convert('L')).type(torch.FloatTensor)
        embedding = self.embeddings[index]
        return image, embedding

    def __len__(self):
        return self.length
