"""
Data format is assumed to be arranged in the following manner

data/
    |
    |-- train/
        |
        |-smiles.txt
        |-embeddings.npy
    |
    |-- val/
        |
        |-smiles.txt
        |-embeddings.npy
    |
    |-- test/
        |
        |-smiles.txt
        |-embeddings.npy


"""


import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
from image_generator import ImageGenerator

class MoleculeData(Dataset):
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        if self.data_dir[-1] != "/":
            self.data_dir += '/'

        # Get all molecules in SMILES representation from the text file
        file = open(self.data_dir + "smiles.txt", "r")
        self.smiles_list = np.array([line.strip() for line in file.readlines()])
        file.close()

        self.length = len(self.smiles_list)
        self.embeddings = torch.from_numpy(np.load(self.data_dir + "embeddings.npy"))

        self.image_generator = ImageGenerator(grayScale=True)
        
        self.image_transform = T.Compose([

            # To add random rotations and fill expansions with white
            T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR, expand=True, fill=255),

            # To add shear to the image and fill extra space with white
            T.RandomAffine(degrees=0, shear=5, interpolation=T.InterpolationMode.BILINEAR, fill=255),

            # To add random brightness
            T.ColorJitter(brightness=[0.85, 1.1]),

            # Resize
            T.Resize(224),
            
        ])

        self.tensor_transform = T.ToTensor()

    def __getitem__(self, index):
        smile = self.smiles_list[index]

        generated_image = self.image_transform(self.image_generator.genImage(smile))
        contrast_image = TF.autocontrast(generated_image)
        image_tensor = self.tensor_transform(contrast_image)

        embedding = self.embeddings[index]
        
        return smile, image_tensor, embedding

    def __len__(self):
        return self.length


def reformat(file):
    f = open(file, 'r')
    data = [line.strip().split(',')[0] for line in f.readlines()]
    f.close()
    f = open(file, 'w+')
    for s in data:
            f.write(s + "\n")
    f.close()


if __name__ == '__main__':
    train_dataset = MoleculeData("data/val")
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                             batch_size=128,
    #                                             shuffle=False, 
    #                                             num_workers=30)

    # for i, (smiles, images, embeddings) in enumerate(train_loader):
    #     print(f"\r{i}/{len(train_loader)}", end="", flush=True)
    # print(error_smiles)
    # print(train_dataset[10179])
    # for i in range(14*128, 16*130):
    #     try:
    #         t = train_dataset[i]
    #         print(f"\r{i}", end="")
    #     except Exception as e:
    #         print("Error", i, e)

    
