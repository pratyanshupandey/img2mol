"""
Data format is assumed to be arranged in the following manner

data/
    |
    |- images/
    |
    |- smiles.txt

"""

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import torch

class ImageData(Dataset):
    def __init__(self, data_dir):
        
        self.data_dir = data_dir
        if self.data_dir[-1] != "/":
            self.data_dir += '/'

        # Get all molecules in SMILES representation from the text file
        file = open(self.data_dir + "smiles.txt", "r")
        data = [line.strip() for line in file.readlines()]
        self.smiles_list = np.array([line.split(",")[0] for line in data])
        self.images_list = np.array([line.split(",")[1] for line in data])
        file.close()

        self.length = len(self.smiles_list)

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

        image = Image.open(self.data_dir + "images/" + self.images_list[index])
        transformed_image = self.image_transform(image)
        contrast_image = TF.autocontrast(transformed_image)
        image_tensor = self.tensor_transform(contrast_image)

        return smile, image_tensor

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
    train_dataset = ImageData("benchmarks/skater")
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                             batch_size=128,
    #                                             shuffle=False, 
    #                                             num_workers=30)

    # for i, (smiles, images) in enumerate(train_loader):
    #     print(f"\r{i}/{len(train_loader)}", end="", flush=True)
    # print(error_smiles)
    # print(train_dataset[10179])
    # err = []
    # for i in range(len(train_dataset)):
    #     try:
    #         t = train_dataset[i]
    #         print(f"\r{i}", end="")
    #     except Exception as e:
    #         err.append(i)
    # print(err)
    # err = [10626, 10627, 10628, 10931, 10945, 11849, 12437, 12460, 12952, 13540, 13766, 13797, 13837, 13947, 14145, 14367, 14370, 14449, 14450, 14638, 15343, 15369, 15887, 16205, 16400, 17866, 17899, 17918, 18324, 18591, 18768, 18999, 19323, 19326, 19876, 20003, 20004, 20008, 20014, 20442, 20579, 20891, 21148, 21369, 21510, 21511, 21759, 22637, 23304, 23470, 23630, 23700, 23760, 24288, 24574, 24743, 25029, 25373, 25665, 26019, 26073, 26076, 26130, 26713, 26714, 26965, 27006, 27197, 27199, 27469, 27470, 27654, 28024, 28077, 28095, 28096, 28172, 28341, 28755, 29092, 29187, 29211, 29263, 29278]
    # file = open("benchmarks/skater/" + "smiles.txt", "r")
    # data = [line for line in file.readlines()]
    # file.close()
    
    # file = open("benchmarks/skater/" + "smiles.txt", "w+")
    # for i,d in enumerate(data):
    #     if i not in err:
    #         file.write(d)
    # file.close()

    
