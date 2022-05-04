import torch 
from torch import nn
from model import IMG2MOL
from dataset import MoleculeData
import requests
import numpy as np
import argparse
from image_dataset import ImageData

class Inference:
    def __init__(self, model, data_loader, device, data_type, host_url="http://127.0.0.1:8000"):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.host_url = host_url
        self.data_type = data_type
        
    # return loss and accuracy on the test dataset
    def print_loss_smile(self):
        self.model.eval()
        total_loss = 0
        # correct_prediction = 0
        # total_data = 1
        self.criterion = nn.MSELoss()
        data = []

        # Test data
        with torch.no_grad():
            for i, (smiles, images, embeddings) in enumerate(self.data_loader):
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)

                # ============ Forward ============
                outputs = self.model(images)
                loss = self.criterion(outputs, embeddings)
                total_loss += loss.data.item()

                data.extend(outputs.tolist())
                print(f"\r{i} / {len(self.data_loader)}", end="", flush=True)
                # =============SMILES from Outputs=======
                # data = requests.post(url=self.host_url + "/embeddings_to_smiles/", 
                #                     json={
                #                         "embeddings": outputs.tolist()
                #                     })
                # print(data.status_code)
                # predicted_smiles = np.array(data.json()["smiles"])
                # total_data += len(predicted_smiles)
                # correct_prediction += sum(smiles == predicted_smiles)

        self.model.train()

        np.save("predicted_embeddings", np.array(data))
        print("Loss = ", total_loss / len(self.data_loader))


    # return loss and accuracy on the test dataset
    def print_loss_image(self):
        self.model.eval()
        total_loss = 0
        # correct_prediction = 0
        # total_data = 1
        self.criterion = nn.MSELoss()
        data = []

        # Test data
        with torch.no_grad():
            for i, (smiles, images) in enumerate(self.data_loader):
                images = images.to(self.device)

                # ============ Forward ============
                outputs = self.model(images)

                data.extend(outputs.tolist())
                print(f"\r{i} / {len(self.data_loader)}", end="", flush=True)

        self.model.train()
        np.save("predicted_embeddings", np.array(data))


    def print_loss(self):
        if self.data_type == 'smile':
            self.print_loss_smile()
        else:
            self.print_loss_image()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    parser = argparse.ArgumentParser(description='Get configurations to run inference')
    parser.add_argument('--model', default="../checkpoints/checkpoint_final.pt", type=str)
    parser.add_argument('--test_data', default="../data/test/", type=str)
    parser.add_argument('--cpu_cores', default="../data/test/", type=int)
    parser.add_argument('--data_type', default="smile", type=str)
    CONFIG = parser.parse_args()

    test_path = CONFIG.test_data
    model_path = CONFIG.model
    cpu_cores = CONFIG.cpu_cores

    batch_size = 128

    if CONFIG.data_type == "smile":
        test_dataset = MoleculeData(test_path)
    elif CONFIG.data_type == "image":
        test_dataset = ImageData(test_path)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores)

    model = IMG2MOL()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    infer = Inference(model, data_loader=test_loader, device=device, data_type=CONFIG.data_type)
    infer.print_loss()