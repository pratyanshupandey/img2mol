import torch 
from torch import nn
from model import IMG2MOL
from utils import Trainer
from dataset import MoleculeData
from early_stopping import EarlyStopping
import argparse

parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--cpu_cores', default=15, type=int)
parser.add_argument('--model', default="", type=str)
CONFIG = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


train_path = "data/train/"
val_path = "data/val/"
test_path = "data/test/"
checkpoint_dir = "checkpoints/"


#Hyperparameters
batch_size = 128
learning_rate = 1e-6#0.00005
epochs = 300
cpu_cores = CONFIG.cpu_cores
print(f"Using {cpu_cores} CPU cores")

# Getting data loaders
train_dataset = MoleculeData(train_path)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True, 
                                            num_workers=cpu_cores)

val_dataset = MoleculeData(val_path)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores)

# Getting the model
model = IMG2MOL()

if CONFIG.model == "":
    print("Training new model")
else:
    print("Using model from", CONFIG.model)
    model.load_state_dict(torch.load(CONFIG.model))

model = model.to(device)


# Optimizer and Criterion
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min', 
                                                        factor=0.5, 
                                                        patience=3, 
                                                        verbose=True)  


# Early Stopping
early_stopping = EarlyStopping(patience=5)

# Train the model
trainer = Trainer(model_name="Img2Mol",
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    early_stopping=early_stopping)

trainer.train()
