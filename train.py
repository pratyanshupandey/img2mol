import torch 
from torch import nn
from model import IMG2MOL
from utils import Trainer
from dataset import MoleculeData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


train_path = "data/data/train/"
val_path = "data/data/val/"
test_path = "data/data/test/"
checkpoint_dir = "checkpoints/"


#Hyperparameters
batch_size = 32
learning_rate = 0.00001
epochs = 300
cpu_cores = 20

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

test_dataset = MoleculeData(test_path)
test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores)


# Getting the model
model = IMG2MOL().to(device)


# Optimizer and Criterion
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min', 
                                                        factor=0.7, 
                                                        patience=10, 
                                                        verbose=True)  




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
                    checkpoint_dir=checkpoint_dir)

trainer.train()
