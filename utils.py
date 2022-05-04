import torch
# from inference import Inference

class Trainer:
    def __init__(self, model_name, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, checkpoint_dir, early_stopping):
        self.model_name = model_name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping = early_stopping
        if self.checkpoint_dir[-1] != '/':
            self.checkpoint_dir += '/'
        
        # self.inference = Inference(model=self.model,
        #                             data_loader=self.val_loader,
        #                             device=self.device)

    def train(self):
        print(f"Started Training of model {self.model_name}")
        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0.0

            # Train for an epoch
            for i, (smiles, images, embeddings) in enumerate(self.train_loader):
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)

                # ============ Forward ============
                outputs = self.model(images)
                loss = self.criterion(outputs, embeddings)

                # ============ Backward ============
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ============ Logging ============
                running_loss += loss.data
                if i % 50 == 49:
                    print('[%d, %d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 50), flush=True)
                    running_loss = 0.0

            # Calculate validation loss
            val_loss = self.calculate_val_loss()
            print('Epoch: %d Validation loss: %.5f' % (epoch + 1, val_loss), flush=True)

            # Take a scheduler step
            self.scheduler.step(val_loss)

            # Take a early stopping step
            self.early_stopping.step(val_loss)
            
            # save a model every 5 epochs and also calculate the accuracy
            if epoch % 5 == 4:
                self.save_model(f"checkpoint_{epoch}")
            # _, accuracy = self.inference.calculate_loss_accuracy()
            # print('Epoch: %d Validation Accuracy: %.5f' %
            #   (epoch + 1, accuracy))
            
            # Check early stopping to finish training
            if self.early_stopping.stop_training:
                print("Early Stopping the training")
                break

        print('Finished Training')
        self.save_model("final")

    def calculate_val_loss(self):
        self.model.eval()
        total_loss = 0

        # Test validation data
        with torch.no_grad():
            for i, (smiles, images, embeddings) in enumerate(self.val_loader):
                images = images.to(self.device)
                embeddings = embeddings.to(self.device)

                # ============ Forward ============
                outputs = self.model(images)
                loss = self.criterion(outputs, embeddings)
                total_loss += loss.data.item()

        self.model.train()
        return total_loss / len(self.val_loader)

    def save_model(self, file_name):
        print(f'Saving Model in {self.checkpoint_dir + file_name}')
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir + file_name}.pt")
