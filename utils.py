
import torch


class Trainer:
    def __init__(self, model_name, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, checkpoint_dir):
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
        if self.checkpoint_dir[-1] != '/':
            self.checkpoint_dir += '/'

    def train(self):
        print(f"Started Training of model {self.model_name}")
        self.model = self.model.to(self.device)

        # Early stopping parameters
        min_loss = 1e5
        patience = 50
        current_count = 0

        for epoch in range(self.epochs):
            running_loss = 0.0

            # Train for an epoch
            for i, (images, embeddings) in enumerate(self.train_loader):
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
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

            # Calculate validation loss
            val_loss = self.calculate_val_loss()

            # Take a scheduler step
            self.scheduler.step(val_loss)

            # Check for early stopping
            print('Epoch: %d Validation loss: %.5f' %
                  (epoch + 1, val_loss))
            if val_loss < min_loss:
                current_count = 0
                min_loss = val_loss
            else:
                current_count += 1
                if current_count >= patience:
                    print("Early Stopping the training")
                    break
            
            # save a model every 50 epochs
            if epoch % 50 == 49:
                self.save_model(f"checkpoint_{epoch // 50}")

        print('Finished Training')
        self.save_model("final")

    def calculate_val_loss(self):
        self.model.eval()
        total_loss = 0

        # Test validation data
        with torch.no_grad():
            for i, (images, embeddings) in enumerate(self.val_loader):
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
