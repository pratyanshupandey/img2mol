

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = 1e5
        self.current_count = 0
        self.stop_training = False
    
    def step(self, val_loss):
        if val_loss < self.min_loss:
            self.current_count = 0
            self.min_loss = val_loss
        else:
            self.current_count += 1
        
        if self.current_count >= self.patience:
            self.stop_training = True