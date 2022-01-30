import numpy as np
import torch



#EARLY STOPPING ERROR
class EarlyException(Exception):
    def __init__(self, message="Training model has been stopped due to enhancement problem!"):
        """Early stopping error. \n
            Args:
                - message (str): message to be printed.
        """
        self.message = message
        super().__init__(self.message)
        


#EARLY STOPPING
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.005, path="last_model.pt"):
        """Early stopping for model's loss improvement. \n
            Args:
            - patience (int): how long to wait after last loss improvement.
                Default: 7
            - verbose (bool): iff True, prints a message for validation loss improvement.
                Defalut: False
            - delta (float): value that how much model should imrove.
                Default: 0.005
            - path (str): path where model's parameters will be saved.
                Default: "last_model.pt"
        """   
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.previous_loss = None
        self.val_loss_min = np.Inf
        self.counter = 0

    #__call__
    def __call__(self, val_loss, model):
        val_loss = np.array(val_loss).mean()

        if self.previous_loss is None:
            self.previous_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif self.previous_loss - self.delta > val_loss:
            self.save_checkpoint(val_loss, model)
            self.previous_loss = val_loss
            self.counter = 0

        else:
            self.previous_loss = val_loss
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                raise EarlyException
        
        self.previous_loss = val_loss

    #Save Checkpoint
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases. \n
            Args:
                - val_loss: validation loss value.
                - model: segmentation model.
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



#SAVE BEST MODEL
class SaveBestModel():
    def __init__(self, verbose=False, path="best_model.pt"):
        """Saves best parameters for validation loss. \n
            Args:
                - verbose (bool): iff True, prints a message for validation loss improvement.
                        Default: False
                - path (str): path where model's parameters will be saved.
                        Default: "best_model.pt"
        """
        self.verbose = verbose
        self.path = path
        self.val_min_loss = np.Inf
        
    #__call__
    def __call__(self, val_loss, model):
        val_loss = np.array(val_loss).mean()
        
        if self.val_min_loss > val_loss:
            self.save_model(val_loss, model)
    
    #Save Model
    def save_model(self, val_loss, model):
        """Saves model when best score got acquired. \n
            Args:
                - val_loss: validation loss value.
                - model: segmentation model.
        """
        if self.verbose:
            print("Model has been saved.")
        self.val_min_loss = val_loss
        torch.save(model.state_dict(), self.path)
