import torch
import torch.nn as nn



#EARLY STOPPING
class EarlyException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print("!!! Training model has been stopped due to enhancement problem !!!")


def early_stop(n_epohcs=7):
    """Early stopping in case of model does not improve anymore."""
    current_loss = []
    last_lost = []
    counter = 0
    if current_loss >= last_lost:
        counter += 1
    else:
        counter = 0

    if counter >= n_epohcs:
        raise EarlyException




#INITIALIZING PARAMETERS > OK
def init_weights(m):
    """Initialization function for model parameters."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)




#SAVES BEST PARAMETERS
def save_best_model(net, val_loss, last_loss):
    """Saves best parameters for validation loss."""
    if val_loss > last_loss:
        val_loss = last_loss
        torch.save(net.state_dict(), "best_model.pt")