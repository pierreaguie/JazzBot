from JazzBot.model import *
from JazzBot.data_loader import *
from JazzBot.config import *
import pytorch_lightning as pl


def train(model, train_dataloader, val_dataloader, epochs):
   trainer = pl.Trainer(max_epochs=epochs)
   trainer.fit(model, train_dataloader, val_dataloader)
   torch.save(model, "./Models/model.pth")
