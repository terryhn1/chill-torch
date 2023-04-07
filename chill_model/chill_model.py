import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl
from . import helper_functions

class ChillModel:
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 problem_type: str,
                 is_custom: bool,
                 optim: nn.Module = None,
                 valid_dataloader: DataLoader = None,
                 lr: float = 1e-3):
        """ Creates an instance of ChillModel. 
            Currently available problem types: [lin-reg, multi-class, binary-class]
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.valid_dataloader = valid_dataloader
        self.problem_type = helper_functions.select_mode(problem_type)
        self.model = helper_functions.select_model(model = model,
                                                   problem_type = self.problem_type,
                                                   custom = is_custom,
                                                   optim = optim,
                                                   lr = lr)

    def train(self):
        trainer = pl.Trainer()
        trainer.fit(model = self.model, train_dataloaders = self.train_dataloader)
    
    def test(self):
        trainer = pl.Trainer()
        trainer.test(model = self.model, dataloaders = self.test_dataloader)

    def validate(self):
        if self.valid_dataloader:
            trainer = pl.Trainer()
            trainer.test(self.model, self.train_dataloader, self.valid_dataloader)
        else:
            raise DataLoaderError("No dataloader initialized. Use model.set_valid_dataloader() to initialize")
    
    def set_valid_dataloader(self, valid_dataloader: DataLoader):
        if valid_dataloader and isinstance(valid_dataloader, DataLoader):
            self.valid_dataloader = valid_dataloader
        else:
            raise DataLoaderError("No input given for valid_dataloader")
            

class DataLoaderError(Exception): pass
