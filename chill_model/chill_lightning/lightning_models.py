import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn

class CustomBinaryClassificationModel(pl.LightningModule):
    def __init__(self, model: nn.module, optim: nn.module = None, lr: float = 1e-3):
        """
            Creates a simple classification model.

            Args:
            model: torch module with layers built from Sequential blocks
            optim: torch optimizer. default is Adam 
        """
        self.layers = list(model.children())
        self.accuracy = torchmetrics.Accuracy()
        self.optim = optim
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x).squeeze()
        loss =  torch.nn.BCEWithLogitsLoss(y_logits, y)
        
        return {"loss": loss, "log": self.log}
    
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logits = self.forward(x).squeeze()
        val_loss = torch.nn.BCEWithLogitsLoss(y_logits, y)
        self.log('val_acc_step', self.accuracy(y_logits, y))
        self.log('val_loss', val_loss)
    
    def configure_optimizer(self):
        if not self.custom_optim:
            optim = torch.nn.Adam(parameters = self.parameters(), lr = self.lr)
        else:
            optim = self.custom_optim(parameters = self.parameters(), lr = self.lr)
        return optim
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.softmax(x, dim = 1)

class BinaryClassificationModel(pl.LightningModule):
    def __init__(self, model: nn.module, is_pretrained: bool):
        super().__init__()
        layers = model.children()
        pass

class LinearRegressionModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        
        pass

