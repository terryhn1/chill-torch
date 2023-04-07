import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn

class RegularClassificationModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 number_of_classes: int,
                 optim: nn.Module = None,
                 forward_override: bool = False,
                 lr: float = 1e-3):
        """
            Creates a simple classification model.

            Args:
            model: torch module with layers built from Sequential blocks
            optim: torch optimizer. default is Adam 
        """
        self.layers = list(model.children())
        self.number_of_classes = number_of_classes
        self.accuracy = torchmetrics.Accuracy()
        self.optim = optim
        self.forward_override = forward_override
        self.torch_forward = model.forward
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x).squeeze()
        if self.number_of_classes == 2:
            loss = torch.nn.BCEWithLogitsLoss(y_logits, y)
        elif self.number_of_classes > 2:
            loss = torch.nn.CrossEntropyLoss(y_logits, y)
        
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
        if self.forward_override:
            return self.torch_forward(x)

        for layer in self.layers:
            x = layer(x)
        return torch.softmax(x, dim = 1)

class LinearRegressionModel(pl.LightningModule):
    def __init__(self, model: nn.Module, forward_override: bool = False, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.forward_override = forward_override
        self.torch_forward = model.forward
        self.layers = list(model.children())

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_preds = self.forward(x)
        loss_fn = nn.MSELoss()
        train_loss = loss_fn(y_preds, y)

        return {"loss": train_loss, "log": self.log}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_preds = self.forward(x)
        loss_fn = nn.MSELoss()
        val_loss = loss_fn(y_preds, y)
        self.log('val_loss', val_loss)
     

    def configure_optimizer(self):
        return torch.optim.SGD(parameters = self.parameters(), lr = self.lr)

    def forward(self, x):
        if self.forward_override:
            return self.torch_forward(x)
        
        for layer in self.layers:
            x = layer(x)
        return x


class ImageClassificationModel(pl.LightningModule):
    pass