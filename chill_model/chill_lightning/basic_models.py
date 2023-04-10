import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn
from typing import Callable

class RegularClassificationModel(pl.LightningModule):
    """ Supports binary and multi-class classification.
        Multi-labelling in the process.
    """
    def __init__(self,
                 model: nn.Module,
                 number_of_classes: int,
                 task: str,
                 optim: torch.optim.Optimizer = None,
                 loss_fn: Callable = None,
                 pretrained: bool = False,
                 classifier: nn.Module = None,
                 forward_override: bool = False,
                 lr: float = 1e-3):
        """
            Creates a simple classification model - does not support image classification.

            Args:
                model: Torch module with layers built from Sequential blocks
                number_of_classes: Input for accuracy metric.
                task: Accepts any of the following values - ['binary','multiclass']
                optim: Torch optimizer. default is Adam
                loss_fn: Custom loss function if added.
                pretrained: Existing model. If true, then all layers are frozen and classifier is updated.
                classifier: If pretrained is True, then this is a required parameter.
                forward_override (optional): True to use your own forward function from model.
                lr: Learning rate given as a float. Default is 0.001 

        """
        super().__init__()
        self.task = task
        self.train_accuracy = torchmetrics.Accuracy(task = self.task, num_classes = number_of_classes)
        self.valid_accuracy = torchmetrics.Accuracy(task = self.task, num_classes = number_of_classes)
        self.test_accuracy = torchmetrics.Accuracy(task = self.task, num_classes = number_of_classes)
        self.optim = optim
        self.forward_override = forward_override
        self.torch_forward = model.forward
        self.lr = lr
        
        if loss_fn:
            self.loss_fn = loss_fn
        elif self.task == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss
        elif self.task == 'multiclass':
            self.loss_fn = nn.CrossEntropyLoss

        if pretrained:
            self.layers = list(model.children()[:-1])
            self.layers.append(classifier)
    
        else:
            self.layers = list(model.children())
        
        self.layers = nn.Sequential(*self.layers)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x).squeeze()
        
        if self.task == 'binary':
            y_preds = torch.round(torch.sigmoid(y_logits))

        elif self.task == 'multiclass':
            y_preds = torch.argmax(y_logits, dim = 1)

        loss = self.loss_fn(y_logits, y)
        self.train_accuracy(y_preds, y)
        self.log('train_acc_step', self.train_accuracy)
        
        return {"loss": loss, "log": self.log}
    
    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_logits = self.forward(x).squeeze()

        if self.task == 'binary':
            y_preds = torch.round(torch.sigmoid(y_logits))

        elif self.task == 'multiclass':
            y_preds = torch.argmax(y_logits, dim = 1)

        val_loss = self.loss_fn(y_logits, y)
        self.valid_accuracy(y_preds, y)
        self.log('val_acc_step', self.valid_accuracy)
        self.log('val_loss', val_loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        y_logits = self.forward(x).squeeze()

        if self.task == 'binary':
            y_preds = torch.round(torch.sigmoid(y_logits))
        
        elif self.task == 'multiclass':
            y_preds = torch.argmax(y_logits, dim = 1)
        
        test_loss = self.loss_fn(y_logits, y)
        self.test_accuracy(y_preds, y)
        self.log('test_acc_step', self.test_accuracy)
        self.log('val_loss', test_loss)
    
    def predict_step(self, batch, batch_idx):
        x, y  = batch

        y_logits = self.forward(x).squeeze()

        if self.task == 'binary':
            y_preds = torch.round(torch.sigmoid(y_logits))
        
        elif self.task == 'multiclass':
            y_preds = torch.argmax(y_logits, dim = 1)
        
        return y_preds
    
    def configure_optimizer(self):
        if not self.optim:
            optim = torch.nn.Adam(params = self.parameters(), lr = self.lr)
        else:
            optim = self.optim(params = self.parameters(), lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = 1)
        return [optim], [lr_scheduler]
    
    def forward(self, x):
        if self.forward_override:
            return self.torch_forward(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x

class LinearRegressionModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 forward_override: bool = False,
                 optim: torch.optim.Optimizer = None,
                 loss_fn: Callable = None,
                 lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.optim = optim
        self.forward_override = forward_override
        self.torch_forward = model.forward

        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.MSELoss

        self.layers = model.children()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.forward(x)
        train_loss = self.loss_fn(y_preds, y)

        return {"loss": train_loss, "log": self.log}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.forward(x)
        val_loss = self.loss_fn(y_preds, y)

        self.log('val_loss', val_loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.forward(x)
        test_loss = self.loss_fn(y_preds, y)
        
        self.log('test_loss', test_loss)
    
    def predict_step(self, batch, batch_idx):
        x, y =batch
        y_preds = self.forward(x)
        return y_preds
     
    def configure_optimizer(self):
        if not self.optim:
            optim = torch.optim.SGD(params = self.parameters(), lr = self.lr)
        else:
            optim = self.optim(params = self.parameters(), lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = 1)
        return [optim], [lr_scheduler]

    def forward(self, x):
        if self.forward_override:
            return self.torch_forward(x)
        
        for layer in self.layers:
            x = layer(x)
        return x