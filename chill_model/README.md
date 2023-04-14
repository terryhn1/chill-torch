## ChillModel

### Table of Contents
1. [Overview](https://github.com/terryhn1/chill_torch/tree/master/chill_model#overview)
2. [Fast Production vs. Customization](https://github.com/terryhn1/chill_torch/tree/master/chill_model#fast-production-vs-customization)
3. [Initialization](https://github.com/terryhn1/chill_torch/tree/master/chill_model#chillmodel-initialization)
4. [Class Methods](https://github.com/terryhn1/chill_torch/tree/master/chill_model#methods)
5. [Workflow 1: Custom Model with Custom Propagation](https://github.com/terryhn1/chill_torch/tree/master/chill_model#sample-workflow-1-instantiating-from-custom-model-with-custom-forward)
6. [Workflow 2: Custom Model with No Custom Propagation](https://github.com/terryhn1/chill_torch/tree/master/chill_model#sample-workflow-2-initializing-custom-model-with-no-custom-forward)
7. [Workflow 3: Transfer Learning using Feature Extraction](https://github.com/terryhn1/chill_torch/tree/master/chill_model#sample-workflow-3-transfer-learning---feature-extraction)
8. [Workflow 4: Training and Testing](https://github.com/terryhn1/chill_torch/tree/master/chill_model#sample-workflow-4-trainingtesting)

### Overview
ChillModel is a class that encapsulates the capabilities of PyTorch Lightning,
holding all the data necessary to enable PyTorch Lightning to work. However, initializing
a ChillModel is intended to stay as pythonic as possible. In order to use ChillModel,
there are still the requirements of needing to know the layers behind custom models and
existing models that are available on PyTorch. Therefore, ChillTorch requires you
to create a PyTorch model before initializing along with creating a forward pass if
you feel the standardized forward given by the Lightning Models created are not
capable for your specific problem.

### Fast Production vs. Customization
ChillModel is a great utility tool for beginners trying to learn more about models themselves
rather than stressing over the other pre-processing workflows. ChillModel reduces the amount
of code you need to write significantly in order to get out results faster rather than
learning more libraries and tools. This leads ChillModel to have a low-ceiling customization as hyperparameters
are typically adjusted through training from PyTorch Lightning. If you are looking to learn more of the process
of training Torch models, it is highly advisable to write your code in pure PyTorch. However, any PyTorch
model can be instantly converted to ChillModel. Lightning Models are straightforward, and you create your
own models as well if the current models do not suit your purpose.

### ChillModel Initialization
```python
def __init__(self,
             model: nn.Module,
             train_dataloader: DataLoader,
             test_dataloader: DataLoader,
             problem_type: str,
             forward_override: bool,
             optim: torch.optim.Optimizer = None,
             loss_fn: Callable = None,
             lr: float = LEARNING_RATE,
             max_epochs: int = MAX_EPOCHS,
             max_time: dict = None,
             deterministic: bool = False,
             callbacks: List[str] = []):
```

#### Required Parameters
* **model**: your custom/pretrained PyTorch Model. Your layers must be written in the order that is needed for propagation.
* **train_dataloader**: A Dataloader for training. You can create custom dataloaders through ChillTorch or use pre-existing ones from PyTorch.
* **test_dataloader**: A Dataloader for testing.
* **problem_type**: A string that depicts what problem you are trying to solve(e.g. classification, regression, labeling, etc). Currently, only 'img-class', 'reg-class', and 'lin-reg' are allowed for the problem_type.
* **forward_override**: A boolean that determines whether to use your own propagation function or to use a standard one.

#### Optional Parameters
* **optim**: Torch Optimizer that has not been instantiated. A default one will be used that is standard in the industry for the specific problem.
* **loss_fn**: A loss function that has not been instantiated. 
* **lr**: Learning rate. Default is set to 1e-3
* **max_epochs**: Tells the Lightning Trainer how many epochs to stop at
* **max_time**: Tells the Lightning Trainer how much time given before termination of training.
* **deterministic**: Tells the Lightning Trainer to set a seed or not to get the same random state.
* **callbacks**: A list of callbacks to help with training and storing data. You may use string arguments or give your own callbacks.

### Class Methods
```python
def train(self):
    """ Uses attributes created from initialization to run training through Lightning Trainer 
    """
    ...
    return

def test(self):
    """ Uses attributes created from initialization to run test through Lightning Trainer
    """
    ...
    return

def validate(self, valid_dataloader):
    """ Requires a separate valid_dataloader if you want to run cross-validation separately from Lightning procedures.
    """
    ...
    return

def predict(self, predict_dataloader):
    """ Requires a predict_dataloader to test out results.
    """
    ...
    return

def convert_predictions(self, labels_to_classes):
    """ In the case that string classse have been converted using sk-learn's Label Encoder, a dictionary can be sent
    in order to change the results from numerical results to string results.
    """
    ...
    return
```

### Sample Workflow 1: Instantiating from Custom Model with Custom Forward
1. Create a Torch Model
```python
class CustomModel(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        # Create layers in order
        self.fc1 = torch.nn.Linear(in_features, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout = torch.Dropout(0.2)
        self.classifier = torch.nn.Linear(hidden_units, out_features)
    def forward(self, input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.classifier(x)        
```
2. Instantiate the ChillModel
```python
model = ChillModel(model = CustomModel(4, 1, 8),
                   train_dataloader = train_dataloader,
                   test_dataloader = test_dataloader,
                   problem_type = 'reg-class',
                   forward_override = True)
```

### Sample Workflow 2: Initializing Custom Model with no Custom Forward
1. Create the layers. You do not need to override forward function here.
```python
class CustomModel2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        self.fc1 = torch.nn.Linear(in_features, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, out_features)
```
2. Instantiate the ChillModel
```python
model = ChillModel(model = CustomModel2(4, 1, 8),
                   train_dataloader = train_dataloader,
                   test_dataloader = test_dataloader,
                   problem_type = 'reg-class',
                   forward_override = False)
```

### Sample Workflow 3: Transfer Learning - Feature Extraction
1. Grab the pretrained model from Torch
```python
from torchvision.models import AlexNet, AlexNet_Weights

weights = AlexNet_Weights.DEFAULT
model = AlexNet(weights = weights)
```
2. Freeze the base layers
```python
for parameter in model.parameters():
    parameter.requires_grad = False
```
3. Create your own extractor
```python

# This variable name will vary depending on the pretrained model used
self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                      torch.nn.Linear(8, 1))
```

4. Instantiate the ChillModel
```python
chill_model = ChillModel(model = model,
                         train_dataloader = train_dataloader,
                         test_dataloader = test_dataloader,
                         problem_type = 'img-class',
                         forward_override = True)
```

### Sample Workflow 4: Training/Testing
```python
# Train
chill_model.train()

# Test
chill_model.test()

# Predict

# If dataset comes from outside ChillTorch
label_to_classes = {0: 'donut', 1: 'sushi', 2: 'french fry'}

# If dataset comes from using ChillTorch's data processing
label_to_classes = original_dataset.label_to_classes

preds = chill_model.predict(predict_dataloader)
preds = chill_model.convert_predictions(preds, label_to_classes)
```