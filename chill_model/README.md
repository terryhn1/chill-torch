## ChillModel

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
             pretrained: bool = False,
             lr: float = LEARNING_RATE,
             max_epochs: int = MAX_EPOCHS,
             max_time: dict = None,
             deterministic: bool = False,
             callbacks: List[str] = []):
```

#### Required Parameters
* model: your custom/pretrained PyTorch Model. Your layers must be written in the order that is needed for propagation.
* train_dataloader: A Dataloader for training. You can create custom dataloaders through ChillTorch or use pre-existing ones from PyTorch.
* test_dataloader: A datalaoder for testing.
* problem_type: A string that depicts what problem you are trying to solve(e.g. classification, regression, labeling, etc). Currently, only 'img-class', 'reg-class', and 'lin-reg' are allowed for the problem_type.
* forward_override: A boolean that determines whether to use your own propagation function or to use a standard one.

#### Optional Parameters
* optim: Torch Optimizer that has not been instantiated. A default one will be used that is standard in the industry for the specific problem.
* loss_fn: A loss function that has not been instantiated. 
* pretrained: Tells the model to use feature extraction instead.
* lr: Learning rate. Default is set to 1e-3
* max_epochs: Tells the Lightning Trainer how many epochs to stop at
* max_time: Tells the Lightning Trainer how much time given before termination of training.
* deterministic: Tells the Lightning Trainer to set a seed or not to get the same random state.
* callbacks: A list of callbacks to help with training and storing data. You may use string arguments or give your own callbacks.