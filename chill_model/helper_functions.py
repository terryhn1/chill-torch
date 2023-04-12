import torch
from torch import nn

from typing import Callable, List


PROBLEM_TYPES = ["lin-reg", "reg-class", "img-class"]

def declare_data_issues(train_data: torch.utils.data.DataLoader,
                        test_data: torch.utils.data.DataLoader,
                        number_of_classes: int = None):
    """ Informs user of potential of bad data before training and reccommends labelled changes.

        Args:
            train_data: torch dataloader containing the data for training
            test_data: torch dataloader containing the data for testing
            number_of_classes(optional): used only for classification problems

    """

    total_length = len(train_data) + test_data
    split = round(len(train_data)/total_length, 2)

    if split > .9:
        print("Consider lowering the split to avoid overfitting.")
    
    elif split < 0.6:
        print("consider raising the split to avoid underfitting")
    
    if number_of_classes:
        # Reviewing for Recall/Precision/F1 Score
        # TO-DO
        pass

    print("DECLARATION FINISHED")
    
def select_mode(problem_type: str):
    """ Returns the mode for ChillModel given a correct problem_type.

        Args:
            problem_type: string indicating the mode selected
    """
    if problem_type not in PROBLEM_TYPES:
        raise ProblemTypeException(f"Incorrect problem type given. Please choose from the following:\n{PROBLEM_TYPES}")

    else:
        return problem_type

def select_model(model: nn.Module,
                 problem_type: str,
                 optim: torch.optim.Optimizer,
                 loss_fn: Callable,
                 lr: float,
                 forward_override: bool = False,
                 pretrained: bool = False):
    """ Given the problem_type, creates a Lightning Model and returns it.

        Args:
            model: torch module that includes all its layers
            problem_type: choice of problem.
            optim: Torch optimizer to be used.
            lr: learning rate. Must be a float.
            forward_override (optional): determinant of using specific model training. default is false.
            pretrained (optional): If true, freezes pre-existing layers for classification.
    """
    if problem_type == "reg-class":
        from chill_torch.chill_model.chill_lightning import basic_models as bm
        return bm.RegularClassificationModel(model = model,
                                             forward_override = forward_override,
                                             optim = optim,
                                             loss_fn = loss_fn,
                                             lr = lr,
                                             pretrained = pretrained)
    elif problem_type == "img-class":
        from chill_torch.chill_model.chill_lightning import computer_vision_models as cvm
        return cvm.ImageClassificationModel(model = model,
                                           forward_override = forward_override,
                                           optim = optim,
                                           loss_fn = loss_fn,
                                           lr = lr,
                                           pretrained = pretrained)
    elif problem_type == "lin-reg":
        from chill_torch.chill_model.chill_lightning import basic_models as bm
        return bm.LinearRegressionModel(model = model,
                                        forward_override = forward_override,
                                        loss_fn = loss_fn,
                                        optim = optim,
                                        lr = lr)

def set_callbacks(callbacks: list):
    """ Turns string arguments into callbacks

        Args:
            callbacks: A list of Callback objects or strings indicating for default setup.

        Accepted Callbacks:
            'early-stopping': creates an early stopping callback to monitor loss
            'grad-accum': creates a gradient accumulator to split batches to mini-batches.
            'lr-finder': finds an optimal learning rate during training.
            'timer-[6,12,24]': shuts down training after 6 hours, 12 hours, or 24 hours

    """
    import lightning.pytorch.callbacks as callbacks

    results = []
    for callback in callbacks:
        if isinstance(callback, callbacks.Callback):
            results.append(callback)
        elif callback == 'early-stopping':
            results.append(callbacks.early_stopping.EarlyStopping(monitor = "val_loss", mode = 'min'))
        elif callback == 'grad-accum':
            results.append(callbacks.GradientAccumulationScheduler(scheduling = {4: 2}))
        elif callback == 'lr-finder':
            results.append(callbacks.LearningRateFinder(min_lr = 1e-5, num_training_steps = 50))
        elif callback == 'timer-6' or callback == 'timer-12' or callback == 'timer-24':
            if callback == 'timer-6': results.append(callbacks.Timer(duration = dict(hours = 6)))
            elif callback == 'timer-12': results.append(callbacks.Timer(duration = dict(hours = 12)))
            elif callback == 'timer-24': results.append(callbacks.Timer(duration = dict(hours = 24)))
    return results


class ProblemTypeException(Exception): pass
