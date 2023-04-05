import torch
from torch import nn
from chill_lightning import lightning_models as lm

PROBLEM_TYPES = ["lin-reg", "binary-class", "multi-class"]

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
    
    if number_of_classes is not None:
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
        raise ProblemTypeException(f"Incorrect problem type given. Please choose from the following:\n{problem_types}")

    else:
        return problem_type

def select_model(model: nn.module, problem_type: str, custom: bool = False):
    """ Given the problem_type, creates a Lightning Model and returns it.

        Args:
        model: torch module that includes all its layers
        problem_type: pretrained
        pretrained: boolean that determines whether the feature extraction should occur instead
    """
    if problem_type == "binary-class" and not custom:
        return lm.BinaryClassificationModel(model)
    elif problem_type == "binary-class" and custom:
        return lm.CustomBinaryClassificationModel(model)
    elif problem_type == "lin-reg" and not custom:
        return lm.LinearRegressionModel(model)

class ProblemTypeException(Exception): pass
