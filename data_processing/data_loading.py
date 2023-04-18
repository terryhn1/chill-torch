# Example file for creating dataloaders
import pandas as pd
import os
from custom_datasets import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, Union, List

def create_dataloaders(dataset: Dataset,
                       batch_size: int = 1,
                       collate_fn = None,
                       train_split: float = 0.8) -> Dict[str, DataLoader]:
    """ Turns a given dataset into train and dataloaders.
        Validation dataset for cross-validation is not an option for this function at the moment.

        Args:
            dataset: custom dataset or pytorch dataset.
            batch_size(optional): Default value is 1. Batch size must be larger than the number of CPUs available
            collate_fn(optional): function to process each batch uniquely
            train_split: Float ranging (0,1). Default is set to 0.8
    """

    train_index, test_index = train_test_split((list(range(len(dataset)))),
                                                train_size = train_split)
    train_dataset, test_dataset = Subset(dataset, train_index), Subset(dataset, test_index)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = os.cpu_count(),
                                  collate_fn = collate_fn)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size = batch_size,
                                  shuffle = False,
                                  num_workers = os.cpu_count(),
                                  collate_fn = collate_fn)

    return {"train": train_dataloader, "test": test_data_loader}

def create_image_dataset(root_dir: str, transform = None):
    """ Generic function to convert image folders to datasets. 
    """
    return ImageClassificationDataset(root_dir, transform = transform)

def create_linear_regression_dataset(csv_file: str, x_headers: List[Union[int, str]], y_header: Union[str, int]):
    """ Uses a csv file to create a dataset for linear regression, recommended for 2-dimensional data.
        More features can be added through x_headers
        Only use if looking for solving linear regression problems with deep learning.

        Args:
            csv_file: a direct file path to the csv file. Cannot include non-ASCII characters.
    
    """
    return LinearRegressionDataset(csv_file, x_headers, y_header)

def create_classification_dataset(csv_file: str, class_header: Union[int, str]):
    """ Uses a csv file to create a dataset that includes labeling capability.
        Transforms non-numerical values to numerical to generate tensors.

        Args:
            csv_file: a direct file path to the csv file. Cannot include non-ASCII characters.
            class_header: class selected as labeling basis. Must be an integer or string. 
    
    """
    return ClassificationDataset(csv_file, class_header)

class HeaderError(Exception): pass