# Example file for creating dataloaders

import torch
import torchvision
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict

def create_dataloaders(dataset: Dataset,
                       train_transform = None,
                       test_transform = None,
                       batch_size: int = 1,
                       collate_fn = None,
                       train_split: float = 0.8) -> Dict[DataLoader]:
    """ Turns a given dataset into train and dataloaders.
        Validation dataset for cross-validation is not an option for this function at the moment.

        Args:
            dataset: custom dataset or pytorch dataset.
            train_transform: function or collection of PyTorch transforms for train
            test_transform: function or collection of PyTorch transforms for test
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
                                  transform = train_transform,
                                  num_workers = os.cpu_count(),
                                  collate_fn = collate_fn)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size = batch_size,
                                  shuffle = False,
                                  transform = test_transform,
                                  num_workers = os.cpu_count(),
                                  collate_fn = collate_fn)

    return {"train": train_dataloader, "test": test_data_loader}

def create_image_dataset(root_dir: str, transform = None):
    """ Generic function to convert image folders to datasets. 
    """
    image_folder = torchvision.datasets.ImageFolder(root_dir, transform = transform)
    return image_folder