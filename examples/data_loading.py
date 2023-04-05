# Example file for creating dataloaders

import pandas as pd
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

def create_linear_regression_dataset(csv_file: str):
    class LinearRegressionDataset(Dataset):
        def __init__(self, csv_file):
            self.dataset = pd.read_csv(csv_file)
            self.index_to_header = {i: self.dataset.columns[i] for i in range(self.dataset.columns)}
            self.header_to_index = {self.dataset.columns[i]: i for i in range(self.dataset.columns)}

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return self.dataset[index, :]
    
    return LinearRegressionDataset(csv_file)

def create_logistic_regression_dataset(csv_file: str, class_header: str | int):
    class LogisticRegressionDataset(Dataset):
        def __init__(self, csv_file, class_header):
            self.dataset = pd.read_csv(csv_file)
            
            self.set_classes(class_header)
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            return (self.dataset[index, :], self.labels[index])
        
        def set_classes(self, class_header):
            if isinstance(class_header, str):
                if class_header not in self.dataset.columns:
                    raise HeaderError("Incorrect header of str type. Please check header input value")
                
                self.classes = self.dataset[class_header].unique()
                self.labels = self.dataset[self.dataset.columns[class_header]]
                self.dataset = self.dataset.drop(class_header, axis = 1)

            elif isinstance(class_header, int):
                if class_header > len(self.dataset) or class_header < 0:
                    raise HeaderError("Incorrect header of int type. Please check header input value")
                
                self.classes= self.dataset[self.dataset.columns[class_header]].unique()
                self.labels = self.dataset[self.dataset.columns[class_header]]
                self.dataset = self.dataset.drop(self.dataset.columns[class_header], axis = 1)
            
            else:
                raise HeaderError(f"Incorrect header given with type {type(class_header)}. Header must be given as an integer or string")
    
    return LogisticRegressionDataset(csv_file, class_header)

class HeaderError(Exception): pass