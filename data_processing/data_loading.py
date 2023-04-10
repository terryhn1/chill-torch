# Example file for creating dataloaders

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset, Subset, DataLoader
import os
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
    image_folder = torchvision.datasets.ImageFolder(root_dir, transform = transform)
    return image_folder

def create_linear_regression_dataset(csv_file: str, x_headers: List[Union[int, str]], y_header: Union[str, int]):
    """ Uses a csv file to create a dataset for linear regression, recommended for 2-dimensional data.
        More features can be added through x_headers
        Only use if looking for solving linear regression problems with deep learning.

        Args:
            csv_file: a direct file path to the csv file. Cannot include non-ASCII characters.
    
    """
    class LinearRegressionDataset(Dataset):
        def __init__(self, csv_file, x_headers, y_header):
            self.dataset = pd.read_csv(csv_file)
            self._set_headers(x_headers, y_header)
            
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            features = tuple([torch.tensor(self.dataset[x_header][index],
                              dtype = torch.float32) for x_header in self.x_headers])
            return ((torch.cat(features, dim = 1)),
                    torch.tensor(self.dataset[self.y_header][index], dtype = torch.float32))

        def _set_headers(self, x_headers, y_header):
            self._check_headers(x_headers, y_header)
        
            self.x_headers = [self._get_header_value(feature) for feature in x_headers]
            self.y_header = self._get_header_value(y_header)
        
        def _check_headers(self, x_headers, y_header):
            for feature in x_headers:
                if not isinstance(feature, str) and not isinstance(feature, int):
                    raise HeaderError(f"x_header was of type {type(feature)}. Please verify input is a string or integer")
            if not isinstance(y_header, str) and not isinstance(y_header, int):
                raise HeaderError(f"y_header was of type {type(y_header)}. Please verify input is a string or integer")
        
        def _get_header_value(self, header):
            if isinstance(header, str):
                if header not in self.dataset.columns:
                    raise HeaderError(f"header of value {header} cannot be found as a column name.")   
                return header
            
            elif isinstance(header, int):
                if header < 0 or x_headers >= len(self.dataset.columns):
                    raise HeaderError(f"header of index {header} is out of range.")
                return self.dataset.columns[header]
            
    return LinearRegressionDataset(csv_file, x_headers, y_header)

def create_classification_dataset(csv_file: str, class_header: Union[int, str]):
    """ Uses a csv file to create a dataset that includes labeling capability.
        Transforms non-numerical values to numerical to generate tensors.

        Args:
            csv_file: a direct file path to the csv file. Cannot include non-ASCII characters.
            class_header: class selected as labeling basis. Must be an integer or string. 
    
    """
    class ClassificationDataset(Dataset):
        def __init__(self, csv_file, class_header):
            self.dataset = pd.read_csv(csv_file)
            self._set_classes(class_header)
            self._encode_string_values()
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            return (torch.tensor(self.dataset.iloc[index, :], dtype = torch.float32), self.labels[index])
        
        def _set_classes(self, class_header):
            if isinstance(class_header, str):
                if class_header not in self.dataset.columns:
                    raise HeaderError("Incorrect header of str type. Please check header input value")
                
            elif isinstance(class_header, int):
                if class_header > len(self.dataset) or class_header < 0:
                    raise HeaderError("Incorrect header of int type. Please check header input value")
                
                class_header = self.dataset.columns[class_header]

            else:
                raise HeaderError(f"Incorrect header given with type {type(class_header)}. Header must be given as an integer or string")
            
            self.classes = self.dataset[class_header].unique()
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(self.classes)
            self.labels = label_encoder.transform(self.dataset[class_header])
            self.label_to_class = {i: label_encoder.inverse_transform([i])[0] for i in label_encoder.transform(self.classes)}
            self.dataset = self.dataset.drop(class_header, axis = 1)
    
        def _encode_string_values(self):
            for i in range(len(self.dataset.columns)):
                column = self.dataset.iloc[:, i]
                if not isinstance(column[0], int) and not isinstance(column[0], float):
                    label_encoder = preprocessing.LabelEncoder()
                    label_encoder.fit(column)
                    self.dataset.iloc[:, i] = label_encoder.transform(column)
            
    return ClassificationDataset(csv_file, class_header)

class HeaderError(Exception): pass