# Example file for creating dataloaders

import torch
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

    train_index, test_index = train_test_split((list(range(len(dataset)))), train_size = train_split)
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