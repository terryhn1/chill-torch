import torch
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset

class LinearRegressionDataset(Dataset):
        def __init__(self, csv_file, x_headers, y_header):
            self.dataset = pd.read_csv(csv_file)
            self._set_headers(x_headers, y_header)
            
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return torch.tensor([self.dataset[x_header][index] for x_header in self.x_headers], dtype = torch.float32), \
                    torch.tensor(self.dataset[self.y_header][index], dtype = torch.float32)

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
                if header < 0 or header >= len(self.dataset.columns):
                    raise HeaderError(f"header of index {header} is out of range.")
                return self.dataset.columns[header]
            
class ClassificationDataset(Dataset):
        def __init__(self, csv_file, class_header):
            self.data_frame = pd.read_csv(csv_file)
            self.class_header = class_header
            self._set_classes()
            self._encode_string_values()
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            return (torch.tensor(self.data_frame.iloc[index, :], dtype = torch.float32),
                    torch.tensor(self.labels[index], dtype = torch.float32))
        
        def _set_classes(self):
            if isinstance(self.class_header, str):
                if self.class_header not in self.data_frame.columns:
                    raise HeaderError("Incorrect header of str type. Please check header input value")
                
            elif isinstance(self.class_header, int):
                if self.class_header > len(self.data_frame) or self.class_header < 0:
                    raise HeaderError("Incorrect header of int type. Please check header input value")
                
                self.class_header = self.data_frame.columns[self.class_header]

            else:
                raise HeaderError(f"Incorrect header given with type {type(self.class_header)}. Header must be given as an integer or string")
            
            self.classes = self.data_frame[self.class_header].unique()
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(self.classes)
            self.labels = label_encoder.transform(self.data_frame[self.class_header])
            self.labels_to_classes = {i: label_encoder.inverse_transform([i])[0] for i in label_encoder.transform(self.classes)}
            self.data_frame = self.data_frame.drop(self.class_header, axis = 1)
    
        def _encode_string_values(self):
            for i in range(len(self.dataset.columns)):
                column = self.dataset.iloc[:, i]
                if not isinstance(column[0], int) and not isinstance(column[0], float):
                    label_encoder = preprocessing.LabelEncoder()
                    label_encoder.fit(column)
                    self.dataset.iloc[:, i] = label_encoder.transform(column)

class ImageClassificationDataset(Dataset):
    def __init__(self):
        pass
    pass

class HeaderError(Exception): pass