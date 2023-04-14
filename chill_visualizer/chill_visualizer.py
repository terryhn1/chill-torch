from ..chill_model import ChillModel
from chill_reccs import ChillVisualRecommender
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class ChillVisualizer:
    def __init__(self, model: ChillModel, dataset: Dataset, dataset_type: str):
        
        self.model = model
        self.dataset = dataset
        self.dataset_type = dataset_type    

    def __call__(self, **kwargs):
        """ Allows the user to change the attributes if desired.
        """
        for key, value in kwargs.items():
            if key not in self.__dict__:
                raise AttributeError(f"Attribute {key} cannot be assigned")
            self.__dict__[key] = value

    def visualize(self):
        pass

    def _count_plot(self):
        pass

    def _hist(self):
        pass

    def _line_plot(self):
        pass

    def _heatmap(self):
        pass

    def _accuracy_to_loss_plot(self):
        pass

    def plot_k_images(self, k = 1, random_seed = None):
        pass

        



        