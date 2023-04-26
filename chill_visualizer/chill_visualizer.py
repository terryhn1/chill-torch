from ..chill_model import ChillModel
from chill_reccs import ChillRecommenderEngine, Recommendation
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from typing import List

class ChillVisualizer:
    def __init__(self, model: ChillModel, dataset: Dataset, dataset_type: str, grid_theme_color: str = "whitegrid"):
        ''' ChillVisualizer abstracts the amount of information needed to graph by allowing the VisualRecommender
        to determine which graphs would be relevant for the user. Can visualize datasets along with setting up graphs
        or using tensorboard to analyze prediction scores.
        
        Args:
            model: ChillModel that can be trained or not yet trained. This only matters for plotting predictions
            dataset: Dataset created from data_loading.py or a Torch dataset.
            dataset_type: A string that indicates the type of data the dataset is made out of.
            grid_theme_color: Sets the matplotlib grid color. Set to white grid by default. 
        '''
        self.model = model
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.grid_theme_color=  grid_theme_color

    def __call__(self, **kwargs):
        """ Allows the user to change the attributes if desired.
        """
        for key, value in kwargs.items():
            if key not in self.__dict__:
                raise AttributeError(f"Attribute {key} cannot be assigned")
            self.__dict__[key] = value

    def visualize(self, graph_requests = List[Recommendation], k = 1):
        ''' The main tool for graphing analysis plots. Does not include graphing prediction plots.

        Args:
            graph_requests: A list of Recommendations. Only to be used if override desired.
            k: An integer indicating the amount of recommendations graphs to be sent back.
        '''
        recc_engine = ChillRecommenderEngine(self.dataset, self.dataset_type)

        if not graph_requests:
            reccs = recc_engine.get_top_k_graph_choices()

            for recc in reccs:
                if recc == 'displot':
                    self._dis_plot()
                elif recc == 'hist':
                    self._hist()
                elif recc == "line_plot":
                    self._line_plot()
                elif recc == "heatmap":
                    self._heatmap()
                elif recc == 'images':
                    self._plot_k_images()
        else:
            pass
            # Do what the user wants


    def plot_preds(self):
        # This is a manual plot, rather than using tensorboard
        train_loss, train_acc = self.model.train_results
        test_loss, test_acc = self.model.test_results

        sns.set_theme("whitegrid")
        pass

    def _dis_plot(self):
        sns.displot()
        pass

    def _hist(self):
        sns.histplot()
        pass

    def _line_plot(self):
        sns.lineplot(...)
        pass

    def _heatmap(self):
        sns.heatmap(...)
        pass

    def _plot_k_images(self, k = 1, random_seed = None):
        # use matplotlib plot charts with PIL images here
        pass

        



        