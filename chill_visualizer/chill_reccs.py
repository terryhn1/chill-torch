import chill_torch.data_processing.custom_datasets as datasets
import seaborn as sns
import pandas as pd

from utils import *
from typing import List

class Recommendation:
    def __init__(self, **kwargs):
        ''' Takes care of storing the information needed for graphing in a condensed manner.
        Arguments will be used by the Visualizer. 
        '''
        for key, val in kwargs.items():
            self.__dict__[key] = val
    
    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
    
    def __getitem__(self, attr):
        return self.__dict__[attr]

class ChillRecommenderEngine:
    def __init__(self, dataset, dataset_type, decision_factor = 'corr'):
        ''' Creates the ChillRecommenderEngine, which takes care of determining
        what is necessary for the Visualizer for plotting by determining relavent graphs
        and ranking them through an information relevance metric.

        Args:
            dataset: Full Dataset, made through DataLoading. Custom datasets can be wrapped if wanted.
            dataset_type: A string, determining what kind of data the dataset is made from.
            decision_factor: metric used to determine what type of relation graphs should be based from.
        
        '''
        
        self.dataset = dataset
        self.decision_factor = decision_factor
        self.recommendations = []
        # Activate certain functionalities depending on the dataset_type
        if dataset_type == 'numerical':
            self._activate_numerical_analysis(decision_factor)
        elif dataset_type == 'image':
            self._activate_image_analysis(decision_factor)
        elif dataset_type == 'audio':
            self._activate_audio_analysis(decision_factor)
        elif dataset_type == 'video':
            self._activate_video_analysis(decision_factor)
        elif dataset_type == 'text':
            self._activate_text_analysis(decision_factor)
    
    def _activate_numerical_analysis(self, decision_factor):
        ''' When pure numbers are only being used, then numerical analysis must be done
        to the dataset in order to gather relationships. Most plotting that comes from this data
        is done using seaborn, which can plot mathematical relationship easily.

        Args:
            decision_factor: String to determine what type of relation graphs should be based from.
        '''
        df = self.dataset.data_frame
        if isinstance(self.dataset, datasets.ClassificationDataset):
            df[self.dataset.class_header] = self.dataset.labels
            if decision_factor == 'corr':
                corr_values = df.corrwith(df[self.dataset.class_header]).sort_values(ascending = False)[1:]
                names = corr_values.keys()

                # TODO: Create a function that will be able to find the optimal x and y
                optimal_x, optimal_y = self._get_optimal_label_and_relation(corr_map = corr_values, column_names = names)

                # Recommended when the x and y are not labeled
                iso_cluster_proportion_rate = self.isolated_clustering_proportion(x = optimal_x,
                                                                                  y = optimal_y)
                if 0 < iso_cluster_proportion_rate < 0.70:
                    self._create_scatter_recommendation(x = optimal_x, y = optimal_y)
                else:
                    self._create_kde_recommendation(x = optimal_x, y= optimal_y)
                
                # Recommended when the one data is labeled
                population_distribution_rate = self._population_distribution(x = optimal_x,
                                                                             y = optimal_y)
                class_bin_relation_rate = self._class_bin_relationship(x = optimal_x,
                                                                       y = optimal_y)
                if population_distribution_rate > 0.5:
                    self._create_violin_recommendation(x = optimal_x, y = optimal_y)
                if class_bin_relation_rate > 0.5:
                    self._create_barplot_recommendation(x = optimal_x, y = optimal_y)
                
                # Recommended for relationship training. Takes the top five values
                correlation_classes_rate = self._correlation_classes(corr_names = names,
                                                                     k = 5)
                if 0 < correlation_classes_rate <  0.65:
                    self._create_heatmap_recommendation(corr_names = names, k = 5)
                else:
                    self._create_clustermap_recommendation(corr_names = names, k = 5)
                

            df.drop(self.dataset.class_header, axis = 1, inplace = True)

        elif isinstance(self.dataset, datasets.LinearRegressionDataset):
            pass
        

    def _activate_image_analysis(self):
        # Images take up a lot of data, so we should only select a random few from the dataset.

        # After that, we want to:
        # 1a. Make sure the images have been grayscaled(check for the amount of channels - should be 2)
        # 1b. If not grayscaled, then have a transform on a random set
        # 2. Use a AvgPool2D Layer on each image with a kernel of size 3.
        # 3. Take the Cosine Similarity Score comparing each result to another.
        # 4. Use a MaxPool2D Layer and repeat step 3 above for another result.
        # 5. Similar to ViT, Flatten and add positional embedding. Then create a positional similarity metric
        #    and score up per patch. Avg Pooling is done per patch, and score added to the position.
        pass

    def _activate_audio_analysis(self):
        pass

    def _activate_video_analysis(self):
        pass

    def _activate_text_analysis(self):
        pass

    def _get_optimal_label_and_relation(self, corr_map: pd.DataFrame, col_names: List[str]):
        pass

    def _create_scatter_recommendation(self, x: str, y: str):
        self.recommendations.append(Recommendation(graph = sns.scatterplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header))
    
    def _create_kde_recommendation(self, x: str, y: str):
        self.recommendations.append(Recommendation(graph = sns.jointplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header))
        
    def _create_violin_recommendation(self, x: str, y: str):
        self.recommendations.append(Recommendation(graph = sns.violinplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header,
                                                   split = True))

    def _create_barplot_recommendation(self, x: str, y: str):
        self.recommendations.append(Recommendation(graph = sns.violinplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header,
                                                   capsize = 0.2))
    
    def _create_heatmap_recommendation(self, corr_names: List[str], k: int):
        self.recommendations.append(Recommendation(graph = sns.heatmap,
                                                   data = self.dataset.data_frame[corr_names[:k]]))

    def _create_clustermap_recommendation(self, corr_names: List[str], k: int):
        self.recommendations.append(Recommendation(graph = sns.clustermap,
                                                   data = self.dataset.data_frame[corr_names[:k]]))

    def _get_fig_size(self, dataset_size):
        level = 5* (dataset_size // 5000)
        return (10 + level, 5 + level)

    def get_top_k_graph_choices(self):
        pass