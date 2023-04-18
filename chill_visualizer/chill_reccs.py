import chill_torch.data_processing.custom_datasets as datasets
import seaborn as sns

class Recommendation:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
    
    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            self.__dict__[key] = val
    
    def __getitem__(self, attr):
        return self.__dict__[attr]

class ChillRecommenderEngine:
    def __init__(self, dataset, dataset_type, decision_factor = 'corr'):
        
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
        if isinstance(self.dataset, datasets.ClassificationDataset):
            df = self.dataset.data_frame
            df[self.dataset.class_header] = self.dataset.labels
            if decision_factor == 'corr':
                corr_values = df.corrwith(df[self.dataset.class_header]).sort_values(ascending = False)[1:]

                # Recommended when the x and y are not labeled
                self.recommendations.append(Recommendation(graph = sns.scatterplot,
                                                           x = corr_values.keys()[0],
                                                           y = corr_values.keys()[1],
                                                           hue = self.dataset.class_header,
                                                           figsize = self._get_fig_size(len(self.dataset))))
                
                # Recommended when the one data is labeled
                self.recommendations.append(Recommendation(graph = sns.violinplot,
                                                           x = corr_values.keys()[0],
                                                           y = corr_values.keys()[1],
                                                           hue = self.dataset.class_header,
                                                           figsize = self._get_fig_size(len(self.dataset))))
                

            df.drop(self.dataset.class_header, axis = 1, inplace = True)

        elif isinstance(self.dataset, datasets.LinearRegressionDataset):
            pass
        

    def _activate_image_analysis(self):
        pass

    def _activate_audio_analysis(self):
        pass

    def _activate_video_analysis(self):
        pass

    def _activate_text_analysis(self):
        pass

    def _get_fig_size(self, dataset_size):
        level = 5* (dataset_size // 5000)
        return (10 + level, 5 + level)

    def get_top_k_graph_choices(self):
        pass