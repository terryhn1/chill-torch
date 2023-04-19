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
        df = self.dataset.data_frame
        if isinstance(self.dataset, datasets.ClassificationDataset):
            df[self.dataset.class_header] = self.dataset.labels
            if decision_factor == 'corr':
                corr_values = df.corrwith(df[self.dataset.class_header]).sort_values(ascending = False)[1:]
                names = corr_values.keys()

                # TODO: Create a function that will be able to find the optimal x and y

                # Recommended when the x and y are not labeled
                self._create_scatter_recommendation(x = names[0], y = names[1])
                self._create_kde_recommendation(x = names[0], y= names[0])
                
                # Recommended when the one data is labeled
                self._create_violin_recommendation(x = names[0], y = names[1])
                self._create_barplot_recommendation(x = names[0], y = names[1])
                
                # Recommended for relationship training. Takes the top five values
                self._create_heatmap_recommendation(corr_names = names, k = 5)
                self._create_clustermap_recommendation(corr_names = names, k = 5)
                

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

    def _create_scatter_recommendation(self, x, y):
        self.recommendations.append(Recommendation(graph = sns.scatterplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header))
    
    def _create_kde_recommendation(self, x, y):
        self.recommendations.append(Recommendation(graph = sns.jointplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header))
        
    def _create_violin_recommendation(self, x, y):
        self.recommendations.append(Recommendation(graph = sns.violinplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header,
                                                   split = True))

    def _create_barplot_recommendation(self, x, y):
        self.recommendations.append(Recommendation(graph = sns.violinplot,
                                                   x = x,
                                                   y = y,
                                                   hue = self.dataset.class_header,
                                                   capsize = 0.2))
    
    def _create_heatmap_recommendation(self, corr_names, k):
        self.recommendations.append(Recommendation(graph = sns.heatmap,
                                                   data = self.dataset.data_frame[corr_names[:k]]))

    def _create_clustermap_recommendation(self, corr_names, k):
        self.recommendations.append(Recommendation(graph = sns.clustermap,
                                                   data = self.dataset.data_frame[corr_names[:k]]))

    def _get_fig_size(self, dataset_size):
        level = 5* (dataset_size // 5000)
        return (10 + level, 5 + level)

    def get_top_k_graph_choices(self):
        pass