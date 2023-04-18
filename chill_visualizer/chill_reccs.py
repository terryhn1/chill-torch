
class ChillRecommenderEngine:
    def __init__(self, dataset, dataset_type, decision_factor = 'corr'):
        
        # Activate certain functionalities depending on the dataset_type
        if dataset_type == 'numerical':
            self._activate_numerical_analysis()
        elif dataset_type == 'image':
            self._activate_image_analysis()
        elif dataset_type == 'audio':
            self._activate_audio_analysis()
        elif dataset_type == 'video':
            self._activate_video_analysis()
        elif dataset_type == 'text':
            self._activate_text_analysis()
    
    def _activate_numerical_analysis(self):
        pass

    def _activate_image_analysis(self):
        pass

    def _activate_audio_analysis(self):
        pass

    def _activate_video_analysis(self):
        pass

    def _activate_text_analysis(self):
        pass

    def get_top_k_graph_choices(self):
        pass