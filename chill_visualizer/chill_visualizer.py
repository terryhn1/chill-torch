from ..chill_model import ChillModel
from chill_reccs import ChillVisualRecommender
import seaborn as sns
import matplotlib.pyplot as plt

class ChillVisualizer:
    def __init__(self):
        """ Creates an instance of the ChillVisualizer, which can store the history
        of all models sent. 
        """
        self.history = {}
        
    
    def visualize(self, model: ChillModel, override: bool = False, option: int = -1):
        if override and option != -1:
            chill_recc = ChillVisualRecommender(model.problem_type, option)
        elif not override and option != -1:
            raise VisualizationOverrideException("Override must be turned to True to use different option than recommended")
        else:
            chill_recc = ChillVisualRecommender(model.problem_type)
        
        self.history[model.name] = chill_recc.graph_choices

        if option >= len(self.history[model.name]):
            raise IndexError(f"Option given needs to be in the range of [0 to {len(self.history[model.name])})")

        # Selection of plot to get from chill_recc  
        if self.history[model.name] == "count-plot":
            pass
        elif self.history[model.name] == "hist":
            pass
        elif self.history[model.name] == "scatter":
            pass
        elif self.history[model.name] == "line-plot":
            pass



class VisualizationOverrideException(Exception): pass
        