import torch
import helper_functions
from torch import nn

class ChillModel:
    def __init__(self,
                 model: nn.module,
                 train_data: torch.utils.data.DataLoader,
                 test_data: torch.utils.data.DataLoader,
                 problem_type: str,
                 is_custom: bool):
        """ Creates an instance of ChillModel. 
            Currently available problem types: [lin-reg, multi-class, binary-class]
        """
        self.train_data = train_data
        self.test_data = test_data
        helper_functions.declare_data_issues(self.train_data, self.test_data)

        self.problem_type = helper_functions.select_mode(problem_type)
        self.model = helper_functions.select_model(model, self.problem_type,)

    def  __call__(self, input):
        return self.model.fit(input)