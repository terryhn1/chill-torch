
class ChillVisualRecommender:
    def __init__(self, problem_type: str):
        if problem_type == "binary-class":
            self.binary_classifier()
        elif problem_type == "lin-reg":
            self.linear_classifier()
        pass

    def binary_classifier(self):
        return [""]

    def linear_classifier(self):
        return [""]