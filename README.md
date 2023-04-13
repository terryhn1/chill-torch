# ChillTorch
ChillTorch is a lightweight package that allows for streamlining PyTorch projects by
allowing the user to visualize their data and to set up deep learning models in little time,
all while limiting the amount of workflow that is required to get the project up. By reducing
the time it takes to get results up, intro-level PyTorch developers can focus more on
learning about the various models in the industry.

## Table of Contents
1. [Installation](https://github.com/terryhn1/chill_torch#installation)
2. [Package Requirements](https://github.com/terryhn1/chill_torch#package-requirements)
3. [Overview](https://github.com/terryhn1/chill_torch#overview)
4. [ChillModel](https://github.com/terryhn1/chill_torch#chillmodel)
5. [ChillVisualizer](https://github.com/terryhn1/chill_torch#chillvisualizer)
6. [Sample Workflow](https://github.com/terryhn1/chill_torch#sample-workflow)

## Installation
Use the command below in a Google Colab or Jupyter Notebooks into your main project directory.

When using in terminal:
```bash
git clone https://github.com/terryhn1/chill_torch.git
```

When using in code block:
```bash
!git clone https://github.com/terryhn1/chill_torch.git
```

## Package Requirements
The following are packages/libraries that require installation for usage of this repository:
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

The following are optional packages that are downloaded if you are planning on working with image processing, text processing, or audio processing:
* [Torchvision](https://pytorch.org/vision/stable/index.html)
* [Torchtext](https://pytorch.org/text/stable/index.html)
* [Torchaudio](https://pytorch.org/audio/stable/index.html)

Package Download Commands using 'pip install':
```bash
pip install torch lightning torchvision torchtext torchmetrics pandas scikit-learn matplotlib seaborn numpy 
```

### Overview
ChillTorch is a package meant for intro-level developers or students that want to reduce
the amount of repeatable work and have a fast streamline process from researching to results.
Therefore, ChillTorch offers two main classes to work with: the **ChillModel** and the **ChillVisualizer**,
along with the capability to quickly convert custom data to Torch DataLoaders for training and testing.

The workflow is simple.
1. ***Use ChillTorch's data processing capabilities*** to upload your own custom dataset. Or, simply download Torch's datasets for usage.
2. ***Create your own PyTorch model*** with the appropriate layers in order. Forward function is optional!
3. ***Upload your model to ChillModel*** with a certain problem type and select custom arguments if desired.
4. ***Use its train method***, no parameters needed and no other arguments need to be handled!
5. ***Create a ChillTorch Visualizer instance***, call it's visualize function, and now you can get faster visualizations recommended for your problem.
6. You're done!

With ChillTorch built on Pytorch Lightning, you can reduce the need of these until you feel more confident with PyTorch:
* Hyperparameters - number of epochs, grid searching techniques, learning rate, etc.
* Methodology - ChillTorch is equipped with standardized training/testing procedures already.
* Callbacks - Callbacks are important, but the standard ones have been set for you. You can add more if you like.
* Visualization - one call is all that's required to get the most recommended graphs to visualize your data.

On top of this, by using PyTorch Lightning as the backbone for all models, you can easily customize the code to your needs,
should you want to add more customizable parts to it or to use different models on different problem types. 

### Sample Workflow

1. Create our own dataset from CSV using ChillTorch for a classification problem.
```python
import chill_torch.data_processing.data_loading as dataloading

custom_dataset = dataloading.create_classification_dataset(csv_file = "foo.csv",
                                                           class_header = "score")
dataloaders = dataloading.create_dataloaders(dataset = custom_dataset,
                                             batch_size = 16)
```

2. Create a Pytorch model capable of classification
```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()

        # Can be written in order as multiple attributes
        self.fc1 = torch.nn.Linear(in_features, hidden_units)
        self.relu = torch.nn.ReLU(0.2)
        self.fc2 = torch.nn.Linear(hidden_units, out_features)

        # Or can be written as a sequential layer
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(in_features, hidden_units),
                                          torch.nn.ReLU(0.2),
                                          torch.nn.Linear(hidden_units, out_features))

    def forward(self):
        pass

torch_model = MyModel(2, 8, 1)
```

3. Integrate it into ChillModel and train.
```python
import chill_torch.chill_model.chill_model as cm

model = cm.ChillModel(model = torch_model,
                      train_dataloader = dataloaders["train"],
                      test_dataloader = datalaoders["test"],
                      problem_type = "reg-class"
                      forward_override = False)

model.train()
preds = model.predict(test_dataloader)
```

4. Create a Visualizer instance to now plot
```python
import chill_Torch.chill_visualizer.chill_visualizer as c_visualizer
visualizer = c_visualizer.ChillVisualizer(model)
visualizer.visualize(all_avail_plots = True)
```