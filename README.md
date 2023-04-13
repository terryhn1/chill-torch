# ChillTorch
ChillTorch is a lightweight package that allows for streamlining PyTorch projects by
allowing the user to visualize their data and to set up deep learning models in little time,
all while limiting the amount of workflow that is required to get the project up. By reducing
the time it takes to get results up, intro-level PyTorch developers can focus more on
learning about the various models in the industry.

## Installation
Use the command below in a Google Colab or Jupyter Notebooks into your main project directory.

When using in terminal:
'''bash
git clone https://github.com/terryhn1/chill_torch.git
'''

When using in code block:
'''bash
!git clone https://github.com/terryhn1/chill_torch.git
'''

## Package Requirements
The following are packages/libraries that require installation for usage of this repository.
    * PyTorch
    * PyTorch Lightning
    * Torchvision
    * Torchtext
    * Torchaudio
    * Torchmetrics
    * Pandas
    * Scikit-learn
    * Matplotlib
    * Seaborn
    * Numpy

Package Download Commands using 'pip install':
'''bash
pip install torch torch-lightning torchvision torchtext torchmetrics pandas sklearn matplotlib seaborn numpy 
'''

### Workflow
ChillTorch is a package meant for intro-level developers or students that want to reduce
the amount of repeatable work and have a fast streamline process from researching to results.
Therefore, ChillTorch offers two main classes to work with: the **ChillModel** and the **ChillVisualizer**,
along with the capability to quickly convert custom data to Torch DataLoaders for training and testing.

The workflow is simple.
    1a. Use ChillTorch's data processing capabilities to upload your own custom dataset.
    1b. Or, simply download Torch's datasets for usage.
    2. Create your own PyTorch model with the appropriate layers in order. Forward function is optional!
    3. Upload your model to ChillModel with a certain problem type and select custom arguments if desired.
    4. Use its train method, no parameters needed and no other arguments need to be handled!
    4. Create a ChillTorch Visualizer instance, call it's visualize function, and now you can get faster visualizations recommended for your problem.
    5. You're done!

With ChillTorch built on Pytorch Lightning, you can reduce the need of these until you feel more confident with PyTorch:
    * Hyperparameters - number of epochs, grid searching techniques, learning rate, etc.
    * Methodology - ChillTorch is equipped with standardized training/testing procedures already.
    * Callbacks - Callbacks are important, but the standard ones have been set for you. You can add more if you like.
    * Visualization - one call is all that's required to get the most recommended graphs to visualize your data.

On top of this, by using PyTorch Lightning as the backbone for all models, you can easily customize the code to your needs,
should you want to add more customizable parts to it or to use different models on different problem types. 

