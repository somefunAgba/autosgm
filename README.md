# Supplementary Material

This supplementary material contains the [Appendices](Appendices_asgm_nips.pdf) to support the submitted main-text in the AutoSGM paper and also reproduce the results shown in the paper. 

Here in this [README.md](README.md), we provide some instructions to run the [code](notebooks/) 

## Installing dependencies

Our code is entirely in Python, so we provide a [requirements.txt](requirements.txt) file through which using `pip` and optionally `virtualenv`, the python environment used by us can be reproduced. You may check 
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for best practices on this.

`pip install -r requirements.txt`


## Directory structure 
The folders in the root directory are structured as shown below:
```
- root folder
  |-- arcs\
  |-- data\
  |-- expstore\
  |-- notebks\
  |-- opts\
  |-- results\
```
The `opts` folder contains the source-code [asgm.py](opts/asgm.py) for the implemented AutoSGM algorithm.

In the ``notebks`` directory we include scripts and jupyter notebooks that aid in using [asgm.py](opts/asgm.py) for training and evaluating a neural network with PyTorch. The `results` directory contains already saved plots obtained from running the files in the `notebks` directory. 

- [demo2nips.ipynb](notebks/demo2nips.ipynb)
Trains a custom fully connected neural network (Attention) on FMNIST data

- [demo4nips.ipynb](notebks/demo4nips.ipynb)
Trains a convolutional neural network (LeNet) on FMNIST data

- [demoxnips.ipynb](notebks/demoxnips.ipynb)
Trains a custom (5-layer) neural network on CIFAR10 data 

- [demo10nips.ipynb](notebks/demo10nips.ipynb)
Trains a convolutional neural network (ResNet-18) on CIFAR10 data 

- [demoxnips_lrdist.ipynb](notebks/demoxnips_lrdist.ipynb)
Trains a custom (5-layer) neural network on CIFAR10 data and plots the learning rate distribution per training iteration 

- [nnrosbrk_lrdist.ipynb](notebks/nnrosbrk_lrdist.ipynb)
Trains a Rosenbrock test function and plots the learning rate distribution per iteration.

The code in the files, mostly follows PyTorch's recipe and the outline is:
- Load Required Libraries
- Setup Configurations (Hyperparameters and so on)
- Load the Dataset
- Construct Model (Differentiable Function) with  necessary methods.
- Train and Evaluate the Network Model

The `data` directory stores raw data which form the basis for the whole learning setup.
The `arcs` directory contains a [resnet.py](arcs/resnet.py) source code.
The `expstore` directory stores training information, graphic plots and saved models obtained from running a jupyter notebook file in the `notebks` directory.

**Training and Evaluation code**
There is a common script [demofcns.py](notebks/demofcns.py) called by all other files in `notebks`. These files serve for both training and evaluating the experimental results shown in the paper. 

We also include a [savedmdls](expstore/savedmdls) directory which contains already trained models obtained from running [demoxnips.ipynb](notebks/demoxnips.ipynb).


## API: Quick Use case

**Load AutoSGM**
```
import opts.asgm as AutoSGM
...
```

**Call AutoSGM**

Say, you have defined a neural network called `nn_model`, then the quickest and minimal way to use this for training, is to call an instance of the loaded AutoSGM
and pass in the model parameters `nn_model.parameters()` and enable `cuda` if desired.
```
optimizer = AutoSGM(nn_model.parameters(), usecuda=True)
```
More possible options are documented in [asgm.py](opts/asgm.py). Most often, the other options rarely need not be changed from their defaults.
For instance, the code snippet below disables the automatic parameter iteration dependent learning rate and uses a constant learning rate value of `lr_init=1e-2` which is the same as running Adam.
```
optimizer = AutoSGM(nn_model.parameters(), lr_init=1e-2, auto=False)
```
You could also renable the automatic iteration dependent learning rate function and optionally start it with an initial value of `lr_init=1e-2`.
```
optimizer = AutoSGM(nn_model.parameters(), lr_init=1e-2)
```

## Contributing

If you'd like to contribute, or have any suggestions, contact the authors.
