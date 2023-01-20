
# AutoSGM
A PyTorch implementation of the automatic learning (or control) algorithm.

<img src="./lasgm_blkview.svg" width="500">

## What is this library about?
This library implements the *first-order gradient method*, for learning the parameters in a pre-defined (artificial) neural network (any appropriate composition of differntiable functions) that is used to fit a data-set. 

This method automatically arises as the empirical form of the best possible learning algorithm (*minimum bayes risk control function*) in the class of linear functions. We call it **AutoSGM**.

Standing on the shoulders of about 7 decades of literature, we provide a clearer understanding of the three practical accelerated learning variants of the (Stochastic) Gradient Method (SGM), namely: *Heavy ball*, *Nesterov Accelerated Gradient*, *Adaptive Moment Estimation* by presenting AutoSGM as their unifying representation for accelerated learning.


## Getting Started (installing or setting up)

- Clone the repo.
```
git clone https://github.com/somefunagba/autosgm.git
```

- Installing necessary python libraries (if not on your machine).

This library uses PyTorch and Numpy, so they should be installed as Python libraries on your machine or virtual environment (if not already present).

## Folder Structure 
Next, the folders in the root directory are structured as shown below:

```
- root folder
  |-- arcs\
  |-- data\
  |-- expstore\
  |-- opts\
```

### Demo Scripts
There are included demo files that both help to demonstrate example neural network training with PyTorch, using AutoSGM.

```
|-- demo1show.ipynb
|-- demo2show.ipynb
|-- demo4show.ipynb
|-- demo5show.ipynb
```

In the root folder, locate and run ``demo1show.ipynb`` to check if your clone of this library is working right.

The code outline in the ``demo[x]show.ipynb`` files, where x is an integer, mostly follows PyTorch's recipe and is:

- Load Required Libraries
- Setup Configurations (Hyperparameters and so on)
- Load the Dataset
- Construct Model (Differentiable Function) with  necessary methods.
- Run the Network Modeling (Training)

The ` opts ` folder contains the source-code for AutoSGM.
The ` data ` folder stores raw data which form the basis for the whole learning setup
The ` arcs ` folder is meant to store any custom neural network architecture.
The `expstore` folder stores training information, graphic plots and saved models, when each `demo[x]show.ipynb` file is run

## API: Quick Use case (how tos?)

> Note: This work is an ongoing research and the function interface might slightly change in the future.

```
import opts.asgm.torchlasgm as asgm
```

```
optimizer = asgm.PID(model.parameters(),steps_per_epoch=100)
```



## Bugs, Issues or Questions (need help?)
Please, create a new issue or email me.

> Note also, that this document will be updated gradually.