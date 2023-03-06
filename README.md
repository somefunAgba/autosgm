
# AutoSGM
Official PyTorch implementation of the (Stochastic) Gradient Method, the automatic iterative learning (or control) algorithm that linearly transforms first-order gradients of a well-defined objective function to estimates of parameters (data representations) in a well-defined differentiable function (artificial neural network).

<img src="./lasgm_blkview.svg" width="500">

## What is this library about?
This library implements the *first-order gradient method*, for learning the parameters in a pre-defined (artificial) neural network (any appropriate composition of differntiable functions) that is used to fit a data-set. 

This method automatically arises as the empirical form of the best possible learning algorithm (*minimum bayes risk control function*) in the class of linear control functions. We call it **AutoSGM**.

Standing on the shoulders of about 7 decades of literature, this work attempts to provide a clearer understanding of the three practical accelerated learning variants of the (Stochastic) Gradient Method (SGM), namely: *Polyak's Heavy ball*, *Nesterov Accelerated Gradient*, *Adaptive Moment Estimation* by presenting AutoSGM as their unifying representation for accelerated learning.


## Getting Started (installing or setting up)

- Download or Clone (cloning requires `git` on your machine).
```
>> git clone https://github.com/somefunagba/autosgm.git
```

- Installing necessary python libraries (if not on your machine).

This library uses PyTorch and Numpy, so they should be installed as Python libraries on your machine or virtual environment (if not already present). 
For example, using pip:
```
>> pip install watermark numpy scipy pandas matplotlib torch torchvision 
```

## Folder Structure 
Next, the folders in the root directory are structured as shown below:

```
- root folder
  |-- .vscode\
  |-- arcs\
  |-- cmps_demo\
  |-- data\
  |-- expstore\
  |-- notebks\
  |-- opts\
```
The ` opts ` folder contains the source-code for AutoSGM.

### Demo Scripts
Included in the ``notebks`` directory are some ``demo[x].show.ipynb`` files that, using AutoSGM both help to demonstrate example neural network training with PyTorch.

For instance, to quickly check if your clone of this library is working right, locate ``demo1show.ipynb`` and run a neural network fitting to a toy dataset.

The code in the ``demo[x]show.ipynb`` files, where `x` is an integer, mostly follows PyTorch's recipe and the outline is:

- Load Required Libraries
- Setup Configurations (Hyperparameters and so on)
- Load the Dataset
- Construct Model (Differentiable Function) with  necessary methods.
- Run the Network Modeling (Training)

The ` data ` directory stores raw data which form the basis for the whole learning setup.
The ` arcs ` directory is meant to store any custom neural network architecture.
The `expstore` directory stores training information, graphic plots and saved models, when each `demo[x]show.ipynb` file is run.
The `cmps_demo` directory contains code to compare two or more saved results in the `expstore` directory

Not exhaustive:

- `demo2show.ipynb`.
Trains a custom fully connected network (Attention) on FMNIST data

- `demo4show.ipynb`.
Trains a custom convolutional network (LeNet) on FMNIST data

- `demo5show.ipynb`.
Trains a custom convolutional network (ResNet-6) on FMNIST data 

## API: Quick Use case (how tos?)

> This work is an ongoing research and the function interface might slightly change in the future.

**Load AutoSGM**
```
import opts.asgm.torchlasgm as asgm
...
```

**Call AutoSGM**

Say, you have defined a neural network called `nn_model`, then the quickest and minimal way to use this for training, is to call an instance of the loaded AutoSGM
and pass in the model parameters `nn_model.parameters()` and the number of batches `num_batches` which is the data-size divided by batch-size.
```
optimizer = asgm.PID(nn_model.parameters(),steps_per_epoch=num_batches)
```
Other than the two arguments above, there are other options in the function's interface, but they, most often, rarely need not be changed from their defaults.

For instance, the code snippet below disables auto initializing the effective step-size (learning rate), and uses, instead, a supplied initial learning rate value of `ss_init=1e-3`.
```
optimizer = asgm.PID(nn_model.parameters(),steps_per_epoch=num_batches, ss_init=1e-3, auto_init_ess=False)
```
Also, the code snippet changes the effective step-size  to a fixed value in each epoch, by setting `eps_ss=1`.
```
optimizer = asgm.PID(nn_model.parameters(),steps_per_epoch=num_batches, eps_ss=1.)
```

Other possible options are documented in the source-code for AutoSGM 


## Bugs, Issues, Suggestions or Questions (need help?)
Please, create a new issue or email me at `somefuno@oregonstate.edu`.

> This doc. might slightly change in the future.