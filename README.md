# AutoSGM : A Unifying Framework for Accelerated Learning
Automatic (Stochastic) Gradient Method (SGM) is a framework for stochastic gradient learning that unifies (Polyak's Heavy Ball (*PHB*), Nesterov's Accelerated Gradient (*NAG*), Adaptive Moment Estimation (*Adam*)) used in deep learning. 

Learning is seen as an interconnection between a gradient-generating system like an artificial neural network (a well-defined differentiable function) with the SGM learning system or control function.

<img src="./asgm_view.svg" width="800">   

This suggests that there is only one (stochastic) gradient method (SGM), with different approaches or metrics to both setting-up the learning rate $\alpha_t$, smoothing the gradient $\mathrm{g}_t$ and smoothing the gradient-generating system parameters $\mathrm{w}_t$ by various lowpass filter implementations $`\mathbb{E}_{t,\beta}\{\cdot\}`$ where $0 \le \beta < 1$. The result is the different momentum-based SGD variants in the literature.

This repo. contains implementation(s) of AutoSGM: ${\rm w}_t = \mathcal{C}\bigl( {{\rm g}_t} \bigr)$

Expected `input` $\mathrm{g}_t$ is a first-order gradient, 
and `output` $\mathrm{w}_t$ is an estimate of each parameter in an (artificial) neural network. 

```math
\begin{align}
\mathrm{g}_t \leftarrow \mathbb{E}_{t,\beta_i}\{ \mathrm{g}_t \}\\
{\rm w}_t \leftarrow \mathbb{I}_{t, \alpha_t}\{ {\rm g_t} \}\\
{\rm w}_t \leftarrow \mathbb{E}_{t,\beta_o}\{{\rm w}_t\}
\end{align}
```
+ a proportional component $\alpha_t$.

+ a time-integration $\mathbb{I}_{t, \alpha_t}$ component. 

+ an optionally active smoothing (lowpass) component $\mathbb{E}_{t, \beta}$  regularizing the gradient generating system, optionally, at both the input where $\beta:= \beta_i$ and the output where $\beta := \beta_o$.

It makes sense of the many variants in use today. 

It explains observed acceleration in the SGM as the consequence of lowpass smoothing. This digital framework leads to many implementations, as seen in the deep learning literature. 

It also allows to derive an optimal choice of learning rate. *Adam* can be seen as one approximation of this optimal choice (normalized gradients). 

<!-- ### Basic signal-processing and control knowledge:  -->

<!-- + crudely implementing the time-difference operation in *NAG* promotes noise and instability. -->

## Dependencies
Code is entirely in Python, using PyTorch. 
<!-- Peek in the [requirements.txt](requirements.txt) file. -->
<!-- 
Our code is entirely in Python, so we provide a [requirements.txt](requirements.txt) file through which using `pip` and optionally `virtualenv`, the python environment used by us can be reproduced. You may check 
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for best practices on this.

`pip install -r requirements.txt` -->

## Getting Started (installing)
Download or clone locally with `git`.
```bash
>> git clone https://github.com/somefunagba/autosgm.git
```

<!-- ## Directory structure  -->

## PyTorch API:

### Calling the implementation

Assume this repository was directly git cloned to the root path of your project. 
```python
from opts.autosgml import AutoSGM
```
This loads an AutoSGM implementation.
****
### Examples
Some examples from the [PyTorch Examples Repo.](https://github.com/pytorch/examples) have been added as demo.
See the [cases](cases) folder.

***
Possible options are documented in [opts/autosgml](opts/autosgml.py). Some of the defaults, might likely need not be changed.

Given a neural network model called `mdl` has been constructed with PyTorch.
The following examples illustrate how parameters of the model `mdl.parameters()`may be optimized or learnt with this AutoSGM implementation.

By default, this implementation, auto-tunes an initial learning rate (correlation estimation) with a normalized gradient (variance/moment estimation) iteratively, which in the code snippet below has been set as `lr_init=1e-3`. 
```python
optimizer = AutoSGM(mdl.parameters(), lr_init=1e-4)
```

To use only moment estimation, the code snippet below disables the iteration dependent learning rate function and for all iteration uses a single initial constant learning rate value of `lr_init=1e-3` with a normalized gradient.
```python
optimizer = AutoSGM(mdl.parameters(), autolr=False, lr_init=1e-3)
```

The code snippet below disables estimation of any optimal learning-rate approximation and uses a single initial learning rate constant `lr_init=1e-3`.
```python
optimizer = AutoSGM(mdl.parameters(), lr_init=5e-4, autolr=None)
```

Also, important parameters to configure apart from the initial learning rate are the 4 `lowpass` (often called momentum) parameters in `beta_cfg`, which in order are for iteratively smoothing the gradient input, smoothing the weight output, estimating the gradient variance/moment, crudely approximating a learning-rate correlation. 

By `smoothing`, we mean the `lowpass` filter is used to carefully filter high frequency noise components from its input signal. By `avergaing`, we mean the `lowpass` filter is used to estimate a statistical expectation function. 

By default, the values in `beta_cfg` are sensible theoretical values, which should be changed depending on what works and the linearity/architecture of the neural network.
```python
optimizer = AutoSGM(mdl.parameters(), lr_init=1e-4, 
beta_cfg=(0.9,0.1,0.999,0.9999))
```
Note that when using the first-order `lowpass` filter: For `smoothing`, the lowpass parameter is often less or equal to `0.9` but for averaging, the lowpass parameter is often greater than `0.9`. 




<!-- self.sgm = AutoSGM(self.parameters(), 
                lr_init=cfgs["ss_init"], 
                spe=steps_per_epoch,epfreq=cfgs['epfreq'], 
                beta_d=cfgs['betad'], beta_o=cfgs['betao'],eps=cfgs['eps'],
                weight_decay=cfgs['weight_decay'], join_wdec=cfgs['joinwdecay'],
                auto_mode=cfgs["star_mode"],
                autolr=cfgs["auto_ss"], 
                foreach=cfgs['foreach'], fused=cfgs['fused'], lrlogstep=cfgs['lrlogstep'], down=cfgs['bayesdown']) --> 

<!-- # Supplementary Material

This supplementary material contains the [Appendices](Appendices_asgm_nips.pdf) to support the submitted main-text in the AutoSGM paper and also reproduce the results shown in the paper. 

Here in this [README.md](README.md), we provide some instructions to run the [code](notebooks/)  -->


## Disclaimer
The `code` and `style` in this repository is still undergoing `active` development as part of my `PhD` work. Feel free to raise an `issue`, if you detect any `bug` or you have any questions.



