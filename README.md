# AutoSGM : A Unifying Framework for Accelerated Learning
Automatic (Stochastic) Gradient Method (SGM) is a framework for stochastic gradient learning that unifies (Polyak's Heavy Ball (*PHB*), Nesterov's Accelerated Gradient (*NAG*), Adaptive Moment Estimation (*Adam*)) used in deep learning. 

It makes sense of the many variants in use today. 

It explains observed acceleration in the SGM as the consequence of lowpass smoothing. This digital framework leads to many implementations, as seen in the deep learning literature. 

It also allows to derive an optimal choice of learning rate.  *Adam* can be seen as one approximation of this optimal choice (which leads to normalized gradients). 

Learning is seen as an interconnection between a gradient-generating system like an artificial neural network (a well-defined differentiable function) with the SGM learning system or control function.

<img src="./asgm_view.svg" width="800">   

This suggests that there is only one (stochastic) gradient method (SGM), with different approaches or metrics to both setting-up the learning rate $\alpha_t$, smoothing the gradient $\mathrm{g}_t$ and smoothing the gradient-generating system parameters $\mathrm{w}_t$ by various lowpass filter implementations $`\mathbb{E}_{t,\beta}\{\cdot\}`$. The result is the different momentum-based SGD variants in the literature.

This repo. contains implementation(s) of AutoSGM: ${\rm w}_t = \mathcal{C}\bigl( {{\rm g}_t} \bigr)$

Expected `input` is a first-order gradient. 
`output` is an estimate of each parameter in an (artificial) neural network. 

$$
\mathrm{g}_t 
$$
<!-- = \mathbb{E}_{t,\beta_i} \bigl[ \mathrm{g}_t \bigr]% {\rm w}_t \leftarrow \mathbb{I}_{t, \alpha_t}\{ {\rm g_t} \}\\
% {\rm w}_t \leftarrow \mathbb{E}_{t,\beta_o}\{{\rm w}_t\} -->
+ a active smoothing (lowpass) component $\mathbb{E}_{t, \beta}$  regularizing the gradient generating system, optionally, at both the input where $\beta= \beta_i$ and the output where $\beta= \beta_o$. 

+ a proportional component $\alpha_t$.

+ a time-integration $\mathbb{I}_{t, \alpha_t}$ component. 

<!-- ### Basic signal-processing and control knowledge:  -->

<!-- + crudely implementing the time-difference operation in *NAG* promotes noise and instability. -->

## Dependencies
Code is entirely in Python, using PyTorch. Peek in the [requirements.txt](requirements.txt) file.
<!-- 
Our code is entirely in Python, so we provide a [requirements.txt](requirements.txt) file through which using `pip` and optionally `virtualenv`, the python environment used by us can be reproduced. You may check 
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for best practices on this.

`pip install -r requirements.txt` -->

## Getting Started (installing)
Download or Clone with `git`.
```
>> git clone https://github.com/somefunagba/autosgm.git
```

<!-- ## Directory structure  -->

## PyTorch API:

**Calling an AutoSGM implementation**

Assume this repo., was directly git cloned to the root path of your project. Load an AutoSGM implementation  and name it `aSGM`. 
```
...
import opts.autosgml as aSGM
...
```
****

Then, say you have constructed a neural network called `mdl` with PyTorch, 
you can call an instance of the loaded `aSGM` by passing in the parameters of the model `mdl.parameters()`, and set other options.
```
optimizer = aSGM(mdl.parameters(), levels=3)
optimizer = aSGM(mdl.parameters(), lr_init=5e-4, spe=len(train_dataloader), restarts=True, movwin=30)
```
`levels` (int, optional): number of learning rates used. Defaults to `1`

`restarts` (bool, optional): use a raised cosine lowpass filter function to shape the gradient (default: False).

`spe` (int, optional): means steps per epoch and refers to the number of batches which is the data-size divided by batch-size. Defaults to `1` if not specified. Helps to detect, when the learning regime has enter a new epoch from a new iteration step. Inactive if `restarts` is `False`.

`movwin` indicates the initial window (in epochs) of a moving raised cosine lowpass filter. Defaults to `1`. The moving window restarts every movwin epoch(s), if `movwin_upfact` is set to 1. Inactive if `restarts` is `False`.

More possible options are documented in [opts/autosgml](opts/autosgml.py). 
Some of the options, might likely need not be changed from the defaults.

For instance, to run *Adam* (an approximation of the optimal step-size), the code snippet below disables the iteration dependent learning rate for each parameter and uses a single initial constant learning rate value of `lr_init=5e-4` .
```
optimizer = AutoSGM(mdl.parameters(), lr_init=5e-4, autolr=False)

```
For instance, the code snippet below disables the optimal step-size approximation for each parameter and uses a single initial constant learning rate value of `lr_init=5e-4`.
```
optimizer = AutoSGM(mdl.parameters(), lr_init=5e-4, autolr=None)
```
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
The code implementations and style in this repository is still undergoing active development as part of my PhD work. Feel free to raise an issue, if you detect any bug or you have any questions.



