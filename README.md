# AutoSGM : A Unifying Framework for Accelerated Learning
Automatic (Stochastic) Gradient Method (SGM) is a framework for stochastic gradient learning that unifies the three popular momentum-based algorithms: (Polyak's Heavy Ball (*PHB*), Nesterov's Accelerated Gradient (*NAG*), Adaptive Moment Estimation (*Adam*)) used in deep learning. 

<img src="./asgm_basic_blk.png" width="700">   

To learn more, take a look here → [AutoSGM](https://somefunagba.github.io/asgm)


## Examples

### 1. GPT-2 on Shakespeare-char  


<img src="assets/tt_grid2x4_asgmpaper_ngpt_shake_30M_extralrc1_cmps_asgm_runs=10_blksz=128_bsize=64.png" width="500" /> 

### 2. VIT on CIFAR10.  


<img src="./assets/tr_grid2x4_svit_cifar10_cmps_asgm_runs=5_bsize=100_spe=500_lr=0.001.png" width="500" />   <img src="./assets/tt_grid2x4_svit_cifar10_cmps_asgm_runs=5_bsize=100_spe=500_lr=0.001.png" width="500" /> 


### 3. ResNet18 on CIFAR10.  


<img src="./assets/tr_grid2x4_asgmpaper_resnet18_cifar10_cmps_asgm_runs=5_bsize=128_iters=spe_391_x_200.png" width="500" />     
<img src="./assets/tt_grid2x4_asgmpaper_resnet18_cifar10_cmps_asgm_runs=5_bsize=128_iters=spe_391_x_200.png" width="500" /> 


### 4. GPT-2 on WikiText-103.  

<img src="./assets/tt_grid2x4_asgmpaper_ngpt_wiki_124M_HALF_cmps_asgm_runs=3_blksz=128_bsize=64.png" width="500" /> 



<!-- ### Basic signal-processing and control knowledge:  -->

<!-- + crudely implementing the time-difference operation in *NAG* promotes noise and instability. -->


<!-- # Supplementary Material

This supplementary material contains the [Appendices](Appendices_asgm_nips.pdf) to support the submitted main-text in the AutoSGM paper and also reproduce the results shown in the paper. 

Here in this [README.md](README.md), we provide some instructions to run the [code](notebooks/)  -->


## Disclaimer
The `code` and `style` in this repository is still undergoing `active` development as part of my `PhD` work. Feel free to raise an `issue`, if you detect any `bug` or you have any questions.

## Minimal example — Using AutoSGM
This section shows a minimal, easy-to-follow example of using the AutoSGM implementation with a PyTorch model and lists the most important configuration options.

```python
from asgm import AutoSGM
import torch

# example model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 1)
)

num_epochs = 1
iters_epoch = 100000

# Example: spawn an AutoSGM instance, with cosine annealing.
opt = AutoSGM(
    model.parameters(),
    lr_cfg=(True, 1e-3, 3),            # setup learning-rate (lr) algorithm
    beta_cfg=(0.9999, 0.999, 0.9, 0, 0, True), # setup lowpass filtering (grad.), and averaging (lr)
    rc_cfg=(1, 0, 0, 2, 1, num_epochs, iters_epoch, 1, 0), # setup window (lr schedule)
    wd_cfg=(0.0, 0),                  # setup weight decay
    eps_cfg=(1e-10, True),            # setup numerical eps
)

# update parameters in training loop
opt.step()
opt.zero_grad()
```

<!-- Notes on common config tuples
- lr_cfg = (aoptlr, lr_init, num_lrc)
  - aoptlr: bool. If True, AutoSGM uses an iteration-dependent optimal lr (moment estimation + partial-correlation estimation). If False, lr_init is used as a constant learning rate.
  - lr_init: float. trust-region constant used by the iteration-dependent learning rate variants when aoptlr=True or the constant lr when aoptlr=False.
  - num_lrc: int. Variant selector for the partial-correlation numerator estimator (0 for a fixed estimate, but 2 and 3 gives iteration-dependent estimates ).

- beta_cfg = (beta_n, beta_a, beta_i, gamma_i, eta_i, debias)
  - beta_n, beta_a, beta_i: smoothing pole constants for internal EMAs and filters.
  - gamma_i: zero/predictive term for the smoothing filter (0 for HB, appropriate value for NAG-like behavior).
  - eta_i: input normalization for LPF (if 0, it's set automatically to 1-beta).
  - debias: bool. If True, debiased outputs are produced by the filters.

- rc_cfg = (rcm, inseq, x, n, m, tau, spe, cfact, e)
  - rcm: window mode (0=inactive, 1=raised-cosine, 2=tri/linear, 3=beta-exp, 4=simple-poly, 5=logistic, 6=other sigmoid).
  - inseq: input sequence type (0=uniform/rectangular, 1=kronecker/randomized ordering).
  - x: minimum fraction of function max (fmin = x * fmax).
  - n: shape parameter used by many window shapes (order, decay rate, etc.).
  - m: half/full mode (1=half/anneal-only, 0=full-window).
  - tau: number of epochs (or used in computing window length depending on cfact).
  - spe: steps per epoch (iterations per epoch).
  - cfact: step unit (0=epoch, 1=iteration, 2=sub-iteration).
  - e: coverage fraction (flat-top), 0 <= e < 1.

- wd_cfg = (wd_cte, wd_lvl)
  - wd_cte: weight-decay (L2) constant.
  - wd_lvl: decoupling level (0 uses classic parameter-level decay; 1 decouples weight decay from smoothing).

- eps_cfg = (eps, repeat_eps)
  - eps: small positive value added for numerical stability.
  - repeat_eps: bool controlling whether eps is applied once or twice in normalization.

Minimal tips
- Starting point: enable aoptlr=True to use an iteration-dependent learning-rate and try out three different variants of the learning rate algorithm (0,2,3).
- Enable rc_cfg windowing (rcm = 1); set tau and spe to match your number of training epochs and total training iterations per epoch; choose cfact according to whether you want the window to operate per-epoch or per-iteration.

Where to look next
- Inspect opts/asgm.py for details of the window functions (WINF, WINF_II) and the different numerator estimators (lrc_2, lrc_3) if you need to customize behavior.

Questions / issues
If something doesn't behave as you expect, open an issue in the repository with a minimal repro (model, optimizer config, and a few training steps). -->



