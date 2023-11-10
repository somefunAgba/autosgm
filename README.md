# AutoSGM
AutoSGM: Automatic (Stochastic) Gradient Method. ```output = AutoSGM{input}```

Expected `input` is a first-order gradient. 
`output` is an estimate of each parameter in an (artificial) neural network. 

<img src="./lasgm_blkview.svg" width="500">   

```
  input <- Et{Dt{-g}} or Dt{Et{-g}}
  state <-  It{state,input,alpha_t} := state + alpha_t*input
  output <- Et{state}
```

## Unified Framework  
From an automatic control perspective, AutoSGM is an accelerated learning framework that has: 

+ an active lowpass filtering component 'Et' regularizing its input. 

+ optional time-derivative 'Dt' component. 

+ a proportional component 'alpha_t'.

+ a time-integral 'It' component. 

+ optional lowpass filtering component 'Et' at its output.

### Basic signal-processing and control knowledge: 

+ the time-derivative component is most always sensitive to input noise, so should usually be turned off.

+ the lowpass filtering 'Et' component at the output often adds unnecessary delay to the output estimates, so should usually be turned off.

## Dependencies
PyTorch. Peek in the [requirements.txt](requirements.txt) file.
<!-- 
Our code is entirely in Python, so we provide a [requirements.txt](requirements.txt) file through which using `pip` and optionally `virtualenv`, the python environment used by us can be reproduced. You may check 
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for best practices on this.

`pip install -r requirements.txt` -->


<!-- ## Directory structure  -->

## PyTorch API:

**Calling AutoSGM**

Load AutoSGM package. (This assumes this library, was cloned to an `opts`` directory on the root path of your project.)
```
...
import opts.asgm as AutoSGM
...
```
****

Using PyTorch, you have constructed a neural network called `mmn`, 
you can then: call an instance of the loaded AutoSGM, pass in the parameters of the model `mmn.parameters()`, and set other options.
```
optimizer = AutoSGM(mnn.parameters(), spe=len(train_dataloader))
optimizer = AutoSGM(mnn.parameters(), spe=len(train_dataloader), epfreq=10)
```
`spe` means steps per epoch and refers to the number of batches which is the data-size divided by batch-size. Defaults to `1` if not specified. Helps to detect, when the learning regime has enter a new epoch from a new iteration step.

`epfreq` indicates the successive per-epoch frequency of restarting the lowpass filter generating the learning rates. Defaults to `1`, restarting every epoch, if not specified.

More possible options are documented in each AutoSGM implementations in the [opts](opts/) directory. 
Most options rarely need not be changed from their defaults.

For instance, the code snippet below disables the automatic parameter iteration dependent learning rate and uses a constant learning rate value of `lr_init=2e-3` which is the same as running Adam.
```
optimizer = AutoSGM(mnn.parameters(), lr_init=2e-3, autolr=False)
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
The code implementations and style in this repo. is still undergoing active development



