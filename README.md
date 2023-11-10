
## Dependencies
PyTorch. Optional, peek in the [requirements.txt](requirements.txt) file.
<!-- 
Our code is entirely in Python, so we provide a [requirements.txt](requirements.txt) file through which using `pip` and optionally `virtualenv`, the python environment used by us can be reproduced. You may check 
[Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
for best practices on this.

`pip install -r requirements.txt` -->


<!-- ## Directory structure  -->

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

<!-- # Supplementary Material

This supplementary material contains the [Appendices](Appendices_asgm_nips.pdf) to support the submitted main-text in the AutoSGM paper and also reproduce the results shown in the paper. 

Here in this [README.md](README.md), we provide some instructions to run the [code](notebooks/)  -->


## Disclaimer
The code implementations and style in this repo. is still undergoing active development



