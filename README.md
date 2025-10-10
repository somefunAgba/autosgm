# AutoSGM : A Unifying Framework for Accelerated Learning
Automatic (Stochastic) Gradient Method (SGM) is a framework for stochastic gradient learning that unifies the three popular momentum-based algorithms: (Polyak's Heavy Ball (*PHB*), Nesterov's Accelerated Gradient (*NAG*), Adaptive Moment Estimation (*Adam*)) used in deep learning. 

Such learning process can be viewed as an interconnection between the gradient-generating system and the learning algorithm, *SGM*.

Take a look here → [AutoSGM](https://somefunagba.github.io/asgm)

<img src="./asgm_basic_blk.png" width="700">   


<!-- ### Basic signal-processing and control knowledge:  -->

<!-- + crudely implementing the time-difference operation in *NAG* promotes noise and instability. -->

## Dependencies
Code is entirely in Python, using PyTorch. 

```python
from asgm import AutoSGM
```


<!-- # Supplementary Material

This supplementary material contains the [Appendices](Appendices_asgm_nips.pdf) to support the submitted main-text in the AutoSGM paper and also reproduce the results shown in the paper. 

Here in this [README.md](README.md), we provide some instructions to run the [code](notebooks/)  -->


## Note
The `code` and `style` in this repository is still undergoing `active` development as part of my `PhD` work. Feel free to raise an `issue`, if you detect any `bug` or you have any questions.



