r"""PID:AutoSGM"""


import math
import numpy as np

try:
    from opts.elpf.torchfo import LPF
    from opts.elpf.torchesfo import esLPF
except:
    from elpf.torchfo import LPF
    from elpf.torchesfo import esLPF

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import _functional as Fcn
from torch import Tensor
from typing import List, Union, Optional

class AGM():
    ''' Gradient Method: automatic algorithm for learning/control/estimation

    Core algorithm implementation (Torch)
    '''
    def __init__(self, mlpf:LPF,vlpf:LPF,wlpf:LPF,esslpf:esLPF, 
                use_optim_choice:bool, smooth_out:bool, maximize:bool, weight_decay:Tensor,eps:Tensor,betas:tuple,ss_init:Tensor,ss_end:Tensor) -> None:
        self.gradref = 0
        self.mlpf = mlpf
        self.vlpf = vlpf
        self.wlpf = wlpf
        self.esslpf = esslpf
        self.use_optim_choice = use_optim_choice
        self.smooth_out = smooth_out
        self.maximize = maximize
        self.weight_decay = weight_decay
        self.eps = eps
        self.beta_o = betas[2]
        self.beta_n = betas[1]
        self.beta_i = betas[0]
        self.beta_ss = betas[3]
        self.ss_init = ss_init
        self.ss_end = ss_end 
        pass
    
    @torch.no_grad()
    def compute(self,step:Tensor|int, step_c:Tensor|int, param:Tensor,       param_grad:Tensor,mk:Tensor,vk:Tensor,qk:Tensor,wk:Tensor,ss_k:Tensor, optparam:Tensor=None):
        
        # -1. input grad.
        # grad = gradi + (weight_decay*param)
        grad = (param_grad.add(self.weight_decay*wk))
            
        error = grad.neg_() # (self.grad_ref - grad) = -grad
        if self.maximize:
            error.neg_()
            
        if self.use_optim_choice:
            bmo = 0
        else:
            bmo = 10
            
        # E[g]: input mean estimation
        errmean, mk = self.mlpf.compute(u=error, x=mk, 
                        beta=self.beta_i, bmode=bmo, step=step)
            
        # - placeholder: change in parameter (weight)
        delta_param = errmean.add_(0)
                
        # -2. linear step-size fcn.
        if optparam is not None:
            if self.smooth_out:
                alphap = torch.abs(qk-optparam).div(torch.abs(errmean)+self.eps)
            else:
                alphap = torch.abs(wk-optparam).div(torch.abs(errmean)+self.eps)
            delta_param.mul_(alphap)
        else:
            if self.use_optim_choice:        
                # derivative part: variance estimation / normalization
                # inner product: <g,g> = E[g^2] 
                # gradnorm = grad.norm("fro")
                
                errsq =  error.mul(error.conj())
                errvar, vk = self.vlpf.compute(u=errsq, x=vk, beta=self.beta_n, bmode=10, step=step)
                gradnorm_den = (errvar.sqrt().add_(self.eps))
                

                ss_k = self.esslpf.compute(
                    u=self.ss_end,x_init=self.ss_init,noise_k=0,beta=self.beta_ss,step=step_c
                    )
                
                delta_param.div_(gradnorm_den) #.mul(ss_k)
                delta_param.mul_(ss_k)
                
                alphap = (ss_k).div(gradnorm_den) 
                # delta_param.mul_(alphap) 
                
                # ss_k = self.ss_init
                # delta_param.div_( 
                    # (errvar.sqrt() / (ss_k)).add_(self.eps / (ss_k)) 
                    # )
            else: 
                # no variance estimation added to alpha_p.
                alphap = ss_k 
                delta_param.mul_(alphap)      
                # kpk = alphap; delta_param = errmean.mul_(kpk)
            
        # -3. update
        if self.smooth_out:
            # update
            wk.add_(delta_param)
            # E[w] output parameter smoothing
            paramf, qk = self.wlpf.compute(u=wk, x=qk,
                    beta=self.beta_o, bmode=0,  step=step)
            # pass values to network's parameter placeholder.
            param.copy_(paramf)

        else:
            # assume param is smooth
            # update
            wk.add_(delta_param)
            # pass values to network's parameter placeholder.
            param.copy_(wk)
            
        # END
        
        return alphap        
        
        
class PID(Optimizer):

    """Implements: "Stochastic" Gradient Method (Torch), an automatic learning algorithm.
    
    The SGM is a discrete-time PID structure, with lowpass regularizing components.
    
    PID structure is a discrete-time structure with Proportional + Integral + Derivative components
    
    The proportional gain component is the effective step-size in the optimal "step-size" (or learning rate) which represents a linear correlation variable.
    
    The derivative gain component is the variance estimation (or normalizing component) in the SGM's optimal "step-size".
    
    The integral component is the additive means of disctrete-time parameter adaptation using the step-size and first-order gradient of a scalar-valued objective-function.
    
    Author: Oluwasegun Ayokunle Somefun. somefuno@oregonsate.edu 
        
    Affilation: Information Processing Group, Orgeon State University
        
    Date: (Changes)

        2022. Nov. (added parameter (weight) adaptation)
        
        2023. Jan. (added effective step-size variation)

    Args:
        params(iterable, required): iterable of parameters to optimize or dicts defining parameter groups

        steps_per_epoch (int, required): per batch iterations >= 1 (default: 1)

        ss_init (float, required): starting eff. proportional gain or step-size (default=1e-3): (0, 1), 
                
        ss_end(float, optional):  ending eff. proportional gain or step-size (default=0): (0, 1), 
        
        eps_ss(float, optional): accuracy of final step_size in an epoch with respect to ss_end (default = 0.5): (0, 1), 

        beta_i (float, optional): input mean est. lowpass filter pole (default = 0.9): (0, 1), 
        
        beta_n (float, optional): input variance est. lowpass filter pole (default = 0.999): (0, 1),         
        
        beta_o (float, optional): output smoothing. lowpass filter pole (default = 1e-4): (0, 1), 

        weight_decay (float, optional): weight decay (L2-norm penalty) (default: 0),

        use_optim_choice (bool, optional): use optimal step-size choice by doing variance estimation (default=True),
        
        smooth_out (bool, optional): smooth output (default=True),

        maximize (bool, optional): maximize the params based on the objective, 
        instead of minimizing (default: False)
        
        optparams (optional): optimum param, if known, else default=None


        .. AutoSGM: Automatic (Stochastic) Gradient Method _somefuno@oregonstate.edu

    Example:
        >>> import opts.asgm.torchlasgm as asgm
        >>> ...
        
        >>> optimizer = asgm.PID(model.parameters(), ss_init=1e-3,steps_per_epoch=100)
        
        >>> optimizer = asgm.PID(model.parameters(), ss_init=1e-3, ss_end=1e-5, betai=1e-1, steps_per_epoch=100, weight_decay=0)
        
        >>> ...
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step
    """
    # 
    def __init__(self, params, *, steps_per_epoch=1, ss_init=1e-3, ss_end=0, eps_ss = 5e-1, beta_i=0.9, beta_n=0.999, beta_o=1e-5, weight_decay=1e-5, use_optim_choice=True, smooth_out=True, maximize=False, optparams=None):

        if not 0.0 < ss_init:
            raise ValueError(f"Invalid value: rho={ss_init} must be in (0,1) for tuning")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        
        if maximize:
            weight_decay = -weight_decay

        self.use_optim_choice = use_optim_choice
        self.debug = True
        
        # -pick computation device
        devcnt = torch.cuda.device_count()
        if devcnt > 0:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # eps added: div. by zero.
        self.eps = torch.tensor(1e-8, dtype=torch.float, device=self.device)
        
        # step-size.
        self.ss_init = torch.tensor(ss_init, dtype=torch.float, device=self.device)
        self.ss_end = torch.tensor(ss_end, dtype=torch.float, device=self.device)
        # error in the convergence accuracy to ss_end.
        eps_ss = torch.tensor(eps_ss, dtype=torch.float, device=self.device)
        # steps
        self.beta_ss = torch.exp(torch.divide(torch.log(eps_ss),steps_per_epoch))
        
        # init. LPFs
        self.esslpf = esLPF(cdevice=self.device)
        self.mlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.vlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.wlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        
        # compute: one-pole filter gain
        self.betai = torch.tensor(beta_i, dtype=torch.float, device=self.device)
        self.betan = torch.tensor(beta_n, dtype=torch.float, device=self.device)
        self.betao = torch.tensor(beta_o, dtype=torch.float, device=self.device)
        # print.
        if self.debug:
            strlog = f"aSGM info:\n[step-size or linear correlation value (starting := {ss_init:.4g}), (ending := {ss_end:.4g}), step-size pole : {self.beta_ss:.4g}],\n[LPF Poles [i,n,o] : {self.betai:.4g}, {self.betan:.4g}, {self.betao:.4g}]"
            txt = "*"
            print(f"{txt * len(strlog)}")    
            print(strlog)
            print(f"{txt * len(strlog)}\n")
            
        defaults = dict(ss_init=self.ss_init, ss_end=self.ss_end, betas=(self.betai,self.betan,self.betao,self.beta_ss), weight_decay=weight_decay,steps_per_epoch=steps_per_epoch, smooth_out= smooth_out, maximize=maximize)
        
        # if ref/opt/desired parameters are known.
        if isinstance(optparams, torch.Tensor):
            raise TypeError("optparams argument given to the optimizer should be an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        #TODO: can this be resolved using pytorch's base optimizer.py?
        
        if optparams is not None:
            self.optparams = Optimizer(optparams,defaults)
        else:
            self.optparams = None

        super(PID, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(PID, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates 
                the model grad and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                # loss = closure()
                loss = closure
                
        if self.optparams is not None:
            optparams_with_grad = []
            for group in self.optparams.param_groups:
                for optp in group['params']:
                    optparams_with_grad.append(optp)
        else:
            optparams_with_grad = None

        for group in self.param_groups:
            
            asgm = AGM(self.mlpf,self.vlpf,self.wlpf,self.esslpf,
            self.use_optim_choice,group['smooth_out'],group['maximize'], 
            group['weight_decay'], self.eps, group['betas'], group['ss_init'], group['ss_end']) 
            
            # list of parameters, gradients, gradient notms
            params_with_grad = []
            
            grads = []

            # list to hold step count
            state_steps = []

            # list to hold one-pole filter memory
            mk, vk, qk, wk, ss_k = [], [], [], [], []


            for p in group['params']:

                if p.grad is not None:
                    # get parameter
                    params_with_grad.append(p)
                    
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'This implementation might not support sparse gradients.')

                    # get its gradient
                    grads.append(p.grad)

                    state = self.state[p]
                    # initialize state, if empty
                    if len(state) == 0:
                        # first step count
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=self.device)

                        # one-pole filter memory (filter)
                        state['m'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)

                        state['v'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)

                        state['q'] = p.clone(memory_format=torch.preserve_format).detach() 
                        #torch.zeros_like(p, memory_format=torch.preserve_format, device=self.device)
                        
                        state['w'] = p.clone(memory_format=torch.preserve_format).detach()
                        
                        state['ss'] = group['ss_init']*torch.ones_like(
                            p, memory_format=torch.preserve_format, device=self.device) 
                        # group['ss_init']*torch.ones((1,), device=self.device)

                    mk.append(state['m'])
                    vk.append(state['v'])
                    qk.append(state['q'])
                    wk.append(state['w'])
                    ss_k.append(state['ss'])

                    # update the step count by 1.
                    state['step'] += 1

                    # record the step
                    state_steps.append(state['step'])

            # Actual Learning Event: 
            alphaps = control_event(asgm, loss,
                params_with_grad, grads, mk, vk, qk, wk, ss_k, group['steps_per_epoch'], state_steps, optparams_with_grad
            )

        return loss, state['step']


# Functional Interface
@torch.no_grad()
def control_event(asgm: AGM, loss:Tensor, 
            params: List[Tensor], grads: List[Tensor], 
            mk: List[Tensor], vk: List[Tensor], qk: List[Tensor], wk: List[Tensor], ss_k: List[Tensor], steps_per_epoch:int, state_steps: List[Tensor], optparams: List[Tensor] | None)->Tensor:
    r"""Functional API that computes the AutoSGM control/learning algorithm for each parameter in the model.

    See : [in future] class:`~torch.optim.AutoSGM` for details.
    """

    step = state_steps[0]

    # cyclic step in each epoch
    step_c = (((step-1) % steps_per_epoch) + 1)
    
    # - At each step, adapt parameters (weights of the model) using 
    # (AutoSGM) PID structure.
    # Et[param] = Et[param + alphap*Et[grad] ] = Et[param] + alphap*Et[grad]  
    alphaps = []
    # for each parameter in the model.
    
    # uniform initial effective step-size (rough linear correlation estimation).
    if step == 1:
        # calc.
        sum_params, param_numels = [],[]
        for i, param in enumerate(params):
            sum_params.append(param.square().sum())
            param_numels.append(param.nelement())
        pvec_norm_2 = torch.tensor(sum_params).sum().sqrt()
        dparam = torch.tensor(param_numels).sum()
        ss_init_calc=(pvec_norm_2/dparam)
        # cond. use
        if (ss_init_calc < 1) and (ss_init_calc < asgm.ss_init): asgm.ss_init.add_(ss_init_calc)
        if (ss_init_calc < 1) and (ss_init_calc > asgm.ss_init): asgm.ss_init.copy_(ss_init_calc)
        # debug
        if step == 1: print(f"ss0: {asgm.ss_init:.6f}")
    
    for i, param in enumerate(params):
        if optparams is not None:
            optparam = optparams[i]
        else:
            optparam = None
        alphap = asgm.compute(step,step_c,param,grads[i],mk[i],vk[i],qk[i],wk[i],ss_k[i],optparam)
        alphaps.append(alphap)
        
    return alphaps

    