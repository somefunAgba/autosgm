r"""PID:AutoSGM"""


import math
import numpy as np


from opts.elpf.torchfo import LPF
from opts.elpf.torchesfo import esLPF

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import _functional as Fcn
from torch import Tensor
from typing import List, Union, Optional

class AGM():
    ''' Gradient Method: automatic algorithm for learning/control/estimation

    Core algorithm implementation (Torch)
    '''
    def __init__(self, wlpf:LPF, 
                mlpf:LPF,vlpf:LPF,esslpf:esLPF,
                use_optim_choice:bool, maximize:bool, 
                eps:Tensor,betas:tuple,
                ss_inits:tuple, ss_end:Tensor, ss_cte:Tensor, auto_init_ess:bool) -> None:
        self.gradref = 0
        self.wlpf = wlpf
        self.mlpf = mlpf
        self.vlpf = vlpf
        self.esslpf = esslpf
        self.use_optim_choice = use_optim_choice
        self.maximize = maximize
        self.eps = eps
        self.beta_o = betas[1]
        self.beta_i = betas[0]
        self.beta_ss = betas[2]
        self.ss_init = ss_inits[0]
        self.ss_base = ss_inits[0]
        self.ss_end = ss_end 
        self.ss_cte = ss_cte
        self.auto_init_ess = auto_init_ess
        pass
    
    @torch.no_grad()
    def init_lin_corr_est(self, params, grads, step):
        if step == 1 and self.auto_init_ess:
            square_params, param_numels = [],[]
            square_grads, prod_pg = [],[]
            
            for i, param in enumerate(params):  
                gradi = ((grads[i]).mul(1))
                errgrad = gradi.neg()
                square_params.append((param.square()).sum())
                #
                square_grads.append(((errgrad.pow(1)).square()).sum())            
                prod_pg.append((param*(errgrad.pow(1))).sum())
                param_numels.append(param.nelement())
                
            numel_param = torch.tensor(param_numels).to(self.eps.device).sum()
            pvec_sq = (torch.tensor(square_params).to(self.eps.device).sum())
            pvec_sq_sqrt = (pvec_sq).sqrt().add(self.eps)
            
            # 1
            gvec_sq = (torch.tensor(square_grads).to(self.eps.device).sum())
            gvec_sq_sqrt = gvec_sq.sqrt().add(self.eps)
            prod_pgvec = (torch.tensor(prod_pg).to(self.eps.device).sum())
            std_pgvec = ((pvec_sq_sqrt)*(gvec_sq_sqrt))
            rho_est =  prod_pgvec/(std_pgvec)
            
            
            if step == 1 and self.auto_init_ess:    
                self.ss_init.copy_(self.ss_cte*rho_est.abs())  
                
                strlog = f"aSGM info:\n[eff. step-size [linear correlation value (starting := {self.ss_init:.4g}), (ending := {self.ss_end:.4g}), step-size pole : {self.beta_ss:.4g}],\n[LPF Poles [i,o] : {self.beta_i:.4g}, {self.beta_o:.4g}]"
                txt = "*"
                print(f"{txt * len(strlog)}")    
                print(strlog)
                
                # debug   
                print(f"total params, d={numel_param.item()}")
                print(f"[ss_init: {self.ss_init:>.6f} <= {rho_est.abs():>.6f} ]")
                
                print(f"{txt * len(strlog)}\n")
                    

    @torch.no_grad()
    def compute_opt(self,step:Tensor|int, step_c:Tensor|int,        
                param:Tensor, param_grad:Tensor,grad_regs:Tensor,qk:Tensor,wk:Tensor,mk:Tensor,vk:Tensor):
        
        # -1. input grad.          
        # - placeholder: change in parameter (weight)            
        # Et[g]: input smoothing, like mean estimation     
        # -2. linear step-size fcn. 
        # derivative part: variance estimation / normalization
        # inner product: <g,g> = Et[g^2]  # gradnorm = grad.norm("fro")
        # -3. update param.
        
        # (grad_ref - grad) = -grad
        error = (param_grad.add(grad_regs)).neg_() 
        if self.maximize: error.neg_()
        errsq = error.square() #error.mul(error.conj())        
        err_smth, mk = self.mlpf.compute(u=error, x=mk, 
                    beta=self.beta_i, step=step)
        
        # IN 1: 
        # - placeholder: change in parameter (weight)
        delta_param = err_smth.pow(1)        
        ss_k = self.esslpf.compute(
            u=self.ss_end,x_init=self.ss_init,noise_k=0,
            beta=self.beta_ss,step=step_c
            )
        ss_k = ss_k.div(1)
        errvar, vk = self.vlpf.compute(u=errsq.pow(1), x=vk, 
                    beta=1-ss_k, step=step)
        errstd = (errvar.sqrt().add_(self.eps))
        delta_param.div_(errstd).mul_(ss_k)
        # update
        wk.add_(delta_param)
  
        # UPDATE
        # E[w] output parameter smoothing
        param_smth, qk = self.wlpf.compute(u=wk, x=qk,
                beta=self.beta_o, step=step)
        # pass values to network's parameter placeholder.
        param.copy_(param_smth)
            
        # END
        alphap = (ss_k).div(errstd) 
        return alphap        
        

    @torch.no_grad()
    def compute_opt_dcp(self,step:Tensor|int, step_c:Tensor|int, param:Tensor,  param_grad:Tensor,grad_regs:Tensor,mk:Tensor,vk:Tensor,qk:Tensor,wk:Tensor,ss_k:Tensor):
        
        # -1. input grad.
        
        # direct error 
        error = param_grad.neg() # (0 - grad) = -grad
        # weight regularization error
        error_reg = grad_regs.neg()
        # total error
        error_comb = error.add(error_reg)
        
        if self.maximize: 
            error.neg_()
            error_comb.neg_()
            error_reg.neg_()
            
        # E[g]: input mean estimation
        errmean, mk = self.mlpf.compute(u=error, x=mk, 
                        beta=self.beta_i, step=step)
            
        # - placeholder: change in parameter (weight)
        delta_param = errmean
        dcp_param = error_reg
                
        # -2. linear step-size fcn.
        # derivative part: variance estimation / normalization
        # inner product: <g,g> = E[g^2] 
        # gradnorm = grad.norm("fro")
        
        errsq =  error_comb.mul(error_comb.conj())
        errvar, vk = self.vlpf.compute(u=errsq, x=vk, beta=self.beta_n, bmode=10, step=step)
        errstd = (errvar.sqrt().add_(self.eps))
        
        ss_k = self.esslpf.compute(
            u=self.ss_end,x_init=self.ss_init,noise_k=0,beta=self.beta_ss,step=step_c
            )
        delta_param.div_(errstd).mul_(ss_k) # delta_param.mul_(ss_k)
        dcp_param.div_(errstd).mul_(ss_k)
        
        alphap = (ss_k).div(errstd) 
        # delta_param.mul_(alphap) 
        
            
        # -3. update
        if self.smooth_out:
            # update
            wk.add_(dcp_param)
            wk.add_(delta_param)
            # E[w] output parameter smoothing
            paramf, qk = self.wlpf.compute(u=wk, x=qk,
                    beta=self.beta_o, bmode=0,  step=step)
            # pass values to network's parameter placeholder.
            param.copy_(paramf)
        else:
            # assume param is smooth
            # update
            wk.add_(dcp_param)
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

        2022. Nov. (added (weight) parameter adaptation)
        
        2023. Jan. (added effective step-size auto init. and variation)

    Args:
        params(iterable, required): iterable of parameters to optimize or dicts defining parameter groups

        steps_per_epoch (int, required): per batch iterations >= 1 (default: 1)

        ss_init (float, required): starting eff. proportional gain or step-size (default=1e-3): (0, 1), 
                
        ss_end (float, optional):  ending eff. proportional gain or step-size (default=0): (0, 1), 
        
        eps_ss (float, optional): accuracy of final step_size in an epoch with respect to ss_end (default = 0.5): (0, 1), 
        
        nfl_cte (float, optional): no-free-lunch constant, how much do you trust the initial eff. step-size. (default = 0.25): (0, 1),

        beta_i (float, optional): input mean est. lowpass filter pole (default = 0.9): (0, 1), 
        
        beta_n (float, optional): input variance est. lowpass filter pole (default = 1-ss_k): (0, 1), auto-computed        
        
        weight_decay (or L2-norm penalty) or beta_o (float, optional): output smoothing. lowpass filter pole (default = 1e-5): (0, 1), 

        use_optim_choice (bool, optional): use optimal step-size choice by doing variance estimation (default=True),
        
        maximize (bool, optional): maximize the params based on the objective, 
        instead of minimizing (default: False)
        
        auto_init_ess, auto (bool, optional, default: True)
        
        optparams (optional): optimum param, if known, else default=None


        .. AutoSGM: Automatic (Stochastic) Gradient Method _somefuno@oregonstate.edu

    Example:
        >>> import opts.asgm.torchlasgm as asgm
        >>> ...
        
        >>> optimizer = asgm.PID(model.parameters(), steps_per_epoch=100, weight_decay=1e-5)
        
        >>> optimizer = asgm.PID(model.parameters(), ss_end=1e-5, beta_i=1e-1, steps_per_epoch=100, weight_decay=0)
        
        >>> ...
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step
    """
    # 
    def __init__(self, params, *, steps_per_epoch=1, ss_init=1e-3, ss_end=0, eps_ss = 5e-1, nfl_cte = 2.5e-1, beta_i=0.9, weight_decay=1e-5, use_optim_choice=True, maximize=False, auto_init_ess=True, optparams=None):

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
        
        self.weight_decay = torch.tensor(weight_decay, dtype=torch.float, device=self.device)        
        
        # eps added: div. by zero.
        self.eps = torch.tensor(1e-8, dtype=torch.float, device=self.device)
        
        # step-size.
        self.ss_init = torch.tensor(ss_init, dtype=torch.float, device=self.device)
        self.ss_end = torch.tensor(ss_end, dtype=torch.float, device=self.device)
        self.ss_cte = torch.tensor(nfl_cte, dtype=torch.float, device=self.device)
        self.auto_init_ess = auto_init_ess
        
        # error in the convergence accuracy to ss_end.
        eps_ss = torch.tensor(eps_ss, dtype=torch.float, device=self.device)
        # steps
        self.beta_ss = torch.exp(torch.divide(torch.log(eps_ss),steps_per_epoch))
        
        # init. LPFs
        self.wlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.mlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.vlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.esslpf = esLPF(cdevice=self.device)
        
        # compute: one-pole filter gain
        self.betai = torch.tensor(beta_i, dtype=torch.float, device=self.device)

        defaults = dict(ss_inits=(self.ss_init,), ss_end=self.ss_end, betas=(self.betai,self.weight_decay,self.beta_ss), weight_decay=self.weight_decay,steps_per_epoch=steps_per_epoch, maximize=maximize)
        
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
            if 'asgm' not in group:
                group['asgm'] = AGM(self.wlpf,self.mlpf,
                  self.vlpf,self.esslpf,
                  self.use_optim_choice,group['maximize'], 
                  self.eps, group['betas'], group['ss_inits'], group['ss_end'], self.ss_cte, self.auto_init_ess) 
            
            asgm = group['asgm']
            
            # list of parameters, gradients, gradient notms
            params_with_grad = []
            grads = []

            # list to hold step count
            state_steps = []

            # list to hold one-pole filter memory
            qk, wk, mk = [],[],[]
            vk = []

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

                        state['q'] = p.clone(memory_format=torch.preserve_format).detach() 
                        
                        state['w'] = p.clone(memory_format=torch.preserve_format).detach()
                        
                        # one-pole filter memory (filter)
                        state['m'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)

                        state['v'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)
                        

                    
                    qk.append(state['q'])
                    wk.append(state['w'])
                    mk.append(state['m'])
                    vk.append(state['v'])                  

                    # update the step count by 1.
                    state['step'] += 1

                    # record the step
                    state_steps.append(state['step'])

            # Actual Learning Event: 
            alphaps = control_event(asgm, loss,
                params_with_grad, grads, qk, wk, mk, 
                vk,
                group['steps_per_epoch'], state_steps, optparams_with_grad
            )

        return loss, state['step']


@torch.no_grad()
def control_event(asgm: AGM, loss:Tensor, 
        params: List[Tensor], grads: List[Tensor], 
        qk: List[Tensor], wk: List[Tensor], mk: List[Tensor], 
        vk: List[Tensor],
        steps_per_epoch:int, state_steps: List[Tensor], optparams: List[Tensor] | None)->Tensor:
    
    r'''Functional API that computes the AutoSGM control/learning algorithm for each parameter in the model.

    See : [in future] class:`~torch.optim.AutoSGM` for details.
    '''
    undcp = True
    step = state_steps[0]
    alphaps = []

    #- At each step, adapt parameters (weights of the model) using 
    # (AutoSGM) PID structure.
    # Et[param] = Et[param + alphap*Et[grad] ] = Et[param] + alphap*Et[grad]  
    
    # cyclic step in each epoch
    step_c = (((step-1) % steps_per_epoch) + 1)
    # uniform initial effective step-size (rough linear correlation estimation).
    asgm.init_lin_corr_est(params, grads, step)
    
    for i, param in enumerate(params):
        
        grad_regs = (wk[i].mul(asgm.beta_o))
        if undcp:
            alphap = asgm.compute_opt(
                        step,step_c,param,grads[i],grad_regs,
                        qk[i],wk[i],mk[i],
                        vk[i]
                        )
        else:
            alphap = asgm.compute_opt_dcp(
                        step,step_c,param,grads[i],grad_regs,
                        qk[i],wk[i],mk[i],
                        vk[i]
                        )
                
        alphaps.append(alphap)
        
    return alphaps
