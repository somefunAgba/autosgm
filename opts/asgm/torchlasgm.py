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
    def __init__(self,p,wlpf:LPF,mlpf:LPF,vlpf:LPF,esslpf:esLPF,elpf:LPF,      
                ak:Tensor, betas:tuple, ss_inits:Tensor, ss_end:Tensor, eps:Tensor, 
                auto_init_ess:bool, use_optim_choice:bool, maximize:bool, joint:bool) -> None:
        self.gradref = 0
        self.wlpf = wlpf
        self.mlpf = mlpf
        self.vlpf = vlpf
        self.esslpf = esslpf
        self.elpf = elpf
        self.use_optim_choice = use_optim_choice
        self.maximize = maximize
        self.eps = eps
        self.beta_i = betas[0]
        self.beta_corr = betas[1]
        self.beta_ss = betas[2]
        self.beta_o = betas[3] 
        self.ss_inits = ss_inits
        self.ss_base = ss_inits
        self.ss_end = ss_end 
        self.ak = ak
        self.auto_init_ess = auto_init_ess
        self.p = p
        self.joint = joint 
        self.f1max = torch.tensor(0.99999, device=eps.device)
    
    @torch.no_grad()
    def init_lin_corr_est(self, params, grads, step, step_c):
        
        # E
        if step_c == 1 and self.auto_init_ess:
            square_eus, est_numels, prod_ev, square_evs = [],[],[],[]
            for id, param in enumerate(params):  
                egrads = (grads[id]+self.eps).neg_()
                egsqsum = (egrads.square()).sum()
                epsqsum = (param.square()).sum()
                epgprodsum = (param.mul(egrads)).sum()                
                if id == 0:
                    square_eus.append(epsqsum)
                    prod_ev.append(epgprodsum)
                    square_evs.append(egsqsum)
                    est_numels.append(param.nelement())
                else:
                    square_eus[0] = square_eus[0] + (epsqsum)
                    prod_ev[0] = prod_ev[0] + (epgprodsum)
                    square_evs[0] = square_evs[0] + (egsqsum)
                    est_numels[0] = est_numels[0] + param.nelement()   
            numel_param = torch.tensor(est_numels[0], device=self.eps.device)
            

            eu_sq = ((square_eus[0])).add(self.eps)
            evgrad_sq = ((square_evs[0]).add(self.eps))
            numrho = prod_ev[0].square()
            denrho = eu_sq*evgrad_sq
            erho_ests = (numrho/denrho).sqrt()
            
            if step == 1: erho_ests = (0.1)*erho_ests              
            tmp, self.ak = self.elpf.compute(u=erho_ests, x=self.ak, 
                    beta=self.beta_corr, step=step, mode=2)
            self.ss_inits = tmp[0]

            # logging.
            txt = "*"
            infostr = "aSGM info:"
            if step == 1:            
                strlog = f"{infostr} (step-size ending := {self.ss_end:.4g}), step-size pole : {self.beta_ss:.4g}],\n[LPF Poles [i,o] : {self.beta_i:.4g}, {self.beta_o:.4g}]"
                # debug  
                print(f"{txt * (len(strlog)) }") 
                print(f"total params, d={numel_param.item()}")
                print(f"[p={self.p}, eff. step-size [linear correlation est. (starting := {self.ss_inits:.6g})]")
                print(strlog) #
                print(f"{txt * (len(strlog)) }\n")
            else:
                print(f"...[eff. step-size (starting := {self.ss_inits:.6g})]...")
                
        pass

                              
    @torch.no_grad()
    def compute_opt(self,step:Tensor|int,step_c:Tensor|int,        
                param:Tensor, param_grad:Tensor,grad_regs:Tensor,qk:Tensor,wk:Tensor,mk:Tensor,vk:List[Tensor]):
        
        # -1. input grad.          
        # -2. linear step-size fcn. (prop./deriv.) 
        # -3. output param. update. (integ.)

        
        # (grad_ref - grad) = -grad
        # joint
        # if joint: error = (param_grad.add(grad_regs)).neg_() 
        error = param_grad.neg()
        regerror = grad_regs.neg()
        if self.maximize: error.neg_(); regerror.neg_()
        if self.joint: error.add_(regerror)
        
        errsq = error.square() #error.mul(error.conj())
        # if not self.joint: regerrsq = regerror.square() #regerror.mul(regerror.conj())

        # Et[-g]: input smoothing,             
        err_smth, mk = self.mlpf.compute(u=error, x=mk, 
                    beta=self.beta_i, step=step)
        
        # eff. step-size (linear correlation) 
        # cyclical growing (could be increasing or decreasing) per epoch. 
        # i.e: varies at each step in a training epoch, 
        # ss.end = 0 (decay), ss.end = 1 (warmup)   
        
        # Et[w] output: smoothed parameter
        # param_smth, qk = self.wlpf.compute(u=wk, x=qk,
        #         beta=self.beta_o, step=step)     

        alphaps = []  
        for pid in range(self.p):
            # in: (0-g)^gp -> (0-Et[g])^gp 
            gp = pid+1     
            
            # optimal (bayes) step-size: ss_k_gp/Et[(-g)^(2gp)].sqrt()
            
            # proportional part: eff. step-size estimation
            ss_k_gp = self.esslpf.compute(
                u=self.ss_end,x_init=self.ss_inits.div(math.factorial(gp)),noise_k=0,beta=self.beta_ss,step=step_c
                )
                     
            # derivative part: variance estimation / input normalization
            # inner product: <g^gp,g^gp> = Et[g^(2gp)] 
            errvar, vk[pid] = self.vlpf[pid].compute(
                u=errsq.pow(gp), x=vk[pid], 
                beta=torch.min(1-ss_k_gp,self.f1max), step=step
                )
            errstd = (errvar.sqrt().add_(self.eps))
            # alphaps.append(1/(errstd.div(ss_k_gp) 
            if self.use_optim_choice:
                delta_param = (err_smth.pow(gp)).div_(errstd).mul_(ss_k_gp)
            else:
                delta_param = (err_smth.pow(gp)).mul_(ss_k_gp)
                
            # out: update
            wk.add_(delta_param)

            # decouple gradient contribution due to constraints:
            # e.g: weight regularization
            if not self.joint:
                regdelta_param = (regerror.pow(gp)).mul_(ss_k_gp)
                # out: update
                wk.add_(regdelta_param)

        
        # Et[w] output: smoothed parameter
        param_smth, qk = self.wlpf.compute(u=wk, x=qk,
                beta=self.beta_o, step=step)

        # pass values to network's parameter placeholder.
        param.copy_(param_smth)  
        # END
        
        return alphaps            
        

                    
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
        
        2023. Feb. (added generalization to higher orders)

    Args:
        params(iterable, required): iterable of parameters to optimize or dicts defining parameter groups

        steps_per_epoch (int, required): per batch iterations >= 1 (default: 1)

        ss_init (float, required): starting eff. proportional gain or step-size (default=1e-3): (0, 1), 
                
        ss_end (float, optional):  ending eff. proportional gain or step-size (default=0): (0, 1), 
        
        eps_ss (float, optional): accuracy of final step_size in an epoch with respect to ss_end (default = 0.5): (0, 1), 
        
        beta_corr (float, optional): initial eff. step-size est. lowpass filter pole (default = 0.25): (0, 1),

        beta_i (float, optional): input mean est. lowpass filter pole (default = 0.9): (0, 1), 
        
        beta_n (float, optional): input variance est. lowpass filter pole (default = 1-ss_k): (0, 1), auto-computed        
        
        weight_decay (or L2-norm penalty) or beta_o (float, optional): output smoothing. lowpass filter pole (default = 1e-5): (0, 1), 

        use_optim_choice (bool, optional): use optimal step-size choice by doing variance estimation (default=True),
        
        maximize (bool, optional): maximize the params based on the objective, 
        instead of minimizing (default: False)
        
        auto_init_ess, auto (bool, optional, default: True)
        
        joint (bool, optional, default: True)
        
        optparams (optional): optimum param, if known, else default=None
        
        p (optional): number of fcn. inputs (default=1)
        
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
    def __init__(self, params, *, p=1, steps_per_epoch=1, ss_init=1e-3, ss_end=0, eps_ss=0.5, beta_corr=0.25, beta_i=0.9, weight_decay=1e-5, use_optim_choice=True, maximize=False, auto_init_ess=True, joint=True, optparams=None):

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
            
        self.p = torch.tensor(p, dtype=torch.int, device=self.device) 
        # eps added: div. by zero.
        self.eps = torch.tensor(1e-8, dtype=torch.float, device=self.device)
        
        # effective step-size.
        self.ss_inits = torch.tensor(ss_init, dtype=torch.float, device=self.device)
        self.ss_end = torch.tensor(ss_end, dtype=torch.float, device=self.device)
        # error in the convergence accuracy to ss_end.
        eps_ss = torch.tensor(eps_ss, dtype=torch.float, device=self.device)
        self.auto_init_ess = auto_init_ess
        
        # compute: one-pole filter gain
        self.beta_corr = torch.tensor(beta_corr, dtype=torch.float, device=self.device)
        # steps
        self.beta_ss = torch.exp(torch.divide(torch.log(eps_ss),steps_per_epoch))
        self.betai = torch.tensor(beta_i, dtype=torch.float, device=self.device)
        self.weight_decay = torch.tensor(weight_decay, dtype=torch.float, device=self.device)  
        self.betas = (self.betai,self.beta_corr,self.beta_ss,self.weight_decay)
        
        # init. LPFs
        self.wlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.mlpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.elpf = LPF(inplace=True, direct=True, cdevice=self.device)
        self.esslpf = esLPF(cdevice=self.device)
        self.vlpf = []
        for pid in range(p):
            self.vlpf.append( LPF(inplace=True, direct=True, cdevice=self.device) )

        defaults = dict(
            ss_inits=self.ss_inits, ss_end=self.ss_end, betas=self.betas, steps_per_epoch=steps_per_epoch, use_optim_choice=use_optim_choice, maximize=maximize, joint=joint
            )
        
        # if ref/opt/desired parameters are known.
        if isinstance(optparams, torch.Tensor):
            raise TypeError("optparams argument given to the optimizer should be an iterable of Tensors or dicts, but got " +
                            torch.typename(params))
            
        if optparams is not None:
            #TODO: can this be resolved using pytorch's base optimizer.py?
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
                ak = torch.tensor(0.,dtype=torch.float, device=self.device)
                group['asgm'] = AGM(self.p, 
                    self.wlpf,self.mlpf,self.vlpf,self.esslpf,self.elpf, ak,
                    group['betas'], group['ss_inits'], group['ss_end'], self.eps, self.auto_init_ess, group['use_optim_choice'], group['maximize'], group['joint']) 
            
            asgm = group['asgm']
            
            # list of parameters, gradients, gradient notms
            params_with_grad = []
            grads = []

            # list to hold step count
            state_steps = []

            # list to hold one-pole filter memory
            qk, wk, mk, vk, ek = [],[],[],[],[]
            
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
                        
                        for pid in range(self.p):
                            state[f"v_{pid+1}"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format, device=self.device)

                    qk.append(state['q'])
                    wk.append(state['w'])
                    mk.append(state['m'])
                    vks = []
                    for pid in range(self.p):
                        vks.append(state[f"v_{pid+1}"])
                    vk.append(vks);             

                    # update the step count by 1.
                    state['step'] += 1

                    # record the step
                    state_steps.append(state['step'])

            # Actual Learning Event: 
            alphaps = control_event(asgm, loss,
                params_with_grad, grads, qk, wk, mk, vk,
                group['steps_per_epoch'], state_steps, optparams_with_grad
            )

        return loss, state['step']


@torch.no_grad()
def control_event(asgm: AGM, loss:Tensor, 
        params: List[Tensor], grads: List[Tensor], 
        qk: List[Tensor], wk: List[Tensor], mk: List[Tensor], 
        vk: List[List],
        steps_per_epoch:int, state_steps: List[Tensor], optparams: List[Tensor] | None= None):
    
    r'''Functional API that computes the AutoSGM control/learning algorithm for each parameter in the model.

    See : [in future] class:`~torch.optim.AutoSGM` for details.
    '''

    step = state_steps[0][0]
    alphaps = []

    #- At each step, adapt parameters (weights of the model) using 
    # (AutoSGM) PID structure.
    # Et[param] = Et[param + alphap*Et[grad] ] = Et[param] + alphap*Et[grad]  
    if step > steps_per_epoch:
        pass
    
    # cyclic step in each epoch
    step_c = (((step-1) % steps_per_epoch) + 1)
    # uniform initial effective step-size (linear correlation estimation).
    asgm.init_lin_corr_est(wk, grads, step, step_c)
    
    for i, param in enumerate(params):
        
        grad_regs = (wk[i].mul(asgm.beta_o))
        alphap = asgm.compute_opt(
                        step,step_c,param,grads[i],grad_regs,
                        qk[i],wk[i],mk[i],vk[i]
                        )
        alphaps.append(alphap)
        
    return alphaps