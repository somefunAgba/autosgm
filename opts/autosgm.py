# Common doc strings among optimizers
_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. (default: None)"""

_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""

_maximize_doc = r"""maximize (bool, optional): maximize the params based on the
            objective, instead of minimizing (default: False)"""

_email_doc = r"""somefuno@oregonstate.edu"""

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer, required, 
    _use_grad_for_differentiable,
    _default_to_fused_or_foreach,
    _get_value, _stack_if_compiling, _dispatch_sqrt,
)
from typing import Any, Dict, List, Optional
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


__all__ = ['AutoSGM', 'autosgm']


#Smooth HPF
def smooth_td(beta_d, curr:Tensor, past:Tensor):
    """ Smooth First-order Digital Differentiator
    
    A smooth digital high-pass filter for time-differentiation
    
    Args:
        beta_d (>= 0)
        
        curr current input
        
        past past input
    """
    
    return (curr-past).mul_(curr+past).mul_(0.5*beta_d)  


# LPF
class LPF():
    """ Generic Digital First-order Low Pass Filter Structure 
        
        Recursively computes a weighted average (exponential or uniform).
        
        Uses: smoothing, averaging.
    """

    def __init__(self, inplace:bool=True, foreach=False, fused=False):
        self.inplace = inplace
        self.foreach = foreach
        self.fused = fused
        self.abnormal = fused or foreach

    @torch.no_grad()
    def compute(self, in_k:Tensor, x:Tensor, 
                beta:Tensor, step:Tensor, freq:int=1, mode:int=1, fix:bool=False, sq:bool=False):
        
        '''Computes in_k -> LPF -> out_k
        
            in_k: input at current time
            x: state at previous time
            beta: LPF pole at current time
            step: current discrete time
            freq: number of times >=1 an update is done w.r.t current iteration,
                filter updates iff (step-1) % freq == 0
            mode: [default: mode=1] unbiased (all, except 3,4) | asympt. unbiased (3,4)
            fix: add one to the iteration

        out_k : output at current time, k
        x : updated state for next time, k+1
        '''
        
        doupdate = (step-1)%freq  == 0
        k = 1*step
        if fix: k = k + 1
        betak = beta       
        
        if not mode in [5]:
            one_minus_betak = (1-betak)
            betak_pow_k = betak.pow(k)
            one_minus_betak_pow_k = (1-betak_pow_k)
                    
        if doupdate and not self.abnormal:
                
            if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
                
                # forward:
                ((x.mul_((betak - betak_pow_k))).add_(one_minus_betak*in_k)).div_(one_minus_betak_pow_k)
                out_k = 1*x
                
            elif mode == 1: # exponential. (stable if 0 \le \beta < 1) 

                # forward:
                (x.mul_(betak)).add_(in_k)
                out_k = x*(one_minus_betak/one_minus_betak_pow_k)     
                
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # forward:
                (x.mul_(betak)).add_(one_minus_betak*in_k)
                out_k = 1*x      
                    
            elif mode == 2: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                (x.mul_(betak)).add_(one_minus_betak*in_k)
                out_k = x.div(one_minus_betak_pow_k)     
                
            elif mode == 5: # uniform (unstable as k \to infinity or as \beta \to 1)
            
                # forward:
                (x.mul_(k-1)).add_(in_k).div_(k)
                out_k = 1*x   
                
            elif mode == -1: # exponential. (can be unstable) 

                # forward:
                (x.mul_(betak)).add_(in_k)
                out_k = x*1     
                    
        elif doupdate and self.abnormal:
            if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
                # ((x.mul_((betak - betak_pow_k))).add_(one_minus_betak*in_k)).div_(one_minus_betak_pow_k)
                # out_k = 1*x                
                # forward:
                torch._foreach_mul_(x, (betak - betak_pow_k))
                if not sq:
                    torch._foreach_add_(x, torch._foreach_mul(in_k, one_minus_betak))
                elif sq:
                    torch._foreach_addcmul_(x, in_k, in_k, one_minus_betak)
                torch._foreach_div_(x, one_minus_betak_pow_k)
                out_k = torch._foreach_mul(x, 1)
                
            elif mode == 1: # exponential. (stable if 0 \le \beta < 1) 
                # (x.mul_(betak)).add_(in_k)
                # out_k = x*(one_minus_betak/one_minus_betak_pow_k)  
                # forward:
                torch._foreach_mul_(x, betak)
                if not sq:
                    torch._foreach_add_(x, in_k)
                elif sq:
                    torch._foreach_addcmul_(x, in_k, in_k)
                out_k = torch._foreach_div(torch._foreach_mul(x, one_minus_betak),one_minus_betak_pow_k)      
                               
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # (x.mul_(betak)).add_(one_minus_betak*in_k)
                # out_k = 1*x 
                # forward:
                torch._foreach_mul_(x, betak)
                if not sq:
                    torch._foreach_add_(x, torch._foreach_mul(in_k, one_minus_betak))
                elif sq:
                    torch._foreach_addcmul_(x, in_k, in_k, one_minus_betak)
                out_k = torch._foreach_mul(x,1)    
                                     
            elif mode == 2: # exponential. (stable if 0 \le \beta < 1) 
                # (x.mul_(betak)).add_(one_minus_betak*in_k)
                # out_k = x/(one_minus_betak_pow_k) 
                # forward:
                torch._foreach_mul_(x, betak)
                if not sq:
                    torch._foreach_add_(x, torch._foreach_mul(in_k, one_minus_betak))
                elif sq:
                    torch._foreach_addcmul_(x, in_k, in_k, one_minus_betak)
                out_k = torch._foreach_div(x,one_minus_betak_pow_k) 

                                   
            elif mode == 5 and not sq: # uniform (unstable as k \to infinity or as \beta \to 1)
                # (x.mul_(k-1)).add_(in_k).div_(k)
                # out_k = 1*x              
                # forward:
                torch._foreach_mul_(x, k-1)
                torch._foreach_add_(x, in_k)
                torch._foreach_div_(x, k)
                out_k = torch._foreach_mul(x,1) 
                
            elif mode == -1: # exponential. (stable if 0 \le \beta < 1) 
                # (x.mul_(betak)).add_(in_k)
                # out_k = x*(one_minus_betak/one_minus_betak_pow_k)  
                # forward:
                torch._foreach_mul_(x, betak)
                if not sq:
                    torch._foreach_add_(x, in_k)
                elif sq:
                    torch._foreach_addcmul_(x, in_k, in_k)
                out_k = torch._foreach_mul(x,1)            
        else:
            if mode in [0,3,2,5]:
                out_k = 1*x   
            else:
                out_k =  x*(one_minus_betak/one_minus_betak_pow_k)        
        '''
        out_k : output at current time, k
        x : updated state for next time, k+1
        '''
               
        return out_k, x

    def previous(self, xkm1, beta:Tensor, step:Tensor, freq:int=1, mode=1, fix:bool=False):
        ''' Get previous ouput, given previous state.
        
        xkm1: state at previous time
        beta: LPF pole at current time
        step: current discrete time
        freq: number of times >=1 an update is done w.r.t current iteration,
            filter updates iff (step-1) % freq == 0
        mode: [default: mode=1] unbiased (all, except 4) | asympt. unbiased (4)
        fix: add one to the iteration

        out_km1 : output at previous time, k-1
        '''    
        
        doupdate = (step-1)%freq  == 0
        
        if doupdate: 
            k = step-1
            if fix or k==0: 
                k=k+1
            betak = beta
            
            if not mode in [5]:
                one_minus_betak = (1-betak)
                betak_pow_k = betak.pow(k)
                one_minus_betak_pow_k = (1-betak_pow_k)  
        
            if not self.abnormal:
                if mode == 1: # exponential. (stable if 0 \le \beta < 1) 

                    out_km1 = xkm1*(one_minus_betak/one_minus_betak_pow_k)     
                        
                elif mode == 2: # exponential. (stable if 0 \le \beta < 1) 

                    out_km1 = xkm1/(one_minus_betak_pow_k) 
                     
                else:
                    
                    out_km1 = xkm1*1  
                    
            else:                              
                if mode == 1: # exponential. (stable if 0 \le \beta < 1) 

                    out_km1 = torch._foreach_mul(xkm1, one_minus_betak)
                    torch._foreach_div_(out_km1, one_minus_betak_pow_k)         
                        
                elif mode == 2: # exponential. (stable if 0 \le \beta < 1) 

                    out_km1 = torch._foreach_div(xkm1,one_minus_betak_pow_k) 
                    
                elif mode == 4: # exponential. (stable if 0 \le \beta < 1) k \to infty

                    out_km1 = torch._foreach_mul(xkm1,one_minus_betak)   
                else:
                    
                    out_km1 = torch._foreach_mul(xkm1,1)
        else:
            if not self.abnormal:
                out_km1 = 1*xkm1 
            else:
                out_km1 = torch._foreach_mul(xkm1,1)         
                
        return out_km1

    def patch(self, xkm1, beta:Tensor, step:Tensor, freq:int=1, mode=1, fix:bool=False):
        ''' transform gradient wrt smoothed weight to gradient wrt unsmoothed weight. 
        xkm1: state at previous time
        beta: LPF pole at current time
        step: current discrete time
        freq: number of times >=1 an update is done w.r.t current iteration,
            filter updates iff (step-1) % freq == 0
        mode: [default: mode=1] unbiased (all, except 4) | asympt. unbiased (4)
        fix: add one to the iteration

        out_km1 : output at previous time, k-1
        '''    
        
        doupdate = (step-1)%freq  == 0
        
        if doupdate: 
            k = step
            if fix or k==0: 
                k=k+1
            betak = beta
            
            if not mode in [5]:
                one_minus_betak = (1-betak)
                betak_pow_k = betak.pow(k)
                one_minus_betak_pow_k = (1-betak_pow_k)  
        
                if not self.abnormal:
                    if mode == 3:
                        out_km1 = xkm1*(one_minus_betak)   
                    else: # mode == 1 
                        out_km1 = xkm1*(one_minus_betak/one_minus_betak_pow_k)   
                else:  # foreach impl.       
                    if mode == 3:
                        out_km1 = torch._foreach_mul(xkm1, one_minus_betak)  
                    else: # mode == 1                      
                        out_km1 = torch._foreach_mul(xkm1, one_minus_betak)
                        torch._foreach_div_(out_km1, one_minus_betak_pow_k)         
            else:
                if not self.abnormal:
                    out_km1 = 1*xkm1 
                else: # foreach impl.   
                    out_km1 = torch._foreach_mul(xkm1,1)         
                
        return out_km1

    @torch.no_grad()
    def compute_dbl(self, in_k:Tensor, x1:Tensor, x2:Tensor, 
                beta1:Tensor, beta2:Tensor, step:Tensor, freq:int=1, mode1:int=1, mode2:int=3, fix:bool=False):
        
        ''' Cleaner way to compute in->LPF->LPF->out 
        
        This allows us to save memory, hence compute time, compared to the typical way it would have been computed.
        
            in_k: input at current time
            x1: first state at previous time
            x2: second state at previous time
            beta1: first LPF pole at current time       
            beta2: second LPF pole at current time

            step: current discrete time
            freq: number of times >=1 an update is done w.r.t current iteration,
                filter updates iff (step-1) % freq == 0
            mode1: [default: 1 | 0,1,2,5] unbiased
            mode2: [default: 3] asympt. unbiased
            fix: add one to the step (iteration), if first step == 0
            
        out_k : output at current time, k
        x1, x2 : updated states for next time, k+1
        '''
        
        y, x1 = self.compute(
            in_k=in_k, x=x1,
            beta=beta1, step=step, mode=mode1
            )
        
        out_k, x2 = self.compute(
            in_k=y, x=x2,
            beta=beta2, step=step, mode=mode2
            )
         
                  
        '''
        out_k : output at current time, k
        x1, x2 : updated states for next time, k+1
        '''
        return out_k, x1, x2

class common_sets():
    """ Commons 
    """
    
    def __init__(self, lpf:LPF, auto_mode, 
                lr_init, epfreq, betas, beta_o, beta_d, 
                eps, spe, wd_cte, 
                autolr:bool, restarts:bool, maximize:bool, join_wdec:bool, lrlogstep:bool, down:bool) -> None:
        self.down = down
        self.lrlogstep = lrlogstep
        self.lpf = lpf
        self.auto_mode = auto_mode
        self.beta_i = betas[0]
        self.beta_e = betas[1] 
        self.beta_a = betas[2] 
        self.beta_o = beta_o 
        self.beta_d = beta_d
        self.lr_init = lr_init
        self.restarts = restarts
        self.eps = eps
        self.spe = spe
        self.autolr = autolr
        self.maximize = maximize
        self.join_wdec = join_wdec 
        self.wd_cte = wd_cte
        self.est_numels = 0
        self.rstfact = 2
        self.epfreq = epfreq
        
    @torch.no_grad()
    def grp_devdt(self, device, dtype):
        fzero = torch.tensor(0., dtype=dtype, device=device)
        fone = torch.tensor(1., dtype=dtype, device=device)
        lr_init = self.lr_init*fone
        beta_i = self.beta_i*fone
        beta_e = self.beta_e*fone
        beta_a = self.beta_a*fone
        beta_o = self.beta_o*fone
        beta_d = self.beta_d*fone
        eps = self.eps*fone
        wd_cte = self.wd_cte*fone
        auto_mode = self.auto_mode 
        return lr_init, beta_i, beta_e, beta_a, beta_o, beta_d, eps, wd_cte, fzero, fone, auto_mode        
    
    @torch.no_grad()
    def log_stats(self):        
        # logging.
        txt = "="
        infostr = "AutoSGM info:\t"          
        strlog = f"{infostr} [total params, d={self.est_numels}]\n[autolr={self.autolr}, auto_mode={self.auto_mode}, pos_dir={self.down}, init_lr={self.lr_init:.5g} |\n lpf [i,e,o] : {self.beta_i:.4g}, {self.beta_e:.4g}, {self.beta_o:.4g} | tdiff. : {self.beta_d:.4g} |\n eps : {self.eps:.4g} | weight-decay: {self.wd_cte:.4g}]"
        # debug  
        print(f"{txt * int(0.35*len(strlog)) }")
        print(strlog) #      
        print(f"{txt * int(0.35*len(strlog)) }\n")
            

# PyTorch Backend   
class AutoSGM(Optimizer):
    '''
    r"""Implements Automatic Stochastic Gradient Method
    
    AutoSGM is the general digital structure in popular optimizers like (Adam, SGD).
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        autolr (bool, optional): lr algorithm (default: True). Set to False to use adam.
        lr_init (float, optional): initial learning rate (default: 1e-3)
        beta_i (float, optional): input smoothing lowpass pole param. (default: 0.9)
        beta_e (float, optional): averaging lowpass pole param. (default: 0.999)
        beta_d (float, optional): input time-differentiator param. (default: 0.)
        eps (float, optional): positive constant to condition the sqrt. of the graident variance (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        maximize (bool, optional): whether the objective is being maximized
            (default: False)
        {foreach}
        {differentiable}
        fused (bool, optional): whether the fused implementation (CUDA only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. Since the fused implementation is usually significantly faster than
            the for-loop implementation, we try to use it whenever possible (all parameters
            are on CUDA and are of a supported type). Else, we attempt to use the foreach
            implementation and lastly fall back to the for-loop implementation. (default: None)

    .. note:: 
            The foreach and fused implementations are typically faster than   the for-loop,
            single-tensor implementation, so we will try to default to them IF the user has
            not specified either flag (i.e., when foreach = fused = None). For example, if
            the user specifies True for foreach but nothing for fused, we will run the foreach
            implementation. If the user specifies False for fused but nothing for foreach, we will
            run the for-loop implementation. If the user specifies True for both foreach and
            fused, we will prioritize fused over foreach. We attempt to use the fastest, so the
            hierarchy goes fused -> foreach -> for-loop.
    .. _AutoSGM\: A Unified Lowpass Regularization Frameowrk for Accelerated Learning
        paper link here.
    
    """.format(maximize=_maximize_doc, foreach=_foreach_doc, differentiable=_differentiable_doc) + r"""
    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = AutoSGM(model.parameters(), foreach=True)
        >>> optimizer.zero_grad()
        >>> loss_fcn(model(input), target).backward()
        >>> optimizer.step()
        
        .. note::
            This is a general implementation and there can be any number of specialized implementations
        .. math::
            \begin{aligned}
            e_{t} &= -g_{t}
            m_{t} &= \beta_{i}*m_{t-1} + (1-\beta_{i})*e_{t}
            v_{t} &= e_{t} + \beta_{d}\,*(m_{t}-m_{t-1})
            p_{t} &= p_{t-1} + \alpha_{t}*v_{t}
            \end{aligned}
            
        where :math:`p`, :math:`g`, :math:`m`, :math:`v` denote the parameters, gradient, smooth gradient by lowpass filtering, smooth gradient by time-differentiation respectively.
    """
    '''
    
    def __init__(self, params, *, lr_init=1e-3, spe=1, epfreq=1,
                beta_i=0.9, beta_e=0.999, beta_a=0.99999,
                eps=1e-8, weight_decay=0,
                beta_o=0, beta_d=0,
                auto_mode=0, # 0: corr lr, 1: 1984 ss, 2: const lr. 
                autolr:bool=True, 
                restarts:bool=True,
                join_wdec:bool=False, 
                lrlogstep:bool=True, down:bool=False,
                maximize:bool=False, 
                foreach: Optional[bool]=None,
                fused: Optional[bool]=False,
                differentiable: bool=False):
        
        '''
        Inits an SGM optimizer
        '''
        
        if any([beta_i, beta_e, eps, beta_d, weight_decay, lr_init, spe]) < 0.0:
            raise ValueError(f"One or more of: \
                            lr_init={lr_init}, \
                            beta_i={beta_i}, \
                            beta_e={beta_e}, \
                            eps={eps}, \
                            beta_d={beta_d}, \
                            weight_decay={weight_decay} \
                            has a negative or an invalid value!")

        if any([beta_i, beta_e, lr_init]) > 1.0:
            raise ValueError(f"One or more of: \
                            lr_init={lr_init}, \
                            beta_i={beta_i}, \
                            beta_e={beta_e}, \
                            has an invalid value greater than 1!")
        
        # init. recursive first-order lowpass filter obj.
        self.lpf = LPF(foreach=foreach, fused=fused)
        
        defaults = dict(lr_init=lr_init,betas=(beta_i,beta_e,beta_a),
                        beta_o=beta_o,
                        beta_d=beta_d, 
                        spe=spe, auto_mode=auto_mode,
                        eps=eps, weight_decay=weight_decay, epfreq=epfreq,
                        restarts=restarts,
                        maximize=maximize, autolr=autolr, join_wdec=join_wdec,lrlogstep=lrlogstep, down=down,
                        foreach=foreach, differentiable=differentiable, 
                        fused=fused, com_sets=None, fusedparams=None)
        
        
        super().__init__(params, defaults)
        
        # Pytorch's AMP issues [not exhaustive]
        # - fused and differentiable can't be True
        # - fused and foreach can't be True
        # - fused=True means params are on CUDA and are floating-point Tensors.
        if fused:
            if differentiable:
                raise RuntimeError("`fused` doesn't support `differentiable`!")
            
            if foreach:
                raise RuntimeError("`fused` doesn't support `foreach`!")
            
            self._step_supports_amp_scaling = True
            if not all(
                p.is_cuda and torch.is_floating_point(p) 
                for pg in self.param_groups for p in pg['params']):
                raise RuntimeError("`fused=True` requires CUDA and floats for all params!")
        
        
        
    def __setstate__(self, state):
        super().__setstate__(state)
        '''
        Set defaults for parameter groups
        '''
        
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('fused', None)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)
            # group.setdefault('beta_o', 0)
            # group.setdefault('beta_d', 0)
            group.setdefault('join_wdec', False)
            group.setdefault('com_sets', None)
            
            
    @torch.no_grad()
    def zero_logged_lr(self):
        """zero lr logged in last epoch.
        
            This will help save time in plotting logged lrs
            since we average the whole lrs logged in an epoch by the total steps in that epoch.
        """
        
        for group in self.param_groups:
            lr_save_list = [
                self.state[p]["lrsave"] 
                for p in group['params'] 
                if p.grad is not None
            ]
            torch._foreach_zero_(lr_save_list)
                
                
    def _init_group(self, group, 
                    params_with_grad_list,
                    weight_list,
                    weight_smth_list,
                    grad_list, 
                    grad_smth_list, 
                    grad_var_list,                
                    lr_avgb_list,
                    lr_avga_list, 
                    lr_save_list,
                    steps,
                    ):
        '''
        Inits state of params
        '''
        
        if 'com_sets' not in group or group['com_sets'] is None:
            group['com_sets'] = common_sets(
                self.lpf,group['auto_mode'],group['lr_init'], group['epfreq'],
                group['betas'], group['beta_o'], group['beta_d'],
                group['eps'], group['spe'], group['weight_decay'],
                group['autolr'],group['restarts'],group['maximize'], group['join_wdec'], group['lrlogstep'], group['down']
            ) 
        
        com_sets = group['com_sets']
        has_sparse_grad = False       
                
        for p in group['params']:
            if p.grad is not None:
                params_with_grad_list.append(p)
                grad_list.append(p.grad)
                if p.grad.is_sparse: has_sparse_grad = True
                
                state = self.state[p]
                # Lazy state init.
                if len(state)==0:
                    state['step'] = torch.tensor(0, dtype=torch.float, device=p.device)
                                                
                    #
                    state['weight'] = p.clone(memory_format=torch.preserve_format).detach()   
                    state['weight_smth'] = p.clone(memory_format=torch.preserve_format).detach() 
                    #
                    state['grad_smth'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=p.device)                 
                    state['grad_var'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=p.device)   
                    #
                    state["lr_avgb"] = group['lr_init']*torch.ones_like(p, memory_format=torch.preserve_format, device=p.device)    
                    state["lr_avga"] = group['lr_init']*torch.ones_like(p, memory_format=torch.preserve_format, device=p.device)                         
                    state["lrsave"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=p.device)
                    
                
                state['step'] += 1
                steps.append(state['step'])
                weight_list.append(state['weight'])
                weight_smth_list.append(state['weight_smth'])      
                grad_smth_list.append(state['grad_smth'])
                grad_var_list.append(state['grad_var'])       
                lr_avgb_list.append(state['lr_avgb'])
                lr_avga_list.append(state['lr_avga'])
                lr_save_list.append(state['lrsave'])
        
        return com_sets, has_sparse_grad
        
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (Callable, optional): A closure taht reevaluates the model and returns the loss
            
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # lr_lists = []
        for group in self.param_groups:
        
            params_with_grad_list = []
            weight_list = []
            weight_smth_list = []
            grad_list = []
            grad_smth_list = []
            grad_var_list = []            
            lr_avgb_list = []
            lr_avga_list = []
            lr_save_list = []
            steps = []
                        
            com_sets, has_sparse_grad = \
                self._init_group(group,
                    params_with_grad_list, weight_list,
                    weight_smth_list, grad_list, 
                    grad_smth_list, grad_var_list,                     
                    lr_avgb_list, lr_avga_list, 
                    lr_save_list, steps)
            
            sgm(com_sets, steps, 
                params_with_grad_list, 
                weight_list,
                weight_smth_list,
                grad_list, 
                grad_smth_list,
                grad_var_list,
                lr_avgb_list,
                lr_avga_list, 
                lr_save_list,
                has_sparse_grad = has_sparse_grad,
                foreach=group['foreach'], 
                differentiable=group['differentiable'],  
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None))
            
            # update state
            pass
        
        return loss
            
AutoSGM.__doc__ = r"""\
    Implements Automatic Stochastic Gradient Method
    """ + r"""\
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr_init (float, optional): initial learning rate (default: 1e-3)
        beta_i (float, optional): input smoothing lowpass pole param. (default: 0.9)
        beta_e (float, optional): averaging lowpass pole param. (default: 0.999)
        beta_d (float, optional): input time-differentiator param. (default: 0.)
        eps (float, optional): positive constant to condition the sqrt. of the graident variance (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        maximize (bool, optional): whether the objective is being maximized
            (default: False)
        {foreach}
        {differentiable}
        fused (bool, optional): whether the fused implementation (CUDA only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. Since the fused implementation is usually significantly faster than
            the for-loop implementation, we try to use it whenever possible (all parameters
            are on CUDA and are of a supported type). Else, we attempt to use the foreach
            implementation and lastly fall back to the for-loop implementation. (default: None)

    .. note:: 
            The foreach and fused implementations are typically faster than   the for-loop,
            single-tensor implementation, so we will try to default to them IF the user has
            not specified either flag (i.e., when foreach = fused = None). For example, if
            the user specifies True for foreach but nothing for fused, we will run the foreach
            implementation. If the user specifies False for fused but nothing for foreach, we will
            run the for-loop implementation. If the user specifies True for both foreach and
            fused, we will prioritize fused over foreach. We attempt to use the fastest, so the
            hierarchy goes fused -> foreach -> for-loop.
    .. _AutoSGM\: A Unified Lowpass Regularization Frameowrk for Accelerated Learning
        paper link here.
    
    """.format(maximize=_maximize_doc, foreach=_foreach_doc, differentiable=_differentiable_doc) + r"""
    Example:
        xdoctest: +SKIP
        >>> optimizer = SGM(model.parameters(), lr=1e-3, beta_i=0.9, beta_d=1e-8)
        >>> optimizer.zero_grad()
        >>> loss_fcn(model(input), target).backward()
        >>> optimizer.step()
        
        .. note::
            This is a general implementation and there can be any number of specialized implementations
        .. math::
            \begin{aligned}
            e_{t} &= -g_{t}
            m_{t} &= \beta_{i}*m_{t-1} + (1-\beta_{i})*e_{t}
            v_{t} &= e_{t} + \beta_{d}\,*(m_{t}-m_{t-1})
            p_{t} &= p_{t-1} + \alpha_{t}*v_{t}
            \end{aligned}
            
        where :math:`p`, :math:`g`, :math:`m`, :math:`v` denote the parameters, gradient, smooth gradient by lowpass filtering, smooth gradient by time-differentiation respectively.
"""    

def sgm(com_sets:common_sets, steps:List[Tensor], params: List[Tensor], 
        weight_list: List[Tensor], weight_smth_list: List[Tensor], 
        grad_list: List[Tensor], grad_smth_list: List[Optional[Tensor]],
        grad_var_list: List[Optional[Tensor]],
        lr_avgb_list: List[Tensor], lr_avga_list: List[Tensor], lr_save_list: List[Tensor],*,
        has_sparse_grad:bool=None,
        differentiable:Optional[bool]=False,
        foreach:Optional[bool]=None,
        fused:Optional[bool]=None, 
        grad_scale:Optional[Tensor]=None,
        found_inf:Optional[Tensor]=None):
    
    r""" Functional API performing the SGM algorithm computation
    
    See :class:`~torch.optim.SGM` for details.
    
    """
    # PyTorch's JIT scripting issues (Conditionals, Optionals)
    # logic to use multi_tensor: foreach, fused or single_tensor
    if foreach is None and fused is None:
        fused, foreach = _default_to_fused_or_foreach(
                [params, grad_list, grad_smth_list],
                differentiable, has_fused=True,
                )
    if fused is None: fused = False
    if foreach is None: foreach = False
            
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach ops.')
    
    if fused and not torch.jit.is_scripting():
        func = _fused_sgm
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgm
    else:
        func = _single_tensor_sgm

    func(com_sets, steps,
        params, weight_list, weight_smth_list,
        grad_list, grad_smth_list, grad_var_list,
        lr_avgb_list, lr_avga_list, lr_save_list,
        has_sparse_grad=has_sparse_grad,        
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf)
    
    
def _single_tensor_sgm(com_sets:common_sets, steps: List[Tensor], 
        params: List[Tensor], weight_list: List[Tensor], weight_smth_list: List[Tensor], 
        grad_list: List[Tensor], grad_smth_list: List[Optional[Tensor]],
        grad_var_list: List[Optional[Tensor]], 
        lr_avgb_list: List[Optional[Tensor]], lr_avga_list: List[Optional[Tensor]], lr_save_list: List[Optional[Tensor]],*,has_sparse_grad:bool,       
        differentiable:Optional[bool],
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor]):
    
    ''' Typical for loop implementation
    '''
    
    assert grad_scale is None and found_inf is None
    
    dtype = params[0].dtype
    device= params[0].device
    
    lr_init, beta_i, beta_e, beta_a, \
    beta_o, beta_d, eps, \
    wdecay, fzero, fone, auto_mode = com_sets.grp_devdt(device,dtype)
        
    # LOG.
    if steps[0] == 1: 
        com_sets.est_numels = sum(p.numel() for p in params)
        com_sets.log_stats()    
                
    for i, param in enumerate(params):
        step = steps[i]

        w_t = weight_list[i]
        w_smth = weight_smth_list[i]
        grad = grad_list[i]
        grad_smth = grad_smth_list[i]
        grad_var = grad_var_list[i]
        lr_avgb = lr_avgb_list[i] 
        lr_avga = lr_avga_list[i] 
        lr = lr_save_list[i]
        
        # handle complex parameters
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            w_t = torch.view_as_real(w_t)
            w_smth = torch.view_as_real(w_smth)
            grad = torch.view_as_real(grad)
            grad_smth = torch.view_as_real(grad_smth)
            grad_var = torch.view_as_real(grad_var)          
            lr_avgb = torch.view_as_real(lr_avgb)
            lr_avga = torch.view_as_real(lr_avga)
            lr = torch.view_as_real(lr)
        
        # START
        mwd = 1-wdecay
        g_t = com_sets.lpf.patch(grad,beta=beta_o,step=step) 
        if com_sets.join_wdec:
            # decay weight directly
            w_t.mul_(mwd)
        else:
            # decay weight as l2-regularized gradient
            g_t.addcmul_(w_t,wdecay)
        if com_sets.down: g_t.neg_()
                     
        # flip sign, if maximizing.
        if com_sets.maximize: g_t.neg_()

        # smooth input [lowpass]
        m_t_min_1 = com_sets.lpf.previous(xkm1=grad_smth, beta=beta_i, step=step)
        
        # gradss_t_min_1 = g_t_min_1.mul(g_t)
        gradss_t_min_1 = m_t_min_1.mul(g_t)
        if step == 1:
            gradss_t_min_1 = 0
        
        m_t, grad_smth = com_sets.lpf.compute(in_k=g_t, x=grad_smth, beta=beta_i, step=step)
        
        # smooth input [add average time-diff] 
        if beta_d > 0:
            if step == 1: v_diff = 0
            else: v_diff = smooth_td(beta_d, m_t, m_t_min_1)
            g_t.add_(v_diff)
            m_t.add_(v_diff)
    
        if auto_mode in [0,2]:
            # denominator of the optimal step_size
            # in: averaging [lowpass] ~ approx. input variance
            v_var_t, grad_var = com_sets.lpf.compute(in_k=(g_t*g_t.conj()), x=grad_var, beta=beta_e, step=step)
            
            # normalized input
            # malt_t = m_t.div((v_var_t.add(eps)).sqrt_())
            (m_t).div_((v_var_t.sqrt()).add_(eps))
   
        
        # com_sets.autolr = False
        # learning rate estimation
        if com_sets.autolr and auto_mode==0:
            # computes numerator of the bayes-optimal step_size
            # computes approx. linear correlation funcion 
            
            # surrogate approx. without w^\star 
            if com_sets.join_wdec:
                ewg_t = m_t.mul(w_t)
            else:
                ewg_t = m_t.mul(w_t.mul(mwd))
                
            # restart logic: idea is that, after the first k epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs, with an averagely constant learning-rate for each parameter.
            # cyclic step
            step_c = (((step-1) % (com_sets.spe*com_sets.epfreq)) + 1)
            # restart lowpass state at the end of every 'k' epoch.
            if com_sets.restarts and step_c == 1 and step > 1:
                lr_avgb.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avgb))
                lr_avga.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avga))
        
            # linear correlation estimate update  
            # (double averaging [lowpass]), 
            # Note: future work: how to estimate this more accurately?
            lrat, lr_avgb, lr_avga = com_sets.lpf.compute_dbl(in_k=ewg_t, x1=lr_avgb, x2=lr_avga, beta1=beta_a, beta2=beta_e, step=step)
            
            # abs. val projection. to ensure positive rates.
            alpha_hat_t = (lrat.abs_())
        elif com_sets.autolr and auto_mode==1: # 1984 impl.
            # no normalization is done here.
            # lr_init 
            
            # denominator of the optimal step_size
            # in: averaging [lowpass] ~ approx. input variance
            v_var_t, grad_var = com_sets.lpf.compute(in_k=(gradss_t_min_1*gradss_t_min_1.conj()), x=grad_var, beta=beta_e, step=step)
            
            # normalized input
            # malt_t = m_t.div((v_var_t.add(eps)).sqrt_())
            (gradss_t_min_1).div_((v_var_t.sqrt()).add_(eps))
            
            
            # surrogate approx. without w^\star 
            ewg_t = gradss_t_min_1.mul(lr_avga)
                
            # restart logic: idea is that, after the first k epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs, with an averagely constant learning-rate for each parameter.
            # cyclic step
            step_c = (((step-1) % (com_sets.spe*com_sets.epfreq)) + 1)
            # restart lowpass state at the end of every 'k' epoch.
            if com_sets.restarts and step_c == 1 and step > 1:
                lr_avgb.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avgb))
                lr_avga.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avga))
        
            # linear correlation estimate update  
            # (double averaging [lowpass]), 
            # Note: future work: how to estimate this more accurately?
            lrat, lr_avgb = com_sets.lpf.compute(in_k=ewg_t, x=lr_avgb, beta=beta_e, step=step, mode=3)
            
             # abs. val projection. to ensure positive rates.
            lr_avga.addcmul_(gradss_t_min_1, lrat.abs())
            
            # restarts:
            # push close to zero, if a negative element occurs.
            alpha_hat_t = (lr_avga.relu().add(eps))            
            
        
            # lr_avgb.mul_(0).add_(lr_init/10)
            # lr_avga.addcmul_(gradss_t_min_1, lr_avgb)
            
            # # restarts:
            # # push close to zero, if a negative element occurs.
            # lrat = lr_avga.relu().add(eps)
            
            # alpha_hat_t = lrat.abs()
            
            # alpha_hat_t = 1*lr_init
        else:
            # auto_mode==2
            # use externally supplied (typically small) linear correlation estimate 
            alpha_hat_t = lr_init      

        # integrate: state update
        if com_sets.down:
            w_t.addcmul_(m_t, alpha_hat_t)
        else:
            w_t.addcmul_(m_t, alpha_hat_t, value=-1)
            
        # smooth out. [lowpass]
        w_est, w_smth = com_sets.lpf.compute(in_k=w_t, x=w_smth, beta=beta_o, step=step, mode=3)
            
        # pass estimated/updated weight values back to the neural network's placeholder.
        param.mul_(0).add_(w_est)
        
        # log lr
        if com_sets.autolr and com_sets.lrlogstep:
            # we want to do this per step
            lr.mul_(0).add_(alpha_hat_t)
        elif com_sets.autolr and not com_sets.lrlogstep:
            # to save time and memory,
            # we want to do this per epoch
            # but get the average over all steps in that epoch. 
            lr.add_(alpha_hat_t)        
        
    # END 
    

def _multi_tensor_sgm(com_sets:common_sets, steps: List[Tensor], 
        params: List[Tensor], 
        weight_list: List[Tensor], weight_smth_list: List[Tensor], 
        grad_list: List[Tensor], grad_smth_list: List[Optional[Tensor]],
        grad_var_list: List[Optional[Tensor]],  
        lr_avgb_list: List[Optional[Tensor]], lr_avga_list: List[Optional[Tensor]], lr_save_list: List[Optional[Tensor]],*,has_sparse_grad:bool,       
        differentiable:Optional[bool],
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor]):
    
    if len(params) == 0:
        return
    
    assert grad_scale is None and found_inf is None

    if steps[0] == 1: 
        com_sets.est_numels = sum(p.numel() for p in params)
        com_sets.log_stats()
    
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_var_list, lr_avgb_list, lr_avga_list, lr_save_list, steps])

    for (device, dtype) in grouped_tensors:
        (
            device_params, device_w, device_w_smth, 
            device_grads,device_grads_smth, device_grads_var, 
            device_lrb, device_lra, device_lr, device_steps
        ) = grouped_tensors[(device, dtype)] 

        lr_init, beta_i, \
        beta_e, beta_a, \
        beta_o, beta_d, \
        eps, wdecay, fzero, fone, auto_mode =  com_sets.grp_devdt(device=device, dtype=dtype)
        
        step_this = device_steps[0]
        
        device_has_sparse_grad = any(grad.is_sparse for grad in device_grads)
        
        # handle complex parameters
        params_ = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]
        device_w = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_w]
        device_w_smth = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_w_smth]
        #
        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_grads_smth = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads_smth]
        device_grads_var = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads_var]    
        #   
        device_lrb = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lrb]
        device_lra = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lra]
        device_lr = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_lr]
        
        
        # START
        g_t = com_sets.lpf.patch(device_grads,beta=beta_o,step=step_this,mode=3) 
        if com_sets.join_wdec: pass # decay weight directly
        else:
            # decay weight as l2-regularized gradient
            torch._foreach_add_(g_t, torch._foreach_mul(device_w,wdecay))
        
        if com_sets.down: torch._foreach_neg_(g_t)    
            
        # flip sign, if maximizing.
        if com_sets.maximize: torch._foreach_neg_(g_t)

        # smooth input [lowpass]
        m_t_min_1 = com_sets.lpf.previous(xkm1=device_grads_smth, beta=beta_i, step=step_this)
        
        gradss_t_min_1 = torch._foreach_mul(m_t_min_1,g_t)
        
        m_t, device_grads_smth = com_sets.lpf.compute(in_k=g_t, x=device_grads_smth, beta=beta_i, step=step_this)
        
        # smooth input [add average time-diff] 
        if step_this == 1: vdiff = 0
        else:                 
            vdiff = [
                smooth_td(beta_d, mti, mti_mone) 
                for mti, mti_mone in zip(m_t, m_t_min_1)
            ]                     
        # in: smooth gradient
        torch._foreach_add_(g_t, vdiff) 
        torch._foreach_add_(m_t, vdiff)
        
        if auto_mode in [0,2]:
            # denominator of the bayes-optimal step_size
            # in: averaging [lowpass] ~ approx. input variance
            v_var_t, device_grads_var = com_sets.lpf.compute(in_k=g_t, x=device_grads_var, beta=beta_e, step=step_this, sq=True)
            
            # normalized input        
            # malt_t = torch._foreach_div(m_t, torch._foreach_sqrt(torch._foreach_add(v_var_t,eps)))
            torch._foreach_div_(m_t, torch._foreach_add(torch._foreach_sqrt(v_var_t), eps)) 


        # learning rate estimation
        if com_sets.autolr and auto_mode==0:
            # computes numerator of the bayes-optimal step_size
            # computes approx. linear correlation funcion 
            
            alpha_hat_t = []
            com_sets.lpf.abnormal = False
            # since we're doing in-place ops., we can't use the zip iterator!
            for i in range(len(params_)):
                # surrogate approx. for w^\star 
                if com_sets.join_wdec:
                    device_w[i].mul_(1-wdecay)
                    ewg_t = 1*device_w[i]
                else: 
                    ewg_t = device_w[i].mul(1-wdecay) 
                ewg_t.mul_(m_t[i])
                
                # restart logic: idea is that, after the first k epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs, with an averagely constant learning-rate for each parameter.
                # cyclic step
                step_c = (((step_this-1) % (com_sets.spe*com_sets.epfreq)) + 1)
                # restart lowpass state at the end of every 'k' epoch.
                if com_sets.restarts and step_c == 1 and step_this > 1:
                    device_lrb[i].mul_(0).add_(com_sets.rstfact*(lr_init+device_lrb[i]))
                    device_lra[i].mul_(0).add_(com_sets.rstfact*(lr_init+device_lra[i]))
                
                # linear correlation estimate update  
                # (double averaging [lowpass]), 
                # Note: future work: how to estimate this more accurately?
                lra_t, device_lrb[i], device_lra[i] = com_sets.lpf.compute_dbl(in_k=ewg_t, x1=device_lrb[i], x2=device_lra[i], beta1=beta_a, beta2=beta_e, step=step_this)
                # lra_t, device_lra[i] = com_sets.lpf.compute(in_k=ewg_t, x=device_lra[i], beta=beta_e, step=step_this, mode=3)
  
                # abs. val projection. to ensure positive rates.
                alpha_hat_t.append(lra_t.abs_())
                
            com_sets.lpf.abnormal = True
            
            # log lr
            if com_sets.lrlogstep:
                # we want to do this per step
                torch._foreach_zero_(device_lr)
            # else:
            # to save time and memory,
            # we want to do this per epoch
            # but get the average over all steps in that epoch. 
            torch._foreach_add_(device_lr, alpha_hat_t)
            
        elif com_sets.autolr and auto_mode==1: # 1984 impl.
            # no normalization is done here.
            # lr_init 
            
            # computes numerator of the bayes-optimal step_size
            # computes approx. linear correlation funcion 
            
            # denominator of the bayes-optimal step_size
            # in: averaging [lowpass] ~ approx. input variance
            v_var_t, device_grads_var = com_sets.lpf.compute(in_k=gradss_t_min_1, x=device_grads_var, beta=beta_e, step=step_this, sq=True)
            
            # normalized input        
            # malt_t = torch._foreach_div(m_t, torch._foreach_sqrt(torch._foreach_add(v_var_t,eps)))
            torch._foreach_div_(gradss_t_min_1, torch._foreach_add(torch._foreach_sqrt(v_var_t), eps)) 

            alpha_hat_t = []
            com_sets.lpf.abnormal = False
            # since we're doing in-place ops., we can't use the zip iterator!
            for i in range(len(params_)):
                
                # surrogate approx. for w^\star 

                ewg_t = device_lra[i].mul(gradss_t_min_1[i]) 
                
                # restart logic: idea is that, after the first k epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs, with an averagely constant learning-rate for each parameter.
                # cyclic step
                step_c = (((step_this-1) % (com_sets.spe*com_sets.epfreq)) + 1)
                # restart lowpass state at the end of every 'k' epoch.
                if com_sets.restarts and step_c == 1 and step_this > 1:
                    device_lrb[i].mul_(0).add_(com_sets.rstfact*(lr_init+device_lrb[i]))
                    device_lra[i].mul_(0).add_(com_sets.rstfact*(lr_init+device_lra[i]))
                
                # linear correlation estimate update  
                # (double averaging [lowpass]), 
                # Note: future work: how to estimate this more accurately?
                # lra_t, device_lrb[i], device_lra[i] = com_sets.lpf.compute_dbl(in_k=ewg_t, x1=device_lrb[i], x2=device_lra[i], beta1=beta_a, beta2=beta_e, step=step_this)
                lra_t, device_lrb[i] = com_sets.lpf.compute(in_k=ewg_t, x=device_lrb[i], beta=beta_e, step=step_this, mode=3)
  
                # abs. val projection. to ensure positive rates.
                device_lra[i].addcmul_(gradss_t_min_1[i], lra_t.abs())

                # device_lrb[i].mul_(0).add_(lr_init)                
                # device_lrb[i].mul_(0).add_(lr_init/10)                
                # device_lrb[i].mul_(0).add_(device_lra[i])
                # device_lra[i].addcmul_(gradss_t_min_1[i], device_lrb[i])
                
                # restarts:
                # push close to zero, if a negative element occurs.
                alpha_hat_t.append(device_lra[i].relu().add(eps))
                
            com_sets.lpf.abnormal = True
            
            # log lr
            if com_sets.lrlogstep:
                # we want to do this per step
                torch._foreach_zero_(device_lr)
            # else:
            # to save time and memory,
            # we want to do this per epoch
            # but get the average over all steps in that epoch. 
            torch._foreach_add_(device_lr, alpha_hat_t)

        else:
            # use externally supplied linear correlation estimate 
            alpha_hat_t = lr_init
        
        # update
        if not device_has_sparse_grad:
            # integrate: state update
            if com_sets.autolr:
                if com_sets.down:
                    torch._foreach_addcmul_(device_w, m_t, alpha_hat_t, value=1)    
                else:
                    torch._foreach_addcmul_(device_w, m_t, alpha_hat_t, value=-1)          
            else:
                if com_sets.down:
                    torch._foreach_mul_(m_t, alpha_hat_t)
                    torch._foreach_add_(device_w, m_t)
                else:
                    torch._foreach_mul_(m_t, -alpha_hat_t)
                    torch._foreach_add_(device_w, m_t)
            
            # smooth out. [lowpass]
            w_est, device_w_smth = com_sets.lpf.compute(in_k=device_w, x=device_w_smth, beta=beta_o, step=step_this, mode=3)
                
            # pass estimated/updated weight values back to the neural network's placeholder.
            torch._foreach_zero_(params_)
            torch._foreach_add_(params_, w_est)
                
        else:
            # foreach APIs don't support sparse
            # integrate: state update
            for i in range(len(params_)):
                if com_sets.down:
                    device_w[i].addcmul_(m_t, alpha_hat_t, value=1)
                else:
                    device_w[i].addcmul_(m_t, alpha_hat_t, value=-1) 
                
                # smooth out. [lowpass]
                w_est, device_w_smth[i] = com_sets.lpf.compute(in_k=device_w[i], x=device_w_smth[i], beta=beta_o, step=step_this, mode=3)
                    
                # pass estimated/updated weight values back to the neural network's placeholder.                
                params_[i].mul_(0).add_(w_est)


def _fused_sgm(com_sets:common_sets, steps: List[Tensor], 
        params: List[Tensor], weight_list: List[Tensor], weight_smth_list: List[Tensor], 
        grad_list: List[Tensor], grad_smth_list: List[Optional[Tensor]],
        grad_var_list: List[Optional[Tensor]],  
        lr_avgb_list: List[Optional[Tensor]], lr_avga_list: List[Optional[Tensor]], lr_save_list: List[Optional[Tensor]],*,has_sparse_grad:bool,       
        differentiable:Optional[bool],
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor]):
    
    if len(params) == 0:
        return
    
    assert grad_scale is None and found_inf is None        
    
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_var_list, lr_avgb_list, lr_avga_list, lr_save_list,
        steps])
    
    firstgrp=True
    for (device, dtype) in grouped_tensors:
        (
            device_params, device_w, device_w_smth, 
            device_grads,device_grads_smth, device_grads_var, 
            device_lrb, device_lra, device_lr, device_steps
        ) = grouped_tensors[(device, dtype)] 

        
        # flatten group
        f_params, f_w, f_w_smth, f_grads, f_grads_smth, f_grads_var, f_lrb, f_lra, f_lr, nel = _fuse_grouped_tensors(
            [device_params, device_w, device_w_smth, device_grads,device_grads_smth, device_grads_var, device_lrb, device_lra, device_lr], 
            device, dtype
            )
        
        if device_steps[0] == 1 and firstgrp:    
            com_sets.lpf.abnormal = False     
            com_sets.est_numels = nel
            # LOG.
            com_sets.log_stats()     
            firstgrp = False
        
        
        lr_init, beta_i, beta_e, beta_a, \
        beta_o, beta_d, eps, \
        wdecay, fzero, fone, auto_mode = com_sets.grp_devdt(device,dtype)
            
        step = device_steps[0]

        w_t = f_w
        w_smth = f_w_smth
        grad = f_grads
        grad_smth = f_grads_smth
        grad_var = f_grads_var
        lr_avgb = f_lrb
        lr_avga = f_lra
        lr = f_lr
            
        # handle complex parameters
        if torch.is_complex(f_params):
            f_params = torch.view_as_real(f_params)
            w_t = torch.view_as_real(w_t)
            w_smth = torch.view_as_real(w_smth)
            grad = torch.view_as_real(grad)
            grad_smth = torch.view_as_real(grad_smth)
            grad_var = torch.view_as_real(grad_var)          
            lr_avgb = torch.view_as_real(lr_avgb)
            lr_avga = torch.view_as_real(lr_avga)
            lr = torch.view_as_real(lr)
            
        
        # START
        mwd = 1-wdecay
        g_t = com_sets.lpf.patch(grad,beta=beta_o,step=step,mode=3) 
        if com_sets.join_wdec:
            # decay weight directly
            w_t.mul_(mwd)
        else:
            # decay weight as l2-regularized gradient
            g_t.addcmul_(w_t,wdecay)
        if com_sets.down: g_t.neg_()
                        
        # flip sign, if maximizing.
        if com_sets.maximize: g_t.neg_()
        
        # smooth input [lowpass]
        m_t_min_1 = com_sets.lpf.previous(xkm1=grad_smth, beta=beta_i, step=step)
        
        gradss_t_min_1 = m_t_min_1.mul(g_t)
        
        m_t, grad_smth = com_sets.lpf.compute(in_k=g_t, x=grad_smth, beta=beta_i, step=step)
            
        # smooth input [add average time-diff] 
        if step == 1: v_diff = 0
        else: v_diff = smooth_td(beta_d, m_t, m_t_min_1)
        g_t.add_(v_diff)
        m_t.add_(v_diff)
        
        if auto_mode in [0,2]:
            # denominator of the bayes-optimal step_size
            # in: averaging [lowpass] ~ approx. input variance
            v_var_t, grad_var = com_sets.lpf.compute(in_k=(g_t*g_t.conj()), x=grad_var, beta=beta_e, step=step)
            
            # normalized input
            # malt_t = m_t.div((v_var_t.add(eps)).sqrt_())
            (m_t).div_((v_var_t.sqrt_()).add_(eps))

        # learning rate estimation
        if com_sets.autolr:
            # computes numerator of the bayes-optimal step_size
            # computes approx. linear correlation funcion 
            
            # surrogate approx. for w^\star 
            if com_sets.join_wdec:
                ewg_t = m_t.mul(w_t)
            else:
                ewg_t = m_t.mul(w_t.mul(mwd))

            # restart logic: idea is that, after the first k>0 epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs [with an averagely constant/cyclic learning-rate for each parameter.] without becoming too small.
            # cyclic step
            step_c = (((step-1) % (com_sets.spe*com_sets.epfreq)) + 1)
            # scaling and restart lowpass state at the end of every 'k' epoch.
            if com_sets.restarts and step_c == 1 and step > 1:
                lr_avgb.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avgb))
                lr_avga.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avga))
                            
            # linear correlation estimate update  
            # (double averaging [lowpass]), 
            # Note: future work: how to estimate this more accurately?    
            lrat, lr_avgb, lr_avga = com_sets.lpf.compute_dbl(in_k=ewg_t, x1=lr_avgb, x2=lr_avga, beta1=beta_a, beta2=beta_e, step=step)
            
            # abs. val projection. to ensure positive rates.
            alpha_hat_t = (lrat.abs_())
            
        elif com_sets.autolr and auto_mode==1: # 1984 impl.
            # no normalization is done here.
            # lr_init 
            
           
            lr_avgb.mul_(0).add_(lr_init)
            lr_avga.addcmul_(gradss_t_min_1, lr_avgb)
            
            # restarts:
            # push close to zero, if a negative element occurs.
            lrat = lr_avga.relu().add(eps)
            
            alpha_hat_t = 1*lrat
        else:
            # use externally supplied linear correlation estimate 
            alpha_hat_t = lr_init      

        # integrate: state update
        if com_sets.down:
            w_t.addcmul_(m_t, alpha_hat_t)
        else:
            w_t.addcmul_(m_t, alpha_hat_t, value=-1)
        
        # smooth out. [lowpass]
        w_est, w_smth = com_sets.lpf.compute(in_k=w_t, x=w_smth, beta=beta_o, step=step, mode=3)
            
        # pass estimated/updated weight values back to the neural network's placeholder.   
        f_params.mul_(0).add_(w_est)
        
        # log lr
        if com_sets.autolr and com_sets.lrlogstep:
            # we want to do this per step
            lr.mul_(0).add_(alpha_hat_t)
        elif com_sets.autolr and not com_sets.lrlogstep:
            # to save time and memory,
            # we want to do this per epoch
            # but get the average over all steps in that epoch. 
            lr.add_(alpha_hat_t)     
        # END    
            

def _fuse_grouped_tensors(tensorlists, device, dtype):
    ''' helper function to fuse tensors in a list'''
    fused_tensorlists = []
    m = []
    nnel = None
    first = True
    for tensorlist in tensorlists:
        
        if first==True:
            # compute len of each param and sum
            m += [p.numel() for p in tensorlist if isinstance(p, torch.Tensor)]
            nnel = sum(im for im in m)
            first = False
            
        n = sum(p.numel() for p in tensorlist if isinstance(p, torch.Tensor))
        
        if n > 0:
            fused_tensorlist = torch.zeros(n, dtype=dtype, device=device)
            
            # fuse ops
            i = 0
            for p in tensorlist: 
                params_slice = fused_tensorlist[i:i + p.numel()]
                with torch.no_grad(): params_slice.copy_(p.flatten())
                p.data = params_slice.view(p.shape)
                i += p.numel()
            fused_tensorlists.append(fused_tensorlist)
        
        else:
            # list of scalars, non-tensor type to match network params
            # for auto_mode.
            sctensorlist = torch.zeros(nnel, dtype=dtype, device=device)
            i = 0
            for p, im in zip(tensorlist, m):
                sctensorlist[i:i + im] = p
                i += im
            fused_tensorlists.append(sctensorlist)
    
    # nnel = fused_tensorlists[0].shape[0]
    fused_tensorlists.append(nnel)
    return fused_tensorlists



# def fuse_parameters_and_gradients(mdlparams):
#     """Move model parameters and gradients to a contiguous tensor, and return that tensor."""
#     n = sum(p.numel() for p in mdlparams)
#     params = torch.zeros(n, requires_grad=True)
#     params.grad = torch.zeros(n)
#     i = 0
#     for p in mdlparams():
#         params_slice = params[i:i + p.numel()]
#         with torch.no_grad(): params_slice.copy_(p.flatten())
#         p.data = params_slice.view(p.shape)
#         p.grad = params.grad[i:i + p.numel()].view(p.shape)
#         i += p.numel()
#     return params

# params = fuse_parameters_and_gradients(model)



