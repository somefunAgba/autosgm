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

def cmplx2real(lols):
    "input: List of Lists"
    return [[torch.view_as_real(tsr) 
            if torch.is_complex(tsr) else tsr 
                for tsr in lst] 
                    for lst in lols]


# LPF0
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

            elif mode == 6: # exponential. (stable if 0 \le \beta < 1) 

                # forward:
                (x.mul_(betak)).add_(in_k)
                out_k = x*(one_minus_betak)  

                
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

            elif mode == 6: # exponential. (stable if 0 \le \beta < 1) 
                # (x.mul_(betak)).add_(in_k)
                # out_k = x*(one_minus_betak)  
                # forward:
                torch._foreach_mul_(x, betak)
                if not sq:
                    torch._foreach_add_(x, in_k)
                elif sq:
                    torch._foreach_addcmul_(x, in_k, in_k)
                out_k = torch._foreach_mul(x, one_minus_betak)  
                               
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

    @torch.no_grad()
    def kf_compute(self, in_t:Tensor, var_in_t:Tensor, x_t:Tensor, var_x_t:Tensor, step:Tensor, beta_init:Tensor):
        '''
        kf_compute _summary_

        kalman-filter LPF

        Args:
            in_t (Tensor): _description_
            var_in_t (Tensor): _description_
            x_t (Tensor): _description_
            var_x_t (Tensor): _description_
            step (Tensor): _description_
            beta_init (Tensor): _description_

        Returns:
            _type_: _description_
        '''
        
        if not self.abnormal:
            if step == 1:
                var_x_t.mul_(0).add_((1-beta_init)*var_in_t)
            
            var_vt = (var_in_t - var_x_t).abs_()
            alph_t = var_x_t/(var_x_t + var_vt)
            beta_t = 1 - alph_t
            # forward:
            x_t.mul_(beta_t).add_(alph_t*in_t)
            var_x_t.mul_(beta_t)
            # out.
            unden = (1-beta_t.pow(step))
            out_t = 1*x_t/(unden)
            var_out_t = 1*var_x_t/(unden.square())
        else:
            
            if step == 1:
                torch._foreach_mul_(var_x_t, 0)
                torch._foreach_add_(var_x_t, torch._foreach_mul(var_in_t,(1-beta_init)))            
            
            var_vt = torch._foreach_abs(torch._foreach_sub(var_in_t,var_x_t))
            alph_t = torch._foreach_div(var_x_t, torch._foreach_add(var_x_t,var_vt))
            beta_t = torch._foreach_add(torch._foreach_neg(alph_t),1)
           
            torch._foreach_mul_(x_t, beta_t)
            torch._foreach_add_(x_t, torch._foreach_mul(in_t, alph_t))      
            torch._foreach_mul_(var_x_t, beta_t)
            
            unden = [1-bpt.pow(step) for bpt in beta_t]
            out_t = torch._foreach_div(x_t,unden)
            var_out_t = torch._foreach_div(var_x_t,torch._foreach_mul(unden,unden))
            
        return out_t, x_t, var_out_t, var_x_t
    
    
class common_sets():
    """ Commons 
    """
    
    def __init__(self, lpf:LPF,norm_all:bool,p, 
                lr_init, epfreq, betas, beta_o, beta_d, 
                eps, spe, wd_cte, 
                autolr:bool, restarts:bool, maximize:bool, join_wdec:bool, lrlogstep:bool, down:bool) -> None:
        self.down = down
        self.lrlogstep = lrlogstep
        self.lpf = lpf
        self.p = p
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
        self.levelnorm = norm_all
        
    @torch.no_grad()
    def grp_devdt(self, device, dtype):
        fzero = torch.tensor(0., dtype=dtype, device=device)
        fone = torch.tensor(1., dtype=dtype, device=device)
        self.dev_lr_init = self.lr_init*fone
        self.dev_beta_i = self.beta_i*fone
        self.dev_beta_e = self.beta_e*fone
        self.dev_beta_a = self.beta_a*fone
        self.dev_beta_o = self.beta_o*fone
        self.dev_beta_d = self.beta_d*fone
        self.dev_eps = self.eps*fone
        self.dev_wd_cte = self.wd_cte*fone
        self.levels = self.p       
    
    @torch.no_grad()
    def log_stats(self, params=None):      
        
        if params is not None:
            self.est_numels = sum(p.numel() for p in params)  
        # logging.
        txt = "="
        infostr = "AutoSGM info:\t"          
        strlog = f"{infostr} [total params, d={self.est_numels}], restarts={self.restarts}\n[autolr={self.autolr}, levels={self.p}, init_lr={self.lr_init:.5g} |\n lpf. [i,e,o] : {self.beta_i:.4g}, {self.beta_e:.4g}, {self.beta_o:.4g} | hpf. : {self.beta_d:.4g} |\n eps : {self.eps:.4g} | weight-decay: {self.wd_cte:.4g}]"
        # debug  
        print(f"{txt * int(0.35*len(strlog)) }")
        print(strlog) #      
        print(f"{txt * int(0.35*len(strlog)) }\n")
            

    # Smooth HPF
    def smooth_td(self, curr:Tensor, past:Tensor):
        """ Smooth First-order Digital Differentiator
        
        A smooth digital high-pass filter for time-differentiation
        
        Args:
            beta_d (>= 0)
            
            curr current input
            
            past past input
        """
        
        return (curr-past).mul_(curr+past).mul_(0.5*self.beta_d)  

    # Trace Gradient Inputs
    def grader(self, step, pl, grad_lev, grad_smth_lev, grad_var_lev, gradsmth_levm1_old, param_lev):
        
        if not self.lpf.abnormal:
            # get gradient for this level
            if pl != 0:
                grad_lev[pl].mul_(0).add_(gradsmth_levm1_old.mul(grad_lev[pl-1]))
            g_t = 1*grad_lev[pl]  
            
            # add weight decay directly or not
            if pl == 0:  
                g_t = self.lpf.patch(g_t,beta=self.dev_beta_o,step=step) 
                if self.dev_wd_cte is not None: 
                    if self.join_wdec:
                        # decay weight directly
                        param_lev[pl].mul_(1-self.dev_wd_cte)
                    else:
                        # decay weight as l2-regularized gradient
                        g_t.addcmul_(param_lev[pl],self.dev_wd_cte)
                
            # flip sign, if maximizing.   
            if self.down or self.maximize: 
                g_t.neg_()
                            
            # smooth gradient input [lowpass]
            if pl == 0:  
                m_t_min_1 = self.lpf.previous(xkm1=grad_smth_lev[pl], beta=self.dev_beta_i, step=step)
                m_t, grad_smth_lev[pl] = self.lpf.compute(in_k=g_t, x=grad_smth_lev[pl], beta=self.dev_beta_i, step=step)
            else:
                m_t_min_1 = 1*grad_smth_lev[pl]
                grad_smth_lev[pl].mul_(0).add_(g_t)
                m_t = 1*g_t

            # add [add average time-diff] to graidient 
            if self.dev_beta_d > 0:
                if step == 1: v_diff = 0
                else: v_diff = self.smooth_td(m_t, m_t_min_1)
                g_t.add_(v_diff)
                m_t.add_(v_diff)
                
            # denominator of the optimal step_size
            # in: averaging [lowpass] ~ approx. input variance   
            if pl == self.levels-1 or self.levelnorm:
                # normalize all levels or normalize last level only  
                if step > 1:
                    v_var_t_min_1 = self.lpf.previous(xkm1=grad_var_lev[pl], beta=self.dev_beta_i, step=step)
                    m_t_min_1.div_((v_var_t_min_1.sqrt()).add_(self.dev_eps))

                v_var_t, grad_var_lev[pl] = self.lpf.compute(in_k=(g_t*g_t.conj()), x=grad_var_lev[pl], beta=self.dev_beta_e, step=step)
                
                # if pl == 0:
                #     # var of smooth grad.
                #     obt_mdp = (1-self.dev_beta_i)/(1+self.dev_beta_i)
                #     bti_t = self.dev_beta_i.pow(step)
                #     obt_pdm_t = (1+bti_t)/(1-bti_t)
                #     v_var_t.mul_(obt_mdp*obt_pdm_t)
                
                # normalized input
                # malt_t = m_t.div((v_var_t.add(eps)).sqrt_())
                (m_t).div_((v_var_t.sqrt()).add_(self.dev_eps))    
        
        elif self.lpf.abnormal: # operating on lists
            # get gradient for this level from all layers
            gpl = [ allist[pl] for allist in grad_lev]
            gsmthpl = [ allist[pl] for allist in grad_smth_lev]
            gvarpl = [ allist[pl] for allist in grad_var_lev]
            parampl = [ allist[pl] for allist in param_lev]
            if pl != 0:
                gpl_m1 = [ allist[pl-1] for allist in grad_lev]
                torch._foreach_mul_(gpl,0)
                torch._foreach_add_(gpl, torch._foreach_mul(gpl_m1,gradsmth_levm1_old))
            g_t = torch._foreach_mul(gpl,1)
            
            # add weight decay directly or not
            if pl == 0:  
                g_t = self.lpf.patch(g_t,beta=self.dev_beta_o,step=step,mode=3) 
                if self.dev_wd_cte is not None: 
                    if self.join_wdec:
                        # decay weight directly
                        torch._foreach_mul_(parampl,1-self.dev_wd_cte)
                    else:
                        # decay weight as l2-regularized gradient
                        torch._foreach_add_(g_t, torch._foreach_mul(parampl,self.dev_wd_cte))
                
            # flip sign, if maximizing.   
            if self.down or self.maximize: 
                torch._foreach_neg_(g_t)
                            
            # smooth gradient input [lowpass]
            if pl == 0:  
                m_t_min_1 = self.lpf.previous(xkm1=gsmthpl, beta=self.dev_beta_i, step=step)
                m_t, gsmthpl = self.lpf.compute(in_k=g_t, x=gsmthpl, beta=self.dev_beta_i, step=step)
            else:
                m_t_min_1 = torch._foreach_mul(gsmthpl,1)
                torch._foreach_mul_(gsmthpl,0)
                torch._foreach_add_(gsmthpl,g_t)
                m_t = torch._foreach_mul(g_t,1)

            # add [add average time-diff] to graidient 
            if self.dev_beta_d > 0:
                if step == 1: v_diff = 0
                else:             
                    v_diff = [self.smooth_td(mti, mti_mone) 
                        for mti, mti_mone in zip(m_t, m_t_min_1)
                    ]   
                torch._foreach_add_(g_t, v_diff) 
                torch._foreach_add_(m_t, v_diff)
                
            # denominator of the optimal step_size
            # in: averaging [lowpass] ~ approx. input variance   
            if pl == self.levels-1 or self.levelnorm:
                # normalize all levels or normalize last level only  
                if step > 1:
                    v_var_t_min_1 = self.lpf.previous(xkm1=gvarpl, beta=self.dev_beta_e, step=step)
                    torch._foreach_div_(m_t_min_1, torch._foreach_add(torch._foreach_sqrt(v_var_t_min_1), self.dev_eps))

                v_var_t, gvarpl = self.lpf.compute(in_k=g_t, x=gvarpl, beta=self.dev_beta_e, step=step, sq=True)
                
                # normalized input           
                # malt_t = torch._foreach_div(m_t, torch._foreach_sqrt(torch._foreach_add(v_var_t,eps)))
                torch._foreach_div_(m_t, torch._foreach_add(torch._foreach_sqrt(v_var_t), self.dev_eps)) 
                      
        return g_t, m_t, m_t_min_1    
                
    # Integrator
    def integrator(self, w_t, m_t, rpl, alpha_hat_t):
        if not self.lpf.abnormal:
            if rpl == 0:
                if self.down:
                    w_t[rpl].addcmul_(m_t[rpl], alpha_hat_t)
                else:
                    w_t[rpl].addcmul_(m_t[rpl], alpha_hat_t, value=-1)
            else:
                if self.down:
                    w_t[rpl].addcmul_(m_t[rpl], alpha_hat_t, value=1)
                else:
                    w_t[rpl].addcmul_(m_t[rpl], alpha_hat_t)
        elif self.lpf.abnormal:
            wrpl = [ allist[rpl] for allist in w_t]
            if rpl== 0:
                if self.autolr:
                    if self.down:
                        torch._foreach_addcmul_(wrpl, m_t[rpl], alpha_hat_t, value=1)    
                    else:
                        torch._foreach_addcmul_(wrpl, m_t[rpl], alpha_hat_t, value=-1)          
                else:
                    if self.down:
                        torch._foreach_mul_(m_t[rpl], alpha_hat_t)
                        torch._foreach_add_(wrpl, m_t[rpl])
                    else:
                        torch._foreach_mul_(m_t[rpl], -alpha_hat_t)
                        torch._foreach_add_(wrpl, m_t[rpl])
            else:
                if self.down:
                    torch._foreach_addcmul_(wrpl, m_t[rpl], alpha_hat_t, value=1)    
                else:
                    torch._foreach_addcmul_(wrpl, m_t[rpl], alpha_hat_t, value=1)                  

    # Smooth output
    def smooth_out(self, step, rpl, param, w_t, w_smth):
        if rpl == 0:
            if not self.lpf.abnormal:
                # optional smooth out. [lowpass]
                if self.dev_beta_o > 0:
                    w_est, w_smth[rpl] = self.lpf.compute(in_k=w_t[rpl], x=w_smth[rpl], beta=self.dev_beta_o, step=step, mode=3)
                    
                    param.mul_(0).add_(w_est)  
                else:
                    param.mul_(0).add_(w_t[rpl])
                # end for
            else: 
                wrpl = [ allist[rpl] for allist in w_t]          
                if self.beta_o > 0: 
                    wsmthrpl = [ allist[rpl] for allist in w_smth]
                    # smooth
                    w_est, wsmthrpl = self.lpf.compute(in_k=wrpl, x=wsmthrpl, beta=self.dev_beta_o, step=step, mode=3)
                    # pass on
                    torch._foreach_zero_(param)
                    torch._foreach_add_(param, w_est)
                else:                    
                    torch._foreach_zero_(param)
                    torch._foreach_add_(param, wrpl)
                    
                
      
    # Compute Proportional Gain
    def lr_compute(self, step, rpl, 
                m_t, w_t, lr_avga, lr_avgb):
        
        # com_sets.autolr = False
        # computes numerator of the bayes-optimal step_size
        # computes approx. linear correlation funcion         
        # learning rate estimation
        if self.autolr:
            
            if not self.lpf.abnormal:
            
                # surrogate approx. without w^\star 
                ewg_t = m_t[rpl].mul(w_t[rpl])
                                
                # restart logic: idea is that, after the first k epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs, with an averagely constant learning-rate for each parameter.
                # cyclic step
                step_c = (((step-1) % (self.spe*self.epfreq)) + 1)
                # restart lowpass state at the end of every 'k' epoch.
                if self.restarts and step_c == 1 and step > 1:
                    reval = self.rstfact*(self.lr_init+lr_avgb[rpl])
                    lr_avgb[rpl].mul_(0).add_(reval)
                    reval = self.rstfact*(self.lr_init+lr_avga[rpl])
                    lr_avga[rpl].mul_(0).add_(reval)
            
                # linear cov. or correlation estimate update  
                # Note: future work: how to estimate this more accurately?
                if rpl == 0: 
                    # (double averaging [lowpass]), 
                    # lrat, lr_avgb[rpl], lr_avga[rpl] = self.lpf.compute_dbl(in_k=ewg_t, x1=lr_avgb[rpl], x2=lr_avga[rpl], beta1=self.dev_beta_a, beta2=self.dev_beta_e, step=step)
                    lrat, lr_avgb[rpl] = self.lpf.compute(in_k=ewg_t, x=lr_avgb[rpl], beta=self.dev_beta_e, step=step, mode=3)  
                    # abs. val projection. to ensure positive rates.
                    alpha_hat_t = (lrat.abs())
                else:
                    lrat, lr_avgb[rpl] = self.lpf.compute(in_k=ewg_t, x=lr_avgb[rpl], beta=self.dev_beta_e, step=step, mode=3)   
                    # abs. val projection. to ensure positive rates.
                    alpha_hat_t = (lrat.abs())
                    # alpha_hat_t = (lrat.relu().add(eps))                 

            elif self.lpf.abnormal:
                wrpl = [ allist[rpl] for allist in w_t]
                lrbrpl = [ allist[rpl] for allist in lr_avgb]
                lrarpl = [ allist[rpl] for allist in lr_avga]    
                
                # surrogate approx. without w^\star 
                ewg_t = torch._foreach_mul(m_t[rpl],wrpl)            
                # restart logic: idea is that, after the first k epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs, with an averagely constant learning-rate for each parameter.
                # cyclic step
                step_c = (((step-1) % (self.spe*self.epfreq)) + 1)
                # restart lowpass state at the end of every 'k' epoch.
                if self.restarts and step_c == 1 and step > 1:

                    
                    reval = torch._foreach_mul(torch._foreach_add(lrbrpl, self.lr_init), self.rstfact)
                    torch._foreach_mul_(lrbrpl,0)
                    torch._foreach_add_(lrbrpl,reval)
                    
                    reval = torch._foreach_mul(torch._foreach_add(lrarpl, self.lr_init), self.rstfact)
                    torch._foreach_mul_(lrarpl,0)
                    torch._foreach_add_(lrarpl, reval)
            
                # linear cov. or correlation estimate update  
                # Note: future work: how to estimate this more accurately?
                if rpl == 0: 
                    # (double averaging [lowpass]), 
                    lrat, lrbrpl, lrarpl = self.lpf.compute_dbl(in_k=ewg_t, x1=lrbrpl, x2=lrarpl, beta1=self.dev_beta_a, beta2=self.dev_beta_e, step=step)
                    
                    # abs. val projection. to ensure positive rates.
                    alpha_hat_t = torch._foreach_abs(lrat)
                else:
                    lrat, lrbrpl = self.lpf.compute(in_k=ewg_t, x=lrbrpl, beta=self.dev_beta_e, step=step, mode=3)    
                               
                    # abs. val projection. to ensure positive rates.
                    alpha_hat_t = torch._foreach_abs(lrat)
                
        else:
            # use a externally supplied (typically small) linear correlation estimate or value
            alpha_hat_t = self.dev_lr_init   
            
        return alpha_hat_t   
    
    # Log LR or SS                    
    def logginglr(self, rpl, lrm, lrsq, alpha_hat_t):
        if rpl == self.p-1:
            if not self.lpf.abnormal:
                if self.autolr and self.lrlogstep:
                    # we want to do this per step
                    lrm[rpl].mul_(0).add_(alpha_hat_t)
                elif self.autolr and not self.lrlogstep:
                    # to save time and memory,
                    # we want to do this per epoch
                    # but get the average over all steps in that epoch. 
                    lrm[rpl].add_(alpha_hat_t)
                    lrsq[rpl].add_(alpha_hat_t.square())
            elif self.lpf.abnormal:            
                if self.autolr and self.lrlogstep:
                    lrmrpl = [ allist[rpl] for allist in lrm]
                    # we want to do this per step
                    torch._foreach_zero_(lrmrpl)
                    torch._foreach_add_(lrmrpl, alpha_hat_t)
                elif self.autolr and not self.lrlogstep:
                    lrmrpl = [ allist[rpl] for allist in lrm]
                    lrsqrpl = [ allist[rpl] for allist in lrsq] 
                    # to save time and memory,
                    # we want to do this per epoch
                    # but get the average over all steps in that epoch. 
                    torch._foreach_add_(lrmrpl, alpha_hat_t)
                    torch._foreach_add_(lrsqrpl, torch._foreach_mul(alpha_hat_t,alpha_hat_t))
            
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
    
    def __init__(self, params, *, lr_init=1e-3, spe=1, epfreq=10,
                beta_i=0.9, beta_e=0.999, beta_a=0.99999,
                eps=1e-8, weight_decay=0,
                beta_o=0, beta_d=0,
                levels=1,
                norm_all=True, 
                autolr:bool=True, restarts:bool=False,
                join_wdec:bool=False, down:bool=False,
                maximize:bool=False, lrlogstep:bool=True, 
                foreach: Optional[bool]=None,
                fused: Optional[bool]=False,
                differentiable: bool=False):
        
        '''
        Inits optimizer: (Auto)SGM of p levels (levels are like layers)
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
        self.nodes = levels
        
        defaults = dict(p=levels,lr_init=lr_init,betas=(beta_i,beta_e,beta_a),
                        beta_o=beta_o,
                        beta_d=beta_d, 
                        spe=spe, norm_all=norm_all,
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
            group.setdefault('beta_o', 0)
            group.setdefault('beta_d', 0)
            group.setdefault('p',1)
            group.setdefault('join_wdec', False)
            group.setdefault('com_sets', None)
            
            
    @torch.no_grad()
    def zero_logged_lr(self):
        """zero lr logged in last epoch.
        
            This will help save time in plotting logged lrs
            since we average the whole lrs logged in an epoch by the total steps in that epoch.
        """
        
        
        for group in self.param_groups:
          lev = group['p']
          
          lrm_save_list = [
              self.state[p]['levels'][f'{lev}']["lr_m_save"] 
              for p in group['params'] 
              if p.grad is not None
          ]
          torch._foreach_zero_(lrm_save_list)
          
          lrsq_save_list = [
              self.state[p]['levels'][f'{lev}']["lr_sq_save"] 
              for p in group['params'] 
              if p.grad is not None
          ]
          torch._foreach_zero_(lrsq_save_list)
                
                
    def _init_group(self, group):
        '''
        Inits state of params
        '''
        
        if 'com_sets' not in group or group['com_sets'] is None:
            group['com_sets'] = common_sets(
                self.lpf,group['norm_all'],group['p'],group['lr_init'], 
                group['epfreq'],
                group['betas'], group['beta_o'], group['beta_d'],
                group['eps'], group['spe'], group['weight_decay'],
                group['autolr'],group['restarts'],group['maximize'], group['join_wdec'], group['lrlogstep'], group['down']
            ) 
        
        com_sets = group['com_sets']
        has_sparse_grad = False       
        
        params_with_grad_list = []
        steps = []
        
        weight_list = []
        weight_smth_list = []
        grad_list = []
        grad_smth_list = []
        grad_var_list = []            
        lr_avgb_list = []
        lr_avga_list = []
        lrm_save_list = []        
        lrsq_save_list = []
        
                
        for p in group['params']:
            if p.grad is not None:
                params_with_grad_list.append(p)
                if p.grad.is_sparse: has_sparse_grad = True
                
                state = self.state[p]
                # Lazy state init.
                if len(state)==0:
                  
                  state['step'] = torch.tensor(0, dtype=torch.float, device=p.device)
                  
                  state['levels'] = dict()
                  for pl in range(group['p']):
                    lev = pl+1
                    state['levels'][f'{lev}'] = dict()
                    # if not state['levels'][f'{lev}']: # if level does not exist
                    # -  for all levels
                    state['levels'][f'{lev}']['grad_smth'] = torch.zeros_like(
                            p.real, memory_format=torch.preserve_format, device=p.device) 
                    if lev == 1:
                      state['levels'][f'{lev}']['grads'] = None
                      state['levels'][f'{lev}']['weight'] = p.real.clone(memory_format=torch.preserve_format).detach()   
                      state['levels'][f'{lev}']['weight_smth'] = p.real.clone(memory_format=torch.preserve_format).detach() 
                      state['levels'][f'{lev}']['grad_var'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                    else:
                      state['levels'][f'{lev}']['grads']= torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=p.device)
                      state['levels'][f'{lev}']['weight'] = group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)     
                      state['levels'][f'{lev}']['weight_smth'] = group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)    
                      state['levels'][f'{lev}']['grad_var'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                      
                    # - only for the last level.                
                    if lev == group['p']:
                      state['levels'][f'{lev}']["lr_avgb"] = group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)    
                      state['levels'][f'{lev}']["lr_avga"] = group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)    
                    else:
                      state['levels'][f'{lev}']["lr_avgb"] = None   
                      state['levels'][f'{lev}']["lr_avga"] = None
                      
                    # - for all levels (stores, mean and second moment for states.)              
                    state['levels'][f'{lev}']["lr_m_save"] = torch.zeros_like(
                            p.real, memory_format=torch.preserve_format, device=p.device)
                    state['levels'][f'{lev}']["lr_sq_save"] = torch.zeros_like(
                            p.real, memory_format=torch.preserve_format, device=p.device) 
                
                state['step'] += 1
                steps.append(state['step'])
                
                # Level Lists for this parameter
                weight_llist = []
                weight_smth_llist = []
                grad_llist = []
                grad_smth_llist = []
                grad_var_llist = []            
                lr_avgb_llist = []
                lr_avga_llist = []
                lrm_save_llist = []        
                lrsq_save_llist = []
                # -  for all levels
                for lev in range(1,group['p']+1):
                  grad_smth_llist.append(state['levels'][f'{lev}']['grad_smth'])
                  if lev == 1:
                    grad_llist.append(p.grad)
                  else:
                    grad_llist.append(state['levels'][f'{lev}']['grads'])
                  weight_llist.append(state['levels'][f'{lev}']['weight'])
                  weight_smth_llist.append(state['levels'][f'{lev}']['weight_smth'])      
                  
                  # - only for the last level.
                  grad_var_llist.append(state['levels'][f'{lev}']['grad_var'])       
                  lr_avgb_llist.append(state['levels'][f'{lev}']['lr_avgb'])
                  lr_avga_llist.append(state['levels'][f'{lev}']['lr_avga'])               
                  # - (stores, mean and second moment for states.)
                  lrm_save_llist.append(state['levels'][f'{lev}']['lr_m_save'])
                  lrsq_save_llist.append(state['levels'][f'{lev}']['lr_sq_save'])             

                # List of Level Lists for each 
                # parameter with a gradient in the ANN.
                grad_list.append(grad_llist)
                weight_list.append(weight_llist)
                weight_smth_list.append(weight_smth_llist)      
                grad_smth_list.append(grad_smth_llist)
                # - only for the last level.
                grad_var_list.append(grad_var_llist)       
                lr_avgb_list.append(lr_avgb_llist)
                lr_avga_list.append(lr_avga_llist)               
                # - for all levels (stores, mean and second moment for states.)
                lrm_save_list.append(lrm_save_llist)
                lrsq_save_list.append(lrsq_save_llist)
        
        return com_sets, has_sparse_grad, \
                params_with_grad_list, weight_list, weight_smth_list,\
                  grad_list, grad_smth_list, grad_var_list, lr_avgb_list,\
                    lr_avga_list, lrm_save_list, lrsq_save_list, steps
        
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
                        
            com_sets, has_sparse_grad, \
            params_with_grad_list, weight_list, weight_smth_list, grad_list, \
            grad_smth_list, grad_var_list, lr_avgb_list, lr_avga_list, \
            lrm_save_list, lrq_save_list, steps = self._init_group(group)
            
            sgm(com_sets, steps, 
                params_with_grad_list, 
                weight_list,
                weight_smth_list,
                grad_list, 
                grad_smth_list,
                grad_var_list,
                lr_avgb_list,
                lr_avga_list, 
                lrm_save_list,
                lrq_save_list,
                has_sparse_grad = has_sparse_grad,
                foreach=group['foreach'], 
                differentiable=group['differentiable'],  
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None))
            
            # update state
            pass
        
        return loss
            

def sgm(com_sets:common_sets, steps:List[Tensor], params: List[Tensor], 
        weight_list: List[List[Tensor]], 
        weight_smth_list: List[List[Optional[Tensor]]], 
        grad_list: List[List[Tensor]], 
        grad_smth_list: List[List[Optional[Tensor]]],
        grad_var_list: List[List[Optional[Tensor]]],
        lr_avgb_list: List[List[Optional[Tensor]]], 
        lr_avga_list: List[List[Optional[Tensor]]], 
        lrm_save_list: List[List[Tensor]],lrsq_save_list: List[List[Tensor]],*,
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
        lr_avgb_list, lr_avga_list, 
        lrm_save_list, lrsq_save_list,
        has_sparse_grad=has_sparse_grad,        
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf)
    
def _single_tensor_sgm(com_sets:common_sets, 
        steps: List[Tensor], params: List[Tensor], 
        weight_list: List[List[Tensor]], 
        weight_smth_list: List[List[Optional[Tensor]]], 
        grad_list: List[List[Tensor]], 
        grad_smth_list: List[List[Optional[Tensor]]],
        grad_var_list: List[List[Optional[Tensor]]],
        lr_avgb_list: List[List[Optional[Tensor]]], 
        lr_avga_list: List[List[Optional[Tensor]]], 
        lrm_save_list: List[List[Tensor]],lrsq_save_list: List[List[Tensor]],*,has_sparse_grad:bool,       
        differentiable:Optional[bool],
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor]):
    
    ''' Typical for loop implementation
    '''
    
    assert grad_scale is None and found_inf is None
    
    dtype = params[0].dtype
    device= params[0].device
    
    com_sets.grp_devdt(device,dtype)
    levels = com_sets.p
        
    # LOG.
    if steps[0] == 1: com_sets.log_stats(params)    
                
    for i, param in enumerate(params):
        step = steps[i]

        w_t = weight_list[i]
        w_smth = weight_smth_list[i]
        grad = grad_list[i]
        grad_smth = grad_smth_list[i]
        grad_var = grad_var_list[i]
        lr_avgb = lr_avgb_list[i]
        lr_avga = lr_avga_list[i]
        lrm = lrm_save_list[i]
        lrsq = lrsq_save_list[i]
        
        # handle if complex parameters
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            # for pl in range(levels):
            #   w_t[pl] = (torch.view_as_real(weight_list[i][pl]))
            #   w_smth[pl] = (torch.view_as_real(weight_smth_list[i][pl]))
            #   grad[pl] = (torch.view_as_real(grad_list[i][pl]))
            #   grad_smth[pl] = (torch.view_as_real(grad_smth_list[i][pl]))
            #   if (pl == levels-1) or com_sets.down: 
            #     grad_var[pl] = (torch.view_as_real(grad_var_list[i][pl]))
            #     #
            #     lr_avgb[pl] = (torch.view_as_real(lr_avgb_list[i][pl]))
            #     lr_avga[pl] = (torch.view_as_real(lr_avga_list[i][pl]))
            #     #
            #     lrm[pl] = (torch.view_as_real(lrm_save_list[i][pl]))
            #     lrsq[pl] = (torch.view_as_real(lrsq_save_list[i][pl]))          
        
        
        # - TRACE gradients
        g_t, m_t = [], []
        oldsmthval = 0
        for pl in range(levels):
          theval, smthval, oldsmthval = com_sets.grader(step, pl, grad, grad_smth, grad_var, oldsmthval, w_t)
          g_t.append(theval)
          m_t.append(smthval)            
        # end trace
         
        # # - TRACE gradients
        # g_t, m_t = [], []
        # oldsmthval = 0
        # for pl in range(levels):
        #   theval, smthval, oldsmthval = com_sets.grader(step, pl, grad, grad_smth, grad_var, grad_smth_var, oldsmthval, w_t)
        #   g_t.append(theval)
        #   m_t.append(smthval)            
        # # end trace
        
                
        # - AutoSGM Flow
        for rpl in range(levels-1, -1, -1):
          # compute step-size or lr.
          if rpl == levels-1:
            # compute lr, at last level
            alpha_hat_t = com_sets.lr_compute(step, rpl, m_t, w_t, lr_avga, lr_avgb)        
          else:
            # assign step_size
            alpha_hat_t = w_t[rpl+1].relu().add(com_sets.dev_eps) 
            
          # integrate: state update
          com_sets.integrator(w_t, m_t, rpl, alpha_hat_t)            
          # pass output: 
          # updated weight values back to the neural network's placeholder.
          com_sets.smooth_out(step, rpl, param, w_t, w_smth)           
            
          # logging lr
          com_sets.logginglr(rpl, lrm, lrsq, alpha_hat_t)

def _multi_tensor_sgm(com_sets:common_sets, 
        steps: List[Tensor], params: List[Tensor], 
        weight_list: List[List[Tensor]], 
        weight_smth_list: List[List[Optional[Tensor]]], 
        grad_list: List[List[Tensor]], 
        grad_smth_list: List[List[Optional[Tensor]]],
        grad_var_list: List[List[Optional[Tensor]]],
        lr_avgb_list: List[List[Optional[Tensor]]], 
        lr_avga_list: List[List[Optional[Tensor]]], 
        lrm_save_list: List[List[Tensor]],lrsq_save_list: List[List[Tensor]],*,has_sparse_grad:bool,       
        differentiable:Optional[bool],
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor]):
    
    if len(params) == 0: return
    
    assert grad_scale is None and found_inf is None

    if steps[0] == 1: com_sets.log_stats(params)
    
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_var_list, lr_avgb_list, lr_avga_list, lrm_save_list, lrsq_save_list, steps])

    for (device, dtype) in grouped_tensors:
        (
            dev_params, dev_wt, dev_wt_smth, 
            dev_grads,dev_grads_smth, dev_grads_var, 
            dev_lrb, dev_lra, dev_lrm, dev_lrsq, dev_steps
        ) = grouped_tensors[(device, dtype)] 

        com_sets.grp_devdt(device,dtype)
        levels =  com_sets.p
        step_this = dev_steps[0]
        
        # if step_this == 1: 
        device_has_sparse_grad = any(grad.is_sparse 
                                        for gradlist in dev_grads 
                                        for grad in gradlist )
        
        # handle complex parameters
        params_ = [torch.view_as_real(x) 
                if torch.is_complex(x) else x 
                for x in dev_params]
        # dev_wt = cmplx2real(dev_wt)
        
        # - TRACE gradients
        g_t, m_t = [], []
        oldsmthval = 0
        for pl in range(levels):
          theval, smthval, oldsmthval = com_sets.grader(step_this, pl, dev_grads, dev_grads_smth, dev_grads_var, oldsmthval, dev_wt)
          g_t.append(theval)
          m_t.append(smthval)            
        # end trace

        # - AutoSGM Flow
        for rpl in range(levels-1, -1, -1):
            # compute step-size or lr.
            if rpl == levels-1:
                # compute lr, at last level
                alpha_hat_t = com_sets.lr_compute(step_this, rpl, m_t, dev_wt, dev_lra, dev_lrb)        
            else:
                # assign step_size
                wrpl_p1 = [ allist[rpl+1] for allist in dev_wt]
                alpha_hat_t = torch._foreach_add(torch._foreach_clamp_min(wrpl_p1,0), com_sets.dev_eps)  
            
            if not device_has_sparse_grad:
                # integrate: state update
                com_sets.integrator(dev_wt, m_t, rpl, alpha_hat_t)            
                # pass output: 
                # updated weight values back to the neural network's placeholder.
                com_sets.smooth_out(step_this, rpl, params_, dev_wt, dev_wt_smth)           
                    
                # logging lr
                com_sets.logginglr(rpl, dev_lrm, dev_lrsq, alpha_hat_t)
                
            elif device_has_sparse_grad:
                com_sets.lpf.abnormal = False
                for i in range(len(params_)):
                    if com_sets.down:
                        dev_wt[i][rpl].addcmul_(m_t[rpl], alpha_hat_t[rpl], value=1)
                    else:
                        dev_wt[i][rpl].addcmul_(m_t[rpl], alpha_hat_t[rpl], value=-1) 
                    
                    # smooth out. [lowpass]
                    if com_sets.beta_o > 0:
                        w_est, dev_wt_smth[i][rpl] = com_sets.lpf.compute(in_k=dev_wt[i][rpl], x=dev_wt_smth[i][rpl], beta=com_sets.dev_beta_o, step=step_this, mode=3)
                        
                        # pass estimated/updated weight values back to the neural network's placeholder.                
                        params_[i][rpl].mul_(0).add_(w_est)
                    else:
                        params_[i][rpl].mul_(0).add_(dev_wt[i][rpl])

# def _fused_sgm(com_sets:common_sets, steps:List[Tensor], params: List[Tensor], 
#         weight_list: List[List[Tensor]], 
#         weight_smth_list: List[List[Optional[Tensor]]], 
#         grad_list: List[List[Tensor]], 
#         grad_smth_list: List[List[Optional[Tensor]]],
#         grad_var_list: List[List[Optional[Tensor]]],
#         lr_avgb_list: List[List[Optional[Tensor]]], 
#         lr_avga_list: List[List[Optional[Tensor]]], 
#         lrm_save_list: List[List[Tensor]],lrsq_save_list: List[List[Tensor]],*,has_sparse_grad:bool,       
#         differentiable:Optional[bool],
#         grad_scale:Optional[Tensor],
#         found_inf:Optional[Tensor]):
    
#     if len(params) == 0:
#         return
    
#     assert grad_scale is None and found_inf is None        
    
#     grouped_tensors = _group_tensors_by_device_and_dtype(
#         [params, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_var_list, lr_avgb_list, lr_avga_list, lrm_save_list, lrsq_save_list, steps])
    
#     firstgrp=True
#     for (device, dtype) in grouped_tensors:
#         (
#             dev_params, dev_wt, dev_wt_smth, 
#             dev_grads,dev_grads_smth, dev_grads_var, 
#             dev_lrb, dev_lra, dev_lrm, dev_lrsq, dev_steps
#         ) = grouped_tensors[(device, dtype)] 

        
        
#         #todo: 
#         # for each variable, get a list of levels 
#         # flatten tensors in each level 
#         # current behaviour, takes a list of variables and flattens each variable with no idea they have multiple levels.
        
#         # flatten group
#         f_params, f_w, f_w_smth, f_grads, f_grads_smth, f_grads_var, f_lrb, f_lra, f_lrm, f_lrsq, nel = _fuse_grouped_tensors(
#             [dev_params, dev_wt, dev_wt_smth, dev_grads,dev_grads_smth, dev_grads_var, dev_lrb, dev_lra, dev_lrm, dev_lrsq], 
#             device, dtype
#             )
        
#         if dev_steps[0] == 1 and firstgrp:    
#             com_sets.lpf.abnormal = False     
#             com_sets.est_numels = nel
#             com_sets.log_stats()  # LOG.   
#             firstgrp = False
        
        
#         com_sets.grp_devdt(device,dtype)
            
#         step = dev_steps[0]

#         w_t = f_w
#         w_smth = f_w_smth
#         grad = f_grads
#         grad_smth = f_grads_smth
#         grad_var = f_grads_var
#         lr_avgb = f_lrb
#         lr_avga = f_lra
#         lrm = f_lrm
#         lrsq = f_lrsq
            
#         # handle complex parameters
#         if torch.is_complex(f_params):
#             f_params = torch.view_as_real(f_params)
#             # w_t = torch.view_as_real(w_t)
#             # w_smth = torch.view_as_real(w_smth)
#             # grad = torch.view_as_real(grad)
#             # grad_smth = torch.view_as_real(grad_smth)
#             # grad_var = torch.view_as_real(grad_var)          
#             # lr_avgb = torch.view_as_real(lr_avgb)
#             # lr_avga = torch.view_as_real(lr_avga)
#             # lr = torch.view_as_real(lr)
            
        
#         # START
#         mwd = 1-wdecay
#         g_t = com_sets.lpf.patch(grad,beta=beta_o,step=step,mode=3) 
#         if com_sets.join_wdec:
#             # decay weight directly
#             w_t.mul_(mwd)
#         else:
#             # decay weight as l2-regularized gradient
#             g_t.addcmul_(w_t,wdecay)
#         if com_sets.down: g_t.neg_()
                        
#         # flip sign, if maximizing.
#         if com_sets.maximize: g_t.neg_()
        
#         # smooth input [lowpass]
#         m_t_min_1 = com_sets.lpf.previous(xkm1=grad_smth, beta=beta_i, step=step)
        
#         gradss_t_min_1 = m_t_min_1.mul(g_t)
        
#         m_t, grad_smth = com_sets.lpf.compute(in_k=g_t, x=grad_smth, beta=beta_i, step=step)
            
#         # smooth input [add average time-diff] 
#         if step == 1: v_diff = 0
#         else: v_diff = smooth_td(beta_d, m_t, m_t_min_1)
#         g_t.add_(v_diff)
#         m_t.add_(v_diff)
        
#         if norm_all in [0,2]:
#             # denominator of the bayes-optimal step_size
#             # in: averaging [lowpass] ~ approx. input variance
#             v_var_t, grad_var = com_sets.lpf.compute(in_k=(g_t*g_t.conj()), x=grad_var, beta=beta_e, step=step)
            
#             # normalized input
#             # malt_t = m_t.div((v_var_t.add(eps)).sqrt_())
#             (m_t).div_((v_var_t.sqrt_()).add_(eps))

#         # learning rate estimation
#         if com_sets.autolr:
#             # computes numerator of the bayes-optimal step_size
#             # computes approx. linear correlation funcion 
            
#             # surrogate approx. for w^\star 
#             if com_sets.join_wdec:
#                 ewg_t = m_t.mul(w_t)
#             else:
#                 ewg_t = m_t.mul(w_t.mul(mwd))

#             # restart logic: idea is that, after the first k>0 epochs, the gradient of the learning system will be far smaller than during initialization. Therefore, we can restart the state of the lowpass filter with a relatively higher initial learning rate than during initialization, so that the learning system can generalize better after the first k epochs [with an averagely constant/cyclic learning-rate for each parameter.] without becoming too small.
#             # cyclic step
#             step_c = (((step-1) % (com_sets.spe*com_sets.epfreq)) + 1)
#             # scaling and restart lowpass state at the end of every 'k' epoch.
#             if com_sets.restarts and step_c == 1 and step > 1:
#                 lr_avgb.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avgb))
#                 lr_avga.mul_(0).add_(com_sets.rstfact*(lr_init+lr_avga))
                            
#             # linear correlation estimate update  
#             # (double averaging [lowpass]), 
#             # Note: future work: how to estimate this more accurately?    
#             lrat, lr_avgb, lr_avga = com_sets.lpf.compute_dbl(in_k=ewg_t, x1=lr_avgb, x2=lr_avga, beta1=beta_a, beta2=beta_e, step=step)
            
#             # abs. val projection. to ensure positive rates.
#             alpha_hat_t = (lrat.abs_())
            
#         elif com_sets.autolr and norm_all==1: # 1984 impl.
#             # no normalization is done here.
#             # lr_init 
            
           
#             lr_avgb.mul_(0).add_(lr_init)
#             lr_avga.addcmul_(gradss_t_min_1, lr_avgb)
            
#             # restarts:
#             # push close to zero, if a negative element occurs.
#             lrat = lr_avga.relu().add(eps)
            
#             alpha_hat_t = 1*lrat
#         else:
#             # use externally supplied linear correlation estimate 
#             alpha_hat_t = lr_init      

#         # integrate: state update
#         if com_sets.down:
#             w_t.addcmul_(m_t, alpha_hat_t)
#         else:
#             w_t.addcmul_(m_t, alpha_hat_t, value=-1)
        
#         # smooth out. [lowpass]
#         w_est, w_smth = com_sets.lpf.compute(in_k=w_t, x=w_smth, beta=beta_o, step=step, mode=3)
            
#         # pass estimated/updated weight values back to the neural network's placeholder.   
#         f_params.mul_(0).add_(w_est)
        
#         # log lr
#         if com_sets.autolr and com_sets.lrlogstep:
#             # we want to do this per step
#             lr.mul_(0).add_(alpha_hat_t)
#         elif com_sets.autolr and not com_sets.lrlogstep:
#             # to save time and memory,
#             # we want to do this per epoch
#             # but get the average over all steps in that epoch. 
#             lr.add_(alpha_hat_t)     
#         # END    
            
# def _fuse_grouped_tensors(tensorlists, device, dtype):
#     ''' helper function to fuse tensors in a list'''
#     fused_tensorlists = []
#     m = []
#     nnel = None
#     first = True
#     for tensorlist in tensorlists:
        
#         if first==True:
#             # compute len of each param and sum
#             m += [p.numel() for p in tensorlist if isinstance(p, torch.Tensor)]
#             nnel = sum(im for im in m)
#             first = False
            
#         n = sum(p.numel() for p in tensorlist if isinstance(p, torch.Tensor))
        
#         if n > 0:
#             fused_tensorlist = torch.zeros(n, dtype=dtype, device=device)
            
#             # fuse ops
#             i = 0
#             for p in tensorlist: 
#                 params_slice = fused_tensorlist[i:i + p.numel()]
#                 with torch.no_grad(): params_slice.copy_(p.flatten())
#                 p.data = params_slice.view(p.shape)
#                 i += p.numel()
#             fused_tensorlists.append(fused_tensorlist)
        
#         else:
#             # list of scalars, non-tensor type to match network params
#             sctensorlist = torch.zeros(nnel, dtype=dtype, device=device)
#             i = 0
#             for p, im in zip(tensorlist, m):
#                 sctensorlist[i:i + im] = p
#                 i += im
#             fused_tensorlists.append(sctensorlist)
    
#     # nnel = fused_tensorlists[0].shape[0]
#     fused_tensorlists.append(nnel)
#     return fused_tensorlists



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


