"""
:mod:`autosgml` is a package implementing the stochastic gradient learning algorithm.
"""

# Common doc strings among pytorch's optimizer impl.
_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. (default: None)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""

_maximize_doc = r"""maximize (bool, optional): maximize the params based on the
            objective, instead of minimizing (default: False)"""

_email_doc = r"""somefuno@oregonstate.edu"""



from dataclasses import dataclass

import math, torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer, required, 
    _use_grad_for_differentiable,
    _default_to_fused_or_foreach,
    _get_value, _stack_if_compiling, _dispatch_sqrt,
)
from typing import Any, Dict, List, Optional

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

from torch.autograd.grad_mode import no_grad

__all__ = ['LPF', 'AutoSGM']

# Forked 'group tensors' from old pytorch install: from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
# with_indices (bool, optional): whether to track previous indices as the last list per dictionary entry.
#   It comes in handy if there are Nones or literals in the tensorlists that are getting scattered out.
#   Whereas mutating a tensor in the resulting split-up tensorlists WILL propagate changes back to the
#   original input tensorlists, changing up Nones/literals WILL NOT propagate, and manual propagation
#   may be necessary.
@no_grad()
def _group_tensors_by_device_and_dtype(tensorlistlist: List[List[Tensor]],
                                       with_indices: Optional[bool] = False) -> \
        Dict[Tuple[torch.device, torch.dtype], List[List[Union[Tensor, int]]]]:
    assert all([not x or len(x) == len(tensorlistlist[0]) for x in tensorlistlist]), (
           "all specified tensorlists must match in length")
    per_device_and_dtype_tensors: Dict[Tuple[torch.device, torch.dtype], List[List[Union[Tensor, int]]]] = defaultdict(
        lambda: [[] for _ in range(len(tensorlistlist) + (1 if with_indices else 0))])
    for i, t in enumerate(tensorlistlist[0]):
        key = (t.device, t.dtype)
        for j in range(len(tensorlistlist)):
            # a tensorlist may be empty/None
            if tensorlistlist[j]:
                per_device_and_dtype_tensors[key][j].append(tensorlistlist[j][i])
        if with_indices:
            # tack on previous index
            per_device_and_dtype_tensors[key][j + 1].append(i)
    return per_device_and_dtype_tensors

def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    if device.type not in ['cpu', 'cuda'] or torch.jit.is_scripting():
        return False
    return all([t is None or type(t) == torch.Tensor for t in tensors])


def cmplx2real(lols):
    "input: List of Lists"
    return [[torch.view_as_real(tsr) 
            if torch.is_complex(tsr) else tsr 
                for tsr in lst] 
                    for lst in lols]

@dataclass
class OAC():
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        
    

# LPF
# The `LPF` class defines a generic first-order low pass filter structure for routine smoothing and averaging operations, with methods for exponential lowpass filtering, and raised cosine window generation.
class LPF():
    """ (Generic Digital) First-order Low Pass Filter Structure (Linear System)
        
        Recursively computes a weighted average (exponential or uniform).
        
        Main Use: routine smoothing, averaging operations.
    """

    def __init__(self, inplace:bool=True, foreach:bool=False, fused:bool=False):
        self.inplace = inplace
        self.foreach = foreach
        self.fused = fused
        self.tensor_lists = fused or foreach

    @torch.no_grad()
    def btrng(self, bt:Tensor|float, p:int=1):
        return (1 - (10**(-p))*(1-bt)) - bt
        
    
    @torch.no_grad()
    def compute(self, in_t:Tensor, x:Tensor, 
                beta:Tensor, step:Tensor, mode:int=1, fix:bool=False, mix:bool=False, sq:bool=False, beta_d:Tensor|float=0, epp:Tensor|float=1):
        
        '''Computes in_t -> LPF -> out_t
        
            in_t: input at current time
            x: state at previous time
            beta: LPF pole at current time
            step: current discrete time
            mode: [default: mode = 1] 
            fix: add one to the iteration
            mix: shelve
            beta_d: HPF param.
            epp: weighting order, depends on mode

        out_t : output at current time, t
        x : updated state for next time, t+1
        '''
        
        if not self.tensor_lists:
            u_t = 1*in_t
        elif self.tensor_lists:
            u_t = torch._foreach_mul(in_t,1) 
        else:
            return in_t
        
        t = 1*step
        if fix: t = t + 1
        beta_t = beta       
        
        one_minus_beta_t = (1-beta_t).pow(epp)
        beta_t_pow_t = beta_t.pow(t)
        one_minus_beta_t_pow_t = (1-beta_t_pow_t)
            
        # poor
        hdiff = 0
        if beta_d > 0 and t > 1:
            one_minus_beta_t_pow_t_min1 = (1-beta_t.pow(t-1))
            #
            hdiff = self.hpf_td(self, in_t, x, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t, one_minus_beta_t_pow_t_min1, beta_d)    
                            
        if not self.tensor_lists:
            if sq:
                in_t = in_t.pow(2)
                
            if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
                # forward:
                ((x.mul_((beta_t - beta_t_pow_t))).add_(one_minus_beta_t*in_t)).div_(one_minus_beta_t_pow_t)
                out_t = 1*x
                
            elif mode == 1: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                # instead of mode=2, use if init x_t = 0
                (x.mul_(beta_t)).add_(in_t)
                out_t = x*(one_minus_beta_t/one_minus_beta_t_pow_t)     
                
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # instead of mode=4, use if init x_t is likely not 0
                # forward:
                (x.mul_(beta_t)).add_((one_minus_beta_t)*in_t)
                out_t = 1*x      
                    
            elif mode == 4: # exponential. (stable if 0 \le \beta < 1) 
                # instead of mode=3, use if init x_t = 0
                # forward:
                (x.mul_(beta_t)).add_(one_minus_beta_t*in_t)
                out_t = x.div(one_minus_beta_t_pow_t)     
                
            elif mode == 5: # uniform (constant as t \to infinity or as \beta \to 1)
                # useful for averaging, even when t is not time but a sample instance
                # forward:
                (x.mul_(t-1)).add_(in_t).div_(t)
                out_t = 1*x   

            elif mode == 6: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging, 
                # trusts memory as t increases
                # forward:
                b = beta_t.mul((t-1)/t)
                (x.mul_(b)).add_((1-b)*in_t)
                out_t = 1*x  
                
            elif mode == 7: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging, 
                #
                # forward:
                b = beta_t.mul((t-1)/t)
                (x.mul_(b)).add_((1-b)*in_t)
                out_t = x/(1-b.pow(t))                
            
            elif mode == 8: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging, 
                # trusts input as t increases
                # forward:
                b = beta_t.div(t)
                (x.mul_(b)).add_((1-b)*in_t)
                out_t = 1*x  
                  
            elif mode == -1: # exponential. (as beta_t -> 1) 
                # often: use mode = 1 instead of this. 
                # forward:
                (x.mul_(beta_t)).add_(in_t)
                out_t = 1*x     
                    
        elif self.tensor_lists:
            if sq:
                in_t = torch._foreach_mul(in_t, in_t)
            if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
                # forward:
                torch._foreach_mul_(x, (beta_t - beta_t_pow_t))
                torch._foreach_add_(x, torch._foreach_mul(in_t, one_minus_beta_t))
                torch._foreach_div_(x, one_minus_beta_t_pow_t)
                out_t = torch._foreach_mul(x, 1)
                
            elif mode == 1: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_div(torch._foreach_mul(x, one_minus_beta_t),one_minus_beta_t_pow_t)      
                               
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, torch._foreach_mul(in_t, one_minus_beta_t))
                out_t = torch._foreach_mul(x,1)    
                                     
            elif mode == 4: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, torch._foreach_mul(in_t, one_minus_beta_t))
                out_t = torch._foreach_div(x,one_minus_beta_t_pow_t) 

                                   
            elif mode == 5: # uniform (constant as k \to infinity or as \beta \to 1)
            
                # forward:
                torch._foreach_mul_(x, t-1)
                torch._foreach_add_(x, in_t)
                torch._foreach_div_(x, t)
                out_t = torch._foreach_mul(x,1) 
                
                
            elif mode == 6: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging,
                # trusts memory as t increases
                # forward:
                # ct = beta_t.mul((1-(1/t)))
                ct = beta_t.mul((t-1)/t)
                in_t = torch._foreach_mul(in_t, 1-ct)
                #
                torch._foreach_mul_(x, ct)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_mul(x,1)         
       
            elif mode == -1: # exponential. (as beta_t -> 1) 
                # often: use mode = 1 instead of this. 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_mul(x,1)                                           

        # MIX: 
        # shelve (enhanced smoothing (~ stateless, 2nd order) 
        # than the highpass addition below)
        if mix:
            self.shelve(u_t, out_t, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t)
            
        # MIX: highpass tap: get out_t - out_{t-1}
        if beta_d > 0 and t > 1:
            if not self.tensor_lists:
                out_t.add_(hdiff)
            elif self.tensor_lists:
                torch._foreach_add_(out_t, hdiff)
            
        '''
        out_t : output at current time, k
        x : updated state for next time, k+1
        '''

        return out_t, x

    def shelve(self, u_t, y_t, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t):
        ''' enhanced smoothing
        '''
        if not self.tensor_lists:
            if mode in [0, 1, 4]: 
                ((y_t.mul_((beta_t - beta_t_pow_t))).add_(one_minus_beta_t*u_t)).div_(one_minus_beta_t_pow_t)
            elif mode in [0, 1, 4, 3, 5]: 
                (y_t.mul_((beta_t)).add_(one_minus_beta_t*u_t))
            else:
                pass
        else:  # foreach impl.       
            if mode in [0, 1, 4]: 
                torch._foreach_mul_(y_t, (beta_t - beta_t_pow_t))
                torch._foreach_add_(y_t, torch._foreach_mul(u_t, one_minus_beta_t))
                torch._foreach_div_(y_t, one_minus_beta_t_pow_t)
            elif mode in [3, 5]:                  
                torch._foreach_mul_(y_t, beta_t)
                torch._foreach_add_(y_t, torch._foreach_mul(u_t, one_minus_beta_t))
            else:
                pass

    # HPF Tap
    def hpf_td(self, in_t, x_t, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t, one_minus_beta_t_pow_t_min1, beta_d):
        """ A Crude First-order Differentiator
        
        High-pass filter/time-differentiation
        
        Args:
            beta_d (>= 0)
            Returns beta_d*E_{t, beta, in_{t} - in_{t-1}}
        """
        if mode not in [0,1,3,4]:
            return 0
        
        pout_t = x_t
        if not self.tensor_lists:
            if mode in [1,4]:
                pout_t.div_(one_minus_beta_t_pow_t_min1)
                if mode in [4]:
                    pout_t.mul_(one_minus_beta_t)
                    
            hdiff = -(pout_t - in_t).mul(one_minus_beta_t)
            if mode in [0,1,4]:       
                hdiff.div_(one_minus_beta_t_pow_t)
                
            hdiff.mul_(beta_d)
            
        elif self.tensor_lists:
            if mode in [1,4]:
                torch._foreach_div_(pout_t,one_minus_beta_t_pow_t_min1)
                if mode in [4]:
                    torch._foreach_mul_(one_minus_beta_t)
                    
            hdiff = torch._foreach_mul(torch._foreach_sub(in_t, pout_t), one_minus_beta_t)
            if mode in [0,1,4]:       
                torch._foreach_div_(hdiff, one_minus_beta_t_pow_t)
                
            torch._foreach_mul_(hdiff, beta_d)
            
        return hdiff
       
    # Exponential Lowpass Parameter
    @torch.no_grad()
    def expbeta(self, t:Tensor|float, dn:int=1, u:float=1, l:float=0):
        '''
        expbeta Exponential beta for Averaging

        Makes (Uniformly Exponentially-Weighted) LPF handle tracking better

        Args:
            t (Tensor | float): current window step
            dn (int, optional): sampling time or size. Defaults to 1.
            u (float, optional): max. freq. param. Defaults to 1.
            l (float, optional): min. freq. param. Defaults to 0.

        Returns:
           beta (float): beta. parameter (t-dn)/t
        '''
        # alpha = 1-beta = freq param = dn/t : from  1 to 0
        
        return (1 - ( l + ( (u-l)*torch.abs(dn/t) ) )).item()
       
       
    # Raised Cosine Window
    @torch.no_grad()
    def rcf(self, freq:Tensor|float, a:float=0.5, rho:float=1, n:int=2, rhlf_win:Optional[bool|int]=True, not_reverse:bool=True):
        '''
        rcf (Digital) Raised Cosine Function (Lowpass Filter)

        Y(z)/X(z) = |H(z)|: Frequency (freq) to LPF Magnitude Mapping

        Args:
            freq (Tensor |float, required): R.H.S unit-circle frequency in [0,1]
            rho (float, optional): damping factor in [0,1]. Defaults to 1 (critically damped binomial filter).
            n (int, optional): all-pole polynomial order, n=1 is first order, n=2 is second-order. Defaults to 2.
            a0 (float, optional): a step input. Defaults to 1.

            freq: (float, required) current normalized frequency value must be in [-1, 1].
            
            n:(float, default=2) filter order. higher order means more smoothing, and delay.
            
            rho: (float, default=0.5) configures the roll-off factor of the filter's gain OR number of oscillating modes (1/rho) in one sinusodial cycle.  
            
            a: (float, default=0.5) configures cosine mangnitude range
            raised cosine (shifted chebyshev), a >= 0.5 -> gain-magnitude in [2a-1, 1]
            
            hann (n=2, a = 0.5) -> gain-magnitude in [0, 1], 
            
            hamming (n=2, a = 0.54) -> gain-magnitude in [4/46, 1], 
            
            pass-through (a = 1 or n=0) -> gain-magnitude in [1, 1], 
            
            chebyshev (cosine), a = 0 -> gain-magnitude in [-1, 1].

            rhlf_win: (bool|int|None, default=True) if True: right-half window. if False: left-half window. if None: full window. if int >=1
            
            not_reverse: (bool, default=True) if False: (set n as int) reverses the window upside down.          
        Returns:
            rcf(f): gain-magnitude, at freq: f
        '''        

        # window's gain mappings
        low = 2*a - 1 # smallest value in the window
        vrng = 1-low # range
        
        assert 0 <= rho <= 1, f"{rho} is not in (0,1]"
        assert 0 <= a <= 1, f"{a} is not in [0,1]."
        assert -1 <= low <= 1, f"{low} is not in [-1,1]."
        
        # window's f-interval mappings
        # [1, N], f:[0,1] -> full [-1, 1]
        # [1, N], f:[0,1] -> left-half [-1, 0]

        if rhlf_win is None: freq = 2*freq - 1 # full window 
        elif not rhlf_win: freq -= 1 # left-half window    
        elif type(rhlf_win) is int:
            # if int: rhlf_win=10, f:[0,1] -> full [-1/10, 1]
            freq = (-1/rhlf_win) + (1+(1/rhlf_win))*freq # a window 
                
        
        # fmax = 1 # max freq.
        fnq = 0.5 # nyquist freq
        fstop = fnq*(1+rho)
        fpass = fnq*(1-rho)
        
        if 0 <= torch.abs(freq) <= fpass:
            if not_reverse: return 1*torch.ones((1,)).item()
            else: return 0*torch.ones((1,)).item()

        elif fpass < torch.abs(freq) < fstop:
            
            # teta = (torch.pi/rho)*(torch.abs(f) - fnq)
            # s = torch.sin(teta)
            # return a0*0.5*(1-s) # n = 2
            
            # 2*phi = teta_plus_halfpi 
            # 2*phi = sc*(teta + fnq*torch.pi)     
            # => sc*(torch.pi/rho)*(torch.abs(f) - fpass)
            # return a0*0.5*(1+c) # n = 2 
            
            phi = 0.5*(torch.pi/rho)*(torch.abs(freq) - fpass)
            cn = torch.cos(phi).pow(n)
            # safeguards: floating-point issues
            if torch.isnan(cn): 
                if not_reverse: cn = low*torch.ones((1,))
                else: cn = 1*torch.ones((1,))
            
            # phi.cos_().pow_(n).mul_(vrng).add_(low)    
            if not_reverse: return (low + (vrng*cn)).item() 
            else: return (1-(low + (vrng*cn))).item() 
        
        else:  # torch.abs(f) > fstop:
            if not_reverse and n > 0: return low*torch.ones((1,)).item()
            else: return vrng*torch.ones((1,)).item()


# Backend    
# The `CommonSets` class defines methods for handling 
# common configurations and computations
class CommonSets():
    """ Commons 
    """
    
    def __init__(self, lpf:LPF, gen_cfg:OAC, misc_cfg:OAC, beta_cfg:OAC, 
                rcf_cfg:OAC, bte_cfg:OAC, bta_cfg:OAC) -> None:

        self.lpf = lpf
        #
        self.gen_cfg = gen_cfg
        self.p = gen_cfg.p
        self.autolr = gen_cfg.autolr
        self.decpl_wd = gen_cfg.decpl_wd
        self.spe = gen_cfg.spe
        self.maximize = gen_cfg.maximize
        #
        self.misc_cfg = misc_cfg
        self.down = misc_cfg.down
        self.lrlogstep = misc_cfg.lrlogstep
        #
        self.beta_cfg = beta_cfg
        #        
        self.rcf_cfg = rcf_cfg
        self.bte_cfg = bte_cfg
        self.bta_cfg = bta_cfg
        self.power = bta_cfg.powr
        #       
        self.est_numels = 0
        
        
                   
    @torch.no_grad()
    def grp_devdt(self, device, dtype):
        fzero = torch.tensor(0., dtype=dtype, device=device)
        fone = torch.tensor(1., dtype=dtype, device=device)
        self.fone = fone
        self.fzero = fzero
        self.epp = self.power*fone
        self.dev_bscaler = self.misc_cfg.batchscaler*fone
        self.dev_lr_init = self.gen_cfg.lr_init*fone
        self.dev_beta_i = self.beta_cfg.beta_i*fone
        self.dev_beta_e = self.beta_cfg.beta_e*fone
        self.dev_beta_o = self.beta_cfg.beta_o*fone
        self.dev_beta_d = self.beta_cfg.beta_d*fone
        self.dev_eps = self.gen_cfg.eps*fone
        self.dev_wd_cte = self.gen_cfg.wd_cte*fone
        self.levels = self.p    
        self.dev_beta_ss = self.beta_cfg.beta_ss*fone
        #
        self.dev_fone = 1*fone
        if not self.down: self.dev_fone.neg_()
              
    @torch.no_grad()
    def log_stats(self, params=None):  
        
        if params is not None:
            self.est_numels = sum(p.numel() for p in params)  
        # logging.
        dtxt = '-'
        eqtxt = "."
        pltxt = "+"
        sptxt = " "
        vsltxt = "|"
        
        infostr = f"AutoSGM info:\t [total params, d={self.est_numels}]"  
           
        if self.rcf_cfg.auto:
            rststr = f"[rcw={self.rcf_cfg.win}, half={self.rcf_cfg.half}, auto={self.rcf_cfg.auto}, upfact={self.rcf_cfg.upfact}, n={self.rcf_cfg.n}, l={2*self.rcf_cfg.a - 1} ]"
        else:
            rststr = f"[rcw={self.rcf_cfg.win}, half={self.rcf_cfg.half}, auto={self.rcf_cfg.auto}, init-width={self.rcf_cfg.width}, up={self.rcf_cfg.upfact}, n={self.rcf_cfg.n}, l={2*self.rcf_cfg.a - 1} ]"
        
        autostr = f"[auto_lr={self.gen_cfg.autolr}, levels={self.gen_cfg.p}, init_lr={self.gen_cfg.lr_init:.5g}]"
        
        filtstr = f"[lpfs. [in, out, var_avg, lr_avg] : {self.beta_cfg.beta_i:.9g}, {self.beta_cfg.beta_o:.9g}, {self.beta_cfg.beta_e:.9g}, {self.beta_cfg.beta_ss:.9g}]"
        
        othstr = f"[eps: {self.gen_cfg.eps:.4g}, weight-decay: {self.gen_cfg.wd_cte:.4g}, full-decoup: {self.gen_cfg.decpl_wd}]"
        
        strlogs = []
        strlogs.append(infostr)
        strlogs.append(autostr)
        strlogs.append(rststr)
        strlogs.append(filtstr)
        strlogs.append(othstr)   
        maxlen = 0
        for astr in strlogs:
            maxlen = max(len(astr), maxlen)
        
        print(f"{pltxt}{dtxt*(maxlen+2)}{pltxt}") # account for two spaces
        for i, astr in enumerate(strlogs):
            splen = ((maxlen) - len(astr))
            if i > 0: print(f"{pltxt}{eqtxt*(maxlen+2)}{pltxt}")
            print(f"{vsltxt} {astr}{sptxt*splen} {vsltxt}")    
        print(f"{pltxt}{dtxt*(maxlen+2)}{pltxt}\n")
                   
    # Trace Gradient inputs for each level (back-prop/autodiff)
    def grader(self, step, pl, grad_lev, grad_smth_lev, grad_var_lev, grad_in_lev_t1, w_lev, wd_cte_t, a_t=1):
        '''
        trace gradient values for all levels (back-prop/auto-diff)
        '''
        if not self.lpf.tensor_lists:
            # get nn. gradient for current level
            if pl != 0:
                grad_lev[pl].mul_(0).add_(-grad_in_lev_t1[pl-1].mul(grad_lev[pl-1]))
            g_t = 1*grad_lev[pl]              
            
            if pl == 0:
                # weight decay term: (l2-regularization)
                ge_t = w_lev[pl].mul(wd_cte_t)
                
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == None:
                    g_t.add_(ge_t)
                elif self.decpl_wd:
                    ge_t.mul_(a_t)
                    w_lev[pl].sub_(ge_t)                
                              
            # gradient's variance/power
            if self.autolr is not None:
                beta_var = self.expwin_cmp(step, self.bte_cfg, self.dev_beta_e)
                # beta_var = self.dev_beta_e
                v_var_t, grad_var_lev[pl] = self.lpf.compute(in_t=g_t, x=grad_var_lev[pl], beta=beta_var, step=step, sq=True)
                
            else: # if not self.normed
                v_var_t = grad_var_lev[pl].add(1)
                            
            # smooth gradient input [lowpass]
            if pl == 0:  
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == True:
                    gin_t = g_t
                elif self.decpl_wd == None:
                    gin_t = 1*grad_lev[pl]
                    
                # current smooth value  
                # with: optional mix or add a [highpass: time-difference value] to the gradient 
                betain = self.dev_beta_i
                m_t, grad_smth_lev[pl] = self.lpf.compute(in_t=gin_t, x=grad_smth_lev[pl], beta=betain, step=step, mix=True)

                # decoupling modes
                if self.decpl_wd == None:
                    (m_t).add_(ge_t) 
            
            else:
                # m_t, g_t equal for higher levels
                grad_smth_lev[pl].mul_(0).add_(g_t)
                m_t = 1*g_t
                
            # normalized input
            if self.autolr is not None:            
                v_sd_t = (v_var_t.add(self.dev_eps).sqrt()).add_(self.dev_eps)
                # integrator input, to be passed to upper levels 
                (m_t).div_(v_sd_t)
                   
            # optional, scale batch_size                 
            m_t.mul_(self.dev_bscaler)
            # flip sign, if maximizing. 
            if pl == 0 and (self.down or self.maximize): 
                m_t.neg_()

        
        elif self.lpf.tensor_lists: # operating on lists
            # get gradient for this 'pl' level from all nn layers
            gpl = [ allist[pl] for allist in grad_lev]
            gsmthpl = [ allist[pl] for allist in grad_smth_lev]
            gvarpl = [ allist[pl] for allist in grad_var_lev]
            
            # get nn. gradient for current level
            if pl != 0:
                # product of the previous level's current and past gradient
                gpl_m1_t = [ allist[pl-1] for allist in grad_lev]
                gpl_m1_t1 = [ allist[pl-1] for allist in grad_in_lev_t1]
                
                torch._foreach_mul_(gpl,0)
                torch._foreach_add_(gpl, torch._foreach_mul(gpl_m1_t,torch._foreach_neg(gpl_m1_t1)))
                            
            g_t = torch._foreach_mul(gpl,1)
            
            if pl == 0:  
                # get weight for this level from all nn layers
                wpl = [ allist[pl] for allist in w_lev]
                                
                # weight decay term: (l2-regularization)
                ge_t = torch._foreach_mul(wpl, wd_cte_t)
                
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == None:
                    torch._foreach_add_(g_t, ge_t)   
                elif self.decpl_wd:                
                    torch._foreach_mul_(ge_t, a_t)
                    torch._foreach_sub_(wpl, ge_t)                    
                           
            # gradient's variance/power
            if self.autolr is not None:
                beta_var = self.expwin_cmp(step, self.bte_cfg, self.dev_beta_e)
                # beta_var = self.dev_beta_e
                v_var_t, gvarpl = self.lpf.compute(in_t=g_t, x=gvarpl, beta=beta_var, step=step, sq=True)             
            else: # if not self.normed
                v_var_t = torch._foreach_add(gvarpl,1)
                                       
            # smooth gradient input [lowpass]            
            if pl == 0:  
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == True:
                    gin_t = g_t
                elif self.decpl_wd == None:
                    gin_t = torch._foreach_mul(gpl, 1)
                                         
                # current smooth value 
                # with: optional mix or add a [highpass: time-difference value] to the gradient 
                m_t, gsmthpl = self.lpf.compute(in_t=gin_t, x=gsmthpl, beta=self.dev_beta_i, step=step, mix=False)
                
                # decoupling modes
                if self.decpl_wd == None:
                    torch._foreach_add_(m_t, ge_t)            
                                         
            else:
                # m_t, g_t equal for higher levels
                torch._foreach_mul_(gsmthpl,0)
                torch._foreach_add_(gsmthpl,g_t)
                m_t = torch._foreach_mul(g_t,1)
                #
                              
            # normalized input  
            if self.autolr is not None:       

                v_sd_t = torch._foreach_add(torch._foreach_sqrt(torch._foreach_add(v_var_t,self.dev_eps)), (self.dev_eps))
                # integrator input, to be passed to upper levels 
                torch._foreach_div_(m_t, v_sd_t)
               
            # optional, scale batch_size  
            torch._foreach_mul_(m_t,self.dev_bscaler)
            # flip sign, if maximizing.   
            if pl == 0 and self.down or self.maximize: 
                torch._foreach_neg_(m_t)
                                  
        return m_t
  
    # Back Trace SGM input for next iteration
    def back_grade(self, rpl, grad_in_t, m_t, a_t=1):
        '''
        store current SGM (smooth) gradient input for the next iteration
        '''
        if not self.lpf.tensor_lists:
            grad_in_t[rpl].mul_(0).add_(m_t[rpl]*a_t)
        else:
            grad_in_rpl = [allist[rpl] for allist in grad_in_t]
            torch._foreach_zero_(grad_in_rpl)
            torch._foreach_add_(grad_in_rpl, torch._foreach_mul(m_t[rpl],a_t))
                   
    # Integrator
    def integrator(self, w_t, m_t, rpl, alpha_hat_t, a_t=1):
        '''
        a state-space function: digital integration
        '''
        # [optional] rcf smoothing, a_t 
        if not self.lpf.tensor_lists:                   
            w_t[rpl].addcmul_(m_t[rpl].mul_(a_t), alpha_hat_t, value=self.dev_fone)
        elif self.lpf.tensor_lists:
            wrpl = [ allist[rpl] for allist in w_t]
            #
            torch._foreach_mul_(m_t[rpl], alpha_hat_t)
            torch._foreach_mul_(m_t[rpl], a_t*self.dev_fone)
            torch._foreach_add_(wrpl, m_t[rpl])

    # Smooth/Averaged output
    def smooth_avg_out(self, step, rpl, w_t, w_smth):
        '''
        Smooth/Averaged output
        '''
        if rpl == 0:
            # smooth out/average. [lowpass]
            if self.dev_beta_o > 0:                    
                
                beta_o = self.dev_beta_o
                
                if not self.lpf.tensor_lists: 
                    wrplin = 1*w_t[rpl]

                    wst, w_smth[rpl] = self.lpf.compute(in_t=wrplin, x=w_smth[rpl], beta=beta_o, step=step, mode=6)
                    param_val = 1*wst
                    # param_val = 0.1*wst + 0.9*wrplin

                else:
                    wrpl = [ allist[rpl] for allist in w_t]          
                    wsmthrpl = [ allist[rpl] for allist in w_smth]
                    wrplin = torch._foreach_mul(wrpl, 1)

                    wst, wsmthrpl = self.lpf.compute(in_t=wrplin, x=wsmthrpl, beta=beta_o, step=step, mode=6)
                    param_val = torch._foreach_mul(wst, 1)
                    
            else:
                if not self.lpf.tensor_lists: 
                    w_smth[rpl].mul_(0).add_(w_t[rpl])
                    param_val = w_smth  
                else:
                    wrpl = [ allist[rpl] for allist in w_t]          
                    wsmthrpl = [ allist[rpl] for allist in w_smth]
                    
                    torch._foreach_zero_(wsmthrpl)
                    torch._foreach_add_(wsmthrpl, wrpl)
                    param_val = wsmthrpl  
        else:
            param_val = None 
        
        return param_val                             
                    
    # Pass state (instantaneous/averaged) to network
    def pass_to_nn(self, rpl, param, w_out):
        '''
        Copy state (instantaneous/averaged) to neural network's placeholder.
        '''
        if rpl == 0:
            if not self.lpf.tensor_lists:
                if isinstance(w_out, list):
                    param.mul_(0).add_(w_out[rpl])
                else:
                   param.mul_(0).add_(w_out) 
            else: 
                if isinstance(w_out[0], list):
                    wrplin = [ allist[rpl] for allist in w_out]  
                else:
                    wrplin = w_out                   
                torch._foreach_zero_(param)
                torch._foreach_add_(param, wrplin)   
   
    # Compute lr (gain)
    def lr_compute(self, autolr, step, rpl, m_t, w_t, lr_avga, w_str, w_str2, a_t=1):# -> Tensor | Any | List[Tensor]:
        '''
        computes an iteration-dependent learning rate that approximates an optimal choice of step-size.
        '''
        # learning rate estimation
        # linear correlation estimate update  
        # can we estimate this more accurately?
        
        # cyclic step at every 'epfreq'
        # step_c = (((step-1) % (self.spe*self.epfreq)) + 1)     
                                               
        if autolr==True:
            beta_ss = self.expwin_cmp(step, self.bta_cfg, self.dev_beta_ss)
            # beta_ss = self.dev_beta_ss
            
            if not self.lpf.tensor_lists:

                wg_t = (w_t[rpl]).mul(m_t[rpl])
                ewg_t = -wg_t
                
                if rpl != 0: 
                    ewg_t.mul_(a_t)                      
                
                # average
                lrat, lr_avga[rpl] = self.lpf.compute(in_t=ewg_t, x=lr_avga[rpl], beta=beta_ss, step=step, mode=3, epp=self.epp)  
                
                # abs. val projection. to ensure positive rates.
                alpha_hat_t = (lrat.abs())    

            elif self.lpf.tensor_lists:
                
                wrpl = [ allist[rpl] for allist in w_t]
                lrarpl = [ allist[rpl] for allist in lr_avga]         

                wg_t = torch._foreach_mul(wrpl,m_t[rpl])
                ewg_t = wg_t
                ewg_t = torch._foreach_neg(wg_t)
                
                if rpl != 0: 
                    torch._foreach_mul_(ewg_t, a_t)           
               
                # average
                lrat, lrarpl = self.lpf.compute(in_t=ewg_t, x=lrarpl, beta=beta_ss, step=step, mode=3, epp=self.epp)   
                                
                # abs. val projection. to ensure positive rates.
                alpha_hat_t = torch._foreach_abs(lrat)          
        else:
            # use an externally supplied (typically small) linear correlation estimate or value
            # alpha_hat_t = a_t*self.dev_lr_init   
            if not self.lpf.tensor_lists:
                alpha_hat_t = 1*lr_avga[rpl]
            else:
                lrarpl = [ allist[rpl] for allist in lr_avga]  
                alpha_hat_t = torch._foreach_mul(lrarpl, 1)
            
        return alpha_hat_t 
    
    
    def expwin_cmp(self, step, cfg, beta_cte):
        '''
        exp param computation
        '''
        beta = beta_cte
        if cfg.win:
            fone = torch.tensor(1., dtype=step.dtype, device=step.device)
            
            # maxiters = (self.spe)*(cfg.width)
            maxiters = torch.ceil(2/(1-beta_cte))
            
            denm = maxiters*(cfg.upfact**(cfg.cnt-1))
            tc = ( (step - cfg.last_t) % (denm) ) + cfg.dn  # tc = t
            tc_end = denm 
            
            beta = (fone)*(self.lpf.expbeta(tc, cfg.dn, u=cfg.u, l=cfg.l))

            if tc == torch.floor(fone.mul(tc_end)):
                cfg.cnt += 1
                cfg.last_t = step+1
            
        return beta
        

    # RCF computation
    def rcf_cmp(self, step, cfg):
        '''
        for current step, returns the modulation value of the configured raised cosine window
        '''
        a_t = 1
        if cfg.win:           
            fone = torch.tensor(1., dtype=step.dtype, device=step.device)
        
            # RCF
            # self.spe, steps per epoch, batches
            # self.epoch_movwin, first moving window width,  epochs >=1
            # self.movwin_upfact, moving window width upsampling factor
            
            #
            if cfg.auto and isinstance(cfg.auto, bool):     
                if cfg.half is None:
                    maxiters = math.ceil((1)/(self.gen_cfg.lr_init))
                else:
                    maxiters = math.ceil((2)/(self.gen_cfg.lr_init))
            else: 
                maxiters = (self.spe)*(cfg.auto)
            
            denm = maxiters*(cfg.upfact**(cfg.cnt-1))
            tc = ((step-cfg.last_t) % denm ) + 1
            tc_end = denm
            fc = fone.mul(tc/denm) 
            
            #
            a_t = self.lpf.rcf(freq=fc, a=cfg.a, rho=cfg.rho, n=cfg.n, rhlf_win=cfg.half)      
            #         
            if tc == torch.floor(fone.mul(tc_end)):
                cfg.cnt += 1
                cfg.last_t = step+1  
  
        return a_t
    
    # Log LR or SS                    
    def logginglr(self, rpl, lrm, lrsq, alpha_hat_t):
        '''
        logs the learning-rate for each parameter
        '''
        if not self.lpf.tensor_lists:
            if self.lrlogstep:
                # we want to do this per step
                lrm[rpl].mul_(0).add_(alpha_hat_t)
            elif not self.lrlogstep:
                # to save time and memory,
                # we want to do this per epoch
                # but get the average over all steps in that epoch. 
                lrm[rpl].add_(alpha_hat_t)
                lrsq[rpl].add_(alpha_hat_t.square())
                
        elif self.lpf.tensor_lists:            
            if self.lrlogstep:
                lrmrpl = [ allist[rpl] for allist in lrm]
                # we want to do this per step
                torch._foreach_zero_(lrmrpl)
                torch._foreach_add_(lrmrpl, alpha_hat_t)
            elif not self.lrlogstep:
                lrmrpl = [ allist[rpl] for allist in lrm]
                lrsqrpl = [ allist[rpl] for allist in lrsq] 
                # to save time and memory,
                # we want to do this per epoch
                # but get the average over all steps in that epoch. 
                torch._foreach_add_(lrmrpl, alpha_hat_t)
                torch._foreach_add_(lrsqrpl, torch._foreach_mul(alpha_hat_t,alpha_hat_t))
                
    # Safeguard
    def assign(self, rpl, w_t):
        '''
        assign higher-level weights to learning rate values
            with a safeguard from initial transient values out of the unit range (0,1)
        '''
        if not self.lpf.tensor_lists:
            alpha_hat_t = torch.clamp_max(w_t[rpl].relu().add(self.dev_eps), 1) # safeguard (0,1]   
        else:       
            # at other nodes
            wrpl_p1 = [ allist[rpl] for allist in w_t]
            alpha_hat_t = torch._foreach_add(torch._foreach_clamp_min(wrpl_p1, 0), self.dev_eps)
            torch._foreach_clamp_max_(alpha_hat_t, 1)
            
        return alpha_hat_t

    # Don't use foreach to integrate when its sparse gradient
    def integrate_sparse(self, rpl, dev_wt, dev_wt_smth, step, params, m_t, alpha_hat_t, a_t=1):
        '''
        loop over all for sparse input gradients, don't use foreach.
        '''
        self.lpf.tensor_lists = False
                    
        for i in range(len(params)):
            self.integrator(dev_wt[i], m_t[i], rpl, alpha_hat_t[i], a_t)
            self.smooth_avg_out(step, rpl, dev_wt[i], dev_wt_smth[i], a_t)
            self.pass_to_nn(rpl, params[i],dev_wt[i])

        self.lpf.tensor_lists = True
            


# PyTorch Front   
class AutoSGM(Optimizer):
    
    r""".. _AutoSGM: A Unified Framework for Accelerated Learning
        paper link here.
    
    """.format(maximize=_maximize_doc, foreach=_foreach_doc, differentiable=_differentiable_doc) + r"""
    Example:
        >>> # xdoctest: +SKIP
        >>> from opts.autosgml import AutoSGM
        >>> optimizer = AutoSGM(model.parameters(), weight_decay=5e-4)
        >>> ....
        >>> ....
        >>> optimizer.zero_grad()
        >>> loss_fcn(model(input), target).backward()
        >>> optimizer.step()
        
        .. note::
            Below is just one implementation. There can be any number of specialized implementations, the structure or idea remains the same.
        .. math::
            \begin{aligned}
            v_t &= E{g_t}
            x_t &= x_{t-1} - \alpha_{t}*F_{t}*v_{t}
            w_t &= E{x_t}
            \end{aligned}
            
        where :math:`w`, denote the parameters to adapt, :math:`g` is its gradient, :math:`F*v`, is its smooth gradient by lowpass-filtering.
    """
    
    def __init__(self, params, *, levels=1, 
                 autolr:Optional[bool]=True,
                 decop_wd:Optional[bool]=None, 
                 lr_init=1e-3, eps=1e-8, weight_decay=0, 
                 beta_cfg=(0.9, 0, 0.999, 0.9999),
                 rcf_cfg=((False), (True,True), (1, 1, 0)), 
                 spe=1, lr_filt_pow=2, batch_scale=1, lvls_scale= 0.001, 
                 loglr_step:Optional[bool]=None,
                 maximize:bool=False, foreach:Optional[bool]=True, differentiable:bool=False):  
        """
        Implements the Stochastic Gradient Method with approximations of an automatic, optimal learning rate function.
        
        AutoSGM is a unified learning framework for the gradient method used in deep learning. Adam is an approximation of this optimal step-size. 
        Polyak's Heavy ball, Nesterov's Momentum are specific cases.
        
        Args:
         `params` (`iterable`): iterable of parameters to optimize or dicts defining parameter groups.
             
         `levels` (`int`, optional): number of SGM levels to use (default: `1`).
         
         `autolr` (`bool` | `None`, optional): estimate an approximation of an optimal lr => normalized gradient and iterative lr (default: `True`).
         set to `False` to only estimate gradient moment (i.e: normalized gradient with a constant lr).
         set to `None` to use un-normalized gradient and constant lr.
             
         `decop_wd` (`bool` | `None`, optional): weight_decay decoupling mode (default: `None`). 
         set to `True` for full decoupling. 
         set to `False` for no decoupling. 
         set to `None` for partial decoupling.
             
         `lr_init` (`float`, optional): initial learning rate (default: `1e-3`)
         
         `eps` (`float`, optional): a positive constant used to condition/stabilize gradient normalization (default: `1e-8`).
         
         `weight_decay` (`float`, optional): weight decay (L2 penalty) (default: `0`).       
                 
         `beta_cfg` (`tuple`, optional): configures lowpass parameters (default: `(0.9,0,0.999,0.9999)`) => (`beta_i, beta_o, beta_e, beta_a`). 
         beta_i (`float`): smoothing lowpass pole param for input gradient. (default:`0.9`). 
         beta_o (`float`): smoothing/averaging lowpass pole param for output parameter. (default:`0`).
         beta_e (`float`): averaging lowpass pole param. to estimate gradient's variance/moment. (default:`0.999`).
         beta_a (`float`): averaging lowpass pole param. to estimate lr correlation function. (default:`0.9999`).
         
         `rcf_cfg` (`tuple`, optional) use a raised cosine window for spectral smoothing (default: `((False),(True,True),(1, 1, 0))` => `((active), (half_win, auto_init_width), (up, order, min))`
             active (`bool`): use to activate or deactivate the window function. (default: `False`).
             half_win (`bool`|`None`): full or half window (default: `True`). if `True`: right-half window. if `False`: left-half window. if `None`: full window.         
             auto_init_width (`bool`|`int`): automates the initial window width using the `lr_init`. (default: `True`).        
             set as `int` (e.g:`30`) to manually configure the initial window iteration-width (often in epochs). if set as `int`, needs `spe` to convert iterations to epochs.
             up (`int`): window width increase factor >= 1. often set to `1` or `2`. (default:`1`). 
             order (`int`): order of the raised cosine function. (default: `2`). Often `>= 1`.
             min: (`float`)  configures smallest mangnitude range. (default: `0`).
            
        `spe` (`int`, optional): steps per epoch => number of batches = `len(trainloader)`. used only if `auto_init_width` in `rcf_cfg` is `int`. (default:`1`).

        `lr_filt_pow` (`int` | `float`, optional): use to indicate degree of linearity of network when estimating the lr. Set to `1`, if network is ~ linear/convex, else set higher. (default: `2`). 
            
        `batch_scale` (`float`, optional): use for batch-size scaling (default:`1`)
        
        `lvls_scale` (`float`, optional): used for initializing learning rates of the higher levels (default:`0.001`)
        
        `loglr_step`:(`bool`, optional) how to log learning-rates: per step (True) or per epoch (False) or don't log to make training faster (None)  (default: `None`).
        
        `maximize` (`bool`, optional): whether the objective is being maximized
            (default: False).    
                        
        `foreach` (`bool`, optional): fast cuda operation on lists instead of looping.
        
        `differentiable` (`bool`, optional): set if tensors can do backpropagation during learning.

        .. note:: 
                foreach and fused implementations are typically faster than the for-loop, single-tensor implementation.
        """
    
        # Inits: (Auto)SGM of p levels (think of levels like layers)
        # if not hasattr(cfg, 'x'): cfg.x = 0
        # init. lowpass filter obj.
        self.lpf = LPF(foreach=foreach)
        self.nodes = levels

        misc_cfg = OAC(down=False, lrlogstep=loglr_step, 
                batchscaler=batch_scale, dfac=lvls_scale)
            
        beta_cfg = OAC(beta_i=beta_cfg[0], beta_e=beta_cfg[2], 
                           beta_o=beta_cfg[1], beta_ss=beta_cfg[3], beta_d=0)
        
       
        rcf_cfg = OAC(win=rcf_cfg[0], half=rcf_cfg[1][0], auto=rcf_cfg[1][1],
                      upfact=rcf_cfg[2][0], rho=1, 
                      n=rcf_cfg[2][1], a=0.5*(rcf_cfg[2][2]+1), 
                      last_t=1, cnt=1,)
        
        bte_cfg = OAC(win=autolr,
                      u=1-beta_cfg.beta_e, 
                      l=(0.1)*(1-beta_cfg.beta_e), 
                      last_t=1, cnt=1, upfact=1, dn=1)
        
        bta_cfg = OAC(win=autolr, powr=lr_filt_pow,
                      u=1-beta_cfg.beta_ss, 
                      l=0.00001, last_t=1, cnt=1, upfact=1, dn=1)
            
        defaults = dict(p=levels, autolr=autolr, lr_init=lr_init,               
                        decpl_wd=decop_wd, eps=eps, spe=spe,weight_decay=weight_decay,                  maximize=maximize, foreach=foreach,
                        beta_cfg=beta_cfg, misc_cfg=misc_cfg, 
                        rcf_cfg=rcf_cfg, bte_cfg=bte_cfg, bta_cfg=bta_cfg,
                        com_sets=None, differentiable=differentiable)
        
        
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        '''
        Set defaults for parameter groups
        '''
        
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('p',1)
            group.setdefault('com_sets', None)   
            
    @torch.no_grad()
    def zero_logged_lr(self):
        """zero lr logged in last epoch.
        
            This will help save time in plotting logged lrs
            since we average the whole lrs logged in an epoch by the total steps in that epoch.
        """
        
        for group in self.param_groups:
          #  lev = group['p']
          for lev in range(1, self.nodes+1):
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
            
            gen_cfg = OAC(p=group['p'], autolr=group['autolr'], lr_init=group['lr_init'], decpl_wd=group['decpl_wd'], eps=group['eps'], wd_cte=group['weight_decay'], maximize=group['maximize'], spe=group['spe'])

            group['com_sets'] = CommonSets(
                self.lpf, gen_cfg, group['misc_cfg'], group['beta_cfg'], 
                group['rcf_cfg'],  group['bte_cfg'], group['bta_cfg']
            ) 
        
        com_sets = group['com_sets']
        has_sparse_grad = False       
        
        params_with_grad_list = []
        steps = []
        
        weight_list = []
        weight_smth_list = []
        w_str_list = []
        w_str2_list = []
        grad_list = []
        grad_in_list = []
        grad_smth_list = []
        grad_var_list = []
        lr_avga_list = []
        
        # wd_param_list = []
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
                    
                    dfac = float(group['misc_cfg'].dfac) # or 0.001, 0.01
                        
                    state['levels'] = dict()
                    for pl in range(group['p']):
                        lev = pl+1
                        
                        state['levels'][f'{lev}'] = dict()
                        
                        # - for all levels
                        state['levels'][f'{lev}']['grad_smth'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device) 
                        state['levels'][f'{lev}']['grad_in'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device) 
                        state['levels'][f'{lev}']['grad_var'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                        state['levels'][f'{lev}']['grad_smth_var'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                        
                        if lev == 1:
                            state['levels'][f'{lev}']['grads'] = None
                        else:
                            state['levels'][f'{lev}']['grads']= torch.zeros_like(p, memory_format=torch.preserve_format, device=p.device)
                        
                        if lev == 1:
                            # weight of the network
                            state['levels'][f'{lev}']['weight'] = p.real.clone(memory_format=torch.preserve_format).detach()   
                            state['levels'][f'{lev}']['weight_smth'] = p.real.clone(memory_format=torch.preserve_format).detach() 
                        else:
                            # [lr in level-1 >= 1] is a weight in level > 1
                            # e.g: lr in level 1 is a weight in level 2
                            modlrinit = group['lr_init'] + 1e-10 # fix for zero lr at level 1, if levels > 1
                            state['levels'][f'{lev}']['weight'] = (dfac**(pl-1))*modlrinit*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                            state['levels'][f'{lev}']['weight_smth'] = (dfac**(pl-1))*modlrinit*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)     
                                                                
                        # - only at the last level.          
                        if lev == group['p']:   # (dfac**pl)  
                            state['levels'][f'{lev}']["lr_avga"] =(dfac**pl)*group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device) 
                            if group['autolr']:
                                state['levels'][f'{lev}']["w_str"] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device) 
                                state['levels'][f'{lev}']["w_str2"] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device)
                            else:
                                state['levels'][f'{lev}']["w_str"] = None
                                state['levels'][f'{lev}']["w_str2"] = None   
                                
                        else:
                            state['levels'][f'{lev}']["lr_avga"] = None
                            state['levels'][f'{lev}']["w_str"] = None
                            state['levels'][f'{lev}']["w_str2"] = None

                        
                        # - history for all levels (stores, first and second moment for alpha_t states.)              
                        state['levels'][f'{lev}']["lr_m_save"] = torch.zeros_like(
                                p.real, memory_format=torch.preserve_format, device=p.device)
                        state['levels'][f'{lev}']["lr_sq_save"] = torch.zeros_like(
                                p.real, memory_format=torch.preserve_format, device=p.device) 
                    
                state['step'] += 1
                steps.append(state['step'])
                
                # Level Lists for this parameter
                weight_llist = []
                weight_smth_llist = []
                w_str_llist = []
                w_str2_llist = []
                grad_llist = []
                grad_in_llist = []
                grad_smth_llist = []
                grad_var_llist = []  
                lr_avga_llist = []

                # wd_param_llist = []

                lrm_save_llist = []        
                lrsq_save_llist = []
                
                # -  for all levels
                for lev in range(1,group['p']+1):
                  grad_smth_llist.append(state['levels'][f'{lev}']['grad_smth'])
                  grad_var_llist.append(state['levels'][f'{lev}']['grad_var']) 
                  grad_in_llist.append(state['levels'][f'{lev}']['grad_in'])
                  
                  if lev == 1:
                    grad_llist.append(p.grad)
                  else:
                    grad_llist.append(state['levels'][f'{lev}']['grads'])
                    
                  weight_llist.append(state['levels'][f'{lev}']['weight'])
                  weight_smth_llist.append(state['levels'][f'{lev}']['weight_smth'])      
                  
                  # - only for the last level.    
                  lr_avga_llist.append(state['levels'][f'{lev}']['lr_avga'])     
                  w_str_llist.append(state['levels'][f'{lev}']['w_str'])   
                  w_str2_llist.append(state['levels'][f'{lev}']['w_str2'])                     
                  

                  # - (history stores, mean and second moment for alpha_hat_t.)
                  lrm_save_llist.append(state['levels'][f'{lev}']['lr_m_save'])
                  lrsq_save_llist.append(state['levels'][f'{lev}']['lr_sq_save'])             

                # List of Level Lists for each 
                # parameter with a gradient in the ANN.
                weight_list.append(weight_llist)
                weight_smth_list.append(weight_smth_llist)
                #
                grad_list.append(grad_llist)
                grad_in_list.append(grad_in_llist)
                grad_smth_list.append(grad_smth_llist)
                grad_var_list.append(grad_var_llist)
                  
                # - only for the last level.
                lr_avga_list.append(lr_avga_llist)  
                w_str_list.append(w_str_llist)
                w_str2_list.append(w_str2_llist)
    
                # - for all levels (stores, mean and second moment for states.)
                lrm_save_list.append(lrm_save_llist)
                lrsq_save_list.append(lrsq_save_llist)
        
        pplists = [params_with_grad_list, weight_list, weight_smth_list, w_str_list, w_str2_list, grad_list, grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list]
        
        return com_sets, has_sparse_grad, pplists, steps
        
    @_use_grad_for_differentiable
    def step(self, rank=0, closure=None):
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
            pplists, steps = self._init_group(group)
            
            sgm(com_sets, steps, pplists,
                has_sparse_grad = has_sparse_grad,
                foreach=group['foreach'], 
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None), rank=rank)
            
            # update state
            pass
        
        return loss
    
    @torch.no_grad()
    def use_avg(self, pick_avg=True):
        """
        if pick_avg is True, uses averaged weights, after learning, else
        it resets the parameter in the network to the last iteration weight found during learning
        """
        for group in self.param_groups:
            
            com_sets = group['com_sets']        
            params_with_grad_list = []
            win_list = []
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad_list.append(p)
                    
                    state = self.state[p]

                    # Level Lists for this parameter
                    win_llist = []
                    
                    # -  for all levels
                    for lev in range(1,2):
                        if pick_avg:
                            win_llist.append(state['levels'][f'{lev}']['weight_smth'])  
                        else:
                            win_llist.append(state['levels'][f'{lev}']['weight'])      
                    
                    # List of Level Lists for each 
                    # parameter with a gradient in the ANN.
                    win_list.append(win_llist)
            
            use_avg_fcn(com_sets,params_with_grad_list, win_list)
            # update state     
        pass
 
    @torch.no_grad()
    def update_bn(self, loader, model, device=None, islearning=False):
        r""" forked from offcial pytorch swa_utils.py
        For the model, updates BatchNorm modules:
            running_mean, running_var buffers.

        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.
        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.
            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.
            device (torch.device, optional): If set, data will be transferred to
                :attr:`device` before being passed into :attr:`model`.
                
            islearning (bool): indicates if model is still being trained with the SGM. The default is False to indicate we want to keep the new batch-norm stats.

        Example:
            >>> from autosgml import AutoSGM
            >>> asgm = AutoSGM(model.parameters(),)
            >>> ...
            >>> # During learning, after a training session
            >>> loader, model = ...
            >>> asgm.useavg(True) # try ensembled weights
            >>> asgm.update_bn(loader, model, device, True) # update batch-stats
            >>> # do some train/test evaluations ...
            >>> asgm.useavg(False) # reset to instantaneous values
            >>> ...    
            >>> # After learning is completed
            >>> loader, model = ...
            >>> asgm.useavg() # use ensembled weights
            >>> asgm.update_bn(loader, model, device) # update batch-stats
            >>> # do some train/test evaluations ...
            

        .. note::
            The `update_bn` utility assumes that each data batch in :attr:`loader`
            is either a tensor or a list or tuple of tensors; in the latter case it
            is assumed that :meth:`model.forward()` should be called on the first
            element of the list or tuple corresponding to the data batch.
        """
        betas = {}
        means = {}
        vars = {}
        # For Batch Norm (BN) modules in the model
        # reset running averages (mean and variance) 
        # and store the BN module's averaging parameter, 
        # which was named momentum in the official PyTorch implementation 
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                
                if islearning:
                    means[module] = module.running_mean
                    vars[module] = module.running_var
                    
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                
                betas[module] = module.momentum
                
                module.momentum = None
                module.num_batches_tracked *= 0

        # Stop this, if no BN module found
        if not betas: return
        
        # Otherwise, put model in training mode
        was_training = model.training
        model.train()

        # Use forward pass without training to
        # recompute BN stats: one-pass over data
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            if device is not None:
                input = input.to(device)
            model(input)

        # Pass back the BN module's averaging parameter
        # and return model to its previous mode
        for module in betas.keys():
            module.momentum = betas[module]
            if islearning:
                module.running_mean = means[module]
                module.running_var = vars[module]
                
        model.train(was_training)

   
def use_avg_fcn(com_sets:CommonSets, params: List[Tensor], 
        win_list: List[List[Optional[Tensor]]]):
    
    r""" Functional API to pass averaged weights to network after learning, else
        it resets the parameters in the network to the last iteration weights found during learning.
    """
    
    # logic to use multi_tensor: foreach, fused or single_tensor
    if not com_sets.lpf.tensor_lists:
        
        for i, param in enumerate(params):
            w_in = win_list[i]
            # handle if complex parameters
            if torch.is_complex(param):
                param = torch.view_as_real(param)
            
            # only at first level
            # pass update to the neural network's placeholder.
            com_sets.pass_to_nn(0, param, w_in)
                      
                
    elif com_sets.lpf.tensor_lists:
        
        grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, win_list])
    
        for (device, dtype) in grouped_tensors:
            (dev_params, dev_wt_in) = grouped_tensors[(device, dtype)] 
            
            # handle complex parameters
            params_ = [torch.view_as_real(x) 
                    if torch.is_complex(x) else x 
                    for x in dev_params]
            
            # only at first level
            # pass update to the neural network's placeholder.
            com_sets.pass_to_nn(0, params_, dev_wt_in)  
            
    else:
        pass
        
       
def sgm(com_sets:CommonSets, steps:List[Tensor], 
        pplists:List[List[List[Optional[Tensor]]]],*,
        has_sparse_grad:bool=None,
        foreach:Optional[bool]=None,
        grad_scale:Optional[Tensor]=None,
        found_inf:Optional[Tensor]=None, rank=0):
    
    r""" Functional API performing the SGM algorithm computation
    
    See :class:`~torch.optim.SGM` for details.
    
    """
    # PyTorch's JIT scripting issues (Conditionals, Optionals)
    # logic to use multi_tensor: foreach, fused or single_tensor
    if foreach is None: foreach = False
            
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach ops.')
    
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgm
    else:
        func = _single_tensor_sgm

    func(com_sets, steps, pplists,
        has_sparse_grad=has_sparse_grad,
        grad_scale=grad_scale,
        found_inf=found_inf, rank=rank)

    
def _single_tensor_sgm(com_sets:CommonSets, steps: List[Tensor], 
        pplists:List[List[List[Optional[Tensor]]]],*,has_sparse_grad:bool,       
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor], rank=0):
    
    ''' Typical for loop implementation
    '''
    
    assert grad_scale is None and found_inf is None
    
    params, weight_list, weight_smth_list, w_str_list, w_str2_list, grad_list, grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list = pplists
    
    dtype = params[0].dtype
    device= params[0].device
    
    com_sets.grp_devdt(device,dtype)
    levels = com_sets.p
    a_t = com_sets.rcf_cmp(steps[0], com_sets.rcf_cfg)    
        
    # LOG.
    if rank==0 and steps[0] == 1: com_sets.log_stats(params)    
                
    for i, param in enumerate(params):
        step = steps[i]

        w_t = weight_list[i]
        w_smth = weight_smth_list[i]
        grad = grad_list[i]
        grad_smth = grad_smth_list[i]
        grad_var = grad_var_list[i]
        
        grad_in_t = grad_in_list[i]
        
        lr_avga = lr_avga_list[i]
        w_str = w_str_list[i]
        w_str2 = w_str2_list[i]
                
        lrm = lrm_save_list[i]
        lrsq = lrsq_save_list[i]
        
        # handle if complex parameters
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            
        # at only the first level
        wdcte_t = com_sets.dev_wd_cte
        
        # - TRACE gradients: top -> bottom node
        m_t = []
        for pl in range(levels):
            smthval = com_sets.grader(step, pl, grad, grad_smth, grad_var, grad_in_t, w_t, wdcte_t, a_t)
            m_t.append(smthval) 
        #::end trace

        # - FLOW: bottom -> top node
        for rpl in range(levels-1, -1, -1):
            
            # back trace for next iteration
            com_sets.back_grade(rpl, grad_in_t, m_t, a_t)
            
            # compute step-size or lr.
            if rpl == levels-1:
                # at bottom node: base lr
                alpha_hat_t = com_sets.lr_compute(com_sets.autolr, step, rpl, m_t, w_t, lr_avga, w_str, w_str2, a_t)        
            else:
                # at other nodes
                alpha_hat_t = com_sets.assign(rpl+1, w_t)          
                
            # integrate: state update
            com_sets.integrator(w_t, m_t, rpl, alpha_hat_t, a_t) 
                       
            # [optional] smooth-averaging of the state
            param_val = com_sets.smooth_avg_out(step, rpl, w_t, w_smth)
            
            # pass update to the neural network's placeholder.
            # com_sets.pass_to_nn(rpl, param, w_t)
            com_sets.pass_to_nn(rpl, param, param_val)
                      
            # log lr
            com_sets.logginglr(rpl, lrm, lrsq, alpha_hat_t)

        #::end flow


def _multi_tensor_sgm(com_sets:CommonSets, steps: List[Tensor], 
        pplists:List[List[List[Optional[Tensor]]]],*,has_sparse_grad:bool,       
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor], rank=0):
    
    assert grad_scale is None and found_inf is None
    
    params, weight_list, weight_smth_list, w_str_list, w_str2_list, grad_list, grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list = pplists
    
    if len(params) == 0: return

    if rank==0 and steps[0] == 1: com_sets.log_stats(params)
    a_t = com_sets.rcf_cmp(steps[0], com_sets.rcf_cfg)   
    
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, weight_list, weight_smth_list, w_str_list, w_str2_list, grad_list, grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list, steps])
    
    for (device, dtype) in grouped_tensors:
        (
            dev_params, dev_wt, dev_wt_smth, dev_wt_str, dev_wt_str2,
            dev_grads,dev_grads_smth, dev_grads_var, dev_grad_in_t,
            dev_lra, dev_lrm, dev_lrsq, dev_steps
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
               
        # at only the first level
        wdcte_t = com_sets.dev_wd_cte
        
        # - TRACE gradients
        m_t = []
        for pl in range(levels):
            smthval = com_sets.grader(step_this, pl, dev_grads, dev_grads_smth, dev_grads_var, dev_grad_in_t, dev_wt, wdcte_t, a_t)
            m_t.append(smthval)             
        #
        #::end trace

        # - FLOW: bottom -> top node
        for rpl in range(levels-1, -1, -1):
            
            # back trace for next iteration
            com_sets.back_grade(rpl, dev_grad_in_t, m_t, a_t) 
            
            # compute step-size or lr.
            if rpl == levels-1:
                # at bottom node: base lr
                alpha_hat_t = com_sets.lr_compute(com_sets.autolr, step_this, rpl, m_t, dev_wt, dev_lra, dev_wt_str, dev_wt_str2, a_t)   
            else:
                # at other nodes
                alpha_hat_t = com_sets.assign(rpl+1, dev_wt)
                      
            if not device_has_sparse_grad:
                # integrate: state update
                com_sets.integrator(dev_wt, m_t, rpl, alpha_hat_t, a_t)            
                
                # [optional] smooth-averaging of the state
                param_val = com_sets.smooth_avg_out(step_this, rpl, dev_wt, dev_wt_smth)
                
                # pass update to the neural network's placeholder.
                # com_sets.pass_to_nn(rpl, params_, dev_wt)
                com_sets.pass_to_nn(rpl, params_, param_val)
                             
            elif device_has_sparse_grad:
                # sparse update
                com_sets.integrate_sparse(rpl, dev_wt, dev_wt_smth, step_this, params_, m_t, alpha_hat_t, a_t)  
                            
            # log lr
            com_sets.logginglr(rpl, dev_lrm, dev_lrsq, alpha_hat_t) 

        #::end flow
        
        

AutoSGM.__doc__ = ""

