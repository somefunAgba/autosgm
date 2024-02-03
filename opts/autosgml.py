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

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

from torch.autograd.grad_mode import no_grad

# Forked from old pytorch install: from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

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


__all__ = ['AutoSGM', 'autosgm']

def cmplx2real(lols):
    "input: List of Lists"
    return [[torch.view_as_real(tsr) 
            if torch.is_complex(tsr) else tsr 
                for tsr in lst] 
                    for lst in lols]


# LPF
class LPF():
    """ (Generic Digital) First-order Low Pass Filter Structure (Linear System)
        
        Recursively computes a weighted average (exponential or uniform).
        
        Main Use: routine smoothing, averaging operations.
    """

    def __init__(self, inplace:bool=True, foreach=False, fused=False):
        self.inplace = inplace
        self.foreach = foreach
        self.fused = fused
        self.tensor_lists = fused or foreach

    @torch.no_grad()
    def compute(self, in_t:Tensor, x:Tensor, 
                beta:Tensor, step:Tensor, mode:int=1, fix:bool=False, mix:bool=False, sq:bool=False, beta_d:Tensor|float=0):
        
        '''Computes in_t -> LPF -> out_t
        
            in_t: input at current time
            x: state at previous time
            beta: LPF pole at current time
            step: current discrete time
            mode: [default: mode = 1] 
            fix: add one to the iteration
            mix: shelve
            beta_d: HPF param.

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
        
        if not mode in [5]:
            one_minus_beta_t = (1-beta_t)
            beta_t_pow_t = beta_t.pow(t)
            one_minus_beta_t_pow_t = (1-beta_t_pow_t)
            
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

            elif mode == 2: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                # instead of mode=1, use if init x_t not 0
                (x.mul_(beta_t)).add_(in_t)
                out_t = x*(one_minus_beta_t)  
                
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # instead of mode=4, use if init x_t is likely not 0
                # forward:
                (x.mul_(beta_t)).add_(one_minus_beta_t*in_t)
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

            elif mode == 2: # exponential. (stable if 0 \le \beta < 1)  
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_mul(x, one_minus_beta_t)  
                               
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
                
            elif mode == -1: # exponential. (as beta_t -> 1) 
                # often: use mode = 1 instead of this. 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_mul(x,1)                                           

        # MIX: 
        # shelve (better smoothing (~ stateless, 2nd order) than the highpass addition below)
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
            if mode in [0,1,4]:
                ((y_t.mul_((beta_t - beta_t_pow_t))).add_(one_minus_beta_t*u_t)).div_(one_minus_beta_t_pow_t)
            elif mode in [3]: 
                (y_t.mul_((beta_t)).add_(one_minus_beta_t*u_t))
            else:
                pass
        else:  # foreach impl.       
            if mode in [0,1,4]:
                torch._foreach_mul_(y_t, (beta_t - beta_t_pow_t))
                torch._foreach_add_(y_t, torch._foreach_mul(u_t, one_minus_beta_t))
                torch._foreach_div_(y_t, one_minus_beta_t_pow_t)
            elif mode in [3]:                  
                torch._foreach_mul_(y_t, beta_t)
                torch._foreach_add_(y_t, torch._foreach_mul(u_t, one_minus_beta_t))
            else:
                pass

    # HPF Tap
    def hpf_td(self, in_t, x_t, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t, one_minus_beta_t_pow_t_min1, beta_d):
        """ A First-order Digital Differentiator
        
        A high-pass filter/time-differentiation
        
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
class common_sets():
    """ Commons 
    """
    
    def __init__(self, lpf:LPF, p, 
                lr_init, movwin, movwin_upfac, rho, n, a, half_win,
                betas, beta_o, beta_d, 
                eps, spe, wd_cte, decpl_wd:Optional[bool],
                autowd:bool, autolr:bool, restarts:bool, maximize:bool, lrlogstep:bool, down:bool) -> None:
        self.down = down
        self.lrlogstep = lrlogstep
        self.lpf = lpf
        self.p = p
                
        self.beta_i = betas[0]
        self.beta_e = betas[1] 
        self.beta_o = beta_o 
        self.beta_d = beta_d
        
        self.lr_init = lr_init
        self.eps = eps
        self.wd_cte = wd_cte
        self.decpl_wd = decpl_wd
        self.autowd = autowd
        self.autolr = autolr
        self.maximize = maximize
        
        self.est_numels = 0
        #
        self.beta_as = (0.9, 0.99, 0.999, 0.9999, 0.99999, \
                        0.999999, 0.9999999, 0.99999999)
        self.ssbeta = self.beta_as[betas[2]-1]
        self.ssmode = 3
        #        
        self.rcf = restarts
        self.spe = spe
        self.epoch_movwin = movwin
        self.movwin_upfact = movwin_upfac
        self.last_t = 1
        self.wincnt = 1
        self.rho = rho
        self.n = n
        self.a = a
        self.half_win = half_win
        
            
    @torch.no_grad()
    def grp_devdt(self, device, dtype):
        fzero = torch.tensor(0., dtype=dtype, device=device)
        fone = torch.tensor(1., dtype=dtype, device=device)
        self.dev_lr_init = self.lr_init*fone
        self.dev_beta_i = self.beta_i*fone
        self.dev_beta_e = self.beta_e*fone
        self.dev_beta_o = self.beta_o*fone
        self.dev_beta_d = self.beta_d*fone
        self.dev_eps = self.eps*fone
        self.dev_wd_cte = self.wd_cte*fone
        self.levels = self.p    
        self.dev_ssbeta = self.ssbeta*fone
        #
        self.dev_fone = fone
              
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
        rststr = f"[restarts={self.rcf}, rho={self.rho}, n={self.n}, a={self.a}, half_win={self.half_win}, epoch-width={self.epoch_movwin}, upfact={self.movwin_upfact}]"
        autostr = f"[auto_lr={self.autolr}, levels={self.p}, init_lr={self.lr_init:.5g}]"
        filtstr = f"[lpfs. [in, var_avg, lr_avg,out, in_diff] : {self.beta_i:.4g}, {self.beta_e:.7g}, {self.ssbeta:.7g}, {self.beta_o:.4g}, {self.beta_d:.4g}]"
        othstr = f"[eps: {self.eps:.4g}, weight-decay: {self.wd_cte:.4g}, full-decoup: {self.decpl_wd}]"
        
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
                   
    # Trace Gradient Inputs
    def grader(self, step, pl, grad_lev, grad_smth_lev, grad_var_lev, grad_in_lev_t1, w_lev, wd_cte_t, a_t=1):
        
        if not self.lpf.tensor_lists:
            # get nn. gradient for current level
            if pl != 0:
                grad_lev[pl].mul_(0).add_(-grad_in_lev_t1[pl-1].mul(grad_lev[pl-1]))
            g_t = 1*grad_lev[pl]              
            

            if pl == 0:   
                if self.dev_beta_o > 0:                              
                    # patch for gradient value, if weight value is smoothed
                    g_t = self.lpf.patch(g_t,beta=self.dev_beta_o,step=step, mode=3) 
                
                # weight decay term: (l2-regularization)
                ge_t = w_lev[pl].mul(wd_cte_t)
                
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == None:
                    g_t.add_(ge_t)
                elif self.decpl_wd:
                    w_lev[pl].sub_(ge_t)                
                              
            # gradient's variance/power
            if self.autolr is not None:
                v_var_t, grad_var_lev[pl] = self.lpf.compute(in_t=g_t, x=grad_var_lev[pl], beta=self.dev_beta_e, step=step, sq=True)
            else: # if not self.normed
                v_var_t = grad_var_lev[pl].add(1)
                            
            # smooth gradient input [lowpass]
            if pl == 0:  
                
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == True:
                    gin_t = g_t
                elif self.decpl_wd == None:
                    gin_t = g_t.sub(ge_t) 
                    
                # current smooth value  
                # with: optional mix or add a [highpass: time-difference value] to the gradient 
                m_t, grad_smth_lev[pl] = self.lpf.compute(in_t=gin_t, x=grad_smth_lev[pl], beta=self.dev_beta_i, step=step, mix=False, beta_d=self.dev_beta_d)
                
                # decoupling modes
                if self.decpl_wd == None:
                    (m_t).add_(ge_t) 
            
            else:
                # m_t, g_t equal for higher levels
                grad_smth_lev[pl].mul_(0).add_(g_t)
                m_t = 1*g_t
                
            # normalized input
            if self.autolr is not None:            
                v_sd_t = (v_var_t.sqrt()).add_(self.dev_eps)
                # integrator input, to be passed to upper levels 
                (m_t).div_(v_sd_t)
                            
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
                    torch._foreach_sub_(wpl, ge_t)                    
                           
            # gradient's variance/power
            if self.autolr is not None:
                v_var_t, gvarpl = self.lpf.compute(in_t=g_t, x=gvarpl, beta=self.dev_beta_e, step=step, sq=True)             
            else: # if not self.normed
                v_var_t = torch._foreach_add(gvarpl,1)
                                       
            # smooth gradient input [lowpass]            
            if pl == 0:  
                # decoupling modes
                if self.decpl_wd == False or self.decpl_wd == True:
                    gin_t = g_t
                elif self.decpl_wd == None:
                    gin_t = torch._foreach_sub(g_t, ge_t)
                                                           
                # current smooth value 
                # with: optional mix or add a [highpass: time-difference value] to the gradient 
                m_t, gsmthpl = self.lpf.compute(in_t=gin_t, x=gsmthpl, beta=self.dev_beta_i, step=step, mix=True, beta_d=self.dev_beta_d)
                
                # decoupling modes
                if self.decpl_wd == None:
                    torch._foreach_add_(m_t, ge_t)            
                                         
            else:
                # m_t, g_t equal for higher levels
                torch._foreach_mul_(gsmthpl,0)
                torch._foreach_add_(gsmthpl,g_t)
                m_t = torch._foreach_mul(g_t,1)
                #
                # m_t, gsmthpl = self.lpf.compute(in_t=g_t, x=gsmthpl, beta=self.dev_beta_i, step=step, mix=True, beta_d=self.dev_beta_d)
                              
            # normalized input  
            if self.autolr is not None:       

                v_sd_t = torch._foreach_add(torch._foreach_sqrt(torch._foreach_add(v_var_t,self.dev_eps)), (self.dev_eps))
                # integrator input, to be passed to upper levels 
                torch._foreach_div_(m_t, v_sd_t)
               
            # torch._foreach_mul_(m_t,a_t)   
            # flip sign, if maximizing.   
            if pl == 0 and self.down or self.maximize: 
                torch._foreach_neg_(m_t)
                                  
        return m_t
                
    # Integrator
    def integrator(self, w_t, m_t, rpl, alpha_hat_t, a_t=1):
        # [optional] rcf smoothing, a_t 
        if not self.lpf.tensor_lists:                      
            if self.down:
                w_t[rpl].addcmul_(m_t[rpl].mul_(a_t), alpha_hat_t)
            else:
                w_t[rpl].addcmul_(m_t[rpl].mul_(a_t), alpha_hat_t, value=-1)
        elif self.lpf.tensor_lists:
            wrpl = [ allist[rpl] for allist in w_t]
            if self.down:
                torch._foreach_mul_(m_t[rpl], alpha_hat_t)
                torch._foreach_mul_(m_t[rpl], a_t)
                torch._foreach_add_(wrpl, m_t[rpl])
            else:
                torch._foreach_mul_(m_t[rpl], alpha_hat_t)
                torch._foreach_mul_(m_t[rpl], a_t)
                torch._foreach_sub_(wrpl, m_t[rpl])

    # Smooth output
    def smooth_out(self, step, rpl, param, w_t, w_smth):
        if rpl == 0:
            if not self.lpf.tensor_lists:
                # optional smooth out. [lowpass]
                if self.dev_beta_o > 0:
                    w_est, w_smth[rpl] = self.lpf.compute(in_t=w_t[rpl], x=w_smth[rpl], beta=self.dev_beta_o, step=step, mode=3)
                    
                    param.mul_(0).add_(w_est)  
                else:
                    param.mul_(0).add_(w_t[rpl])
                # end for
            else: 
                wrpl = [ allist[rpl] for allist in w_t]          
                if self.beta_o > 0: 
                    wsmthrpl = [ allist[rpl] for allist in w_smth]
                    # smooth
                    w_est, wsmthrpl = self.lpf.compute(in_t=wrpl, x=wsmthrpl, beta=self.dev_beta_o, step=step, mode=3)
                    # pass on
                    torch._foreach_zero_(param)
                    torch._foreach_add_(param, w_est)
                else:                    
                    torch._foreach_zero_(param)
                    torch._foreach_add_(param, wrpl)                  
   
    # Compute lr (gain)
    def lr_compute(self, autolr, step, rpl, m_t, w_t, lr_avga, a_t=1):# -> Tensor | Any | List[Tensor]:
            
        # learning rate estimation
        # linear correlation estimate update  
        # can we estimate this more accurately?
        
        # cyclic step at every 'epfreq'
        # step_c = (((step-1) % (self.spe*self.epfreq)) + 1)
        # a_t = self.rcf_cmp(step)                                               

        if autolr==True:

            if not self.lpf.tensor_lists:
                             
                # surrogate approx.
                ewg_t = m_t[rpl].mul(w_t[rpl])
      
                # optional. pre rcf smoothing before averaging
                # ewg_t.mul_(a_t)
                # average
                lrat, lr_avga[rpl] = self.lpf.compute(in_t=ewg_t, x=lr_avga[rpl], beta=self.dev_ssbeta, step=step, mode=self.ssmode)  
                
                # optional. rcf smoothing after averaging
                lrat.mul_(a_t)
                # abs. val projection. to ensure positive rates.
                alpha_hat_t = (lrat.abs())    


            elif self.lpf.tensor_lists:
                
                wrpl = [ allist[rpl] for allist in w_t]
                lrarpl = [ allist[rpl] for allist in lr_avga]               
                
                ewg_t = torch._foreach_mul(m_t[rpl],wrpl)                
                    
                # optional. pre rcf smoothing before averaging
                # torch._foreach_mul_(ewg_t, a_t)
                # average
                lrat, lrarpl = self.lpf.compute(in_t=ewg_t, x=lrarpl, beta=self.dev_ssbeta, step=step, mode=self.ssmode)                  
                     
                # optional. post rcf smoothing after averaging +
                # abs. val projection. to ensure positive rates.
                torch._foreach_mul_(lrat, a_t)
                alpha_hat_t = torch._foreach_abs(lrat)          
        else:
            # use a externally supplied (typically small) linear correlation estimate or value
            # with optional. rcf smooth 
            # alpha_hat_t = a_t*self.dev_lr_init   
            
            if not self.lpf.tensor_lists:
                alpha_hat_t = a_t*lr_avga[rpl]
            else:
                lrarpl = [ allist[rpl] for allist in lr_avga]  
                alpha_hat_t = torch._foreach_mul(lrarpl, a_t)
            
        return alpha_hat_t 

    # RCF computation
    def rcf_cmp(self, step):
        a_t = 1
        fone = torch.tensor(1., dtype=step.dtype, device=step.device)
        if self.rcf:           
            # RCF
            # self.spe, steps per epoch, batches
            # self.epoch_movwin, first moving window width,  epochs >=1
            # self.movwin_upfact, moving window width upsampling factor
            
            #     
            denm = (self.spe*self.epoch_movwin)*(self.movwin_upfact**(self.wincnt-1))
            tc = ((step-self.last_t) % denm ) + 1
            tc_end = denm
            fc = fone.mul(tc/denm) 
            #
            a_t = self.lpf.rcf(freq=fc, a=self.a, rho=self.rho, n=self.n, rhlf_win=self.half_win)   
            # print(a_t)              
            #         
            if tc == torch.floor(fone.mul(tc_end)):
                self.wincnt += 1
                self.last_t = step+1  
  
        return a_t
    
    # Log LR or SS                    
    def logginglr(self, rpl, lrm, lrsq, alpha_hat_t):
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
                
            
# PyTorch Front   
class AutoSGM(Optimizer):
    '''
    r"""Implements Automatic Stochastic Gradient Method
    
    AutoSGM is a unified learning framework for the gradient method used in deep learning. Popular optimizers like Adam, SGD are specific cases.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        autolr (bool | None, optional): normalized gradient and iterative lr (default: True). Set to 'False' to get Adam (normalized gradient with a constant lr). Set to 'None' to use un-normalized gradient and constant lr.
        decoup_wd: (bool | None, optional): weight_decay decoupling mode(default: None). Set to 'True' for full decoupling. Set to 'False' for no decoupling. Set to 'None' for partial decoupling.
        
        
        lr_init (float, optional): initial learning rate (default: 1e-3)
        beta_i (float, optional): input smoothing lowpass pole param. (default: 0.9).
        beta_d (float, optional): input time-differencing highpass param. (default: 0.).
        beta_e (float, optional): input averaging lowpass pole param. (default: 0.999).
        beta_o (float, optional): output smoothing lowpass pole param. (default: 0.).
        beta_a (int, optional): number of significant digits for averaging lowpass pole param used in the lr. (default: 6 sig. digits => 0.999999).
        eps (float, optional): a positive constant added to condition the sqrt. of the graident variance (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).       
         
        levels (int, optional) number of levels to use in this AutoSGM implementation (default: 1).
        
        restarts (bool, optional) use a raised cosine lowpass filter function to shape the learning-rate (default: False).
        spe (int, optional): steps per epoch, also called number of batches = len(trainloader). Set if restarts is True (default:1).
        movwin (int, optional): frequency width in epochs. Set >= 1 if restarts is True (default:1).
        movwin_upfac (int, optional): frequency width upsampling factor. Set >= 1, if restarts is True (default:2).        
        rho (float, optional): positive damping constant in [0,1] for rcf filter  (default: 1).
        n (int, optional): rcf filter order >=1 (default: 1).          
        a: (float, optional)  configures cosine mangnitude range
        raised cosine (shifted chebyshev), a >= 0.5 -> gain-magnitude in [2a-1, 1]. (default: 0.5).
        half_win: (bool|None, optional) if True: right-half window. if False: left-half window. if None: full window. (default: True).      
        
        maximize (bool, optional): whether the objective is being maximized
            (default: False).

        lrlogstep:(bool, optional) how to log learning-rates: per step (True) or per epoch (False)  (default: True).
            
        foreach (bool, optional): fast cuda operation on lists instead of looping.
        differentiable (bool, optional): set if tensors can do backpropagation during learning.
        fused (bool, optional): whether the fused implementation (CUDA only) is used. (NOTE: Currently not implemented).
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
    .. _AutoSGM\: A Unified Lowpass Regularization Framework for Accelerated Learning
        paper link here.
    
    """.format(maximize=_maximize_doc, foreach=_foreach_doc, differentiable=_differentiable_doc) + r"""
    Example:
        >>> # xdoctest: +SKIP
        >>> from opts.autosgml import AutoSGM
        >>> optimizer = AutoSGM(model.parameters(), levels=2, foreach=True)
        >>> ....
        >>> ....
        >>> optimizer.zero_grad()
        >>> loss_fcn(model(input), target).backward()
        >>> optimizer.step()
        
        .. note::
            Below is jsut one implementation. There can be any number of specialized implementations, the structure or idea remains the same.
        .. math::
            \begin{aligned}
            v_{t} &= E_t{g_t}
            w_{t} &= w_{t-1} - \alpha_{t}*F_{t}*v_{t}
            \end{aligned}
            
        where :math:`w`, denote the parameters to adapt, :math:`g` is its gradient, :math:`F*v`, is its smooth gradient by lowpass-filtering.
    """
    '''
    
    def __init__(self, params, *, lr_init=1e-3, spe=1, 
                movwin=1, movwin_upfac=2, rho=1, n=1, a=0.5, half_win=True,
                beta_i=0.9, beta_e=0.999, beta_a=6,
                eps=1e-8, weight_decay=0, 
                beta_o=0, beta_d=0,
                levels=1,
                decoup_wd:Optional[bool]=None,
                autolr:Optional[bool]=True, restarts:bool=False, down:bool=False,
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
                        eps=eps, weight_decay=weight_decay, 
                        spe=spe, movwin=movwin, movwin_upfac=movwin_upfac,
                        restarts=restarts, rho=rho, n=n, a=a, half_win=half_win,
                        maximize=maximize, decpl_wd=decoup_wd, autowd=None,
                        autolr=autolr, lrlogstep=lrlogstep, down=down,
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
            # safeguard
            if group['weight_decay'] == 0:
                group['autowd'] = None
                
            group['com_sets'] = common_sets(
                self.lpf, group['p'], group['lr_init'], 
                group['movwin'], group['movwin_upfac'], 
                group['rho'], group['n'], group['a'], group['half_win'],
                group['betas'], group['beta_o'], group['beta_d'],
                group['eps'], group['spe'], group['weight_decay'], 
                group['decpl_wd'],group['autowd'],
                group['autolr'],group['restarts'],group['maximize'], group['lrlogstep'], group['down']
            ) 
        
        com_sets = group['com_sets']
        has_sparse_grad = False       
        
        params_with_grad_list = []
        steps = []
        
        weight_list = []
        weight_smth_list = []
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
                    
                    dfac = 0.001 # or 0.001, 0.01
                        
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
                            # weight
                            state['levels'][f'{lev}']['weight'] = p.real.clone(memory_format=torch.preserve_format).detach()   
                            state['levels'][f'{lev}']['weight_smth'] = p.real.clone(memory_format=torch.preserve_format).detach() 
                        else:
                            # lr
                            state['levels'][f'{lev}']['weight'] = (dfac**(pl-1))*group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                            state['levels'][f'{lev}']['weight_smth'] = (dfac**(pl-1))*group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)    
                                                                
                        # - only at the last level.          
                        if lev == group['p']:   # (dfac**pl)             
                            state['levels'][f'{lev}']["lr_avga"] = (dfac**pl)*group['lr_init']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)        
                        else:
                            state['levels'][f'{lev}']["lr_avga"] = None
                        
                        # for wd:
                        # if lev == 1 and group['autowd'] is not None:
                        #     state['levels'][f'{lev}']["wd_param"] = group['weight_decay']*torch.ones_like(p.real, memory_format=torch.preserve_format, device=p.device)    
                        
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
                  
                  #   if lev == 1 and group['autowd'] is not None:        
                  #     wd_param_llist.append(state['levels'][f'{lev}']['wd_param'])    
                            
                  # - (history stores, mean and second moment for alpha_hat_t.)
                  lrm_save_llist.append(state['levels'][f'{lev}']['lr_m_save'])
                  lrsq_save_llist.append(state['levels'][f'{lev}']['lr_sq_save'])             

                # List of Level Lists for each 
                # parameter with a gradient in the ANN.
                weight_list.append(weight_llist)
                weight_smth_list.append(weight_smth_llist)
                grad_list.append(grad_llist)
                grad_in_list.append(grad_in_llist)
                grad_smth_list.append(grad_smth_llist)
                grad_var_list.append(grad_var_llist)  
                # - only for the last level.
                lr_avga_list.append(lr_avga_llist)  
                # - only for the first level.   
                # wd_param_list.append(wd_param_llist)       
                # - for all levels (stores, mean and second moment for states.)
                lrm_save_list.append(lrm_save_llist)
                lrsq_save_list.append(lrsq_save_llist)
        
        return com_sets, has_sparse_grad, \
                params_with_grad_list, weight_list, weight_smth_list,\
                  grad_list, grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list, steps
        
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
            grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list, steps = self._init_group(group)
            
            sgm(com_sets, steps, 
                params_with_grad_list, 
                weight_list,
                weight_smth_list,
                grad_list, 
                grad_smth_list,
                grad_var_list,  
                grad_in_list,             
                lr_avga_list, 
                lrm_save_list,
                lrsq_save_list,
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
        grad_in_list: List[List[Optional[Tensor]]],
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
        # func = _fused_sgm
        raise NotImplementedError('please: use foreach, instead of fused!')
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgm
    else:
        func = _single_tensor_sgm

    func(com_sets, steps,
        params, weight_list, weight_smth_list,
        grad_list, grad_smth_list, grad_var_list, grad_in_list,
        lr_avga_list, 
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
        grad_in_list: List[List[Optional[Tensor]]],
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
    af_t = com_sets.rcf_cmp(steps[0])    
        
    # LOG.
    if steps[0] == 1: com_sets.log_stats(params)    
                
    for i, param in enumerate(params):
        step = steps[i]

        w_t = weight_list[i]
        w_smth = weight_smth_list[i]
        grad = grad_list[i]
        grad_smth = grad_smth_list[i]
        grad_var = grad_var_list[i]
        
        grad_in_t = grad_in_list[i]
        
        lr_avga = lr_avga_list[i]
                
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
            smthval = com_sets.grader(step, pl, grad, grad_smth, grad_var, grad_in_t, w_t, wdcte_t)
            m_t.append(smthval) 
        #::end trace

        # - FLOW: bottom -> top node
        a_t = af_t
        for rpl in range(levels-1, -1, -1):
            
            # back trace for next iteration
            grad_in_t[rpl] = grad_in_t[rpl].mul_(0).add_(m_t[rpl]*a_t)
            
            # compute step-size or lr.
            if rpl == levels-1:
                # at bottom node: base lr
                alpha_hat_t = com_sets.lr_compute(com_sets.autolr, step, rpl, m_t, w_t, lr_avga)        
            else:
                # at other nodes
                alpha_hat_t = torch.clamp_max(w_t[rpl+1].relu().add(com_sets.dev_eps), 1) # safeguard (0,1]            
                
            # integrate: state update
            com_sets.integrator(w_t, m_t, rpl, alpha_hat_t, a_t)            
            # pass update to the neural network's placeholder.
            com_sets.smooth_out(step, rpl, param, w_t, w_smth)           
            # log lr
            com_sets.logginglr(rpl, lrm, lrsq, alpha_hat_t)

        #::end flow

def _multi_tensor_sgm(com_sets:common_sets, 
        steps: List[Tensor], params: List[Tensor], 
        weight_list: List[List[Tensor]], 
        weight_smth_list: List[List[Optional[Tensor]]], 
        grad_list: List[List[Tensor]], 
        grad_smth_list: List[List[Optional[Tensor]]],
        grad_var_list: List[List[Optional[Tensor]]],
        grad_in_list: List[List[Optional[Tensor]]],
        lr_avga_list: List[List[Optional[Tensor]]], 
        lrm_save_list: List[List[Tensor]],lrsq_save_list: List[List[Tensor]],*,has_sparse_grad:bool,       
        differentiable:Optional[bool],
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor]):
    
    if len(params) == 0: return
    
    assert grad_scale is None and found_inf is None

    if steps[0] == 1: com_sets.log_stats(params)
    af_t = com_sets.rcf_cmp(steps[0])   
    
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [params, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_var_list, grad_in_list, lr_avga_list, lrm_save_list, lrsq_save_list, steps])
    
    for (device, dtype) in grouped_tensors:
        (
            dev_params, dev_wt, dev_wt_smth, 
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
            smthval = com_sets.grader(step_this, pl, dev_grads, dev_grads_smth, dev_grads_var, dev_grad_in_t, dev_wt, wdcte_t, a_t=af_t)
            m_t.append(smthval)                 
        #
        #::end trace
        
        a_t = af_t
        # - FLOW: bottom -> top node
        for rpl in range(levels-1, -1, -1):
            
            # back trace for next iteration
            grad_in_rpl = [allist[rpl] for allist in dev_grad_in_t]
            torch._foreach_zero_(grad_in_rpl)
            torch._foreach_add_(grad_in_rpl, torch._foreach_mul(m_t[rpl],a_t)) 
            
            # compute step-size or lr.
            if rpl == levels-1:
                # at bottom node: base lr
                alpha_hat_t = com_sets.lr_compute(com_sets.autolr, step_this, rpl, m_t, dev_wt, dev_lra)   
            else:
                # at other nodes
                wrpl_p1 = [ allist[rpl+1] for allist in dev_wt]
                # safeguard
                alpha_hat_t = torch._foreach_add(torch._foreach_clamp_min(wrpl_p1, 0), com_sets.dev_eps)
                torch._foreach_clamp_max_(alpha_hat_t, 1)
        
            if not device_has_sparse_grad:
                # integrate: state update
                com_sets.integrator(dev_wt, m_t, rpl, alpha_hat_t, a_t)            
                # pass update to the neural network's placeholder.
                com_sets.smooth_out(step_this, rpl, params_, dev_wt, dev_wt_smth)           
                # log lr
                com_sets.logginglr(rpl, dev_lrm, dev_lrsq, alpha_hat_t) 

            elif device_has_sparse_grad:
                com_sets.lpf.tensor_lists = False
                
                for i in range(len(params_)):
                  # integrate: state update
                  if com_sets.down:
                      dev_wt[i][rpl].addcmul_(m_t[i][rpl]*a_t, alpha_hat_t[i][rpl], value=1)
                  else:
                      dev_wt[i][rpl].addcmul_(m_t[i][rpl]*a_t, alpha_hat_t[i][rpl], value=-1) 
                  
                  # smooth out. [lowpass] and 
                  # pass update to the neural network's placeholder.
                  if com_sets.beta_o > 0:
                      w_est, dev_wt_smth[i][rpl] = com_sets.lpf.compute(in_t=dev_wt[i][rpl], x=dev_wt_smth[i][rpl], beta=com_sets.dev_beta_o, step=step_this, mode=3)
                      
                      params_[i][rpl].mul_(0).add_(w_est)
                  else:
                      params_[i][rpl].mul_(0).add_(dev_wt[i][rpl])
                
                com_sets.lpf.tensor_lists = True
        #::end flow

AutoSGM.__doc__ = ""

