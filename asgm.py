
"""
:mod:`autosgm` stochastic gradient learning algorithm.
"""
'''

# old labels
`0` -- inactive, flat shape, 
`1` -- raised-cosine fcn, 
`2` -- simple poly. approx, i=1.8, 
`3` -- simple poly. approx, i=2, 
`4` -- linear/triangular/trap. shape, 
`5` -- sigmoid .

# # new labels.
# `1` -- raised-cosine fcn, n = 2
## `2` -- linear decay. shape, previously does not use n
# `2` -- polynomial,  n=1 is linear decay
# `3` -- beta-exponential fcn,  beta <-- 0 < n < 1 beta=[0.9, 0.5, 0.05]
# `4` -- simple poly.,  k <-- 1 < n < infty, k=[1.8, 2, 20]
# `5` -- logistic sigmoid, mid inflection pt. k <-- 0 < n < infty, k=[4,]
# `6` -- logistic sigmoid, higher inflection pt. k <-- 0 < n < infty, k=[4,]

# # rccfg: half
# (1, 0, 0, 2, 1, ...)
# (2, 0, 0, 2, 1, ...)
# (4, 1, 0, 1.8, 1, ...)
# (4, 1, 0, 2, 1, ...)
# (5, 1, 0, 4, 1, ...)
# (3, 0, 0, 0.9, 1, ...)
# (3, 0, 0, 0.5, 1, ...)
# (3, 0, 0, 0.05, 1, ...)
# (6, 1, 0, 4, 1, ...)

# # rccfg: full
# (1, 0, 0, 2, 0, ...)
# (2, 0, 0, 2, 0, ...)
# (4, 1, 0, 1.8, 0, ...)
# (4, 1, 0, 2, 0, ...)
# (5, 1, 0, 4, 0, ...)
# (3, 0, 0, 0.9, 0, ...)
# (3, 0, 0, 0.5, 0, ...)
# (3, 0, 0, 0.05, 0, ...)
# (6, 1, 0, 4, 0, ...)

'''
from dataclasses import dataclass
import string
import random
import math
import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, required, )
from types import SimpleNamespace
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

from torch.autograd.grad_mode import no_grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import heapq

__all__ = ['WINF', 'WINF_II', 'AutoSGM']

N2Rs = {2: 4.324}


# ---------- Utils. ----------
def _pprt(t, x_txt, x_var):
    '''Print signal, x_var'''
    print(t, x_txt, torch.mean(x_var).item(), torch.std(x_var).item() if x_var.numel() > 1  else 0)

def _nz(w, eps):
    we = w.clone()
    mask = we.abs() < eps
    we[mask] += eps
    return we

def _sq(val):
    return val*val

def _abs(val):
    return val.abs()

def _supn(v: torch.Tensor):
    return (torch.max(torch.abs(v))).cpu().numpy().item()

def _capt(tensor):
    return tensor.detach().cpu().clone().numpy()

def _captmn(tensor): 
    return torch.mean(tensor).item()

def _capt_mn_sd(tensor):
    m, s = torch.std_mean(tensor)
    return m.item(), s.item()


#-------- Robustifiers --------------------

def _huber(val, cap, c=4):
    return torch.sign(val) * torch.clip(val.abs(), max=c*cap)

def _tuckey(val):
    valc = torch.clip(val, max=1)
    return _sq(1 - _sq(valc))*valc

'''Markov-based template'''
def _huber_m(x, mag, expt_mag, a=4):
    '''markov inequality, a=4 => cf.'
    ' mag = |x| > 0, expt_mag = E[|x|] > 0'''
    # ssign = x/mag
    # cap = a*expt_mag
    return (x/mag) * torch.clip(mag, max=a*expt_mag)

'''Chebyshev-based template'''
def _huber_c(x, mag, expt_sd, a=4):
    '''chebyshev inequality a=4 => cf.'
    ' mag = |x - E[x]| > 0, expt_sd = sqrt[E[|x- E[x]|^2]] > 0 '''
    # ssign = x/mag
    # cap = a*expt_sd
    return (x/mag) * torch.clip(mag, max=a*expt_sd)

def _hub_mark_v(x, s, t, beta, a, eps):
    ''' vectorized '''
    xn = torch.abs(x)
    xmag = xn.add(eps)
    xm, s = _lpf_can_t(s, xmag, t, beta)
    return _huber_m(x, xmag, xm, a)

# def _hub_mark_vd(x, s, t, beta, v, eps):
#     ''' vectorized, dynamic scale'''
#     xn = torch.abs(x)
#     xmag = xn.add(eps)
#     xm, s = _lpf_can(s, xmag, t, beta)
#     a, v = _lpf_can(v, _sq(xmag-xm), t, beta)
#     a = torch.sqrt(a)
#     return _huber_m(x, xmag, xm, a)

# def _hub_mark_s(x, s, t, beta, a, eps):
#     xn = torch.norm(x) 
#     xmag = xn.add(eps)
#     xm, s = _lpf_can(s, xmag, t, beta)
#     return _huber_m(x, xmag, xm, a)

def _hub_cheb_s1(x, s1, s2, t, beta, a, eps):
    ''' scalar '''
    xn = torch.norm(x)
    xm, s1 = _lpf_can(s1, xn, t, beta)
    xmag = torch.norm(x-xm).add(eps)
    xvar, s2 = _lpf_can(s2, xmag*xmag, t, beta)
    torch.sqrt_(xvar).add_(eps)
    return _huber_c(x, xmag, xvar, a)

# def _hub_cheb_s2(x, s1, s2, t, beta, a, eps):
#     xn = torch.norm(x)
#     xm, s1 = _lpf_can(s1, xn, t, beta)
#     xmag = torch.norm(xn-xm).add(eps)
#     xvar, s2 = _lpf_can(s2, xmag*xmag, t, beta)
#     torch.sqrt_(xvar).add_(eps)
#     return _huber_c(x, xmag, xvar, a)

def _hub_cheb_v1(x, s1, s2, t, beta, a, eps):
    ''' vectorized '''
    xm, s1 = _lpf_can(s1, x, t, beta)
    xmag = torch.abs(x-xm).add(eps)
    xvar, s2 = _lpf_can(s2, _sq(xmag), t, beta)
    torch.sqrt_(xvar).add_(eps)
    return _huber_c(x, xmag, xvar, a)

# def _hub_cheb_v2(x, s1, s2, t, beta, a, eps):
#     xn = torch.abs(x)
#     xm, s1 = _lpf_can(s1, xn, t, beta)
#     xmag = torch.abs(xn-xm).add(eps)
#     xvar, s2 = _lpf_can(s2, xmag*xmag, t, beta)
#     torch.sqrt_(xvar).add_(eps)
#     return _huber_c(x, xmag, xvar, a)

# def _hub_cheb_hs1(x, s1, s2, t, beta, a, eps):
#     xm, s1 = _lpf_can(s1, x, t, beta)
#     xmag = torch.norm(x-xm).add(eps)
#     xvar, s2 = _lpf_can(s2, xmag*xmag, t, beta)
#     torch.sqrt_(xvar).add_(eps)
#     return _huber_c(x, xmag, xvar, a)

# def _hub_cheb_hs2(x, s1, s2, t, beta, a, eps):
#     xn = torch.abs(x)
#     xm, s1 = _lpf_can(s1, xn, t, beta)
#     xmag = torch.norm(xn-xm).add(eps)
#     xvar, s2 = _lpf_can(s2, xmag*xmag, t, beta)
#     torch.sqrt_(xvar).add_(eps)
#     return _huber_c(x, xmag, xvar, a)


# ---------- LPF captures HB, NAG and more ----------
def _lpf_can_t(v, x, t:float, 
    beta:float=0.9, gamma:float=0, eta:float=0, 
    debias:bool=False):
    '''
    First-order Lowpass filter (unit kernel, Transposed canonical form)

    Uses: smoothing and averaging operations.
    state or memory `v`, input `x`, 

    requires: pole: 0 <= beta < 1,  zero: gamma < beta, norm: 0 <= eta <= 1, 
    '''

    # normalization factor
    eta_n = 1-beta
    u_gam = 1-gamma
    # explicitly normalize pole-only TF 
    # if input `eta` is `0`.
    if eta == 0: eta = eta_n

    # weighted input
    x = eta*x
    # update: state
    v += x

    # update: output
    y = v/u_gam
    # update: state
    v *= beta
    v -= gamma*x

    # max. debias
    if debias:
        try: y *= (eta_n/eta)/(1-torch.pow(beta,t))
        except: y *= (eta_n/eta)/(1-math.pow(beta,t))

    return y, v

'''First-order Lowpass Filter LPF'''
def _lpf_can(v, x, t:float, 
    beta:float=0.9, gamma:float=0, eta:float=0, 
    debias:bool=False):
    '''
    First-order Lowpass filter (unit kernel, canonical form)

    Uses: smoothing and averaging operations.
    state or memory `v`, input `x`, 

    requires: pole: 0 <= beta < 1,  zero: gamma < beta, norm: 0 <= eta <= 1, 
    '''

    # normalization factor
    eta_n = 1-beta
    u_gam = 1-gamma
    # explicitly normalize pole-only TF 
    # if input `eta` is `0`.
    if eta == 0: eta = eta_n

    # placeholder output
    y = gamma*v
    # weighted input
    x = eta*x

    # update: state
    v *= beta
    v += x
    # update: output
    y = (v-y)/u_gam

    # max. debias
    if debias:
        try: y *= (eta_n/eta)/(1-torch.pow(beta,t))
        except: y *= (eta_n/eta)/(1-math.pow(beta,t))

    return y, v


# ---------- Schedules ----------
'''Window Functions. Raised Cosine, et. al.'''

def seq_kron(t):
    # expects input `t` starts at 0, s.t first f in the seq. is 0 
    k = t 
    # uniform distribution
    alpha = 0.5*(-1+math.sqrt(5))
    f = (alpha*k)%1
    return f

def _rcos_aprxt(f, n:int=2, x:float=0, i:float=1.8):
    ''' Approximates a raised cosine shape, using simpler low-cost functions.

    `f` is the input sequence, `f` is in `[0, 1]`.
    `n` is the window's order, `n >= 1`.
    `i` is a real-valued fitting constant, `n` is in `[1, 2]`.
    `x` is the smallest value of the window function, `x=0` by default.
    '''
    # print(i)
    y = (1 - (f**i))**n
    y *= (1-x)
    y += x
    return y

def _rcos_exact(f, n:int=2, x:float=0):
    ''' Exact raised cosine shape.
    
    `f` is the input sequence, `f` is in `[0, 1]`.
    `n` is the window's order, `n >= 1`.
    `x` is the smallest value of the window function, `x=0` by default.
    '''
    y = math.cos(0.5*math.pi*f)**n
    y *= (1-x)
    y += x
    return y

def _rcos_aprxt1(f, n:int=1, x:float=0, i:int=1):
    ''' Approximates a raised cosine shape, using simpler low-cost functions.

    `f` is the input sequence, `f` is in `[0, 1]`.
    `n` is the window's order, `n >= 1`.
    `i` is a real-valued fitting constant, `n` is in `[1, 2]`.
    `x` is the smallest value of the window function, `x=0` by default.
    '''
    y = (1 - (f**i))**n
    y *= (1-x)
    y += x
    return y

def _tri(f, n:int=1, l:float=0, dy:float=1):
    ''' Triangular window

    `f` is the input sequence, `f` is in `[0, 1]`.
    `l` is the smallest value of the window function, `l=0` by default.
    '''
    y = (1 - f)**n
    y *= dy
    y += l
    return y

def _spoly_m(f, m:int=1, n:int=2, k:float=2, l:float=0, dy:float=1):
    ''' Simple polynomial function approximations of `m` raised cosine shapes of order `n` over a window.

    `f`. input sequence, `f` is in `(0, 1]`;
    `n`. window's polynomial order, `n >= 1`. `n=2` by default;
    `k`. shape constant, `1 < k < infty`.  `k=2` by default;
    `l`. smallest ouput of the window function in `[l, 1]`. `l=0` by default.
    '''
    
    # period, fs, with i = 1 to m subintervals
    # fs =  1/m

    # step function
    i = max(1, math.ceil(f*m))

    zn = (1 - (m*f - i + 1)**k)**n
    y = (m-i+zn)/m

    # shift
    y *= dy
    y += l

    return y

def _rcos_m(f, m:int=1, n:int=2, l:float=0, dy:float=1):
    ''' Exact `m` raised cosine shapes of order `n` over a window.

    `f`. input sequence, `f` is in `(0, 1]`;
    `n`. window's polynomial order, `n >= 1`. `n=2` by default;
    `l`. smallest ouput of the window function in `[l, 1]`. `l=0` by default.
    '''
    
    # period, fs, with i = 1 to m subintervals
    # fs =  1/m

    # step function
    i = max(1, math.ceil(f*m))

    zn = math.cos(0.5*math.pi*(m*f - i + 1))**n
    y = (m-i+zn)/m

    # shift
    y *= dy
    y += l

    return y

def _lsig_m(f, b:float=2, r:float=4, l:float=0, m:int=1, n:int=2, dy:float=1):
    ''' `m` logistic sigmoid shapes over a window [Smooth stops].

    `f`. input sequence, `f` is in `(0, 1]`;
    `n`. window's polynomial order, `n=2` by default;
    `r`. logistic growth rate.  `r=4 or 4.324` by default;
    `b`. exponential base, `b >= 2`. `b=2` by default;
    `l`. smallest ouput of the window function in `[l, 1]`. `l=0` by default.
    '''    
    
    # r = N2Rs[n]
    alpha = 2*m*r
   
    # period, fs, with i = 1 to m subintervals
    fs = 1/m
    # deltas = [fs*(i-0.5) for i in range(1, m+1) ]
    deltas = [fs*(i+0.5) for i in range(m) ] # midpoints

    # init
    y, ymin, ymax = 0, 0, 0
    for i in range(m):
        
        Dmax = 1 + (b**(alpha*(-deltas[i])))
        ymax += 1/Dmax

        Dmin = 1 + (b**(alpha*(1-deltas[i])))
        ymin += 1/Dmin

        Di = 1 + (b**(alpha*(f-deltas[i])))
        vi = 1/Di
        y += vi

    ymin *= fs 
    ymax *= fs
    delta_y = ymax - ymin

    # 0-1: y-range
    y *= fs/delta_y
    y -= ymin/delta_y    

    # shift
    y *= dy
    y += l

    return y

def _rlsig_m(f, b:float=2, r:float=4, l:float=0, m:int=1, n:int=2, dy:float=1):
    ''' `m` logistic sigmoid shapes over a window.

    `f`. input sequence, `f` is in `(0, 1]`;
    `n`. window's polynomial order, `n=2` by default;
    `r`. logistic growth rate.  `r=4.324` by default;
    `b`. exponential base, `b >= 2`. `b=2` by default;
    `l`. smallest ouput of the window function in `[l, 1]`. `l=0` by default.
    '''    
    
    # r = N2Rs[n]
    infpt = 0.5
   
    # period, fs, with i = 1 to m subintervals
    # fs = 1/m

    # step function
    i = max(1, math.ceil(f*m))

    ymax = 1/(1 + b**(-r))
    ymin = 1/(1 + b**(r))

    delta_y = (ymax - ymin)
    vi = 1/(1 + b**(2*r*(m*f - i + 1 - infpt)))

    zn = (vi-ymin)/(delta_y)
    y = (m - i + zn)/m

    # shift
    y *= dy
    y += l

    return y

def _rlsig_inf_m(f, b:float=2, r:float=4, l:float=0, m:int=1, n:int=2, dy:float=1):
    ''' `m` logistic sigmoid shapes over a window.

    `f`. input sequence, `f` is in `(0, 1]`;
    `n`. window's polynomial order, `n=2` by default;
    `r`. logistic growth rate.  `r=4.324` by default;
    `b`. exponential base, `b >= 2`. `b=2` by default;
    `l`. smallest ouput of the window function in `[l, 1]`. `l=0` by default.
    '''    
    
    # r = N2Rs[n]
    infpt = 0.875
   
    # period, fs, with i = 1 to m subintervals
    # fs = 1/m

    # step function
    i = max(1, math.ceil(f*m))

    ymax = 1/(1 + b**(-r))
    ymin = 1/(1 + b**(r))

    delta_y = (ymax - ymin)
    vi = 1/(1 + b**(2*r*(m*f - i + 1 - infpt)))

    zn = (vi-ymin)/(delta_y)
    y = (m - i + zn)/m

    # shift
    y *= dy
    y += l

    return y

def _beta_exp_m(f, beta:float=0.95, l:float=0, n:int=1, m:int=1, dy:float=1):
    ''' `m` beta-exp. shapes over a window.

    `f`. input sequence, `f` is in `(0, 1]`;
    `n`. window's polynomial order, `n=2` by default;
    `r`. logistic growth rate.  `r=4.324` by default;
    `beta`. exponential decay rate beta, `0 < b < 1`. `b=0.95` by default;
    `l`. smallest ouput of the window function in `[l, 1]`. `l=0` by default.
    '''    
    
    # period, fs, with i = 1 to m subintervals
    # fs = 1/m

    # step function
    i = max(1, math.ceil(f*m))

    bf = beta**((m*f - i + 1))
    bl = beta
    zn = (bf-bl)/(1-bl)
    y = (m - i + zn)/m

    # shift
    y *= dy
    y += l

    return y

# Uniform input sequence
class WINF():
    def __init__(self, rcm:int=2,
            x:float=0, n:int=2, m:int=1,
            tau:int=51, spe:int=1, cfact:int=1, e:float=0,
            imax:int=float("inf")
        ):
        """Window Function, # Rectangular input sequence (ORDERED)

        Args:
            `rcm` (`int`):
                rcm (default: `2`). Set operating mode as `0` -- inactive, flat shape, `1` -- exact raised-cosine shape, `2` -- approximate raised-cosine shape,  `3` -- approximate raised-cosine shape2, `4` -- approximate raised-cosine shape1. 

            `full` (`bool`):
                full (default: `False`). Set `True` if full window (warm-up and anneal), otherwise set as `False`, if half-window (anneal only).
            
            `x` (`float`): 
                x (default: `0`). Set the function's lowest value as a fraction of its largest value. `fmin` = `x * fmax`

            `n` (`int`):
                n  (default: `2`). set shaping variable for the selected window function. For raised-cosine, this is the order n >= 1 of the raised cosine (default is 2). For simple-polynomials, this is n > 1. For logistic sigmoid, this is n > 0. For beta-exp. this is 0 < n < 1.

            `m` (`int`):
                m  (default: `1`). mode of the window's shape: half-period (`1`) or full-period (`0`). A half-period shape is right-half of the full-period shape.

            `tau` (`int`): 
                tau (default: `1`). Set number of epochs.

            `spe` (`int`):       
                spe (default: `100`). Set the learning iterations per epoch.

            `cfact` (`int`): 
                cfact (default: `1`). Set step-unit of window. `0` means `(epoch units) | `1`: (iteration units). `2`: (sub-iteration units). 

            `e` (`float`):
                e (default: `0`). Coverage fraction of the window length. `0 <= e < 1`. For example, `e=0.1`, for annealing, makes the window to cover a fraction of only (1-0.1) of the actual window length.

            `imax` (`int`): 
                imax (default: `float("inf")`). upper limit on the amount of window frames that can be created.

        """
        if tau*spe < 2: 
            raise ValueError(f"Expects `tau*spe` far greater than 1. Saw `tau`={tau}, `spe`={spe}!")

        # cfgs.
        self.full = (m == 0)
        self.rcm = rcm
        self.n = n
        self.m = 1
        self.eps = e # coverage fraction
        self.cfact = cfact
        self.x = x
        self.imax = imax
        self.yx = x

        # init. step index
        self.q = 0
        # count window frames
        self.i = 1
        """
        Note: For `cfact=1`, `tau*spe` gives the length of one window. For `cfact=2`, `spe` gives the length of one window. For `cfact=0`, `tau` gives the length of one window. 
        """
        if cfact == 1:
            # fuction steps in iteration units
            # total epochs * total iters per epoch, iters units
            self.tau_init = tau*spe
            self.tau = tau*spe 
            self.ne = tau
        elif cfact == 2:
            # fuction steps in sub-iteration units within epochs
            # spe is total iters per epoch, iters units 
            self.tau_init = spe
            self.tau = spe 
            self.ne = tau
        elif cfact == 0:
            # fuction steps in epoch units
            # tau is total epochs, epoch units
            self.tau_init = tau
            self.tau = tau 
            self.ne = tau
            if tau <= 1: 
                raise ValueError(f"`cfact`={cfact}, so I expect `tau` as the max. epoch number to be far greater than 1. But saw `tau`={tau}, `spe`={spe}!")
        elif cfact > 2:
            # fuction steps in iteration units
            # total epochs * total iters per epoch, iters units
            self.tau_init = tau*spe
            self.tau = tau*spe 
            self.ne = tau
            self.m = self.cfact

    def spawn_new(self):
        '''spawn new window frame
        '''
        self.q += self.tau
        self.tau = self.tau_init  # * (self.cfact ** self.i)
        self.i += 1

    def step(self, t, k):
        '''one step of the raised cosine'''
        # disabled
        if self.rcm == 0: return 1
        
        # active
        if self.i < self.imax: y = self.active(t, k)
        else: y = self.x

        return y

    def active(self, t, k):
        '''one step of the raised cosine when `self.rcm > 0` '''
        # get f
        '''converts actual iteration index, `t` to a local version, `tc` within the current window frame/interval `i`.
        tc = 0 -> N-1 ... Assumes t starts at 1, but starts tc at 0
        self.q is the actual iteration at the last index of the previous window
        '''

        if self.cfact == 0: t = k  # treat t as an epoch not iteration.
        # if self.cfact == 0: print((t, k), (k-1-self.q), self.tau, end=', ')
        tc = t-1-self.q
        f = tc/self.tau

        # if k gets to self.tau
        if f < 0: f = (self.q-1)/self.tau
        
        if self.full: f = (2*f) - 1
        f = abs(f)

        # variable coverage (flat-top effect)
        if f < self.eps or self.eps==1: return 1
        f = (f-self.eps)/(1-self.eps)

        # standard
        dy = 1-self.x
        l = self.x

        # one step
        if self.rcm == 1:
            ''''exact shape'''
            y = _rcos_m(f, m=self.m, n=self.n, l=l, dy=dy)
        elif self.rcm == 2:
            '''tri. shape'''
            y = _tri(f, n=self.n, l=l, dy=dy)
        elif self.rcm == 3:
            '''beta-parametric exponential fcn '''
            y = _beta_exp_m(f, m=self.m, beta=self.n, l=l, dy=dy)
        elif self.rcm == 4:
            '''simple poly, low-cost rcos. approxs. set i=2'''
            y = _spoly_m(f, m=self.m, k=self.n, l=l, dy=dy)
        elif self.rcm == 5:
            '''logistic sigmoid, low-cost approxs. set i=4'''
            y = _lsig_m(f, r=self.n, m=self.m, l=l, dy=dy)
        elif self.rcm == 6:
            '''logistic sigmoid, low-cost approxs. set i=4'''
            y = _rlsig_inf_m(f, r=self.n, m=self.m, l=l, dy=dy)
            
        # if self.i > 1:
        #     # Smooth from last frame.
        #     y, self.yx = _lpf_can(self.yx, y, t, beta=0.67)

        # optional: spawn new
        '''check if current window length has elapsed.
        If True, spawn (start) a new window frame'''
        if tc == self.tau-1: 
            self.spawn_new()

        
        # if self.cfact == 0: print(f"{y:.4e}")

        return y

# Kronecker input sequence
class WINF_II():
    def __init__(self, rcm:int=2, x:float=0, n:int=2, m:int=1,
            tau:int=1, spe:int=1, cfact:int=1, e:float=0,
        ):
        '''Half-Window Function II, Kronecker input sequence

       Note: Does not assume an automatic, infinite half-window, with a non-monotonic input sequence.

        Args:
            `rcm` (`int`):
                rcm (default: `2`). Set operating mode as `0` -- inactive, flat shape, `1` -- exact raised-cosine shape, `2` -- approximate raised-cosine shape,  `3` -- approximate raised-cosine shape2, `4` -- approximate raised-cosine shape1. 

            `x` (`float`): 
                x (default: `0`). Set the function's lowest value as a fraction of its largest value. `fmin` = `x * fmax`

            `n` (`int`):
                n  (default: `2`). set shaping variable for the selected window function. For raised-cosine, this is the order n >= 1 of the raised cosine (default is 2). For simple-polynomials, this is n > 1. For logistic sigmoid, this is n > 0. For beta-exp. this is 0 < n < 1.

            `m` (`int`):
                m  (default: `1`). mode of the window's shape: half-period (`1`) or full-period (`0`). A half-period shape is right-half of the full-period shape.

            `tau` (`int`): 
                tau (default: `1`). Set number of epochs.

            `spe` (`int`):       
                spe (default: `100`). Set the learning iterations per epoch.

            `cfact` (`int`): 
                cfact (default: `1`). Set step-unit of window. `0` means `(epoch units) | `1`: (iteration units). `2`: (sub-iteration units). 

            `e` (`float`):
                e (default: `0`). Coverage fraction of the window length. `0 <= e < 1`. For example, `e=0.1`, for annealing, makes the window to cover a fraction of only (1-0.1) of the actual window length.

        '''

        if tau*spe < 2: 
            raise ValueError(f"Expects `tau*spe` far greater than 1. Saw `tau`={tau}, `spe`={spe}!")
        
        # cfgs.
        self.full = (m == 0)        
        self.rcm = rcm
        self.n = n
        self.m = 1
        self.eps = e # coverage fraction
        self.cfact = cfact
        self.x = x
        self.yx = x


        # init. step index
        self.q = 0
        # count window frames
        self.i = 1
        """
        Note: For `cfact=1`, `tau*spe` gives the length of one window. For `cfact=2`, `spe` gives the length of one window. For `cfact=0`, `tau` gives the length of one window. 
        """
        if cfact == 1:
            # fuction steps in iteration units
            # total epochs * total iters per epoch, iters units
            self.tau_init = tau*spe
            self.tau = tau*spe 
            self.ne = tau
        elif cfact == 2:
            # fuction steps in sub-iteration units within epochs
            # spe is total iters per epoch, iters units 
            self.tau_init = spe
            self.tau = spe 
            self.ne = tau
        elif cfact == 0:
            # fuction steps in epoch units
            # tau is total epochs, epoch units
            self.tau_init = tau
            self.tau = tau
            self.ne = tau
            self.lastk = 0
            self.lasty = 1
            if tau <= 1: 
                raise ValueError(f"`cfact`={cfact}, so I expect `tau` as the max. epoch number to be far greater than 1. But saw `tau`={tau}, `spe`={spe}!")
        elif cfact > 2:
            # fuction steps in iteration units
            # total epochs * total iters per epoch, iters units
            self.tau_init = tau*spe
            self.tau = tau*spe 
            self.ne = tau
            self.m = self.cfact

        # previously `cfact` was used to make a new window greater than or equal to the previous window length.

        self.spawn_new(init=True)

    def spawn_new(self, init=False):
        '''spawn new window frame
        '''
        if not init:
            self.q += self.tau
            self.tau = self.tau_init  # * (self.cfact ** self.i)
            self.i += 1
        
        if self.rcm != 0 and self.tau > 1:       

            # handle half-period, full-period and 
            # variable coverage (flat-top effect)
            brkpt = 2 if self.full else 1
            taue = self.tau/brkpt
            cove =  int(self.eps*taue)
            genpts = int(taue - cove)
            if brkpt*(genpts + cove) < self.tau: cove += 1
            # flat-top
            yp = [1 for _ in range(0, cove) ]

            # standard
            dy = 1-self.x
            l = self.x

            if self.rcm == 1:
                ''''exact shape'''
                y = [_rcos_m(seq_kron(t), m=self.m, n=self.n, l=l, dy=dy)
                    for t in range(genpts) ]
            elif self.rcm == 2:
                '''tri. shape '''
                y = [_tri(seq_kron(t), n=self.n, l=l, dy=dy) for t in range(self.tau) ]
            elif self.rcm == 3:
                '''beta-parametric exponential fcn '''
                y = [_beta_exp_m(seq_kron(t), m=self.m, beta=self.n, l=l, dy=dy) for t in range(genpts) ]
            elif self.rcm == 4:
                '''simple poly, low-cost rcos. approxs. set i=2'''
                y = [_spoly_m(seq_kron(t), m=self.m, k=self.n, l=l, dy=dy) for t in range(genpts) ]
            elif self.rcm == 5:
                '''logistic sigmoid, low-cost approxs. set i=4'''
                y = [_lsig_m(seq_kron(t), r=self.n, m=self.m, l=l, dy=dy)  
                     for t in range(genpts) ]
            elif self.rcm == 6:
                '''logistic sigmoid, low-cost approxs. set i=4'''
                y = [_rlsig_inf_m(seq_kron(t), r=self.n, m=self.m, l=l, dy=dy)  
                     for t in range(genpts) ]   
            
            y.sort() # sml -> big
            y.extend(yp)
            if self.full: y.extend(sorted(y, reverse=True))
            self.y = y  

    def step(self, t, k):
        '''one step of the raised cosine'''
        # disabled
        if self.rcm == 0: return 1
        
        # active
        # print(t, k)
        y = self.active(t, k)

        return y

    def active(self, t, k):
        '''one step of the raised cosine when `self.rcm > 0` '''
        if self.cfact == 0:  t = k  # treat t as an epoch not iteration.
            
        tc = int(t-1-self.q)

        # take current biggest, and reduce list size by 1   
        # equivalent to: y = self.y[tc]     
        if self.cfact!= 0: 
            y = self.y.pop() 
        else:
            # print(k, self.y)
            if k > self.lastk:
                self.lastk = k
                y = self.y.pop()
                self.lasty = y
            else:            
                y = self.lasty
            # print(f"{y:.4e}")

        '''check if current window length has elapsed.
        If True, spawn (start) a new window frame'''
        if tc + 1 == self.tau : 
            self.spawn_new()             

        return y


# def ndecorr(x, xvar, dvar, dpc, dpcx, t, betas, eps, cf, lr0):
#     '''Naive Decorrelate x'''

#     ds = -(x - torch.sum(x))

#     _, xvar = _lpf_can(xvar, x*x, t, betas[0])     
#     # moment estimate
#     dpow, dvar = _lpf_can(dvar, ds*ds, t, 
#         betas[1], eta=betas[4], debias=betas[5])     
#     torch.sqrt_(dpow).add_(eps)
#     ds /= dpow    

#     # robust par-corr estimate: markov-style
#     lr0m = lr0/(1+xvar)
#     inp = lr0m * _hub_mark_v(x*ds, dpcx, t, betas[0], cf, eps)
#     out, dpc = _lpf_can(dpc, inp, t, betas[0], eta=1)
#     # clip `out` around trust-region `lr0`
#     out.abs_()
#     out.clip_(min=_tuckey(dpow)*lr0m)
#     out.clip_(max=lr0*(1+xvar))

#     # update
#     return x - (out*ds)    

# ---------- PyTorch Optim. Class ----------

'''Stochastic Gradient Learning SGM'''
class AutoSGM(Optimizer):
    
    """AutoSGM is a unifying framework for the gradient method used in deep learning. Popular variants: Polyak's Heavy ball, Nesterov's Momentum, Adam, are specific cases.    
    """
        
    def __init__(self, params, *,
                 lr_cfg:Tuple=(True, 1e-3, 0), 
                 beta_cfg:Tuple=(0.9999, 0.999, 0.9, 0, 0, True), 
                 rc_cfg:Tuple=(0, 0, 0, 2, 1, 1000, 1, 1, 0), 
                 wd_cfg:Tuple=(0, 0), eps_cfg:Tuple=(1e-10, True), 
                 miscs:Tuple=(1,), maximize:bool=False, cf:float=4, debug:bool=False):  
        """
        Implements AutoSGM with approximate variants of an optimal learning rate (lr) function, and cosine annealing (rc).
        
        
        Args:
         `params` (`iterable`): iterable of parameters to optimize or dicts defining parameter groups.
                      
         `lr_cfg` (`bool`,`float`,`int`): lr_cfg (default: (`True`, `0.001`, `0`), optional). Choose an optimal lr approximation in AutoSGM.

            `aoptlr` (`bool`, default: `True`) set as `True` to use approx. realizations of an optimal learning rate (moment estimation (normalized gradient) and a possibly varying partial correlation estimation). set to `False` indicating to use an un-normalized gradient and constant lr.

            `lr_init` (`float`, default: `1e-3`) is the learning rate constant if `aoptlr` is False. However, if `aoptlr` is `True`, it serves as a scale factor for realizing a partial correlation numerator estimate of an optimal learning rate. 

            `num_lrc` (`int`, default: `0`) use to choose variant for estimating the optimal lr's partial corrleation numerator estimate when `aoptlr` is True. Expects a value in `{0,1,2}`. The value `0` indicates a maximum partial correlation constant equal to `lr_init`. The values `1` and `2` each indicate iteration-dependent partial correlations varied from an initial value of 0, and `lr_init` is a small constant that can be tuned to ensure uniformly stable updates across parameters.

         `wd_cfg` (`float`, `int`): wd_cfg (defaut: (0, 0), optional), set weight decay regularization.

            `wd_cte` (`float`, default: `0`). Set a small positive constant that may be used to stabilize the weight learning process, (reflects an L_2-norm penalty). 

            `wd_lvl` is the decoupling level (`int`, default: `0`). level `0` is weight-decay at the the parameter-level. level-1 is weight-decay at the parameter-change level. set to `1` to decouple from grad smoothing and normalization. set to `0` for no decoupling. 
                 
         `beta_cfg` (`float`, `float`, `float`, `float`, `float`, `bool`): beta_cfg (default: `(0.9999, 0.999, 0.9, 0, 0, True)`, optional). Set parameters of the lowpass filter for smoothing the gradient input, and averaging ops. et.c.
            
            `beta_n` (`float`, default:`0.9999`): lowpass pole for averaging. Set much greater than `0.9`. 

            `beta_a` (`float`, default:`0.999`): lowpass pole for averaging. Set much greater than `0.9`. 

            `beta_i` (`float`, default:`0.9`): lowpass pole for smoothing the gradient. Set less or equal to `0.9`. 

            `gamma_i` (`float`, default:`0`): lowpass zero (set as non-negative to reduce over-smoothing, set as negative to increase smoothing). `gamma_i=0` is `HB`, `gamma_i=beta_i/(1+beta_i` is `NAG`. Can improve steady-state error bound.  Expects `gamma < beta_i`  

            `eta_i` (`float`, default:`0`): Expects eta_i in `[0, 1]`. If `eta_i = 0` this resets `eta_i = 1-beta_i`, otherwise the given `eta_i` is used as is. 

            `debias` (`bool`, default:`True`): Enable debaised output from the smoothing filter. Can also improve convergence. Can be disabled, if `aoptlr` is not `True`.        
 
         `rc_cfg` (`int`, `int`, `float`, `float`, `int`, `int`, `int`, `int`, `float`): rc_cfg (default: `(0, 0, 0, 2, 1, 1000, 1, 1, 0)`). Activates windowing/annealing, user must choose the window function, input sequence, et.c. To set window length, must configure number of epcohs `tau`, steps per epoch `spe`, and window step unit, `cfact` which has 3 options. Note: For `cfact=1`, `tau*spe` gives the length of one window. For `cfact=2`, `spe` gives the length of one window. For `cfact=0`, `tau` gives the length of one window. 

            `rcm` (`int`, default: `0`). Choose operating mode as `0` -- inactive, flat shape, `1` -- raised-cosine fcn, `2` -- linear decay. shape,`3` -- beta-exponential fcn, `4` -- simple poly.,  `5` -- logistic sigmoid, mid inflection pt. `6` -- logistic sigmoid, higher inflection pt.

            `inseq` (`int` default: `0`)  Set as `0` to use a `uniform rectangular` input sequence. set as `1` to use a `kronecker` input sequence.  

            `x` (`float`, default: `0`). Set the function's lowest value as a fraction of its largest value. `fmin` = `x * fmax`

            `n` (`float`, default: `2`). Set shaping variable for the selected window function. For raised-cosine, this is the order `n >= 1` of the raised cosine (default is `2`). For simple-polynomials, this is ` n > 1`. For logistic sigmoid, this is `n > 0`. For beta-exp. this is `0 < n < 1`.

            `m` (`int`, default: `1`). anneal only mode (1) or full window (0).

            `tau` (`int`, default: `1`). Set the max. number of learning epochs. Expects value greater than or equals 1. 

            `spe` (`int`, default: `1000`). Choose the number of iterations in each epoch. Expects a value far greater than 1. If `tau=1`, then `spe` is the total learning iterations, since there is only 1 epoch.

            `cfact` (`int`, default: `1`). Sets the unit of each step of the window function. `0` means  (epoch-level units) | `1`:  (iteration-level unit changes) | `2`:  (sub-iteration-level unit changes).            
            
            `e` (`float`, default: `0`). Coverage fraction of the window length. `0 <= e < 1`. For example, `e=0.1`, for annealing, makes the window to cover a fraction of only (1-0.1) of the actual window length.
        
         `eps_cfg` (`float`, `bool`): eps_cfg (defaut: (1e-10, True), optional), set to make normalizations well-behaved.

            `float` is a small positive constant used to condition or stabilize the gradient normalization operation (default: `1e-10`). 

            `bool` controls how to add the small `float` value, once or twice. (default: `True`).
         
         `miscs` (`int`): miscs (default: (1), optional). miscellanous cfg.
            
            `layer_wise` (`int`, default:`1`). indicate if parameters are presented layer-by-layer.
        
         `maximize` (`bool`): maximize (default: `False`, optional). indicate the optimization direction. `False` if minimizing, otherwise, it indicates maximizing the learning objective.         
         
         `cf` (`float`): cf (default: `4`, optional). scale constant in markov's inequality. confidence iinterval ~ (1/cf, 1-(1/cf)).
                         

        .. note:: 
            This is an implementation of a stochastic gradient learning framework for deep learning models.
                
        """
        # if not hasattr(cfg, 'x'): cfg.x = 0
        # access, e.g: optimizer.defaults['lr_cfg']

        # init. store inputs as optimizer `defaults`
        defaults = dict(dbg=debug, cf=cf, lr_cfg=lr_cfg, wd_cfg=wd_cfg, eps_cfg=eps_cfg, maximize=maximize, beta_cfg=beta_cfg, msc_cfg=miscs, rcf_cfgs=rc_cfg, com_sets=None)

        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        '''
        Set defaults for parameter groups
        '''       
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('com_sets', None)   


    def lazyinit_param(self, p, group):  
        # Lazy Init [self.state]       
        state = self.state[p]
        if len(state) == 0:
            dtype, device = p.dtype, p.device         
            
            dbg = group['dbg']
            aoptlr, lr0, num_lrc = group['lr_cfg']
            rc_cfg = group['rcf_cfgs']
            per_lay, = group['msc_cfg']   
            eps1 = group['eps_cfg'][0]     

            state['dtype_dev'] = (dtype, device)

            state['step'] = torch.tensor(0, dtype=torch.float, device=p.device)
            state['epoch'] = torch.tensor(0, dtype=torch.float, device=p.device)

            # grads.
            state['g_sm'] = torch.zeros_like(p.real) 
            state['g_var'] = torch.zeros_like(p.real) 

            # network weights (parameters)
            state['w'] = _nz(p.real.clone().detach(), eps1) 
                
            # lr.
            state['pc'] = None            
            state['wsq'] = None 
            state['pc_mag'] = None                           
            state['pc_var'] = None    
            state['hcmag'] = None
            state['hcvar'] = None    
            if aoptlr:
                if num_lrc in [1, ]:
                    state['hcmag'] = torch.zeros(1, device=p.device)
                    state['hcvar'] = torch.zeros(1, device=p.device)      

                if num_lrc in [2, 3, 4,]:
                    state['pc'] = torch.zeros_like(p.real)
                    state['wsq'] = torch.zeros_like(p.real)

                if num_lrc in [3, ]:
                    state['pc_mag'] = torch.zeros_like(p.real)
                    state['pc_var'] = 1 # torch.ones_like(p.real)

                if num_lrc in [4, ]:
                    state['pc_mag'] = torch.zeros_like(p.real)
                    state['pc_var'] = torch.zeros_like(p.real)

                state["lr_num"] = torch.zeros(1, device=p.device) if per_lay else torch.zeros_like(p.real)       
            else:
                state["lr_num"] = torch.zeros(1, device=p.device) 

            if dbg:
                state['g[t]'] = torch.zeros_like(p.real) 

                state['alpha[t]'] = torch.ones_like(p.real)

                state['d[t]'] = torch.ones_like(p.real)
    
                state['w[t]'] = torch.zeros_like(p.real)

            # win fcn.
            if rc_cfg[0] != 0 and rc_cfg[1] == 1: # kron. input seq
                state['rcf'] = WINF_II(rc_cfg[0], *rc_cfg[2:])
            else: # rect. input seq
                state['rcf'] = WINF(rc_cfg[0], *rc_cfg[2:])

        return state
        
    def update_group(self, group, closure=None, history=None):
        '''
        Inits and Updates state of params that have gradients.
        '''
        self.dbg = group['dbg']
        self.eps = group['eps_cfg'][0]
        self.spe = group['rcf_cfgs'][5]
        self.tau_win = group['rcf_cfgs'][4]*group['rcf_cfgs'][5]
        self.cf = group['cf']

        aoptlr, lr0, num_lrc = group['lr_cfg']
        ngrad = aoptlr

        betas = group['beta_cfg'] # beta_i,gamma_i,eta,debias = betas[2:]
        wcf = group['wd_cfg'] # wd_cte,wd_lvl = group['wd_cfg']
        scf = group['eps_cfg'] # eps1,repeat_eps = group['eps_cfg']

        per_lay, = group['msc_cfg']
        self.per_lay = per_lay

        maximize = group['maximize']
        
        for p in group['params']:
            if p.grad is None: continue

            ##STATES.
            state = self.lazyinit_param(p, group)      
            
            state['step'] += 1
            if state['step'] % self.spe == 1: state['epoch'] += 1

            t = state['step'].item() 
            k = state['epoch'].item() 

            g = 1*p.grad # cloned
            w = state['w']

            ##----*
            ##UPDATEs.          

            # (window function) defaults to 1, if not active.
            rca = state['rcf'].step(t, k)
            
            # grad-stats!
            g, gpow, gn, v = self.ong(state, w, g, ngrad,
                t, betas, scf, wcf)  

            # lr numerator @ current iteration
            numlr = self.lr_t(state, aoptlr, num_lrc, lr0, betas, 
                t, w, gpow, gn, v)
            
            if self.dbg and history is not None:
                with torch.no_grad():
                    self.update_ltv_hist(p, history, betas, wcf, t, w, g, gpow, v, rca*numlr)

            # param. update !
            if maximize: v.neg_()
            w -= (rca*v*numlr)

            # pass update to the net. (model) !
            with torch.no_grad(): p.copy_(w)


    def ong(self, state, w, g, ngrad, 
            t, betas, scf, wcf):
        '''gradient stats.'''
        gsm = state['g_sm']
        gvar = state['g_var']

        eta = (1-betas[2])/(1-betas[3])
        # weight-decay!
        wd = wcf[0]*w
        # no decoupling of weight from smooth grad?
        if wcf[1] == 0: g += wd   

        # lr denominator @ current iteration
        # grad's power est!
        gpow = 1
        if ngrad:
            gpow, gvar = _lpf_can_t(gvar, g*g, t, 
                betas[1], eta=betas[4], debias=betas[5])     
            if scf[1]: gpow += scf[0]
            torch.sqrt_(gpow)
            gpow += scf[0]
        # normalized grad (via lr denominator).
        gn = g/gpow 

        # smooth grad!
        v, gsm = _lpf_can_t(gsm, g, t, 
            betas[2], betas[3], betas[4], debias=betas[5])
        # normalized input (via lr denominator).
        v /= gpow
        # partial weight decoupling: from grad. smoothing, and energy.
        if wcf[1]: v += eta*wd

        return g, gpow, gn, v

    def lr_t(self, state, aoptlr, num_lrc, lr0, betas, 
            t, w, gpow, gn, v):
        ''' pick learning rate numerator 

        If `aoptlr` is `True`, `lr0` is a trust-region constant for the iteration-dependent learning rate. Otherwise `lr0` is the constant learning rate, the denominator is 1.

        '''
        # iteration-dep. partial weight-grad correlation estimator 
        lrn = state['lr_num']
        if aoptlr:     
            # opt lr's numerator
            if num_lrc in [0, 1]: # denominator only.
                return lr0
            elif num_lrc == 1: # conditional numerator
                grm = state['hcmag']
                grv = state['hcvar']
                numlr = self.a1(lr0, v, grm, grv, t, betas[0])
                return numlr
            elif num_lrc == 2: # relaxed estimate.
                s = state['pc']
                numlr = self.a2(lr0, betas, t, w, gn, s)  
                numlr, lrn = _lpf_can(lrn, numlr, t, beta=0.1) # smooth!   
                return numlr
            elif num_lrc == 3: # default. robust par-corr estimate. 
                s = state['pc']
                wsq = state['wsq']
                pcx = state['pc_mag']
                pcv = state['pc_var']
                _, wsq = _lpf_can_t(wsq, _sq(w), t, betas[0])
                lr0m = lr0/(1+wsq)
                numlr = self.a3(lr0m,betas,t,w,gn,wsq,s,pcx,pcv,gpow)  
                numlr, lrn = _lpf_can(lrn, numlr, t, beta=0.1) # smooth!  
                return numlr
            elif num_lrc == 4: # robust par-corr estimate.             
                s = state['pc']
                wsq = state['wsq']
                pcx = state['pc_mag']
                pcv = state['pc_var']
                _, wsq = _lpf_can(wsq, _sq(w), t, betas[0])
                lr0m = lr0/(1+wsq)
                numlr = self.a4(lr0m,betas,t,w,gn,wsq,s,pcx,pcv,gpow)  
                numlr, lrn = _lpf_can(lrn, numlr, t, beta=0.1) # smooth!  
                return numlr
          
        else: # constant, no normalization
            return lr0
        
    def a1(self, lr0, x, s1, s2, t, beta):
        ''' scalar huber cheb.'''
        xn = torch.norm(x)
        xm, s1 = _lpf_can(s1, xn, t, beta)
        xmag = torch.norm(x-xm).add(self.eps)
        xvar, s2 = _lpf_can(s2, xmag*xmag, t, beta)
        torch.sqrt_(xvar).add_(self.eps)
        # see _huber_c
        return (lr0 / xmag) * torch.clip(xmag, max=self.cf*xvar)
    
    def a2(self, lr0, betas, t, w, gn, s):
        '''[relaxed upbnd.] par-corr. variant

        Here, `lr0` serves as a trust-region constant that bounds the admissible scale of the par-corr update. This function computes an EMA estimate of the maximum partial-correlation relaxed by `lr0`. Ensuring that the effective estimate remains within the trust-region envelope, the resulting EMA estimate is clipped within `[0, lr0]`, Finally, the estimates are averaged across the layer to yield a single, robust par-corr estimate.      
        '''
        # safes: clip (in + out), take mean
        out, s = _lpf_can(s, (_sq(w)+_sq(gn))*lr0, t, betas[0])
        return torch.mean(out.clip(max=lr0)) if self.per_lay else out
    
    def a3(self, lr0m, betas, t, w, gn, wsq, s, pcx, pcv, gpow):
        '''[robust-markov] par-corr. est.
        
        Here, `lr0` serves as a trust-region constant that bounds the admissible scale of the par-corr update. This function computes a robust EMA estimate of the partial-correlation via a Huber-transformed weight-gradient product `w*gn` relative to its expected magnitude `pcx`. To ensure that the effective estimate remains within the trust-region envelope, a Tukey-transform is additionally used to clip the resulting EMA estimate to be within `[0, lr0*(1+E[wsq])]`, Finally, the estimates are averaged across the layer to yield a single, robust par-corr estimate.     
        '''
        # markov-style: clip input
        inp = lr0m * _hub_mark_v(w*gn, pcx, t, betas[0], self.cf, self.eps)
        # estimate
        out, s = _lpf_can_t(s, inp, t, betas[0], eta=1)
        # clip estimate around trust-region `lr0`
        out.abs_().clip_(min=_tuckey(gpow)*lr0m, max=lr0m*_sq(1+wsq))
        # layer-aware smoothing
        return torch.mean(out) if self.per_lay else out

    def a4(self, lr0m, betas, t, w, gn, wsq, s, pcx, pcv, gpow):
        '''[robust-cheb] par-corr. est.
        
        Here, `lr0` serves as a trust-region constant that bounds the admissible scale of the par-corr update. This function computes a robust EMA estimate of the partial-correlation via a Huber-transformed weight-gradient product `w*gn` relative to its expected magnitude `pcx` and variance `pcv`. To ensure that the effective estimate remains within the trust-region envelope, a Tukey-transform is additionally used to clip the resulting EMA estimate to be within `[0, lr0*(1+E[wsq])]`, Finally, the estimates are averaged across the layer to yield a single, robust par-corr estimate.     
        '''
        # chebyshev-style: clip input
        inp = lr0m * _hub_cheb_v1(w*gn, pcx, pcv, t, betas[0], 1, self.eps)
        # estimate
        out, s = _lpf_can(s, inp, t, betas[0], eta=1)
        # clip output around trust-region `lr0`
        out.abs_().clip_(min=_tuckey(gpow)*lr0m, max=lr0m*_sq(1+wsq))
        # layer-aware smoothing
        return torch.mean(out) if self.per_lay else out
    
    def step(self, closure=None, history=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (Callable, optional): A closure that re-evaluates the model and returns the loss.
            history (List[List], optional): external record for each parameter's states.     
            loss (Callable, unused): pass in the Loss function.
            rank (int, unused): GPU rank
            
        """
        for group in self.param_groups: 
            self.update_group(group, closure, history)
 
    def init(self):
        """
        Performs param. initialization, before `step`.
        """
        for group in self.param_groups: 
            for p in group['params']:
                if not getattr(p, 'requires_grad', True): continue
                _ = self.lazyinit_param(p, group)

    def update_ltv_hist(self, p, history, betas, wcf, t, 
                w, g, gpow, v, numlr):
            
            state = self.state[p]
            # filter params
            beta = betas[2]
            gamma = betas[3]
            eta = (1-beta)/(1-gamma)

            etan = 1-beta
            etab = etan if betas[-2] == 0 else betas[-2]
            nf = (etan/etab)/(1-math.pow(beta,t)) if betas[-1] else 1

            # wd params
            rho = wcf[0]

            w_t = w
            w_tm1 = 1*state['w[t]']
            state['w[t]'].copy_(w_t)

            g_t = g
            g_tm1 = 1*state['g[t]']
            state['g[t]'].copy_(g_t)
    
            d_t = gpow
            d_tm1 = 1*state['d[t]']
            state['d[t]'].copy_(d_t)

            # output
            dw_tp1 = -numlr*v
            dw_t = w_t - w_tm1

            # scaled output
            wdw_t = d_t*w_t
            wdw_tm1 = d_tm1*w_tm1

            # lr
            alpha_t = torch.ones_like(g)*numlr*nf/gpow
            alpha_tm1 = 1*state['alpha[t]']
            state['alpha[t]'].copy_(alpha_t)

            # lr ratio
            if t == 1: r_t = torch.ones_like(g_t)
            else: r_t = alpha_t/alpha_tm1
            # print(t, torch.mean(alpha_t))

            # input
            e_t = ((gamma*g_tm1) - g_t)
            if wcf[1]:
                e_t += (rho/eta)*((beta*wdw_tm1) - wdw_t)
            else:
                e_t += (rho)*((gamma*w_tm1) - w_t)
    
            # update-log
            history[p]['e[t]_sup'].append(_supn(e_t))
            history[p]['dw[t]_sup'].append(_supn(dw_t))
            history[p]['dw[t+1]_sup'].append(_supn(dw_tp1))
            history[p]['r[t]_sup'].append(_supn(r_t))
            history[p]['alpha[t]_sup'].append(_supn(alpha_t))
            #
            ems = _capt_mn_sd(e_t)
            dwms = _capt_mn_sd(dw_t)
            dwpms = _capt_mn_sd(dw_tp1)
            rms = _capt_mn_sd(r_t)
            lrms = _capt_mn_sd(alpha_t)
            
            # mean-value per layer
            history[p]['e[t]_mean'].append(ems[0])
            history[p]['dw[t]_mean'].append(dwms[0])
            history[p]['dw[t+1]_mean'].append(dwpms[0])
            history[p]['r[t]_mean'].append(rms[0])
            history[p]['alpha[t]_mean'].append(lrms[0])
            # std
            history[p]['e[t]_sd'].append(ems[1])
            history[p]['dw[t]_sd'].append(dwms[1])
            history[p]['dw[t+1]_sd'].append(dwpms[1])
            history[p]['r[t]_sd'].append(rms[1])
            history[p]['alpha[t]_sd'].append(lrms[1])

