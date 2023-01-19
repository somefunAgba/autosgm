r"""PID:AutoSGM"""

import math
import numpy as np

try:
    from opts.elpf.fo import LPF
    from opts.elpf.sfo import sLPF
except:
    from elpf.fo import LPF
    from elpf.sfo import sLPF

from copy import copy as cpy

from typing import List,Union,Optional

class AGM():
    ''' Gradient Method: automatic algorithm for learning/control/estimation

    Core algorithm implementation (Torch)
    '''
    def __init__(self, mlpf:LPF,vlpf:LPF,wlpf:LPF, 
                use_optim_choice:bool, smooth_out:bool,maximize:bool, weight_decay,eps,betas:tuple) -> None:
        self.gradref = 0
        self.mlpf = mlpf
        self.vlpf = vlpf
        self.wlpf = wlpf
        self.use_optim_choice = use_optim_choice
        self.smooth_out = smooth_out
        self.maximize = maximize
        self.weight_decay = weight_decay
        self.eps = eps
        self.beta_o = betas[2]
        self.beta_n = betas[1]
        self.beta_i = betas[0]
        pass
    
    def compute(self,step:int,param,param_grad,
                mk,vk,qk,wk,ss_k,optparam):
        
        # -1. input grad.
        # grad = gradi + (weight_decay*param)
        grad = (param_grad+(self.weight_decay*wk))
            
        error = -grad # (self.grad_ref - grad) = -grad
        if self.maximize:
            error = -error
            
        if self.use_optim_choice:
            bmo, stepm = 0, step
        else:
            bmo, stepm = 10, step
        # E[g]: input mean estimation
        errmean, mk = self.mlpf.compute(u=error, x=mk, 
                        beta=self.beta_i, bmode=bmo, step=stepm)
            
        # - change in parameters (weights)
        delta_param = errmean+0
                
        # -2. linear step-size fcn.
        if optparam is not None:
            if self.smooth_out:
                alphap = np.abs(qk-optparam)/(np.abs(errmean)+self.eps)
            else:
                alphap = np.abs(wk-optparam)/(np.abs(errmean)+self.eps)
            delta_param = delta_param*(alphap)
        else:
            if self.use_optim_choice:        
                # derivative mode: variance estimation / normalization
                # inner product: <g,g> = E[g^2] 
                # gradnorm = grad.norm("fro")
                errsq =  error*error
                errvar, vk = self.vlpf.compute(u=errsq, x=vk, beta=self.beta_n, bmode=10, step=step)
            
                    
                alphap = ((ss_k)/(np.sqrt(errvar)+(self.eps)))
                #
                den = (np.sqrt(errvar)/(ss_k)) + (self.eps/(ss_k))
                # alphap = (1 / den) 
                delta_param = delta_param/(den)
            else: 
                # no variance estimation added to alpha_p.
                alphap = ss_k 
                delta_param = delta_param*(alphap)      
                # kpk = alphap; delta_param = errmean.mul_(kpk)
            
        # -3. update
        if self.smooth_out:
            # update
            wk += (delta_param)
            # E[w] output parameter smoothing
            paramf, qk = self.wlpf.compute(u=wk, x=qk,
                    beta=self.beta_o, bmode=0,  step=step)
            # pass values to network's placeholder.
            param = paramf

        else:
            # rightly assume param is smooth
            # update
            wk += (delta_param)
            # pass values to network's placeholder.
            param = wk
            
        # END
        
        return param, alphap        
        
    

class PID():

    """Implements: "Stochastic" Gradient Method (Numpy), an automatic learning algorithm.
    
    The SGM is a discrete-time PID structure, with low-pass regularization.
    
    PID structure is a discrete-time structure with Proportional + Integral + Derivative components
    
    The proportional gain component is the SGM's "step-size" (or learning rate) which represents a linear correlation variable.
    
    The derivative component is the first-order gradient of the objective function wrt the parameter
    
    The integral component is the additive means of adaptation using the step-size and first-order gradient
    
    Author: Oluwasegun Ayokunle Somefun. somefuno@oregonsate.edu 
        
    Affilation: Information Processing Group, Orgeon State University
        
    Date: (Changes)

        2022. Nov. (Heavy Refactoring + added model parameter (weight) adaptation)
        
        2023. Jan. (Heavy Refactoring + added step-size adaptation)

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
        >>> import opts.asgm.nplasgm as asgm
        >>> # optimizer = asgm.PID(param_init, steps_per_epoch=1, ss_init=1e-3, beta_i=0.9, weight_decay=1e-5, optparams=None):
    """

    # 
    def __init__(self, param_init, steps_per_epoch=1, ss_init=1e-3, ss_end=0, eps_ss = 5e-1, beta_i=0.9, beta_n=0.999, beta_o=1e-5, weight_decay=1e-5, use_optim_choice=True, smooth_out=True, maximize=False, optparams=None):

        if not 0.0 < ss_init:
            raise ValueError(f"Invalid value: rho={ss_init} must be in (0,1) for tuning")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        self.ssslpf = sLPF()
        self.mlpf = LPF(inplace=True)
        self.vlpf = LPF(inplace=True)
        self.wlpf = LPF(inplace=True)
        
        if maximize:
            weight_decay = -weight_decay
            
        self.use_optim_choice = use_optim_choice
        self.smooth_out = smooth_out
        self.debug = True
        self.steps_per_epoch = steps_per_epoch
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.optparams = optparams

        # compute: one-pole filter gain
        # eps added: div. by zero.
        self.eps = 1e-8
        
        # step-size.
        self.ss_init = ss_init
        self.ss_end = ss_end
        # eps_ss: convergence accuracy to ss_end.
        self.beta_ss = np.exp((np.log(eps_ss)/steps_per_epoch))
        
        self.betas = (beta_i,beta_n,beta_o,self.beta_ss)
        # print.
        if self.debug:
            strlog = f"aSGM info:\n[step-size or linear correlation value (starting := {ss_init:.4g}), (ending := {ss_end:.4g}), step-size pole : {self.beta_ss:.4g}],\n[LPF Poles [i,n,o] : {beta_i:.4g}, {beta_n:.4g}, {beta_o:.4g}]"            
            txt = "*"
            print(f"{txt * len(strlog)}")    
            print(strlog)
            print(f"{txt * len(strlog)}\n")
                        
        self.state = dict()  
        for i in range(param_init.size):
            self.state[i] = dict()
    


    def step(self, params, grads, stepnone=None, alphapnone=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates 
                the model gradnd returns the loss.
        """
        loss = None

        # list of parameters, gradients, gradient notms
        # params_with_grad = []
        # grads = []

        # list to hold step count
        state_steps = []

        # list to hold one-pole filter memory
        mk, vk, qk, wk = [], [], [],[]

        # beta_ss = self.betas[3]
                        
        asgm = AGM(self.mlpf,self.vlpf,self.wlpf,
        self.use_optim_choice,self.smooth_out,self.maximize, 
        self.weight_decay, self.eps,self.betas) 

        for i,p in enumerate(params):

            state = self.state[i]
            # initialize state, if empty
            if len(state) == 0:
                # first step count
                state['step'] = 0

                # one-pole filter memory (filter)
                state['m'] = np.zeros_like(p)

                state['v'] = np.zeros_like(p)

                state['q'] = np.copy(p) #np.zeros_like(p) # if biased.

                state['w'] = np.copy(p)
            
            mk.append(state['m'])
            vk.append(state['v'])
            qk.append(state['q'])
            wk.append(state['w'])

            # update the step count by 1.
            state['step'] += 1

            # record the step
            state_steps.append(state['step'])
        
        # Compute this algorithm for each parameter    
        ## Actual Learning Event: 
        alphaps, ss_k = control_event(asgm, self.ssslpf, loss,
            params, grads, mk, vk, qk, wk, self.steps_per_epoch, state_steps, self.ss_init, self.ss_end, self.beta_ss, self.optparams
        )
        
        # update changes to low-pass filter states: needed here
        for i,_ in enumerate(params):
            state = self.state[i]
            state['m'] = mk[i]
            state['v'] = vk[i]
            state['q'] = qk[i]
            state['a'] = wk[i]

        return params, state['step'], alphaps



# Functional Interface
def control_event(asgm: AGM, ssslpf:sLPF, loss, 
            params, grads, mk, vk, qk, wk, 
            steps_per_epoch:int, state_steps: List[int],
            ss_init, ss_end, beta_ss, 
            optparams):
    
    r"""Functional API that computes the AutoSGM control/learning algorithm for each parameter in the model.

    See : class:`~torch.optim.AutoSGM` for details.
    """

    step = state_steps[0]
    
    # - At each step, stably adapt step_size_init to step_size_end using 
    # (AutoSGM) PID structure reduced to a stateless LPF
    # cyclic step
    stepc = (((step-1) % steps_per_epoch) + 1)
    ss_k = ssslpf.compute(u=ss_end, x_init=ss_init, beta=beta_ss, step=stepc)
    
    # - At each step, adapt parameters (weights of the model) using 
    # (AutoSGM) PID structure.
    # E[param] = E[param + alphap*sy] = E[param] + alphap*E[grad]  
    alphaps = []
    # for each parameter in the model.
    for i, _ in enumerate(params):
        if optparams is not None:
            optparam = optparams[i]
        else:
            optparam = None
        params[i], alphap = asgm.compute(step,params[i],grads[i],mk[i],vk[i],qk[i],wk[i],ss_k,optparam)
        alphaps.append(alphap)
        
    return alphaps, ss_k