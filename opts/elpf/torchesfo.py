r"""sLPF:Torch"""

# Stateless First-order Low Pass Filter Structure

import torch
from torch import Tensor
from typing import Union


# Root LPF object
class esLPF():
  '''
  Stateless First-order Low Pass Filter Structure
  
  It's input is required to be a constant.
  '''

  def __init__(self, cdevice:any="cpu"):
    
    self.ak = torch.ones((1,), dtype=torch.float, device=cdevice)
    self.bk = torch.ones((1,), dtype=torch.float, device=cdevice)
    self.ck = torch.ones((1,), dtype=torch.float, device=cdevice)

  @torch.no_grad()
  def gains(self, betai:Tensor, step:Tensor=torch.ones((1,))):
        
    # mode:
    # asymptotically unbiased
    
    # input pole
    beta = betai
        
    # asymptotically unbiased, constant
    # - ak
    # input gain
    self.ak = (1-beta)

    # - bk
    # filter pole
    self.bk = beta
    
    # - ck
    # output gain
    # self.ck = 1
    
    return self.bk.pow(step-1)

  def out(self,u:Tensor,x_init:Tensor,noise_k:Tensor,cte:Tensor)->Tensor:
    # integrating
    x = u + (cte*(x_init - u)) + ((1-cte)*noise_k) 
    # filter's output at each step
    # y = x
    return x
    
  def compute(self, u:Tensor, x_init:Tensor, noise_k:Tensor, beta:Tensor, 
              step:Tensor=torch.ones((1,)))-> Tensor:
    
    # stateless
    # req=> input: u, init_state: x_init
    
    # compute gains
    # compute output
    return self.out(u,x_init,noise_k,self.gains(beta,step))
    