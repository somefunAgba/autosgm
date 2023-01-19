r"""LPF:Torch"""

# General Unbiased First-order Low Pass Filter Structure

import torch
from torch import Tensor
from typing import Union, List, Optional


# Root LPF object
class LPF():
  # General Unbiased First-order Low Pass Filter Structure

  def __init__(self, inplace:bool=True, direct:bool=False, cdevice:any="cpu"):
    
    self.ak = torch.ones((1,), dtype=torch.float, device=cdevice)
    self.bk = torch.ones((1,), dtype=torch.float, device=cdevice)
    self.ck = torch.ones((1,), dtype=torch.float, device=cdevice)
    self.inplace = inplace
    self.direct = direct
  
  @torch.no_grad()
  def gains(self, betai:Tensor, step:Tensor=torch.ones((1,)), bmode:int=0,  mode:int=1):
        
    # modes: 1,2,3:
    # 1. fully unbiased impl., 2. asymptotically unbiased 
    # 3. unbiased impl. (for redundancy)
    
    # input pole variations
    if bmode == 0: # constant, or otherwise
      beta = betai
    elif bmode == 11: # increasing
      beta = (step-1)/step
    elif bmode == 1: # increasing
      beta = (step)/(1+step)
    elif bmode == 10: # increasing with a constant in (0,1]
      beta = (betai*step)/(1+step)
    elif bmode == -1: # decreasing, # same as mode 0.
      beta = (1)/(1+step)
    elif bmode == -10: # decreasing with a constant in (0,1]
      beta = (betai)/(1+step)
    else:
      beta = betai
          
    if mode == 1: 
      # unbiased: varying
      # - ak
      # input gain
      self.ak = (1-beta) / (1-(beta.pow(step)))

      # - bk
      # filter pole
      self.bk = 1-self.ak
      
      # - ck
      # output gain
      # self.ck = 1
    elif mode == 3:
      #  unbiased: constant
      # - ak
      # input gain
      self.ak = (1-beta) 

      # - bk
      # filter pole
      self.bk = beta
      
      # - ck
      # output gain
      self.ck = 1/(1-(beta.pow(step)))
    elif mode == 2:
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
    

  def out_state(self,u:Tensor,x:Tensor)->Union[Tensor,Tensor]:
    
    v = (self.ak)*(u - x)
    if self.inplace:
      # inplace: preserve state memory
      if self.direct:
        x.mul_(self.bk).add_(self.ak*u)
      else:
        x += v
    else:
      # overwrite: clear state memory
      if self.direct:
        # direct form
        x = ((self.bk)*x) + ((self.ak)*u)
      else:
        # integrating form
        x = (x) + (v)
        
    y = (self.ck)*x
    return y, x
    
  def compute(self, u:Tensor, x:Tensor, beta:Tensor, 
              step:Tensor=torch.ones((1,)), 
              bmode:int=0, mode:int=1)-> Union[Tensor,Tensor]:
    
    # given input: u
    
    # compute gains
    with torch.no_grad():
      self.gains(beta,step,bmode,mode)
      
    # compute output, state
    y,x = self.out_state(u,x)
    
    return y,x