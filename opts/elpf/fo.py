r"""LPF"""


#import math,statistics

# Root LPF object
class LPF():
  
  def __init__(self, inplace:bool=False, direct:bool=False):
    
    self.ak = 1.0
    self.bk = 0.0
    self.ck = 1.0
    self.inplace = inplace
    self.direct = direct
    self.var_mk = 1.0
  
  
  def gains(self, betain=0.9, step:int=1, bmode:int=0, mode=1):
    
    # vary | True: fully unbiased, False: asymptotically unbiased
    # modes: 1,2,3:
    # 1. fully unbiased impl., 2. asymptotically unbiased 
    # 3. unbiased impl. (for redundancy)
    
    # - ak
    # filter gain or step-size
    # eps = 1e-8
    
    # input pole variations
    if bmode == 0:
      beta = betain
    elif bmode == 11: # increasing
      beta = ((step-1)/(1+step))
    elif bmode == 1: # increasing
      beta = ((step)/(1+step))
    elif bmode == 10: # increasing with a constant in (0,1]
      beta = (betain*((step)/(1+step)))    
    elif bmode == -1: # decreasing
      beta = (1/(1+step))
    elif bmode == -10: # decreasing with a constant in (0,1]
      beta = (betain*(1/(1+step)))
    else: # elif bmode == 0: # constant, or otherwise
      beta = betain
    
    if mode == 1: 
      # unbiased: varying
      # steady
      self.ak =  (1 - beta)/(1-(beta**step))
      # - bk
      # filter pole
      self.bk = 1 - self.ak
      # - ck
      # output gain      
      # self.ck = 1    
    elif mode == 3: # for comparison
      # unbiased: constant
      # steady
      self.ak =  (1 - beta)
      # - bk
      # filter pole
      self.bk = beta
      # - ck
      # output gain      
      self.ck = 1/(1-(beta**step)) 
    elif mode == 2:
      # asymptotically unbiased, constant
      # steady
      self.ak =  (1 - beta)
      # - bk
      # filter pole
      self.bk = beta
      # - ck
      # output gain      
      # self.ck = 1   

  
  def out_state(self,u,x):
    
    v = ((self.ak)*(u - x)) 

    if self.inplace:
      # inplace: preserve state memory
      x += v #((self.ak)*(u - x))
    else:
      # overwrite: clear state memory
      if self.direct:
        # direct form
        x = ((self.bk)*x) + ((self.ak)*u)
      else:
        # integrating form
        x = (x) + v #(((self.ak)*(u - x)))
    
    y = (self.ck)*x
    
    return y, x
    
  
  def compute(self, u, x, 
              beta=0.9, step:int=1, 
              bmode:int=0, mode:int=1):
    # step > 0 and int
    # given input: u
    
    # compute gains
    self.gains(beta,step,bmode,mode)
    
    # compute output, state
    y,x = self.out_state(u,x)
    
    return y,x
