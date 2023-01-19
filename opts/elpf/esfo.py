r"""sLPF"""

# Stateless Asymptotically Unbiased First-order Low Pass Filter Structure


# Root LPF object
class esLPF():
  '''
  Stateless First-order Low Pass Filter Structure
  
  It's input is required to be a constant.
  '''
  
  def __init__(self):
    
    self.ak = 1.
    self.bk = 1.
    self.ck = 1.

  def gains(self, betai, step=1):
        
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
    
    return self.bk**(step-1)

  def out(self,u,x_init,noise_k,cte):
    # integrating
    x = u + (cte*(x_init - u)) + ((1-cte)*noise_k) 
    # direct: x = (1-cte)*u + cte*x_init + ((1-cte)*noise_k) 
    # filter's output at each step
    # y = x
    return x
    
  def compute(self, u, x_init,noise_k, beta, step=1):
    
    # stateless
    # req=> input: u, init_state: x_init
    
    # compute gains
    # compute output
    return self.out(u,x_init,noise_k,self.gains(beta,step))
    