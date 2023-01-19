
import torch.nn as nn

"""
CNN: ResNet
"""

class ResBlk(nn.Module):
  '''
  Core Residual block Module for a ResNet (strided convs. for spatial reduction)
  strides = 2 (default)
  '''
  def __init__(self,in_chans, botneck, blk_chans=2,blk_strides=2,ratio_out_to_blk_chans=1):
    super().__init__()
    # => resblk
    # define out_chans:
    out_chans = blk_chans*ratio_out_to_blk_chans
    # 
    self.norm_last = nn.BatchNorm2d(num_features=out_chans)
    #
    # self.norm_last = nn.BatchNorm2d(num_features=blk_chans)    
    if not botneck:
      self.resblk = nn.Sequential(
        # -linear weight layer : channel width x 2, spatial resolution / 2
        # -non.linear fcn
        nn.Conv2d(in_channels=in_chans,out_channels=blk_chans,kernel_size=3,padding=1,stride=blk_strides,bias=False),
        nn.BatchNorm2d(num_features=blk_chans),nn.ReLU(),
        # -linear weight layer: stride = 1
        # -non.linear fcn
        nn.Conv2d(in_channels=blk_chans,out_channels=out_chans,kernel_size=3,padding=1, bias=False),
        self.norm_last,
        
        #
        # nn.BatchNorm2d(num_features=in_chans),nn.ReLU(),        
        # nn.Conv2d(in_channels=in_chans,out_channels=blk_chans,kernel_size=3,padding=1,stride=blk_strides,bias=False),        
        # self.norm_last,nn.ReLU(),  
        # nn.Conv2d(in_channels=blk_chans,out_channels=out_chans,kernel_size=3,padding=1, bias=False),
        
      )
    else: # bottleneck, with out_chans != blk_chans
      self.resblk = nn.Sequential(
        # -bottleneck: linear weight layer : stride = 1
        # -non.linear fcn
        nn.Conv2d(in_channels=in_chans,out_channels=blk_chans,kernel_size=1,padding=0,bias=False),
        nn.BatchNorm2d(num_features=blk_chans),nn.ReLU(),
        # -non.linear fcn
        # -linear weight layer : channel width x 2, spatial resolution / 2
        nn.Conv2d(in_channels=blk_chans,out_channels=blk_chans,kernel_size=3,padding=1, stride=blk_strides, bias=False),    
        nn.BatchNorm2d(num_features=blk_chans),nn.ReLU(),
        # -bottleneck: linear weight layer : blk_chans x 4, stride=1 
        # -non.linear fcn  
        nn.Conv2d(in_channels=blk_chans,out_channels=out_chans,kernel_size=1,padding=0,bias=False),   
        self.norm_last,
        
        #
        # nn.BatchNorm2d(num_features=in_chans),nn.ReLU(),
        # nn.Conv2d(in_channels=in_chans,out_channels=blk_chans,kernel_size=1,padding=0,bias=False),
        # nn.BatchNorm2d(num_features=blk_chans),nn.ReLU(),
        # nn.Conv2d(in_channels=blk_chans,out_channels=blk_chans,kernel_size=3,padding=1, stride=blk_strides, bias=False),    
        # self.norm_last,nn.ReLU(),
        # nn.Conv2d(in_channels=blk_chans,out_channels=out_chans,kernel_size=1,padding=0,bias=False), 
        
      )
      
    # => Resolve if resblk's output channel width is different from input channel width, or strides is not 1
    if ((in_chans != out_chans) or (blk_strides != 1)):
      self.lnkblk = nn.Sequential(
        # 
        nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=1, stride=blk_strides,bias=False),
        nn.BatchNorm2d(num_features=out_chans),
        
        #
        # nn.BatchNorm2d(num_features=in_chans), nn.ReLU(),
        # nn.Conv2d(in_channels=in_chans,out_channels=out_chans,kernel_size=1, stride=blk_strides,bias=False),
        
      )  
    else:
      self.idblk = nn.Sequential(nn.Identity())
      
    self.relu_out = nn.ReLU()
    # self.smooth_out = RFLowpassHLayer(beta=0.1, device=cfgs['device'], group=False) 
    
  def forward(self, X):
    Y = self.resblk(X)
    Y += self.lnkblk(X)
    Y = self.relu_out(Y)
    # Y = self.smooth_out(Y)
    return Y
    
class ResStem(nn.Module):
  '''
  External input Stem block (pre-processing) to a ResNet Body
  
  This downsamples resolution of network input by 'stem_stride' and makes 
  the input network channel width to be 'stem_chans'
  '''
  def __init__(self,in_chans,stem_chans=64,stem_kern=7,stem_stride=4):
    super(ResStem,self).__init__()
    padding = stem_kern-((stem_kern+1)//2) # odd kernel width assumed
    self.stem_chans = stem_chans
    self.stemblk = nn.Sequential(
      # 
      nn.Conv2d(in_channels=in_chans,out_channels=stem_chans,kernel_size=stem_kern,padding=padding,stride=stem_stride, bias=False),
      nn.BatchNorm2d(num_features=stem_chans),nn.ReLU(),
      
      #
      # nn.BatchNorm2d(num_features=in_chans),nn.ReLU(),      
      # nn.Conv2d(in_channels=in_chans,out_channels=stem_chans,kernel_size=stem_kern,padding=padding,stride=stem_stride, bias=False),
      
    )
    
  def forward(self, X):
    Y = self.stemblk(X)
    return Y
  
class CHead(nn.Module):
  '''
  A output classification head after ResNet Body
  '''
  def __init__(self,in_chans,num_classes):
    super(CHead,self).__init__()
    infeats = in_chans
    self.blk1 = nn.Sequential(
      # or nn.AvgPool2d(sdim,stride=1)
      nn.AdaptiveAvgPool2d((1,1)), 
      nn.Flatten(),
      nn.Linear(in_features=infeats,out_features=num_classes, bias=True)
    )
    
  def forward(self, X):
    Y = self.blk1(X)
    return Y

class ResStage(nn.Module):
  '''
  A transform stage in a ResNet: 
    configures a residual block with a number of steps
  '''
  def __init__(self,in_chans,botneck,ratio_out_to_blk_chans,blk_chans,resblk_steps=1,stage_strides=2):
    super(ResStage,self).__init__()
    blk=nn.ModuleList()
    # ratio_out_to_blk_chans = 4 if botneck else 1
    for _ in range(resblk_steps):
      blk.append(ResBlk(in_chans,botneck,blk_chans,stage_strides, ratio_out_to_blk_chans))
      # out chan := blk chan * ratio o/b.
      in_chans = blk_chans*ratio_out_to_blk_chans 
      # in a stage: keep resolution after first blk_step
      stage_strides = 1 
      
    self.blkseq = nn.Sequential(*blk)
    
  def forward(self, X):
    Y = self.blkseq(X)
    return Y

class ResNet(nn.Module):
  '''
  A ResNet Module

  in_chans:int channels of external input to ResNet
  stem_chans:int channels of input stem blk
  stem_kern:int kernel size for input stem blk
  stem_stride:int stride for input stem blk
  bodyarch: defines resblk/stage configs: list of tuples (blk_chans,resblk_steps)
  num_classes:int output classes in each head
  botneck:bool=False i/o bottleneck in core res blk
  ratio_out_to_blk_chans:int ratio of out to block channels in core res blk.
  num_heads:int=1  number of class heads
  '''
  def __init__(self,in_chans,stem_chans,stem_kern,stem_stride,bodyarch,num_classes=10, botneck=False, ratio_out_to_blk_chans=1, num_heads = 1):
    super(ResNet,self).__init__()
    
    # ratio_out_to_blk_chans = 4 if botneck else 1
    
    # Stem
    stem = ResStem(in_chans,stem_chans=stem_chans,stem_kern=stem_kern, stem_stride=stem_stride)
    self.net = nn.Sequential()
    self.net.add_module(f"in-stem",nn.Sequential(stem))
    in_chans = stem.stem_chans
    
    # Body
    for i,b in enumerate(bodyarch):
      self.net.add_module(f"body-{i+1}",ResStage(in_chans,botneck,ratio_out_to_blk_chans,*b))
      in_chans = b[0]*ratio_out_to_blk_chans
      
    # Head
    self.num_heads = num_heads
    self.heads = nn.ModuleList()  
    for _ in range(num_heads):
      self.heads.append(
        nn.Sequential(CHead(in_chans=in_chans,num_classes=num_classes))
      )
    # self.net.add_module(f"out-head",CHead(in_chans,num_classes))
    
  def forward(self, X):
      
    Y = self.net(X)
    if self.num_heads > 1:
      logits = []
      for hid in range(self.num_heads):
        head = self.heads[hid]
        logit = head(Y)
        logits.append(logit)
    else:
      head = self.heads[0]
      logits = head(Y)
    return logits
  
    
# ---- CUSTOM RESNETs

# change stem_chans, stem kern, stem stride as desired for input image

# RESNET C (CUSTOM): 
# C Layers stem:(1) + body:(sum of resblk_steps for each stage)*(2 if botneck else 3) + head:(num_heads)
class ResNetC(ResNet):
  ''' 
  Custom ResNet (strided convs.) for in:image (W x H x C)
  '''
  
  def __init__(self,in_chans,stem_chans=64,layers="6",botneck=False,ratio_obc=1,num_classes=10,num_heads=1):
    
    # custom:
    stem_kern, stem_stride = 3, 1
    
    # custom: arch= list: (blk_chans,resblk_steps)
    if not botneck:
      # ratio_obc = 1 
      if layers == "6":
        # RESNET 6: stem:(1) + body:(1+1)*2 = (4) + head:(1)
        archs = [(stem_chans,1),
              (stem_chans*2,1)
              ]
      elif layers == "8":
        # RESNET 8: stem:(1) + body:(1+1+1)*2 = (6) + head:(1)  
        archs = [(stem_chans,1),
                (stem_chans*2,1),
                (stem_chans*4,1)
              ]
      elif layers == "10":
        # RESNET 10: stem:(1) + body:(1+1+1+1)*2 = (8) + head:(1)
        archs = [(stem_chans,1),
                (stem_chans*2,1),
                (stem_chans*4,1),
                (stem_chans*8,1)
              ]
      elif layers == "18":
        # RESNET 18: stem:(1) + body:(2+2+2+2)*2 = (16) + head:(1)
        archs = [(stem_chans,2),
                (stem_chans*2,2),
                (stem_chans*4,2),
                (stem_chans*8,2)
              ]
      elif layers == "34":
        # RESNET 34: stem:(1) + body:(3+4+6+3)*2 = (32) + head:(1)
        archs = [(stem_chans,3),
                (stem_chans*2,4),
                (stem_chans*4,6),
                (stem_chans*8,3)
              ]
      else:
        raise ValueError(f"For bottleneck={botneck}, layers = '{layers}' is currently not defined!")
    elif botneck:
      # bottleneck res blk
      
      # override user selection.
      ratio_obc = 4
      
      if layers == "8":
        # RESNET 8: stem:(1) + body:(1+1)*3 = (6) + head:(1)
        archs = [(stem_chans,1),
              (stem_chans*2,1)
              ]     
      elif layers == "11":
        # RESNET 11: stem:(1) + body:(1+1+1)*3 = (9) + head:(1)
        archs = [(stem_chans,1),
                (stem_chans*2,1),
                (stem_chans*4,1)
              ]     
      elif layers == "14":
        # RESNET 14: stem:(1) + body:(1+1+1+1)*3 = (12) + head:(1)
        archs = [(stem_chans,1),
                (stem_chans*2,1),
                (stem_chans*4,1),
                (stem_chans*8,1)
              ]  
      elif layers == "50":
        # RESNET 50B: stem:(1) + body:(3+4+6+3)*3 = (48) + head:(1)
        archs = [(stem_chans,3),
                (stem_chans*2,4),
                (stem_chans*4,6),
                (stem_chans*8,3)
              ] 
      elif layers == "101":
        # RESNET 101: stem:(1) + body:(3+4+23+3)*3 = (99) + head:(1)
        archs = [(stem_chans,3),
                (stem_chans*2,4),
                (stem_chans*4,23),
                (stem_chans*8,3)
              ] 
      elif layers == "152":
        # RESNET 152: stem:(1) + body:(3+8+36+3)*3 = (150) + head:(1)
        archs = [(stem_chans,3),
                (stem_chans*2,8),
                (stem_chans*4,36),
                (stem_chans*8,3)
              ] 
      else:
        raise ValueError(f"For bottleneck={botneck}, layers = '{layers}' is currently not defined!")
        
          
    super().__init__(in_chans,stem_chans,stem_kern,stem_stride,archs,num_classes,botneck,ratio_out_to_blk_chans=ratio_obc,num_heads=num_heads)
  
  def __getstate__(self):
    state = self.__dict__.copy()
    return state
  
  def __setstate__(self, state):
    self.__dict__.update(state)
    
