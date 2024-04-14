import copy, time, glob, itertools,shutil, re, json

import collections
import dataclasses

from typing import Optional, Tuple, TypeVar, List, Dict, Protocol
from collections import defaultdict, deque
from queue import Queue, PriorityQueue, LifoQueue

import math,random, cmath

import numpy as np

import pandas as pd

import torch, torchvision
torchvision.disable_beta_transforms_warning()

import torch.nn as nn
import torch.nn.functional as tf
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, v2

#
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

    
from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

plt.rcParams['axes.linewidth'] = 0.25
# sans: Helvetica, 'Computer Modern Serif'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica", 
    "font.sans-serif": ["Computer Modern Serif"],
    })
# # for Palatino and other serif fonts use: 'New Century Schoolbook', 'Bookman', 'Times', 'Palatino', 'Charter', 'Computer Modern Roman'
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    "font.serif": ["Times"],
})

torch.random.manual_seed(0)

def reformat_large_tick_values(tick_val, pos=0):
    """
    https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)
    
    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]
            
    return new_tick_format
  
def reformat_small_tick_values(tick_val, pos=0):
    """
    Formats small tick values
    """
    negsign = False
    sgn = str(tick_val)[0]
    if sgn == '-':
        negsign = True
    
    # fix for values greater than 0.01
    if tick_val < 10:
        
        estr = 'e-'
        ise = estr in str(tick_val)
        if not ise:
            estr = 'E-'
            ise = estr in str(tick_val)
            
        if ise:
            val, sigits = str(tick_val).split(estr)
            val = round(float(val),1)
            rdigits = str(val).split('.')
            if rdigits[-1] == '0':
                val = rdigits[0]
            if float(sigits) == 0:
                new_tick_format = f"{val}"
            else:
                new_tick_format = f"{val}E-{int(float(sigits))}"            
        else:
            tick_val = float(tick_val)/(10**2)
            
            estr = 'e-'
            ise = estr in str(tick_val)
            if not ise:
                estr = 'E-'
                ise = estr in str(tick_val)
                
            if ise:
                val, sigits = str(tick_val).split(estr)
                val = round(float(val),1)
                rdigits = str(val).split('.')
                if rdigits[-1] == '0':
                    val = rdigits[0]
                if int(float(sigits)-2) == 0:
                    new_tick_format = f"{val}"
                else:
                    new_tick_format = f"{val}E-{int(float(sigits)-2)}"
                    
            else:  
                sigits = len(str(tick_val))-2
                val = tick_val
                if sigits > 0:
                    sv = str(tick_val)
                    cnt = -1
                    for chr in sv.split('0'):
                        cnt+=1
                        if chr not in ['','.','-']:
                            val = chr
                            break
                    sigits = cnt
                    try:
                        digs = len(val)-1
                        val = round(float(val)/(10**digs),1)
                        rdigits = str(val).split('.')
                        if rdigits[-1] == '0':
                            val = rdigits[0]
                        if sigits-2 <= 0:
                            new_tick_format = f"{val}"
                        else:
                            new_tick_format = f"{val}E-{sigits-2}"
                    except:
                        new_tick_format = tick_val
                else:
                    new_tick_format = tick_val
    else:
        new_tick_format = reformat_large_tick_values(tick_val, pos)
            
    if negsign: 
        new_tick_format = '-' + new_tick_format
        
    return new_tick_format


# print(reformat_small_tick_values(1.6055063e-5))
# print(reformat_small_tick_values(0.000016055))
# print(reformat_small_tick_values(0.001))
# print(reformat_small_tick_values(-0.05))



