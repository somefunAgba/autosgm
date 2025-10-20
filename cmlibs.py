import os, copy, time, glob, itertools,shutil, re, json

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
# 

plt.rc('text.latex', preamble=r'\usepackage{xcolor} \usepackage{amsmath} \usepackage{amssymb} \usepackage{times} \usepackage{array}')

plt.rcParams['mathtext.fontset'] = 'stixsans'
# plt.rcParams['mathtext.rm'] = 'Avenir Next'
# plt.rcParams['mathtext.it'] = 'Avenir Next:italic'
# plt.rcParams['mathtext.bf'] = 'Avenir Next:bold'
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
    "font.family": "serif",
    # "font.serif": ["Times"],
    "font.serif": ["Computer Modern Roman"],
})


torch.random.manual_seed(0)

def cbinsearch(val,l=0, h=500):

    val = abs(val)
    if val == 1:
        # id = 0
        return 0
    
    id = None
    while l <= h:
        m = (l+h)//2
        if val >= 1:
            eval = 10**m
            if eval >= val:
                id = m
                h = m-1
            else:
                l = m+1
        elif val > 0:
            eval = 10**(-m)
            if val >= 10**(-m) :
                id = -m
                h = m-1
            else:
                l = m+1
        elif np.isnan(val) or np.isinf(val):
            id = None
    # print(val, id)

    return id

def logpow_tick_values(tick_val, pos=0):
    """
    logplot tick vals
    """
    # print('val', tick_val)
    if not np.isinf(tick_val) or np.isnan(tick_val):
        # print(tick_val)
        if tick_val == 0:
            return str(r'$\mathrm{\mathsf{\hbox{0}}}$')
        else:
            id = cbinsearch(tick_val)
            if id is not None:
                fntsel = r'\fontsize{0.5}{0.05}\selectfont'
                new_tick_format = r'\hbox{10}^{\hbox{'+fntsel+f'{id}'+'}}'
                if tick_val < 0:
                    new_tick_format = r'-' + new_tick_format
            else:
                new_tick_format = tick_val
            
            # make new_tick_format into a string value
            new_tick_format =  str(r'$\mathrm{\mathsf{'+ new_tick_format + r'}}$')
    else:
        new_tick_format = tick_val

    # if id >=0: 
    #     divisor = 10**id
    #     fmtval = round(tick_val/divisor, 3)
    # else: 
    #     muliplier = 10**abs(id)
    #     fmtval = round(tick_val*muliplier, 3)
    # tick_format =  rf'{fmtval} \times' + new_tick_format  
    # print(id, tick_format)

    # make new_tick_format into a string value
    

    return new_tick_format

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
    elif int(tick_val) == 0:
        new_tick_format = int(tick_val)
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
  
def reformat_small_tick_values(tick_val, pos=1):
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

# print(logpow_tick_values(23.4))
# print(logpow_tick_values(-0.001))
# print(cbinsearch(0.006))
# print(cbinsearch(1000))

def join_name_to_dir(file_dir:str,file_name:str):
  '''
  join_name_to_dir _summary_

  _extended_summary_

  Args:
      file_dir (str): _description_
      file_name (str): _description_

  Returns:
      _type_: _description_
  '''
  os.makedirs(file_dir, exist_ok=True)
  filepath = f"{file_dir}/{file_name}"
  return filepath

def savelist_df2csv(filedir, filename, thelist, index=True):
  '''
  savelist_df2csv _summary_

  _extended_summary_

  Args:
      filedir (_type_): _description_
      filename (_type_): _description_
      thelist (_type_): _description_
  '''
  PATHpreds =  join_name_to_dir(filedir, filename)
  df = pd.DataFrame(thelist)
  df = df.T
  df.to_csv(PATHpreds+".csv", index=index)
  # to_csv's index is true by default, can change to false to remove row ids
  # print(df.head()) print(df.tail())
    


"""
Custom SymLog Scale Example with Explicit Linear Zone Labels
------------------------------------------------------------
This file defines a custom symmetric log scale ("mysymlog") for Matplotlib,
including forward/inverse transforms, tick locators, and formatters.
The linear region around zero is explicitly labeled as "linear zone".
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter


# ----------------------------
# Transform definitions
# ----------------------------
class MySymLogTransform(mtransforms.Transform):
    input_dims = output_dims = 1
    is_separable = True

    def __init__(self, linthresh=1e-2, base=10.0):
        super().__init__()
        self.linthresh = linthresh
        self.base = base

    def transform_non_affine(self, a):
        a = np.array(a, copy=False)
        sign = np.sign(a)
        abs_a = np.abs(a)

        out = np.zeros_like(a, dtype=float)
        mask = abs_a <= self.linthresh
        out[mask] = a[mask]  # linear region
        out[~mask] = sign[~mask] * (
            self.linthresh + np.log(abs_a[~mask] / self.linthresh) / np.log(self.base)
        )
        return out

    def inverted(self):
        return InvertedMySymLogTransform(self.linthresh, self.base)


class InvertedMySymLogTransform(mtransforms.Transform):
    input_dims = output_dims = 1
    is_separable = True

    def __init__(self, linthresh=1e-2, base=10.0):
        super().__init__()
        self.linthresh = linthresh
        self.base = base

    def transform_non_affine(self, a):
        a = np.array(a, copy=False)
        sign = np.sign(a)
        abs_a = np.abs(a)

        out = np.zeros_like(a, dtype=float)
        mask = abs_a <= self.linthresh
        out[mask] = a[mask]  # linear region
        out[~mask] = sign[~mask] * (
            self.linthresh * (self.base ** (abs_a[~mask] - self.linthresh))
        )
        return out

    def inverted(self):
        return MySymLogTransform(self.linthresh, self.base)


# ----------------------------
# Custom Formatter
# ----------------------------
class MySymLogFormatter(ScalarFormatter):
    def __init__(self, linthresh=1e-2, base=10.0, **kwargs):
        super().__init__(**kwargs)
        self.linthresh = linthresh
        self.base = base

    def __call__(self, x, pos=None):
        if abs(x) <= self.linthresh:
            return f"{x:.2g}\n(linear)"
        else:
            return super().__call__(x, pos)


# ----------------------------
# Scale definition
# ----------------------------
class MySymLogScale(mscale.ScaleBase):
    name = 'mysymlog'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.linthresh = kwargs.get('linthresh', 1e-2)
        self.base = kwargs.get('base', 10.0)

    def get_transform(self):
        return MySymLogTransform(self.linthresh, self.base)

    def set_default_locators_and_formatters(self, axis):
        # Major ticks: logarithmic outside the linear region
        axis.set_major_locator(LogLocator(base=self.base))
        axis.set_major_formatter(MySymLogFormatter(linthresh=self.linthresh, base=self.base))

        # Minor ticks: log-style
        axis.set_minor_locator(LogLocator(base=self.base, subs=np.arange(2, self.base) * 0.1, numticks=10))
        axis.set_minor_formatter(NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax


# ----------------------------
# Register the scale
# ----------------------------
mscale.register_scale(MySymLogScale)


# ----------------------------
# Demo plot
# ----------------------------
if __name__ == "__main__":
    x = np.linspace(-100, 100, 500)
    y = x**3

    fig, ax = plt.subplots()
    ax.plot(x, y, label="y = x^3")

    # Use the custom scale
    ax.set_yscale('mysymlog', linthresh=1e-16, base=10)

    ax.set_title("Custom SymLog Scale with Explicit Linear Zone Labels")
    ax.set_xlabel("x")
    ax.set_ylabel("y (mysymlog scale)")
    ax.legend()
    ax.grid(True, which='both', ls='--', lw=0.5)

    plt.show()

