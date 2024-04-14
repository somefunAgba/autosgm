# Tutorial adapted: https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html

import math
import os
from typing import Optional, Tuple, TypeVar, List, Dict, Protocol
from collections import defaultdict, deque
from queue import Queue, PriorityQueue, LifoQueue

import numpy as np
from numpy import Infinity

 
  # X = np.zeros((K,1), dtype='complex_')
  
  # H = np.ones((K,N), dtype='complex_')
  # H[K-1,N-1] = D[1]
  # def halfmat(k):
  #   if 0 < k < N-1:
  #     ns = slice(k,N)
  #     ids = (k*np.arange(ns.start,ns.stop) % N)
  #     H[k,ns] = D[ids]
  #     H[ns,k] = D[ids]
      
  #   # X[k,0] = np.vdot(H[k],x)
  #   X[k,0] = np.vdot(D[ids],x)
  # for k in range(N): halfmat(k)
  
  # H = np.ones((K,N), dtype='complex_')
  # def fullhalfmat(k):

  #   #1 |-
  #   ns = slice(k,N-k,1)
  #   ids = (k*np.arange(ns.start,ns.stop,ns.step)) % N
  #   H[k,ns] = D[ids]
  #   H[ns,k] = D[ids]
    
  #   #2. _|
  #   kr = (N-1)-k
  #   ns = slice(kr, k, -1)
  #   ids = (kr*np.arange(ns.start,ns.stop,ns.step)) % N # ids = ((N-ids) % N)
  #   H[kr,ns] = D[ids]
  #   H[ns,kr] = D[ids]
      
  #   X[k,0] = np.vdot(H[k],x)
  #   X[kr,0] = np.vdot(H[kr],x)
  #   # X[[k,kr]] = H[[k,kr],:]@x
  # for k in range((N//2) + ((N%2)!=0)): fullhalfmat(k)
  