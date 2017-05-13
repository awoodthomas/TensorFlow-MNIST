# -*- coding: utf-8 -*-

#<GRADED>
import numpy as np
#</GRADED>
import sys
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

def main():
    train = loadmat("train.mat")
    xTr  = train['x'].astype(np.float32, copy=False)
    yTr  = train['y']
    n,d = xTr.shape
    
    test = loadmat("test.mat")
    xTe = test['x']
    



if __name__ == "__main__":
    main()