# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:55:12 2017

@author: awtho
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
import csv
from myCNN import *

def main():
  test = loadmat("test.mat")
  xTe = test['x']
  
  x_image = xTe.reshape([-1,28,28])
  n,_,_ = x_image.shape
  NUM_I = 5
  
  for i in range(0, n, NUM_I):
    fig = plt.figure()
    for j in range(0,NUM_I):
      a=fig.add_subplot(3,2,j+1)
      imgplot = plt.imshow(x_image[i+j], cmap='gray')
      a.set_title(i+j)
    plt.show()
      #plt.imshow(x_image[i], cmap='gray', interpolation='nearest') 
    input("Press Enter to continue...")

if __name__ == "__main__":
  main()