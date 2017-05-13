# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:53:41 2017

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

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


tf.logging.set_verbosity(tf.logging.INFO)

def main(unused):
    
    #LOAD DATA
    train = loadmat("train.mat")
    xTr  = train['x'].astype(np.float32, copy=False)
    yTr  = train['y']
    n,d = xTr.shape
    
    test = loadmat("test.mat")
    xTe = test['x']
    
    print(n, ' training points')
    NUM_BAGS = 1
    alphas = np.ones(NUM_BAGS)/NUM_BAGS
    nTe, _ = xTe.shape
    preds = np.zeros(nTe)
    
    for b in range(0,NUM_BAGS):
      bag_indices = np.random.randint(0, n-1, n)
      #TRAIN THE NET
      CNN = myCNN()
      CNN.train(xTr[bag_indices], yTr[bag_indices], 10, 100)
      #ADD ON TO PREDS
      print("Finished net #%d, added to ensemble."%(b+1))
      preds += alphas[b]*CNN.predict(xTe)
      CNN.close()
      
#    saver.restore(sess, "/tmp/model.ckpt")
#    print("Model restored.")
    
    with open('submission.csv', 'w', newline='') as csvfile:
      csv_w = csv.writer(csvfile, delimiter=',')
      csv_w.writerow(('id', 'digit'))
      for i in range(len(preds)):
        csv_w.writerow((i, preds[i]))
      print('Wrote submission file!')
    
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
  
  
  
if __name__ == "__main__":
  tf.app.run()