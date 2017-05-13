# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:43:03 2017

@author: awtho
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
import csv
import random
from scipy.ndimage.interpolation import rotate

class myCNN:
  
  def __init__(self):
    self.sess = tf.Session()
    
    self.x = tf.placeholder(tf.float32, shape=[None, 784])
    self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    x_image = tf.reshape(self.x, [-1,28,28,1])
    
    #CONVOLUTIONAL 1
    W_conv1 = myCNN._weight_variable([5, 5, 1, 32])
    b_conv1 = myCNN._bias_variable([32])
    
    h_conv1 = tf.nn.relu(myCNN._conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = myCNN._max_pool_2x2(h_conv1)
    
    #CONVOLUTIONAL 2
    W_conv2 = myCNN._weight_variable([5, 5, 32, 64])
    b_conv2 = myCNN._bias_variable([64])
    
    h_conv2 = tf.nn.relu(myCNN._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = myCNN._max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    
    #DENSE 1, POOL, AND ACTIVATION
    W_fc1 = myCNN._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = myCNN._bias_variable([1024])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #DENSE 2, POOL, AND ACTIVATION
    W_fc2 = myCNN._weight_variable([1024, 512])
    b_fc2 = myCNN._bias_variable([512])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    #DROPOUT
    self.keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
    
    #ONEHOT OUT
    W_fc3 = myCNN._weight_variable([512, 10])
    b_fc3 = myCNN._bias_variable([10])
    
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))
    
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    self.predictions = tf.argmax(y_conv, 1)
    
    self.sess.run(tf.global_variables_initializer())
    
  def train(self, xTr, yTr, num_epochs, batch_size):
    n,d = xTr.shape
    print("Training network.")
    for i in range(1,num_epochs+1):
      start = 0
      end = batch_size
      epoch_indices = np.random.choice(n, n, replace=False)
      t0 = time.time()
      while start <= n:#for i in range(1,NUM_EPOCHS+1):
        batch_indices = epoch_indices[start:end]#np.random.choice(n,BATCH_SIZE,replace=False)#
        batch_x = self._random_rotation(xTr[batch_indices], 30)
        batch_y = yTr[batch_indices]
        self.train_step.run(feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 0.7}, session=self.sess)
        start += batch_size
        end += batch_size
      
      if i%1 == 0:
        t1 = time.time()
        train_accuracy = self.accuracy.eval(feed_dict={self.x:xTr, self.y_: yTr, self.keep_prob: 1.0}, session=self.sess)
        print("Epoch %d (%%%2.2f), training accuracy %%%g (%gs)"%(i, float(i)/num_epochs*100, train_accuracy*100, t1-t0))
    
    print("Finished training, training accuracy %%%g."%(100*self.accuracy.eval(feed_dict={self.x:xTr, self.y_: yTr, self.keep_prob: 1.0}, session=self.sess)))
          
  def predict(self, xTe):
    return self.predictions.eval(feed_dict={self.x: xTe, self.keep_prob: 1.0},session=self.sess)
  
  def close(self):
    self.sess.close()
    
  def save(self):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('experiments', self.sess.graph)
    save_path = saver.save(self.sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    writer.flush()
    writer.close()  
    
  @staticmethod
  def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
  @staticmethod
  def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
  @staticmethod
  def _max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    
    
  def _random_rotation(self, batch, max_angle):
    batch = batch.reshape([-1, 28, 28])
    for i in range(len(batch)):
      if bool(random.getrandbits(1)):
        # Random angle
        angle = random.uniform(-max_angle, max_angle)
        batch[i] = rotate(batch[i], angle, reshape=False)
    batch = batch.reshape([-1, 784])
    return batch