from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from mnist_softmax import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# Inputs, y_ is the computed output, not the actual output
x = tf.placeholder(tf.float32, shape=[None,784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

# Model variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

# Model and loss function
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Training node, we will use gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Training loop, recall that next_batch() returns a 2x1 vector
for _ in range(1000):
   batch = mnist.train.next_batch(100)
   train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate accuracy of trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# above returns booleans, we must cast directly to FP-type and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Next, we will go deeper

# Weight initialization functions
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_variable(shape)
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

# Convolution and pooling functions
def conv2d(x, W):
   return tf.nn.conv2s(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# When you come back start working on understanding and implementing
# the convolutions and poolins
