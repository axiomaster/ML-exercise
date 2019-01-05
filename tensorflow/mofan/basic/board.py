# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

    # 导入数据
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

# 输入层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 隐藏层
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# tf.initialize_all_variables() no long valid from
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    # train
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    #
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        writer = tf.summary.FileWriter("logs/", sess.graph)
