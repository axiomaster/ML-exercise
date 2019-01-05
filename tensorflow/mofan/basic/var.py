# -*- coding: utf-8 -*-

import tensorflow as tf

state = tf.Variable(0, name="counter")
one = tf.constant(1)

new_val = tf.add(state, one)
update = tf.assign(state, new_val)

# 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print(sess.run(update))
        print(sess.run(state))