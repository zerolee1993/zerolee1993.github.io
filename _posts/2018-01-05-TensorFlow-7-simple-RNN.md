---
layout:     post
title:      "TensorFlow入门-(7)循环神经网络架构模型"
subtitle:   "TensorFlow学习笔记（7/7）"
date:       2018-01-05
author:     "Zero"
#cover: "/assets/in-post/seaborn1/bg.jpg"
categories: technology
tags: TensorFlow
---

### Tensorflow循环神经网络RNN模型


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
# 训练图片，训练标签，测试图片，测试标签
trainimgs, trainlabels, testimgs, testlabels \
 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# 训练集数量，测试集数量，训练集图像像素点个数，标签类别个数
ntrain, ntest, dim, nclasses \
 = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]

print ('MNIST DATA READY')
```

    Extracting data/train-images-idx3-ubyte.gz
    Extracting data/train-labels-idx1-ubyte.gz
    Extracting data/t10k-images-idx3-ubyte.gz
    Extracting data/t10k-labels-idx1-ubyte.gz
    MNIST DATA READY



```python
# 输入数据为28*28的，这里指定_X层输入为28，需要将单个数据预处理为28个序列数据
diminput = 28
# _H隐层，128个神经元
dimhidden = 128
# 最终分类个数
dimoutput = nclasses
# 将单个数据预处理分成为28个序列数据，分28步走完
nsteps = 28

# 创建占位符，之后一个batch一个batch往里传入
x = tf.placeholder(tf.float32, [None, nsteps, diminput])
y = tf.placeholder(tf.float32, [None, dimoutput])

# 初始化权重参数
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}

# 初始化偏置项参数
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}
print ('NETWORK READY')
```

    NETWORK READY



```python
# 神经网络框架函数搭建
def _RNN(_X, _W, _b, _nsteps, _name):
    # 转换输入的维度位置:[batchsize, nsteps, diminput]==>[nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # reshape
    _X = tf.reshape(_X, [-1, diminput])
    # 计算_H层
    _H = tf.matmul(_X, _W['hidden']) + _b['hidden']
    # 将整体运算的结果切分成序列
    _Hsplit = tf.split(_H, _nsteps, 0)
    # RNN
    with tf.variable_scope(_name, reuse=tf.AUTO_REUSE):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_0, _LSTM_S = tf.contrib.rnn.static_rnn(lstm_cell, _Hsplit, dtype=tf.float32)
    # 输出
    _O = tf.matmul(_LSTM_0[-1], _W['out']) + _b['out']
    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit, 'LSTM_O': _LSTM_0,
        'LSTM_S': _LSTM_S, 'O': _O
    }
print ('RNN READY')
```

    RNN READY



```python
# 计算图op定义
pred = _RNN(x, weights, biases, nsteps, 'basic')['O']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

print ('FUNCTION READY')
```

    FUNCTION READY



```python
training_epochs = 5
batch_size = 16
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
#         total_batch = 100
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
            sess.run(optm, feed_dict={x: batch_xs, y:batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y:batch_ys})/total_batch
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch+1, training_epochs, avg_cost))
            feeds = {x: trainimgs.reshape((ntrain, nsteps, diminput)), y: trainlabels}
            train_acc = sess.run(accr, feed_dict=feeds)
            print (" Training accuracy: %.3f" % (train_acc))
            feeds = {x: testimgs.reshape((ntest, nsteps, diminput)), y: testlabels}
            test_acc = sess.run(accr, feed_dict=feeds)
            print (" Test accuracy: %.3f" % (test_acc))

print ("FINISHED")
```

    Epoch: 001/005 cost: 0.573300335
     Training accuracy: 0.821
     Test accuracy: 0.821
    Epoch: 002/005 cost: 0.215184872
     Training accuracy: 0.881
     Test accuracy: 0.885
    Epoch: 003/005 cost: 0.147054715
     Training accuracy: 0.913
     Test accuracy: 0.913
    Epoch: 004/005 cost: 0.109967285
     Training accuracy: 0.910
     Test accuracy: 0.905
    Epoch: 005/005 cost: 0.087234279
     Training accuracy: 0.930
     Test accuracy: 0.929
    FINISHED
