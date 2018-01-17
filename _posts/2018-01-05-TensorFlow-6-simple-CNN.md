---
layout:     post
title:      "TensorFlow入门-6-卷积神经网络架构模型"
subtitle:   "TensorFlow学习笔记（6/7）"
date:       2018-01-05
author:     "Zero"
#cover: "/assets/in-post/seaborn1/bg.jpg"
categories: technology
tags: TensorFlow
---

### TensorFlow卷积神经网络架构模型


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
```


```python
# 准备数据
mnist = input_data.read_data_sets("data", one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print ('DATA READY')
```

    Extracting data\train-images-idx3-ubyte.gz
    Extracting data\train-labels-idx1-ubyte.gz
    Extracting data\t10k-images-idx3-ubyte.gz
    Extracting data\t10k-labels-idx1-ubyte.gz
    DATA READY



```python
# 搭建神经网络架构
n_input = 784 # 输入层像素点个数
n_output = 10 # 输出层分类个数

# 创建占位符，之后一个batch一个batch往里传入
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
# dropout比例，保留的比例
keepratio = tf.placeholder(tf.float32)

# 初始化权重参数
# 指定标准差
stddev = 0.1
weights = {
    # 卷积层1 输入数据28*28*1深度为1，filter为3*3*1，得到64个特征图
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
    # 卷积层2 输入数据深度为64，filter为3*3*64，得到128个特征图
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
    # 全连接层1
    'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=stddev)),
    # 全连接层2
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=stddev))
}

# 初始化偏置项参数
biases = {
    # 卷积层1 64个特征图，64个偏置项
    'bc1': tf.Variable(tf.random_normal([64], stddev=stddev)),
    # 卷积层2 64个特征图，128个偏置项
    'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
    # 全连接层1：1024个神经元，1024个偏置项
    'bd1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
    # 全连接层2：10个神经元，10个偏置项
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=stddev))
}

print ('NETWORK READY')
```

    NETWORK READY



```python
# 卷积加池化操作函数
def conv_basic(_input, _w, _b, _keepratio):
    # 将输入转化为tensorflow的格式 [n, h, w, c]，-1表示让tensorflow自己推断
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # 卷积层1 ,卷积完成后使用relu函数激活
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # 池化层1
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout随机杀死一些节点
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # 卷积层2 ,卷积完成后使用relu函数激活
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    # 池化层2
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout随机杀死一些节点
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # 全连接层1
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # 输出层
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool_dr1': _pool_dr1,
          'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
          'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out}

    return out

print ('CNN READY')
```

    CNN READY



```python
# 计算图op定义
pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

# 模型保存参数
# 每隔几个epoch进行一次保存
save_step = 1
# 指定max_to_keep表示只保留最后三组模型
saver = tf.train.Saver(max_to_keep=3)

print ('FUNCTION READY')
```

    FUNCTION READY



```python
# 是否训练 1-使用数据训练，0-读取模型进行预测
do_train = 1
# 1个epoch表示所有数据跑一遍
training_epochs = 10
# 每个batch多少数据
batch_size = 16
# 每4个epoch
display_step = 1

with tf.Session() as sess:
    if do_train == 1:
        # 初始化
        sess.run(tf.global_variables_initializer())

        # 迭代每个epoch
        for epoch in range(training_epochs):
            # 平均损失值
            avg_cost = 0.
            # 总batch个数：数据总数比上每个batch的数据数量
            # total_batch = int(trainimg.shape[0] / batch_size)
            # 这里为了跑的快一些示范直接定义为一个较小的值
            total_batch = 10
            # 迭代每个batch
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.7})
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch
            avg_cost = avg_cost / total_batch
            # 展示
            if epoch % display_step == 0:
                print ('Epoch:%03d/%03d cost:%.9f' % (epoch, training_epochs, avg_cost))
                # 训练集正确率
                train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
                # 测试集正确率
                test_acc = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels, keepratio:1.})
                print ('train accuracy : ', train_acc, ',test accuracy:', test_acc)
            # 模型保存
            if epoch % save_step == 0:
                saver.save(sess, 'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))
        print ('FINISHED')

    if do_train == 0:
        epoch = training_epochs-1
        saver.restore(sess, 'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))
        ttest_acc = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels, keepratio:1.})
        print ('train accuracy : ', train_acc, ',test accuracy:', test_acc)
```

    Epoch:000/010 cost:0.733549533
    train accuracy :  0.0625 ,test accuracy: 0.1
    Epoch:001/010 cost:0.256137819
    train accuracy :  0.3125 ,test accuracy: 0.384
    Epoch:002/010 cost:0.165178217
    train accuracy :  0.625 ,test accuracy: 0.4692
    Epoch:003/010 cost:0.153198793
    train accuracy :  0.4375 ,test accuracy: 0.3948
    Epoch:004/010 cost:0.152241067
    train accuracy :  0.75 ,test accuracy: 0.6468
    Epoch:005/010 cost:0.134036309
    train accuracy :  0.9375 ,test accuracy: 0.7452
    Epoch:006/010 cost:0.115600968
    train accuracy :  0.8125 ,test accuracy: 0.7744
    Epoch:007/010 cost:0.079777042
    train accuracy :  0.5625 ,test accuracy: 0.8066
    Epoch:008/010 cost:0.086858889
    train accuracy :  0.9375 ,test accuracy: 0.844
    Epoch:009/010 cost:0.067108465
    train accuracy :  0.8125 ,test accuracy: 0.8487
    FINISHED



```python
# 是否训练 1-使用数据训练，0-读取模型进行预测
do_train = 0
# 1个epoch表示所有数据跑一遍
training_epochs = 10
# 每个batch多少数据
batch_size = 16
# 每4个epoch
display_step = 1

with tf.Session() as sess:
    if do_train == 1:
        # 初始化
        sess.run(tf.global_variables_initializer())

        # 迭代每个epoch
        for epoch in range(training_epochs):
            # 平均损失值
            avg_cost = 0.
            # 总batch个数：数据总数比上每个batch的数据数量
            # total_batch = int(trainimg.shape[0] / batch_size)
            # 这里为了跑的快一些示范直接定义为一个较小的值
            total_batch = 10
            # 迭代每个batch
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.7})
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch
            avg_cost = avg_cost / total_batch
            # 展示
            if epoch % display_step == 0:
                print ('Epoch:%03d/%03d cost:%.9f' % (epoch, training_epochs, avg_cost))
                # 训练集正确率
                train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
                # 测试集正确率
                test_acc = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels, keepratio:1.})
                print ('train accuracy : ', train_acc, ',test accuracy:', test_acc)
            # 模型保存
            if epoch % save_step == 0:
                saver.save(sess, 'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))
        print ('FINISHED')

    if do_train == 0:
        epoch = training_epochs-1
        saver.restore(sess, 'save/nets/cnn_mnist_basic.ckpt-' + str(epoch))
        ttest_acc = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels, keepratio:1.})
        print ('train accuracy : ', train_acc, ',test accuracy:', test_acc)
```

    INFO:tensorflow:Restoring parameters from save/nets/cnn_mnist_basic.ckpt-9
    train accuracy :  0.8125 ,test accuracy: 0.8487
