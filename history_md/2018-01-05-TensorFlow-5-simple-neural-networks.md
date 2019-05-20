---
layout:     post
title:      "TensorFlow 简单神经网络架构模型"
subtitle:   "TensorFlow简单使用（5/7）"
date:       2018-01-05
author:     "Zero"
#cover: "/assets/in-post/tensorflow-bg.jpg"
categories: technology
tags: TensorFlow
---

### TensorFlow简单神经网络架构模型


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
trainlab = mnist.train.labels
testimg = mnist.test.images
testlab = mnist.test.labels
print ('DATA READY')
```

    Extracting data\train-images-idx3-ubyte.gz
    Extracting data\train-labels-idx1-ubyte.gz
    Extracting data\t10k-images-idx3-ubyte.gz
    Extracting data\t10k-labels-idx1-ubyte.gz
    DATA READY



```python
# 搭建神经网络架构
n_hidden_1 = 256 # 第一层神经网神经元个数
n_hidden_2 = 128 # 第二层神经网神经元个数
n_input = 784 # 输入层像素点个数
n_output = 10 # 输出层分类个数

# 创建占位符，之后一个batch一个batch往里传入
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_output])

# 初始化权重参数
# 指定标准差
stddev = 0.1
weights = {
    # 随机高斯初始化，指定维度为784*256，标准差
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output], stddev=stddev)),
}
# 初始化偏置项参数
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output])),
}
print ('NETWORK READY')
```

    NETWORK READY



```python
# 定义神经网络操作函数
def multilayer_perceptron(_X, _weights, _biases):
    # 第一层运算，y = Wx + b, 注意完成后要使用激活函数激活，这里使用sigmoid，一般用ReLU
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    # 第二层运算
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    # out层不需要sigmoid
    return tf.add(tf.matmul(layer_2, _weights['out']), _biases['out'])

# 定义计算图的一个op，计算模型预测值
pred = multilayer_perceptron(x, weights, biases)
# 使用tensorflow实现好的损失函数
# 如这里使用的：softmax_cross_entropy_with_logits交叉熵函数,输入预测值与实际值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
# 使用梯度下降优化器最小化误差，指定学习率为0.001
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
# 定义参数用于对比是否需预测正确
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 定义正确率，为corr的均值
accr = tf.reduce_mean(tf.cast(corr, 'float'))
# 定义初始化操作
init = tf.global_variables_initializer()

print ('FUNCTIONS READY')
```

    FUNCTIONS READY



```python
# 1个epoch表示所有数据跑一遍
training_epochs = 100
# 每个batch多少数据
batch_size = 100
# 每4个epoch
display_step = 4

with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())

    # 迭代每个epoch
    for epoch in range(training_epochs):
        # 平均损失值
        avg_cost = 0.
        # 总batch个数：数据总数比上每个batch的数据数量
        total_batch = int(trainimg.shape[0] / batch_size)
        # 迭代每个batch
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y:batch_ys}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost = avg_cost / total_batch
        # 展示
        if (epoch + 1) % display_step == 0:
            print ('Epoch:%03d/%03d cost:%.9f' % (epoch, training_epochs, avg_cost))
            # 训练集正确率
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
            # 测试集正确率
            test_acc = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print ('train accuracy : ', train_acc, ',test accuracy:', test_acc)
    print ('FINISHED')
```

    Epoch:003/100 cost:2.278202408
    train accuracy :  0.17 ,test accuracy: 0.1619
    Epoch:007/100 cost:2.242510963
    train accuracy :  0.33 ,test accuracy: 0.2933
    Epoch:011/100 cost:2.203844197
    train accuracy :  0.35 ,test accuracy: 0.4195
    Epoch:015/100 cost:2.159955178
    train accuracy :  0.55 ,test accuracy: 0.5192
    Epoch:019/100 cost:2.108661973
    train accuracy :  0.62 ,test accuracy: 0.5768
    Epoch:023/100 cost:2.047843454
    train accuracy :  0.68 ,test accuracy: 0.611
    Epoch:027/100 cost:1.975841208
    train accuracy :  0.61 ,test accuracy: 0.6442
    Epoch:031/100 cost:1.891604822
    train accuracy :  0.64 ,test accuracy: 0.6563
    Epoch:035/100 cost:1.795694553
    train accuracy :  0.67 ,test accuracy: 0.6725
    Epoch:039/100 cost:1.690594893
    train accuracy :  0.7 ,test accuracy: 0.6874
    Epoch:043/100 cost:1.580412333
    train accuracy :  0.69 ,test accuracy: 0.7102
    Epoch:047/100 cost:1.470284497
    train accuracy :  0.74 ,test accuracy: 0.7214
    Epoch:051/100 cost:1.364862667
    train accuracy :  0.76 ,test accuracy: 0.7406
    Epoch:055/100 cost:1.267509181
    train accuracy :  0.74 ,test accuracy: 0.7561
    Epoch:059/100 cost:1.179900314
    train accuracy :  0.74 ,test accuracy: 0.7701
    Epoch:063/100 cost:1.102300486
    train accuracy :  0.79 ,test accuracy: 0.7812
    Epoch:067/100 cost:1.034170843
    train accuracy :  0.8 ,test accuracy: 0.7918
    Epoch:071/100 cost:0.974460358
    train accuracy :  0.81 ,test accuracy: 0.799
    Epoch:075/100 cost:0.922065820
    train accuracy :  0.83 ,test accuracy: 0.8076
    Epoch:079/100 cost:0.875892122
    train accuracy :  0.87 ,test accuracy: 0.8132
    Epoch:083/100 cost:0.835029942
    train accuracy :  0.86 ,test accuracy: 0.8202
    Epoch:087/100 cost:0.798700458
    train accuracy :  0.84 ,test accuracy: 0.8269
    Epoch:091/100 cost:0.766269952
    train accuracy :  0.86 ,test accuracy: 0.8329
    Epoch:095/100 cost:0.737154432
    train accuracy :  0.82 ,test accuracy: 0.836
    Epoch:099/100 cost:0.710901295
    train accuracy :  0.82 ,test accuracy: 0.8403
    FINISHED
