---
layout:     post
title:      "TensorFlow入门-2-线性回归实例"
subtitle:   "TensorFlow学习笔记（2/7）"
date:       2018-01-05
author:     "Zero"
#cover: "/assets/in-post/seaborn1/bg.jpg"
categories: technology
tags: TensorFlow
---

### TensorFlow线性回归实例

---

#### 1、实例概述
- 任意指定 a, b 初始化直线 y = ax + b 周围的一些点
- 对这些点进行线性回归机器学习
- 查看学习后的参数，是否接近指定的 a, b

---

#### 2、步骤
- 构造误差为正态分布的数据样本点
- 声明预估值模型 y = Wx + b
- 最小二乘法构造loss函数，即误差
- 初始化梯度下降优化器
- 使用优化器最小化误差
- 在session中进行训练


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
# 构造数据:1000个点，围绕在 y = 0.1x + 0.3 周围
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

# 生成一些样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

sns.set_style('whitegrid')
plt.scatter(x_data, y_data, c='r', marker='o')
```




    <matplotlib.collections.PathCollection at 0x1afde7d41d0>




![png](/assets/in-post/tensorflow2/output_2_1.png)



```python
# 生成参数W，取值为[-1, 1]间随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 生成偏移量b
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得出预估值y 这里没有用矩阵乘法，是因为都是一维的
y = W * x_data + b

# 预估值y与实际值y_data均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 初始化梯度下降法优化器，用于优化参数，0.5表示学习率，一般设置较小0.001
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练过程就是使用优化器，最小化误差值
train = optimizer.minimize(loss, name='train')

# 初始化变量并打印
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ('W=', W.eval(), "b=", b.eval(), 'loss=', loss.eval())

    # 执行20次训练
    for step in range(20):
        sess.run(train)
        print ('W=', W.eval(), "b=", b.eval(), 'loss=', loss.eval())

    # 展示拟合后的直线图
    plt.scatter(x_data,y_data,c='r')
    plt.plot(x_data, sess.run(W)*x_data+sess.run(b))
```

    W= [ 0.13379335] b= [ 0.] loss= 0.0908692
    W= [ 0.11754344] b= [ 0.29955] loss= 0.000925067
    W= [ 0.11329054] b= [ 0.29915193] loss= 0.000893839
    W= [ 0.1102476] b= [ 0.29904777] loss= 0.000877949
    W= [ 0.10807481] b= [ 0.29897323] loss= 0.000869847
    W= [ 0.10652333] b= [ 0.29892001] loss= 0.000865717
    W= [ 0.10541551] b= [ 0.29888201] loss= 0.000863611
    W= [ 0.10462447] b= [ 0.29885489] loss= 0.000862537
    W= [ 0.10405964] b= [ 0.29883552] loss= 0.00086199
    W= [ 0.10365632] b= [ 0.29882169] loss= 0.00086171
    W= [ 0.10336833] b= [ 0.29881179] loss= 0.000861568
    W= [ 0.1031627] b= [ 0.29880473] loss= 0.000861495
    W= [ 0.10301586] b= [ 0.29879969] loss= 0.000861459
    W= [ 0.10291102] b= [ 0.29879612] loss= 0.00086144
    W= [ 0.10283615] b= [ 0.29879355] loss= 0.00086143
    W= [ 0.1027827] b= [ 0.29879171] loss= 0.000861425
    W= [ 0.10274453] b= [ 0.2987904] loss= 0.000861423
    W= [ 0.10271727] b= [ 0.29878947] loss= 0.000861421
    W= [ 0.10269781] b= [ 0.29878879] loss= 0.000861421
    W= [ 0.10268392] b= [ 0.29878831] loss= 0.00086142
    W= [ 0.10267399] b= [ 0.29878798] loss= 0.00086142



![png](/assets/in-post/tensorflow2/output_3_1.png)
