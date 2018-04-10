---
layout:     post
title:      "TensorFlow 数据集MNIST简介"
subtitle:   "TensorFlow简单使用（3/7）"
date:       2018-01-05
author:     "Zero"
#cover: "/assets/in-post/tensorflow-bg.jpg"
categories: technology
tags: TensorFlow
---

### TensorFlow数据集MNIST简介

---

#### 1、查看数据的结构


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
%matplotlib inline

# 引入数据
mnist = input_data.read_data_sets("data", one_hot=True)

# 查看数据类型
print (type(mnist))
print (type(mnist.train))
print (type(mnist.train.images))

# 查看数据量
print (mnist.train.num_examples)
print (mnist.test.num_examples)
print (mnist.validation.num_examples)

# 查看数据shape：训练集、测试集、验证集
print (mnist.train.images.shape, mnist.train.labels.shape)
print (mnist.test.images.shape, mnist.test.labels.shape)
print (mnist.validation.images.shape, mnist.validation.labels.shape)
```

    Extracting data\train-images-idx3-ubyte.gz
    Extracting data\train-labels-idx1-ubyte.gz
    Extracting data\t10k-images-idx3-ubyte.gz
    Extracting data\t10k-labels-idx1-ubyte.gz
    <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
    <class 'tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet'>
    <class 'numpy.ndarray'>
    55000
    10000
    5000
    (55000, 784) (55000, 10)
    (10000, 784) (10000, 10)
    (5000, 784) (5000, 10)

---

#### 2、展示数据图像


```python
# 在训练集中随机取出五条数据的索引
index = np.random.randint(mnist.train.images.shape[0], size=5)
trainimg = mnist.train.images
trainlabel = mnist.train.labels

for i in index:
    img = np.reshape(trainimg[i, :], (28, 28))
    label = np.argmax(trainlabel[i, :])
    plt.matshow(img, cmap=plt.get_cmap('gray'))
    plt.title('index:' + str(i) + ',label:' + str(label))
```


![png](/assets/in-post/tensorflow3/output_3_0.png)



![png](/assets/in-post/tensorflow3/output_3_1.png)



![png](/assets/in-post/tensorflow3/output_3_2.png)



![png](/assets/in-post/tensorflow3/output_3_3.png)



![png](/assets/in-post/tensorflow3/output_3_4.png)

---

#### 3、取出这些数据用于训练


```python
# 训练神经网络时，一个batch一个batch进行
batch_size = 100
batch_xs, batch_ys = mnist.train.next_batch(batch_size)
print (type(batch_xs))
print (type(batch_ys))
print (batch_xs.shape)
print (batch_ys.shape)
```

    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>
    (100, 784)
    (100, 10)
