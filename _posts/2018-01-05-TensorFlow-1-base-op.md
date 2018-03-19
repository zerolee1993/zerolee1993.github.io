---
layout:     post
title:      "TensorFlow入门-1-基本操作"
subtitle:   "TensorFlow学习笔记（1/7）"
date:       2018-01-05
author:     "Zero"
cover: "/assets/in-post/tensorflow-bg.jpg"
categories: technology
tags: TensorFlow
---

### TensorFlow中的基本操作

---

### 目录

* 纲要
{:toc}

---

#### 1、使用变量


```python
# 完成两个变量的相乘操作
import tensorflow as tf
import numpy as np

# 定义tensor类型变量，及乘积操作
w = tf.Variable([[1, 2]])
x = tf.Variable([[2], [3]])
y = tf.matmul(w, x)

# 此时只是定义好了w,x,y的tensor类型的模型/框架，w,x,y并没有值
print (w)
print (x)
print (y)

# 定义一个初始化操作给w,x,y赋值
init_op = tf.global_variables_initializer()

# 需要把之前的东西添加到计算图中在session中执行
# session表示一次会话，可理解为计算区域
with tf.Session() as sess:
    sess.run(init_op)
    # 执行完初始化后，可以查看y的值了
    # 需要注意，y是tensor类型的数据，需要使用eval函数查看值
    print (y.eval())
```

    <tf.Variable 'Variable_8:0' shape=(1, 2) dtype=int32_ref>
    <tf.Variable 'Variable_9:0' shape=(2, 1) dtype=int32_ref>
    Tensor("MatMul_3:0", shape=(1, 1), dtype=int32)
    [[8]]

---

#### 2、创建tensor类型的数据
- 类似numpy中的数据创建，zeros/ones/random等


```python
# 数据的创建
# 习惯使用numpy的，可以使用numpy语法初始化数据，再转换成tensor（并不推荐）
npdata = np.zeros((4, 4))
tfdata = tf.convert_to_tensor(npdata)

# 指定shape创建一个值均为0的矩阵
a = tf.zeros([3, 4], tf.int32)
# 指定shaoe创建一个值均为1的矩阵
b = tf.ones([2, 2], tf.int32)
# 按照a的shape，创建一个值均为1的矩阵
c = tf.ones_like(a)
# 按照b的shape，创建一个值均为0的矩阵
d = tf.zeros_like(b)

# 定义tensor的一维常量
cons1 = tf.constant([1, 2, 3])
# 定义tensor的二维常量
cons2 = tf.constant(-1.0, shape=[2, 2])

# 定义linspace，在先x,y范围内平均取z个数
lins = tf.linspace(1.0, 3.0, 3, name='linspace')

# 定义一维列表，数据在x,y范围内，每隔z取一个数值，与numpu的arange相同
rang = tf.range(0, 10, 2)

# 指定shape创建一个随机矩阵，满足高斯分布，期望-1，方差4
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# 将原始数据进行洗牌(横向顺序不变，随机调换行位置)
ori = tf.constant([[1, 2, 3], [6, 4, 9]])
shuf = tf.random_shuffle(ori)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print (tfdata.eval())
    print (a.eval())
    print (b.eval())
    print (c.eval())
    print (d.eval())
    print (cons1.eval())
    print (cons2.eval())
    print (lins.eval())
    print (rang.eval())
    print (norm.eval())
    print (shuf.eval())

```

    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]
    [[1 1]
     [1 1]]
    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]
    [[0 0]
     [0 0]]
    [1 2 3]
    [[-1. -1.]
     [-1. -1.]]
    [ 1.  2.  3.]
    [0 2 4 6 8]
    [[-2.26341629 -0.19280821  5.69746351]
     [ 7.89927769  2.62305665 -2.54620504]]
    [[1 2 3]
     [6 4 9]]

---

#### 3、实现一个变量自加的例子


```python
# 创建一个初值为0的变量
state = tf.Variable(0)
# 创建一个相加操作：当前state再加1
new_value = tf.add(state, tf.constant(1))
# 创建一个赋值操作将new_value的返回值赋值给state
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (state.eval())
    for _ in range(3):
        sess.run(update)
        print(state.eval())
```

    0
    1
    2
    3

---

#### 4、占位符的使用
- 使用placeholder占位
- 使用它feed_dict在调用时传入数据


```python
# placeholder,现在session计算图中占个位置，使用时再赋值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print (sess.run([output], feed_dict={input1:[2.], input2:[3.]}))
```

    [array([ 6.], dtype=float32)]
