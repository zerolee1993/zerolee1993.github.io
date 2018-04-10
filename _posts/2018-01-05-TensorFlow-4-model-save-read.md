---
layout:     post
title:      "TensorFlow 模型保存和读取"
subtitle:   "TensorFlow简单使用（4/7）"
date:       2018-01-05
author:     "Zero"
#cover: "/assets/in-post/tensorflow-bg.jpg"
categories: technology
tags: TensorFlow
---

### TensorFlow模型保存和读取

---

#### 1、模型的保存
- 使用tf.train.Saver()声明saver对象
- 计算图执行后，使用saver.save(sess, path)保存模型


```python
import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1, 2]), name='v1')
v2 = tf.Variable(tf.random_normal([2, 3]), name='v2')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_path = saver.save(sess, 'save/model.ckpt')
    print ('保存路径：', saver_path)
    print ('v1：', sess.run(v1))
    print ('v2：', sess.run(v2))
```

    保存路径： save/model.ckpt
    v1： [[ 0.10252484  1.11316967]]
    v2： [[ 1.1336571  -0.14345047 -0.60618758]
     [ 1.3016479  -1.35617161  0.70471817]]

---

#### 2、模型的读取
- 使用tf.train.Saver()声明saver对象
- 计算图执行后，使用saver.restore(sess, path)读取模型
- 注意，执行完第一部分代码，restart后，在执行第二部分代码，否则会报错


```python
import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1, 2]), name='v1')
v2 = tf.Variable(tf.random_normal([2, 3]), name='v2')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'save/model.ckpt')
    print ('v1：', sess.run(v1))
    print ('v2：', sess.run(v2))
```

    INFO:tensorflow:Restoring parameters from save/model.ckpt
    v1： [[ 0.10252484  1.11316967]]
    v2： [[ 1.1336571  -0.14345047 -0.60618758]
     [ 1.3016479  -1.35617161  0.70471817]]
