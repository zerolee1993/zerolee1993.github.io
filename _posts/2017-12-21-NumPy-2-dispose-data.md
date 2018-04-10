---
layout:     post
title:      "NumPy 处理数据"
subtitle:   "Python科学计算库NumPy简单使用（2/2）"
date:       2017-12-21
author:     "Zero"
#cover: "/assets/in-post/python1/bg.jpg"
categories: technology
tags: NumPy
---

### 简介

在学会使用Numpy构造数据后，我们将要对adarray对象进行必要的处理
其中包含数学运算以及数据结构的调整
这里简述了常用的矩阵处理方法，并提供了可执行的代码示例（Python3版本）

---

### 目录

* 纲要
{:toc}

---

#### 1、逻辑运算
- 使用==判断ndarray中是否包含某值，返回值是元素为boolean类型的ndarray
- 返回的值可以充当索引，进行取值
- 返回值可以进行与和或的运算


```python
import numpy as np
vector = np.array([1,2,3,4])
result = (vector == 3)
print (type(result))
print (result)
print (vector[result])
result = (vector == 1) & (vector ==2)
print (result)
result = (vector ==1) | (vector == 2)
print (result)
```

    <class 'numpy.ndarray'>
    [False False  True False]
    [3]
    [False False False False]
    [ True  True False False]

---

#### 2、基本算数运算
- 列举符号运算
- dot函数使用矩阵乘法


```python
# 矩阵和int值的运算
data = np.array([[1,0],
              [3,4]])
print (data)
# 所有数据-1
print (data-1)
# 所有数据+1
print (data+1)
# 所有数据乘方
print (data**2)
# 所有数据判断
print (data<2)
```

    [[1 0]
     [3 4]]
    [[ 0 -1]
     [ 2  3]]
    [[2 1]
     [4 5]]
    [[ 1  0]
     [ 9 16]]
    [[ True  True]
     [False False]]



```python
# 矩阵和矩阵的运算
data1 = np.array([[0,0],
              [1,1]])
data2 = np.array([[1,2],
              [3,4]])
print (data1)
print (data2)
# 对应位置相加
print (data1 + data2)
# 对应位置相乘
print (data1 * data2)
# 矩阵乘法的两种用法
print (data1.dot(data2))
print (np.dot(data1, data2))
```

    [[0 0]
     [1 1]]
    [[1 2]
     [3 4]]
    [[1 2]
     [4 5]]
    [[0 0]
     [3 4]]
    [[0 0]
     [4 6]]
    [[0 0]
     [4 6]]

---

#### 3、求极值
- 使用max/min函数求最大/小值
- 不传入参数返回数据集的最大/最小值，dtype即为数据的类型
- 可以传入axis=0/1可选参数，按列/行取最大/小值，返回值仍为ndarray


```python
vector = np.array([1,2,3])
print (vector.min())
matrix = np.array([[2,3],
                   [1,9]])
print (matrix.min())
print (matrix.min(axis=1))
print (matrix.max(axis=0))
```

    1
    1
    [2 1]
    [2 9]

---

#### 4、数值求和
- 可以使用sum函数进行求和
- 不传入参数返回数据集求和值，dtype即为数据的类型
- 可以传入axis=0/1可选参数，按列/行求和，返回值仍为ndarray


```python
matrix = np.array([[1, 2],
                   [3, 4],])
print (matrix.sum())
print (matrix.sum(axis=1))
print (matrix.sum(axis=0))
```

    10
    [3 7]
    [4 6]

---

#### 5、求逆矩阵
- 调用对象的T属性即可


```python
import numpy as np
data = np.arange(4).reshape(2,-1)
print (data)
print (data.T)
```

    [[0 1]
     [2 3]]
    [[0 2]
     [1 3]]

---

#### 6、计算开方
- 使用sqrt函数可以计算数值的开方值
- 可以传入ndarray，返回同结构的ndarray，为每一个数据求开放


```python
print (np.sqrt(4))
data = np.arange(5)
print (data)
print (np.sqrt(data))
```

    2.0
    [0 1 2 3 4]
    [ 0.          1.          1.41421356  1.73205081  2.        ]

---

#### 7、取整操作
- 使用floor函数进行向下取整操作，即取出小数部分
- 可传入ndarray，返回同结构的ndarray，为每一个数据向下取整


```python
data = 10*np.random.random_sample((2,2))
print (data)
data = np.floor(data)
print (data)
print (np.floor(2.6))
```

    [[ 4.94558713  7.37492681]
     [ 7.92742655  9.89601688]]
    [[ 4.  7.]
     [ 7.  9.]]
    2.0

---

#### 8、exp指数计算
- 使用exp函数可以计算e的n次幂
- 可传入ndarray，返回同结构的ndarray，为每一个数据求exp指数计算值


```python
data = np.arange(3)
print (np.exp(1))
print (data)
print (np.exp(data))
```

    2.71828182846
    [0 1 2]
    [ 1.          2.71828183  7.3890561 ]

---

#### 9、矩阵的拼接和切分
- 使用hstack/vstack进行横向/纵向矩阵拼接操作
- 使用hsplit/vsplit进行横向/纵向矩阵切分操作
    - 传入int平均等分
    - 传入tuble，指定位置切分


```python
# 矩阵拼接
data1 = np.floor(10*np.random.random_sample((2,2)))
data2 = np.floor(10*np.random.random_sample((2,2)))
print (data1)
print (data2)
print (np.hstack((data1, data2)))
print (np.vstack((data1, data2)))
```

    [[ 5.  6.]
     [ 6.  3.]]
    [[ 1.  0.]
     [ 0.  7.]]
    [[ 5.  6.  1.  0.]
     [ 6.  3.  0.  7.]]
    [[ 5.  6.]
     [ 6.  3.]
     [ 1.  0.]
     [ 0.  7.]]



```python
# 矩阵切分
data = np.arange((18)).reshape(3, 6)
print (data)
print (np.hsplit(data,3))
print (np.hsplit(data,(2,)))
print (np.hsplit(data,(1, 3, 5)))
```

    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]]
    [array([[ 0,  1],
           [ 6,  7],
           [12, 13]]), array([[ 2,  3],
           [ 8,  9],
           [14, 15]]), array([[ 4,  5],
           [10, 11],
           [16, 17]])]
    [array([[ 0,  1],
           [ 6,  7],
           [12, 13]]), array([[ 2,  3,  4,  5],
           [ 8,  9, 10, 11],
           [14, 15, 16, 17]])]
    [array([[ 0],
           [ 6],
           [12]]), array([[ 1,  2],
           [ 7,  8],
           [13, 14]]), array([[ 3,  4],
           [ 9, 10],
           [15, 16]]), array([[ 5],
           [11],
           [17]])]

---

#### 10、数据的复制
- 用=直接赋值，并不是赋值，两个引用指向同一内存
- view()浅复制：复制出一个新的对象，和原来的对象共用数据值，不共用结构
- copy()复制：复制出一个新对象，拥有独自的新的数据值和结构


```python
# 赋值
data1 = np.arange(12).reshape(2,6)
data2 = data1
print (data2 is data1)
data2.shape = (3,4)
print (data1.shape)
```

    True
    (3, 4)



```python
# 浅复制
data3 = data1.view()
print (data3 is data1)
data3.shape = (4,3)
print (data1.shape)
data3[0,1] = 9999
print (data1[0,1])
```

    False
    (3, 4)
    9999



```python
# 复制
data4 = data1.copy()
print (data4 is data1)
data4.shape = (6,2)
print (data1.shape)
data4[0,1] = 7777
print (a[0,1])
```

    False
    (3, 4)
    1

---

#### 11、矩阵的扩展
- 使用np.tile对矩阵进行扩展
- 指定int参数，表示横向扩展n倍
- 指定tuple参数，表示按多维度扩展


```python
data = np.arange(4).reshape(2, -1)
print (data)
# 横向扩展2倍
print (np.tile(data, 2))
# 行扩展2倍，列扩展2倍
print (np.tile(data, (2, 2)))
```

    [[0 1]
     [2 3]]
    [[0 1 0 1]
     [2 3 2 3]]
    [[0 1 0 1]
     [2 3 2 3]
     [0 1 0 1]
     [2 3 2 3]]

---

#### 12、求最大值索引
- 使用argmax函数找到数据最大值索引
- 不传入参数，表示按一维结构寻找索引
- 传入可选参数axis=0/1表示求每列/行最大值索引列表


```python
data = np.sin(np.arange(20)).reshape(5,4)
print (data)
# 找到整体数据最大值索引，按一维数据索引
print (data.argmax())
# 找到每列的最大值索引
ind = data.argmax(axis=0)
print (ind)
# 通过索引依次打印每列最大值
data_max = data[ind,range(data.shape[1])]
print (data_max)
# 找到每行的最大值索引
ind = data.argmax(axis=1)
print (ind)
data_max = data[range(data.shape[0]),ind]
print (data_max)
```

    [[ 0.          0.84147098  0.90929743  0.14112001]
     [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
     [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
     [-0.53657292  0.42016704  0.99060736  0.65028784]
     [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
    14
    [2 0 3 1]
    [ 0.98935825  0.84147098  0.99060736  0.6569866 ]
    [2 3 0 2 3]
    [ 0.90929743  0.6569866   0.98935825  0.99060736  0.14987721]

---

#### 13、对矩阵数据进行排序
- 使用sort函数对数据进行排序
- 使用np.sort函数对数据进行排序
- 使用argsort函数得到排序索引


```python
# 使用对象的sort函数，将对象转换为排序后的对象，注意，这里方法调用的返回值为None
data = np.array([[4,3],[2,1]])
print (data)
data.sort(axis=1)
print (data)
data.sort(axis=0)
print (data)
```

    [[4 3]
     [2 1]]
    [[3 4]
     [1 2]]
    [[1 2]
     [3 4]]



```python
# 使用np.sort函数，将转换后的对象作为函数的返回值
data = np.array([[4,3],[2,1]])
print (data)
print (np.sort(data, axis=0))
```

    [[4 3]
     [2 1]]
    [[2 1]
     [4 3]]



```python
# 使用argsort得到排序索引值的矩阵,并可通过得到的索引值打印出排序后的矩阵
data = np.array([4,3,2,1])
index = np.argsort(data)
print (index)
print (data[index])
```

    [3 2 1 0]
    [1 2 3 4]
