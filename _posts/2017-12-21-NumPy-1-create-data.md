---
layout:     post
title:      "NumPy入门-1-构造数据"
subtitle:   "Python科学计算库NumPy的学习笔记（1/2）"
date:       2017-12-21
author:     "Zero"
#cover: "/assets/in-post/python1/bg.jpg"
categories: technology
tags: NumPy
---

### 简介

Numpy提供了强大的N维数据对象ndarray以及针对于这种对象的各种处理
本文简述了ndarray的几种初始化方式及一些基本操作，并提供了可运行代码示例（Python3版本）
在学会使用Numpy构造数据后，可以学习《使用Numpy处理数据》

---

### 目录

* 纲要
{:toc}

---

#### 1、核心数据结构
- Numpy中核心数据结构是ndarray
- 使用type函数可以查看数据类型


```python
import numpy as np
data = np.arange(1)
print (type(data))
```

    <class 'numpy.ndarray'>

---

#### 2、学会使用帮助文档
- 使用help函数可以在命令行打印相关函数的API及示例
- 之后提到的函数记录了常用用法，可以通过help查看API中的详细用法及说明


```python
help(np.arange)
```

    Help on built-in function arange in module numpy.core.multiarray:

    arange(...)
        arange([start,] stop[, step,], dtype=None)

        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval ``[start, stop)``
        (in other words, the interval including `start` but excluding `stop`).
        For integer arguments the function is equivalent to the Python built-in
        `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
        but returns an ndarray rather than a list.

        When using a non-integer step, such as 0.1, the results will often not
        be consistent.  It is better to use ``linspace`` for these cases.

        Parameters
        ----------
        start : number, optional
            Start of interval.  The interval includes this value.  The default
            start value is 0.
        stop : number
            End of interval.  The interval does not include this value, except
            in some cases where `step` is not an integer and floating point
            round-off affects the length of `out`.
        step : number, optional
            Spacing between values.  For any output `out`, this is the distance
            between two adjacent values, ``out[i+1] - out[i]``.  The default
            step size is 1.  If `step` is specified, `start` must also be given.
        dtype : dtype
            The type of the output array.  If `dtype` is not given, infer the data
            type from the other input arguments.

        Returns
        -------
        arange : ndarray
            Array of evenly spaced values.

            For floating point arguments, the length of the result is
            ``ceil((stop - start)/step)``.  Because of floating point overflow,
            this rule may result in the last element of `out` being greater
            than `stop`.

        See Also
        --------
        linspace : Evenly spaced numbers with careful handling of endpoints.
        ogrid: Arrays of evenly spaced numbers in N-dimensions.
        mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.

        Examples
        --------
        >>> np.arange(3)
        array([0, 1, 2])
        >>> np.arange(3.0)
        array([ 0.,  1.,  2.])
        >>> np.arange(3,7)
        array([3, 4, 5, 6])
        >>> np.arange(3,7,2)
        array([3, 5])

---

#### 3、使用np.array函数构建数据
- 作用是将list转换为ndarray类型的向量or矩阵
- ndarray具有元素一致性，可使用astype函数做类型转换
- shape属性可查看数据结构
- 可以按照索引取值

```python
# numpy将自动转换所有的值，使ndarray中的数据类型保持一致
# 可以通过dtype属性查看
vector = np.array([1,2,3,4.0])
print (vector.dtype)

# 使用astype函数可以进行整体类型转换
vector = np.array([1,2,3])
print (vector.dtype)
vector = vector.astype(float)
print (vector.dtype)
```

    float64
    int32
    float64



```python
# 使用shape属性查看数据的结构
vector = np.array([1,2,3,4])
print (vector.shape)
matrix = np.array([[11,12,13],[21,22,23],[31,32,33]])
print (matrix.shape)
```

    (4,)
    (3, 3)



```python
# 按照索引进行取值
# 先看一个一维数据怎么取值
vector = np.array([1,2,3,4,5,6,7,8,9])
# 传入单个参数，用索引取值
print (vector[1])
# 传入两个参数，指定索引范围取值，左闭右开区间
print (vector[0:2])
# 传入三个参数，指定索引范围，并指定取值间隔
print (vector[0:7:3])
# 类推到二维数据
matrix = np.array([[11,12,13],[21,22,23],[31,32,33]])
print (matrix[1,1])
print (matrix[0:2,0:2])
```

    2
    [1 2]
    [1 4 7]
    22
    [[11 12]
     [21 22]]

---

#### 4、使用np.arrange快速构建数据
- 构建的ndarray为一维的，可以通过reshape函数转换为多维
- 方式1：传入单个参数x,创建[0, n)的数据集，可传入小数
- 方式2：传入两个参数x,y,指定范围[x, y)
- 方式3：传入三个参数x,y,z,指定范围[x, y),取值间隔为z


```python
# 方式1
data = np.arange(10)
print (data)
data = np.arange(6.3)
print (data)
```

    [0 1 2 3 4 5 6 7 8 9]
    [ 0.  1.  2.  3.  4.  5.  6.]



```python
# 方式2
data = np.arange(1, 6)
print (data)
data = np.arange(1.1, 6.8)
print (data)
```

    [1 2 3 4 5]
    [ 1.1  2.1  3.1  4.1  5.1  6.1]



```python
# 方式3
data = np.arange(0, 10, 2)
print (data)
data = np.arange(0,5,0.7)
print (data)
```

    [0 2 4 6 8]
    [ 0.   0.7  1.4  2.1  2.8  3.5  4.2  4.9]



```python
# 通过reshape将arange创建的一维变为多维
data = np.arange(6)
print (data)
print (data.shape)
data = data.reshape(2, 3)
print (data)
print (data.shape)
# 指定参数时，允许有一个参数为-1，系统通过其他参数自动计算这个值
data = data.reshape(3, -1)
print (data)
print (data.shape)
```

    [0 1 2 3 4 5]
    (6,)
    [[0 1 2]
     [3 4 5]]
    (2, 3)
    [[0 1]
     [2 3]
     [4 5]]
    (3, 2)



```python
# 当然针对reshape有一个逆操作，ravel，可将多维变为一维
data = np.arange(0, 4).reshape(2, -1)
print (data)
print (data.shape)
data = np.ravel(data)
# 这个操作可以等价替换为： a = a.ravel() 或  a = a.reshape(-1)
print (data)
print (data.shape)
```

    [[0 1]
     [2 3]]
    (2, 2)
    [0 1 2 3]
    (4,)

---

#### 5、通过ndim查看ndarray的维度
- 打印ndarray对象的ndim属性，可以查看维度
- 使用numpy的ndim函数也可以查看维度
- 标量的ndim为0


```python
data = np.arange(10)
print (data.ndim)
data = data.reshape(2, 5)
print (np.ndim(a))
print (np.ndim(1))
```

    1
    1
    0

---

#### 6、通过size查看ndarray的元素个数
- 打印ndarray对象的size属性
- 使用numpy的size函数
- size函数支持传入第二个参数0/1表示按列/行统计size


```python
data = np.arange(10).reshape(2, 5)
print (data.size)
print (np.size(data))
# 指定可选参数为0/1查看列数/行数
print (np.size(data, 0))
print (np.size(data, 1))
```

    10
    10
    2
    5

---

#### 7、使用zeros函数或ones函数构建固定数值的数据
- zeros传入int参数，创建一维ndarray，数据值均为0
- zeros传入包含n个参数的tuple，创建n维ndarray，数据均为0
- 数据值类型默认为浮点型，传入可选参数dtype修改数据类型
- ones的使用方法和zeros相同


```python
# 传入int创建
a = np.zeros(4)
print (a)
# 传入tuple创建
a = np.zeros((2, 2))
print (a)
# 执行dtype为int类型
a = np.zeros((2, 2), dtype=np.int)
print (a)
# 使用ones()构建元素值为1的ndarray，具体用法同zeros
a = np.ones(5)
print (a)
```

    [ 0.  0.  0.  0.]
    [[ 0.  0.]
     [ 0.  0.]]
    [[0 0]
     [0 0]]
    [ 1.  1.  1.  1.  1.]

---

#### 8、random构建随机数值的数据
- 使用np.random模块下的random方法创建随机数据，范围0.0~1.0
- python2.7中仅提供了random_sample方法
- 无参数调用，返回随机标量
- int参数调用，返回一维数据
- tuple调用，返回n维数据


```python
# 无参调用
data = np.random.random_sample()
print (data)
# int参调用
data = np.random.random_sample(2)
print (data)
# 传入tuple调用
data = np.random.random_sample((3,4))
print (data)
```

    0.23834160801104665
    [ 0.4399487   0.75315239]
    [[ 0.94897646  0.34252114  0.53499265  0.83915492]
     [ 0.89656745  0.89092317  0.88215086  0.38528955]
     [ 0.48198362  0.55292249  0.98633396  0.85066574]]

---

#### 9、使用linspace创建范围内的均值
- np.linspace(x, y, z) x,y指定取值范围，z为取值数量


```python
# 开始值0，结束值5，平均取6个数
data = np.linspace(0,5,6)
print (data)
```

    [ 0.  1.  2.  3.  4.  5.]

---

#### 10、文件读取构建
- genfromtxt函数可以从txt中读取数据
- 参数表示文件路径
- delimiter指定数据之间的分隔符
- dtype指定读取的类型
- 使用文件：[txtdata.txt](/assets/in-post/numpy1/txtdata.txt)


```python
txtdata = np.genfromtxt('data/txtdata.txt',delimiter=',',dtype=str)
print (txtdata.shape)
```

    (998, 5)
