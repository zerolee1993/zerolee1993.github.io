---
layout:     post
title:      "Python入门-4-函数编程"
subtitle:   "Python学习笔记（4/7）"
date:       2017-12-18
author:     "Zero"
#cover: ""
categories: technology
tags: Python
---

### 目录

* 纲要
{:toc}

---

#### 1、高阶函数

- 变量可以指向函数，并通过变量调用

```python
f = abs
f(-1)
```

- 函数名其实就是指向函数的变量

```python
abs = len
abs([1,2,3])
```

- 高阶函数，能接受函数作为参数

```python
def add(x, y, f):
    return f(x) + f(y)
add(-5,9,abs)
```

---

#### 2、常用内置函数

- zip()
    - 将两个list组装成一个

```python
zip([1,2,3],['a','b','c'])
[(1,'a'),(2,'b'),(3,'c')]
```

- isinstance(x,str)
    - 判断x是否是字符串`isinstance(1,str)`

- upper()
    - 返回大写字母`'abc'.upper()`

- strip()
    - `s.strip(rm)` 删除 s 字符串中开头、结尾处的 rm 序列的字符。当rm为空时，默认删除空白符（包括'\n', '\r', '\t', ' ')

- map()
    - 接收一个函数f和一个list，并通过把函数f依次作用在list的每个元素上，得到一个新的list并返回
    - map()函数不改变原有的 list，而是返回一个新的 list

```python
def f(x):
    return x * x
map(f, range(1,11))
```

- reduce()
    - 在python3中，reduce函数移动到了functools模块中，如果想使用，需要先导如模块
    - 接收一个函数f和一个list，f必须接受两个参数，reduce()对list的每个元素反复调用函数f，并返回最终结果值
    - list[0]和list[1]通过f计算得出结果，并使用结果和list[2]通过f计算，依次类推。
    - reduce()还可以接收第3个可选参数，作为计算的初始值，如果传入第三个参数，则第一步计算为初始值和list[0]通过f进行计算

```python
from functools import reduce
def f(x, y):
    return x + y
reduce(f,[1,3,5,7,9])
```

- filter()
    - 接收一个函数f和一个list，这个函数f的作用是对每个元素进行判断，返回True或False
    - filter()根据判断结果自动过滤掉不符合条件的元素，返回由符合条件元素组成的新list
    - 在python3中，filter函数返回结果需要经过list函数转换成list

```python
def f(x):
    return x%2==1
l = filter(f,range(1,11))
print (list(l))
```

- sorted()
    - 对list进行排序`sorted([36, 5, 12, 9, 21])`(由小到大)
    - 指定可选参数reverse=True，可实现由大到小排序

```python
def reversed_cmp(x, y):
    if x > y:
        return -1
    if x < y:
        return 1
    return 0
sorted([36, 5, 12, 9, 21], reverse=True)
```

---

#### 3、返回函数

- 函数可以return一个函数
- 返回函数时，不加()，此时没有进行abs计算，只是将a指向了temp函数，延时了计算，当调用a()是才计算了绝对值

```python
def f(x):
    return abs(x)
def g(x):
    def temp():
        return abs(x)
    return temp
a = f(-1)
a = g(-1)
a()
```

---

#### 4、闭包

- 内层函数引用了外层函数的变量（参数也算变量），然后返回内层函数的情况，称为闭包（Closure）
    - 闭包的特点是返回的函数还引用了外层函数的局部变量
    - 正确使用闭包，就要确保引用的局部变量在函数返回后不能变
    - 返回函数不要引用任何循环变量，或者后续会发生变化的变量

```python
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs
f1, f2, f3 = count()
print (f1(), f2(), f3()) # 9 9 9

def count():
    fs = []
    for i in range(1, 4):
        def f(m=i):
             return m*m
        fs.append(f)
    return fs
f1, f2, f3 = count()
print (f1(), f2(), f3()) # 1 4 9
```

---

#### 5、匿名函数

- lambda x: x * x
    - 关键字lambda 表示匿名函数，冒号前面的 x 表示函数参数
    - 匿名函数有个限制，就是只能有一个表达式，不写return，返回值就是该表达式的结果
    - 逆序排列`sorted([1,4,6,7], lambda x,y: -cmp(x,y) )`
    - 返回匿名函数`myabs = lambda x: -x if x < 0 else x`

---

#### 6、装饰器

- 无参装饰器
    - 接受一个参数，对其包装，然后返回一个新函数
    - 打印函数执行时间，注意，此处f(x)实际执行了两次

```python
import time
def performance(f):
    def fn(x):
        a = time.time()
        f(x)
        b = time.time()
        print ('call %s() in %s' % (f.__name__,(b-a)))
        return f(x)
    return fn
@performance
def factorial(n):
    print ('#')
    return reduce(lambda x,y: x*y, range(1, n+1))
print (factorial(10))
```

- 有参装饰器

```python
import time
def performance(unit):
    def new(f):
        def fn(x):
            a = time.time()
            f(x)
            b = time.time()
            print ('call %s() in %s%s' % (f.__name__,(b-a),unit))
            return f(x)
        return fn
    return new
@performance('ms')
def factorial(n):
    return reduce(lambda x,y: x*y, range(1, n+1))
print (factorial(10))
```

---

#### 7、偏函数

- functools.partial可以把一个参数多的函数变成一个参数少的新函数，少的参数需要在创建时指定默认值

```python
import functools
int2 = functools.partial(int, base=2)
int2('1000000')
sorted_ignore_case = functools.partial(sorted, key=str.lower)
print (sorted_ignore_case(['bob', 'about', 'Zoo', 'Credit']))
```
