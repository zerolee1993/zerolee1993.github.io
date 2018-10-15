---
layout:     post
title:      "Python入门-2-基本语法"
subtitle:   "Python学习笔记（2/7）"
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

#### 1、常用操作

- 输出
    - 使用print语句输出
    - `print '123','abc' `会连续打印两个字符串，并用空格隔开
- 注释
    - 仅支持单行注释`#注释内容`
    - 可以单独占用一行，也可以在一行代码末尾

---

#### 2、数据类型

- 整数
    - Python中整数的表示方法和数学上的写法一致
    - 十六进制整数用0x前缀表示，如：`0xff00`
- 浮点数
    - 普通表示方法：`12.3`
    - 科学计数法：`123e-1`
- 注意
    - *整数和浮点数在计算机内部存储的方式是不同的，整数运算永远是精确的（除法难道也是精确的？是的！），而浮点数运算则可能会有四舍五入的误差。*
- 字符串
    - 使用`'abc'`或`"abc"`表示字符串
    - r字符，当一个字符串中包含多个需要转义的字符时，可以使用r开头声明，省略转义符，但不能对单引号双引号自动转义，如：`r'\1\2'`
    - `''`内的`"`不需要转义，`""`内的`'`不需要转义
    - 多行字符串可以使用`'''abc'''`声明
    - Python默认使用ASCII编码，需要输出中文时，应使用unicode字符，u前缀声明，如`u'中文'`
    - u、r、`''' '''`三种方式可以任意叠加使用
- 注意
    - *如果中文字符串在Python环境下遇到 UnicodeDecodeError，这是因为.py文件保存的格式有问题。可以在第一行添加注释`# -*- coding: utf-8 -*-`，目的是告诉Python解释器，用UTF-8编码读取源代码。然后用Notepad++ 另存为... 并选择UTF-8格式保存。*
- 布尔值
    - 只有`True`、`False`两种值（注意大小写）
    - 可以用`and`、`or`、`not`进行逻辑运算
- 关于逻辑运算符
    - `and`, `or`, `not`
    - 布尔类型可以通过逻辑运算符和其他数据类型运算
    - Phthon把`0`,空字符串`''`,`None`看成`False`，其他数值以及非空字符串看成`True`
    - 举个李子`a = True`,`print a and 'a=T' or 'a=F'`,运行结果为`'a=T'`,a and 'a=T'运算，a为True，结果为'a=T'，下一步运算，'a=T'看做True，不会进行后面的判断，直接返回'a=T'
    - 短路计算，a and b，若a是False，直接返回a不再往后计算，a or b，若a是True。直接返回a不再往后计算。
- 空值
    - Python中的特殊值，用None表示
    - None不是0，0是有意义的

---

#### 3、流程控制

- 分支语句if
    - if可以和elif、else配合使用完成分支控制
    - Python中具有相同缩进的代码被视为代码块，为了避免因缩进引起的语法错误，Python习惯使用四个空格进行缩进

```python
age = 20
if age >= 18:
  print ('adult')
elif age >= 6:
  print ('teenager')
elif age >= 3:
  print ('kid')
else:
  print ('baby')
```

- 循环语句for
    - for循环可以将list、tuple中的元素依次迭代出来

```python
foods = ['apple','banana','orange']
for food in foods:
    print (food)
```

- 循环语句while
    - while判断条件为True，则执行循环体代码块，否则退出循环
    - 举个李子：打印1~10

```python
N = 11
x = 0
while x < N:
    print (x)
    x += 1
```

- 退出循环体break
    - 用于在循环体内直接退出本循环
    - 举个李子：打印1~100累加和

```python
sum = 0
x = 1
while True:
    sum = sum + x
    x = x + 1
    if x > 100:
        break
print (sum)
```

- 退出本次循环continue
    - 用于跳过本次循环后续的代码，开始下一次循环

---

#### 4、集合

- 有序集合list
    - 声明方式`foods = ['apple','banana','orange']`
    - Phthon是动态语言，list中的元素数据类型不一定一致，`L=['a',1,True]`
    - 通过索引访问元素，正序`foods[0]`,倒序`foods[-1]`，使用索引时，注意不要越界。
    - 向list尾部添加新的元素：`foods.append('pear')`
    - 向list指定索引位置添加新元素：`foods.insert(2, 'peach')`
    - 弹出list尾部元素：`foods.pop()`
    - 弹出list执行索引位置的元素：`foods.pop(2)`
- 有序集合tuple
    - 元组，与list区别，一旦创建完成，则不可更改
    - 声明方式`foods = ('apple','banana','orange')`
    - 创建单个元素tuple`t = (1,)`,不加,计算机将认为声明的是整数1
    - tuple的不变性，指的是在内存中的指向不变，如果tuple中有个list，list里的元素是可以变得`t = (1,2,['a','b'])`

- 无序键值对集合dict
    - dict内的元素是无序的
    - dict中作为key的元素不可变。基本类型如字符串、整数、浮点数都是不可变的，以及tuple，都可以作为 key。但是list是可变的，就不能作为 key。
    - 声明方式：{key:value,key:value,...} 最后一个key:value后的,可省略，如：`dict = {'apple':1,'pear':2}`
    - list使用索引返回元素，dict使用key查找对应的value`dict['apple']`，如果key不存在，会抛出KeyError
    - 使用in判断key是否存在`'apple' in dict`返回布尔值
    - dict提供了get方法，当Key不存在时不抛出异常，返回None`dict.get('apple')`
    - 向dict中添加新的键值对，直接用赋值语句`dict['orange']=3`，如果key已存在，则覆盖原来的value
    - 使用最简单的for循环可以遍历dict的key

```python
dict = {'apple':1,'pear':2}
for a in dict:
    print (a,':',dict[a])
```

- len()函数
    - 用于查看集合的大小，可将集合作为参数传入`len(dict)`

- 无序集合set
    - set内的元素是无序的，且不能包含重复的元素
    - 存储的元素和dict中的key类似，必须是不变对象
    - 声明方式：`s = set(['apple','apple','orange'])`，此时set自动去掉重复的元素
    - 使用in判断set是否包含元素`'apple' in s`返回布尔值
    - 可以通过for循环来遍历set的元素
    - 添加元素：`s.add('pear')`，若元素已经存在，不会报错但不再添加
    - 删除元素：`s.remove('pear')`，若元素不存在，会抛出KeyError异常

---

#### 5、函数

- abs()取绝对值
    - 可查看官方文档http://docs.python.org/2/library/functions.html#abs
    - 可在Python命令行使用help(abs)查看帮助信息
    - `abs(100)`，`abs(-80)`
    - 传入参数个数、类型不匹配时候会抛出异常TypeError

- cmp(x,y)比较函数
    - 比较两个参数的大小x小于y返回-1，相等返回0，x大于y返回1
    - 可以传入字符
- int()转换整数函数
    - 可以将其他数据类型转换为整数
    - `int(11.2)`，`int('10')`当传入字符时，字符必须为整数，否则抛出ValueError异常
- str()转换字符串函数
    - 将其他数据类型转换为字符串
    - `str(123)`，`str(1.1)`
- 声明函数
    - 使用def定义函数，return语句返回
    - 如果没有return语句，指定返回结果为None
    - return None可简写为return
    - `return a, b`可以返回多个值，实际上是返回了一个tuple

```python
def my_abs(x):
    if x > 0:
        return x
    else:
        return -x
my_abs(-10)
```

- 递归函数
    - 函数在内部调用自身，就是递归函数
    - 自定义一个阶乘函数
    - 使用递归函数需要注意防止栈溢出，函数调用是通过栈（stack）这种数据结构实现的，每当进入一个函数调用，栈就会加一层栈帧，每当函数返回，栈就会减一层栈帧。由于栈的大小不是无限的，所以，递归调用的次数过多，会导致栈溢出。如本函数fact(1000)会抛出异常

```css
def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n-1)
fact(10)
```

- 默认参数
    - 定义函数时可以设置参数默认值，简化调用
    - 如int()函数，实际有两个参数，第二个参数是转换进制，默认为10，`int('100'，8)`
    - 默认参数需要定义在必须参数的后面
    - 设计一个函数计算x的N次方`power(2,3)`，默认计算x的二次方`power(2)`

```python
def power(x,n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
power(3)
```

- 可变参数
    - 在参数名前加*定义为可变参数，可以传入一个或多个值
    - Python会将一组参数组装为一个tuple
    - 定义一个求平均值的函数，可以传入人一个数值

```python
def average(*args):
    sum = 0.0
    for x in args:
        sum += x
    return sum / len(args)
average(2,3,4)
```

---

#### 6、切片

- list/tuple的切片
    - 取list的前三个元素`list[0:3]`，包含索引0，不包含索引3位置的元素
    - `list[:]`表示取全部元素
    - 可以指定第三个参数表示每隔几个元素取一个。`list[::2]`
    - 倒叙切片`list[-3,-1]`
- 字符串切片
    - 操作方式和list一致，可以将字符串看成字符list
    - 操作结果仍然是字符串`'abcdef'[0:3]`
