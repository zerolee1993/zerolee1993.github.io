---
layout:     post
title:      "Python入门教程-3-常用迭代"
subtitle:   "第三篇（共七篇）：常用迭代"
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

#### 1、list/tuple迭代

- list/tuple索引迭代
    - 普通的迭代取出list中的元素，如果需要取出索引，使用enumerate()函数，并在for循环中绑定两个参数（如果仅绑定了一个参数，则迭代的是一个tuple）
    - enumerate()函数实际是把`['apple','pear']`变成了类似`[(0,'apple'),(1,'pear')]`，每一个元素是一个tuple

```python
list = ['apple','pear']
for t in enumerate(list):
    print (t[0],t[1])
for index,name in enumerate(list):
    print (index,name)
```

---

#### 2、dict迭代

- dict迭代value
    - dict.values()可以将dict转换成一个包含所有value的list，从而实现迭代dict的value
    - 在2.7版本中，有一个itervalues方法
    - dict.itervalues()效果和dict.values()相同，但不会转换list，而是在迭代过程中依次取出value，更加节省内存
    - 在3.5版本中，只有values方法，可能新版本自动做了性能优化

```python
d = {'apple':1,'pear':2}
for v in d.values():
    print (v)
```

- dict迭代key和value
    - dict.items()可以将dict转换成一个包含tuple的list，从而实现迭代dict的key和value
    - 在2.7版本中，有一个iteritems方法
    - dict.iteritems()效果和dict.items()相同，但不会转换list，而是在迭代过程中依次取出value，更加节省内存
    - 在3.5版本中，只有items方法，可能新版本自动做了性能优化

```python
d = {'apple':1,'pear':2}
for k,v in d.items():
    print (k,v)
```

---

#### 3、列表生成式

- 简单式
    - 生成一个[1x1,2x2,3x3,...,10x10]
    - `list = [x * x for x in range(1,11)]`
    - `range(x,y)`用于创建一个x~y-1的int列表
- 复杂式
    - 生成一个html表格表示水果及对应数量
    - `','.join(list)`函数将list元素用','拼接字符串
    - 字符串可以通过`%`进行格式化，用指定的参数替代`%s`

```python
d = {'apple':1,'pear':2}
tds = ['<tr><td>%s</td><td>%s</td></tr>' % (fruit, count) for fruit, count in d.items()]
print ('<table>')
print ('<tr><th>Fruit</th><th>Count</th><tr>')
print ('\n'.join(tds))
print ('</table>')
```

- 条件过滤
    - 列表生成式的for循环后可以加上if判断
    - 求偶数的平方`[ x * x for x in range(1,11) if x % 2 == 0 ]`
