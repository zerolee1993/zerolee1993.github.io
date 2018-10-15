---
layout:     post
title:      "Python入门-7-特殊方法"
subtitle:   "Python学习笔记（7/7）"
date:       2017-12-18
author:     "Zero"
#cover: "/assets/in-post/python1/bg.jpg"
categories: technology
tags: Python
---

### 目录

* 纲要
{:toc}

---

#### 1、特点

- 定义在class中
- 不需要直接调用
- Python的某些函数或操作符会调用对应的特殊方法

---

#### 2、常见的特殊方法

- `__str__()`相当于java的toString，用于print
- `__len__()`用于输出长度
- `__cmp__()`用于比较

- 数学运算
  - `__add__()`对应运算符+
  - `__sub__()`对应运算符-
  - `__mul__()`对应运算符*
  - `__div__()`对应运算符/
  - `__int__()`对应转换函数int()
  - `__float__()`对应转换函数float()

- getter&setter
    - python习惯用.调用属性和赋值，如果希望控制.调用的时的逻辑，需要再类中定义方法
    - 方法用`@property`装饰器或`@attrName.setter`装饰器修饰

```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.__score = score
    @property
    def score(self):
        return self.__score
    @score.setter
    def score(self, score):
        if score < 0 or score > 100:
            raise ValueError('invalid score')
        self.__score = score
    @property
    def grade(self):
        if self.__score > 80:
            return 'A'
        elif self.__score<60:
            return 'B'
        else:
            return 'C'
s = Student('Bob', 59)
print (s.grade)
s.score = 60
print (s.grade)
s.score = 99
print (s.grade)
```

- slots限制属性
    - 如下例子，限制person只能有name,gender属性
    - 限制student只能在person基础上再有score属性

```python
class Person(object):
    __slots__ = ('name', 'gender')
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
class Student(Person):
    __slots__ = ('score')
    def __init__(self, name, gender, score):
        self.name = name
        self.gender = gender
        super(Student,self).__init__(name,gender)
s = Student('Bob', 'male', 59)
s.name = 'Tim'
s.score = 99
print (s.score)
```

- 可调用对象
    - 将类的实例变成一个可调用对象，只需要在类中实现call

```python
class Person(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
    def __call__(self, friend):
        print ('my name is %s,my friend\'s name is %s' % (self.name,friend))
p = Person('Bob','male')
p('Tom')
```
