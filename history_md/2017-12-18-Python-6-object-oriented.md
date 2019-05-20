---
layout:     post
title:      "Python入门-6-面向对象"
subtitle:   "Python学习笔记（6/7）"
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

#### 1、类

- 类的定义

```python
class Person:
    pass
tom = Person()
```

- 类的属性

```python
class Person:
    pass
tom = Person()
tom.name = 'Tom'
```

- 初始化时指定类的属性
    - 为Person类添加init方法，第一个参数必须是self，可定义为别的名字，习惯为self，类似于java中this
    - 创建实例是，可以指定init的参数

```python
class Person:
    def __init__(self, name):
        self.name = name
tom = Person('Tom')
```

- 初始化是指定init定义外的属性
    - 需要对init方法进行处理，使能够接受自定义属性

```python
class Person:
    def __init__(self, name, gender, **map):
        self.name = name
        self.gender = gender
        for k,v in map.items():
            setattr(self, k, v)
tom = Person('Tom','male',job='teacher')
```

- 属性访问权限控制
    - 以双下划线开头的属性不能直接被外部访问

```python
class Person(object):
    def __init__(self, name, score):
        self.name = name
        self.__score = score
p = Person('Bob', 59)
print (p.name)
try:
    print (p.__score)
except AttributeError:
    print ('attributeError')
```

- 类属性
    - 定义类的class中声明或通过类名声明
    - 所有实例均可以访问类属性
    - 类属性只有一份
    - 实例不能更改类的属性
    - 在类中访问类属性需要加类名
    - 实例上修改类属性，它实际上并没有修改类属性，而是给实例绑定了一个实例属性。当实例定义属性与类属性重名时，优先调用实例属性，如果想通过实例访问类属性，需要先删除实例的重名属性`del tom.address`
    - 类属型设置为私有时，仅可以在类内部`Person.addr`访问

```python
class Person(object):
    address = 'China'
tom = Person()
tom.address
```

- 实例方法类方法
    - class中定义的全部是实例方法
    - 实例方法的第一个参数必须传入对象本身，习惯命名为self
    - 如果需要定义类方法，需要标记`@classmethod`
    - 类方法的第一个参数必须传入类本身，习惯命名cls

---

#### 2、继承

- 子类
    - 从现有的类继承，自动拥有现有类的所有功能
    - 一个类总是从某个类继承
    - 不要忘记调用`super().__init__`初始化父类属性
    - 可以多重继承，用,隔开

```python
class Person(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
class Teacher(Person):

    def __init__(self, name, gender, course):
        super(Teacher, self).__init__(name, gender)
        self.course = course
t = Teacher('Alice', 'Female', 'English')
print (t.name)
print (t.course)
```

#### 3、获取对象信息

- type()获取对象类型
- dir()获取对象所有属性
- getattr(obj,k)获取对象属性
- setattr(obj,k,v)设置对象属性
