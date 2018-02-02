---
layout:     post
title:      "Python入门-5-模块和包"
subtitle:   "Python学习笔记（5/7）"
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

#### 1、概念

- 模块：就是指一个.py文件
    - 如math就是一个模块
    - `import math`引入模块
    - `math.pow(2,10)`调用模块的函数

- 包：类似于java的包，文件夹
    - 如引用p1.util的包`import p1.util`
    - 使用包下的方法：`p1.util`
    - 在python中，包不是一个简单的文件目录，包下面有一个`__init__.py`文件

---

#### 2、导入模块

- 普通导入
    - 导入方式`import math`
    - 调用方式`math.pow(2,10)`
- 调用特定方法
    - 导入方式`from math import pow`
    - 调用方式`pow(2,10)`
- 调用特定方法并重命名
    - 导入方式`from math import pow as pow1`
    - 调用方式`pow1(2,10)`
- 使用form...import..方式导入时，如果两个包的方法名相同，则会冲突，必须重命名

---

#### 3、动态导入模块

- 导入模块不存在时，抛出ImportError异常，使用try动态导入不同模块
    - python早期没有json模块，但提供了功能相同的simplejson模块
    - 可以使用try...except动态导入

```python
try:
    import json
except ImportError:
    import simplejson as json
print json.dumps({'python':2.7})
```

---

#### 4、使用future

- Python的新功能在旧版本中试用，可通过引入`__future__`
- 在python2中可以使用如下方法引入python3的除法
- 需要注意：这里future两边各是两个下划线！

```python
from __future__ import division
print (10 / 3)
```

---

#### 5、安装第三方模块

- 使用easy_install或pip安装第三方模块
    - 推荐使用pip
    - 将安装目录下的Scripts目录添加到path环境变量
    - 以安装web模块为例，在cmd命令行中执行`pip install web.py`
    - 如果你使用了anaconda，不必要进行任何配置，直接打开Anaconda Prompt，执行`pip install tensorflow`
