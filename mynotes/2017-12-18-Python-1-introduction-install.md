---
layout:     post
title:      "Python入门-1-简介及安装"
subtitle:   "Python学习笔记（1/7）"
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

#### 1、适用范围

- 适用于
  - Web网站和各种网络服务
  - 系统工具和脚本
  - 作为“胶水”语言把其他语言开发的模块包装起来方便使用

- 不适用
  - 贴近硬件的代码（C）
  - 移动开发（ObjC，Swift/Java）
  - 游戏开发（C/C++）

- 实际应用场景
  - YouTube
  - 豆瓣
  - openstack开源云计算平台
  - Google
  - Yahoo
  - NASA美国航空航天局

---

#### 2、语言对比

|语言|类型|运行速度|代码量|
|----|---|-------|------|
|C|编译为机器码|非常快|非常多|
|Java|编译为字节码|快|多|
|Python|解释执行|慢|少|

---

#### 3、一个缺点

- 原码不能加密

---

#### 4、版本信息

- 2.7 & 3.x 两者不兼容
- 本套教程共7片，所有python代码在3.5版本中运行

---

#### 5、集成环境Anaconda

- anaconda是python的集成环境，安装anaconda后，默认安装了Python以及Python的一些必要库，并可以使用anaconda方便的安装其他库。
- 常用命令
  - conda list 列出当前安装的软件包
  - conda install numpy 安装numpy库（默认是安装好的）
- 安装Tensorflow实例
  - anaconda search -t conda tensorflow 查看软件所有版本
  - anaconda show dhirschfeld/tensorflow 查看版本"dhirschfeld/tensorflow"的安装命令
  - 根据提示命令进行安装
- PS：在windows下安装tensorflow必须使用python3版本，所以Anaconda也应该安装python3对应的版本，并且建议使用`pip install tensorflow`而不是上述命令去安装tensorFlow，因为这样安装的tensorflow会缺失一些东西，比如刚接触tensorflow的朋友们学习MNIST数据集，会使用到一个input_data.py，使用anaconda安装的tensorFlow是没有这个模块的~
- PS：如何安装低版本的tensorflow呢，很简单，打开Anaconda Prompt，在命令行中输入`pip install tensorflow==0.xx`，因为版本号指定的不明确，命令行中会提示tensorflow的所有版本号，然后再次使用该命令，指定明确的版本号，就可以安装了，python的其他工具包低版本安装可以用同样方法

---

#### 6、下载安装

- 官方安装（不推荐）：[官方下载地址](https://www.python.org/)
  - 安装msi文件到本地
  - 系统环境变量中Path添加python的安装目录（安装过程可勾选自动配置环境变量）
  - cmd命令行：`python`进入python命令行
- 集成环境Anaconda安装（推荐）：[官方下载](https://www.anaconda.com/download/)
  - 选择合适的系统版本，为了避免一些不可预知的问题，Anaconda最好使用默认安装路径
  - 为了后期更好的支撑tensorflow，本教程采用anaconda3版本，python版本为3.5
