---
layout:     post
title:      "在windows中尝试人脸识别"
subtitle:   "基于github开源项目face_recognition的人脸识别简单实现"
date:       2018-4-26
author:     "Zero"
cover: "/assets/in-post/face_recognition/bg.jpg"
categories: technology
tags: deeplearning
---


---

## 目录

* 纲要
{:toc}


---

## 在windows中尝试人脸识别

人脸识别作为人工智能的冰山一角，已经走进我们的现实生活，如人脸识别打卡、人脸识别手机解锁等

通过编码，使用深度神经网络训练一个精准的人脸识别模型绝非易事，但想要实现人脸识别功能却并不复杂

本文将带你在windows7系统中(windows10据说相差不大，未做实际尝试)，以Python为基础，使用 [face_recognition](https://github.com/ageitgey/face_recognition) 库，实现简单的人脸识别，具体功能：
- 定位一个图片中的人脸的位置
- 定位图片中人脸的各个面部特征，如下巴、左眉毛等
- 识别图中的人脸是谁（前提是提供已知人脸是谁的相关图片）

关于 [face_recognition](https://github.com/ageitgey/face_recognition) ：
- github上有很多关于人工智能的开源项目， [face_recognition](https://github.com/ageitgey/face_recognition) 就是其中之一
- 世界上最简单的人脸识别库
- 支持通过Python或命令行识别和操作人脸

### 1. 安装并配置所需环境

**安装Visual Studio Community 2017**

具体步骤百度

安装目录不要出现中文

安装过程，记得勾选windows SDK 10

**安装anaconda**

下载地址：https://www.anaconda.com/download/#windows

下载3.6版anaconda并安装

**安装cmake**

下载地址：https://cmake.org/download/

下载msi文件并安装

安装过程请勾选添加path到系统变量

这里的path只是添加到了环境变量Path中，CLASSPATH需要手动添加

cmd运行cmake --version验证是否正确安装

**安装face_recognition**

在Anaconda Prompt中依次执行如下命令：

conda create -n dlib python=3.5

conda activate dlib  

conda install -c menpo dlib=19.9  

pip install face_recognition

### 2. 定位图片中人脸的位置

使用方法：[`face_locations(img, number_of_times_to_upsample=1, model='hog')`](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_locations)

参数：
- img 数组形式的图像数据
- number_of_times_to_upsample 搜索次数，次数越多，可以识别到越小的脸
- model 默认为"hog"，使用CPU，跟快，但不够准确，可以指定为"cnn"，使用更准确的深度学习模型，但是需要GPU/CUDA环境

返回一个list，里面有多个tuple，每个tuple中包含四个整数值，表示上下左右像素的边界值

注意：本人在win7环境直接执行model为cnn的方法，12G内存占满，电脑卡住，GPU环境不完美的建议不要使用

GPU环境具备的还可以进行批量识别，参考[官方示例](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)

**简单使用**


```python
import face_recognition
# 将图片加载为数组形式
image = face_recognition.load_image_file("zero_and_ashley.jpg")
# 定位图中人脸的位置
face_locations = face_recognition.face_locations(image)
print(face_locations)
```

    [(558, 854, 825, 587), (468, 1062, 736, 795)]


**拓展示例**

为了展示直观，参考[官方示例](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)，改用matplotlib作图（matplotlib作图资料参考：[展示图片](https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py)，[画线](https://matplotlib.org/tutorials/advanced/path_tutorial.html#sphx-glr-tutorials-advanced-path-tutorial-py)），展示标记人脸后的图像


```python
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

# 需要识别的图像路径
image_path = "zero_and_ashley_and_so_on.jpg"
# 将图片加载为数组形式
image = face_recognition.load_image_file(image_path)
# 定位图中人脸的位置
face_locations = face_recognition.face_locations(image)

print("图中共有{}张人脸。".format(len(face_locations)))

# 使用matplotlib作图，展示原图
img = mpimg.imread(image_path)
fig, ax = plt.subplots(figsize=(img.shape[0]/100,img.shape[1]/100))
imgplot = ax.imshow(img)

# 遍历识别到的人脸位置信息
for face_location in face_locations:
    # 获得上、右、下、左人脸边界像素值
    top, right, bottom, left = face_location

    # 使用matplotlib在原图中用矩形框出人脸
    verts = [
       (left, bottom),  # left, bottom
       (left, top),  # left, top
       (right, top),  # right, top
       (right, bottom),  # right, bottom
       (left, bottom),  # ignored
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='cyan', ls='-', lw=2)
    ax.add_patch(patch)

# 展示图像
plt.show()
```

    图中共有11张人脸。



![png](/assets/in-post/face_recognition/output_4_1.png)


### 3. 定位图片中人脸的面部特征位置

使用方法：[face_landmarks(face_image, face_locations=None)](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.face_landmarks)

参数：
- face_image 数组形式的图像数据
- face_locations 可以将图像定位信息传入，识别途中特定人脸的各个面部特征

返回一个list，每个值是一个map，表示一个人脸，map中的key表示面部特征名称，value是一个list，包含多个多个tuple坐标点

**简单使用**


```python
import face_recognition
# 将图片加载为数组形式
image = face_recognition.load_image_file("zero_and_ashley.jpg")
# 定位图片中所有人脸的各个部位位置
face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list)
```

[{'top_lip': [(685, 752), (714, 749), (738, 748), (748, 752), (757, 751), (766, 757), (766, 765), (761, 765), (754, 760), (745, 759), (735, 757), (691, 754)], 'chin': [(583, 640), (580, 676), (582, 713), (589, 748), (608, 778), (635, 802), (664, 824), (695, 840), (722, 846), (744, 838), (758, 815), (773, 793), (789, 771), (804, 749), (814, 725), (816, 702), (813, 679)], 'right_eye': [(767, 665), (781, 656), (794, 658), (801, 668), (793, 669), (780, 667)], 'nose_bridge': [(759, 657), (761, 677), (764, 696), (767, 715)], 'left_eyebrow': [(656, 617), (677, 605), (701, 600), (725, 604), (744, 616)], 'right_eyebrow': [(776, 624), (791, 620), (806, 622), (815, 631), (816, 647)], 'bottom_lip': [(766, 765), (759, 784), (748, 794), (738, 794), (727, 790), (706, 778), (685, 752), (691, 754), (731, 777), (742, 779), (751, 778), (761, 765)], 'nose_tip': [(727, 728), (740, 733), (752, 737), (762, 737), (771, 733)], 'left_eye': [(677, 644), (694, 639), (709, 641), (719, 653), (707, 651), (692, 649)]}, {'top_lip': [(873, 655), (890, 655), (906, 652), (920, 656), (932, 651), (950, 654), (970, 655), (963, 659), (933, 663), (920, 665), (906, 663), (881, 659)], 'chin': [(797, 518), (799, 553), (805, 589), (811, 623), (823, 655), (842, 684), (863, 710), (889, 732), (923, 738), (957, 732), (987, 710), (1010, 681), (1029, 649), (1042, 615), (1048, 579), (1053, 541), (1054, 505)], 'right_eye': [(955, 526), (969, 516), (986, 514), (1001, 519), (987, 527), (970, 528)], 'nose_bridge': [(916, 525), (917, 554), (917, 581), (917, 610)], 'left_eyebrow': [(811, 497), (828, 484), (850, 483), (872, 486), (893, 493)], 'right_eyebrow': [(942, 489), (962, 478), (985, 471), (1010, 471), (1029, 485)], 'bottom_lip': [(970, 655), (951, 677), (934, 686), (920, 688), (905, 686), (889, 677), (873, 655), (881, 659), (906, 664), (920, 667), (933, 664), (963, 659)], 'nose_tip': [(891, 619), (904, 624), (919, 628), (934, 623), (948, 617)], 'left_eye': [(838, 529), (850, 522), (866, 521), (883, 530), (867, 534), (851, 534)]}]


**拓展示例**

为了展示直观，参考[官方示例](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)，改用matplotlib作图（matplotlib作图资料参考：[展示图片](https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py)，[画线](https://matplotlib.org/tutorials/advanced/path_tutorial.html#sphx-glr-tutorials-advanced-path-tutorial-py)），展示标记人脸面部特征后的图像


```python
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

# 需要识别的图像路径
image_path = "zero_and_ashley.jpg"
# 将图片加载为数组形式
image = face_recognition.load_image_file(image_path)
# 定位图片中所有人脸的面部特征位置
face_landmarks_list = face_recognition.face_landmarks(image)

print("图中共有{}张人脸。".format(len(face_landmarks_list)))

# 使用matplotlib作图，展示原图
img = mpimg.imread(image_path)
fig, ax = plt.subplots(figsize=(img.shape[0]/100,img.shape[1]/100))
imgplot = ax.imshow(img)

# 遍历识别到每个人脸
for face_landmarks in face_landmarks_list:

    # 面部特征map中的key
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    # 遍历每一个面部特征
    for facial_feature in facial_features:
        # 使用matplotlib在原图中用线框出面部特征
        verts = face_landmarks[facial_feature]
        # 出开始结尾的点，共有点的个数
        middle_point_num = len(verts)-2
        # 定义path中的起始点
        codes = [Path.MOVETO]
        # 定义path中的中间点
        for i in range(middle_point_num):
            codes.append(Path.LINETO)
        # 定义path中的结尾点
        if(verts[0]==verts[-1]):
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor='cyan', ls='-', lw=1)
        ax.add_patch(patch)

plt.show()
```

    图中共有2张人脸。



![png](/assets/in-post/face_recognition/output_8_1.png)


### 4. 识别图中的人脸是谁

使用方法：[compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)](https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.compare_faces)

参数：
- known_face_encodings 已知人脸的编码信息
- face_encoding_to_check 需要检测的图像编码信息
- tolerance 默认为0.6，越小对比越严格，默认值会使我做实验时总是识别不太准，改用0.4，很准确

返回一个list，每个元素为True/False，表示被检测图像中是否包含已知人脸，和known_face_encodings列表一一对应

**展示已知图像**


```python
fig= plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(121)
img = mpimg.imread("zero.jpg")
imgplot = ax1.imshow(img)
ax1.set_title("Zero")
ax2 = fig.add_subplot(122)
img = mpimg.imread("ashley.jpg")
imgplot = ax2.imshow(img)
ax2.set_title("Ashley")
plt.show()
```


![png](/assets/in-post/face_recognition/output_10_0.png)


**简单实例**


```python
import face_recognition

#加载已知的人脸图片，并获得编码数据
picture_of_me = face_recognition.load_image_file("zero.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# 加载另一个人的人脸图片，作为测试
unknown_picture = face_recognition.load_image_file("ashley.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# 对比my_face_encoding表示的人脸，是否出现在unknown_face_encoding的表示的图中
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding, tolerance=0.4)

if results[0] == True:
    print("图像中有zero")
else:
    print("图像中没有zero")
```

    图像中没有zero


**拓展示例**

为了展示直观，参考[官方示例](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)，改用matplotlib作图（matplotlib作图资料参考：[展示图片](https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py)，[画线](https://matplotlib.org/tutorials/advanced/path_tutorial.html#sphx-glr-tutorials-advanced-path-tutorial-py)，[添加文字](https://matplotlib.org/tutorials/text/text_intro.html#sphx-glr-tutorials-text-text-intro-py)），展示识别人脸后的图像，框出没张人脸并标记是谁


```python
import face_recognition
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

# 从照片中加载已知的两个人脸
zero_image = face_recognition.load_image_file("zero.jpg")
zero_face_encoding = face_recognition.face_encodings(zero_image)[0]

ashley_image = face_recognition.load_image_file("ashley.jpg")
ashley_face_encoding = face_recognition.face_encodings(ashley_image)[0]

known_face_encodings = [
    zero_face_encoding,
    ashley_face_encoding,
]
known_face_names = [
    "Zero",
    "Ashley",
]

# 加载需要识别的图片
image_path = "zero_and_ashley_and_so_on.jpg"
unknown_image = face_recognition.load_image_file(image_path)

# matplotlib作图
img = mpimg.imread(image_path)
fig, ax = plt.subplots(figsize=(img.shape[0]/100,img.shape[1]/100))
imgplot = ax.imshow(img)

# 找到图中所有人脸的位置
face_locations = face_recognition.face_locations(unknown_image)
# 根据位置加载人脸编码的列表
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# 遍历所有人脸编码，与已知人脸对比
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    # 获得对比结果
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
    # 获得姓名
    name = "Unknown"
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    # matplotlib构造框图
    verts = [
       (left, bottom),  # left, bottom
       (left, top),  # left, top
       (right, top),  # right, top
       (right, bottom),  # right, bottom
       (left, bottom),  # ignored
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='cyan', ls='-', lw=2)
    ax.add_patch(patch)
    # matplotlib 标记人名
    ax.text(left, top, name, style='italic',fontsize=20,
        bbox={'facecolor': 'cyan', 'edgecolor':'cyan','alpha': 1, 'pad': 2})

# 如果你的人名包含中文，可以加上这句，用于正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']

plt.show()
```


![png](/assets/in-post/face_recognition/output_14_0.png)
