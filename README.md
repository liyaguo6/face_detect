# face_detect
利用caffe训练人脸检测，并且标记人脸框

### 目的：检测到人脸框

 ![result1.jpg](D:\face_detect-master\result1.jpg)

### 一、准备数据

#### (1) 数据获取

1.face detection benchmark -- 行业基准(数据库、论文、源码、结果)。  
2.优秀论文，通常试验阶段会介绍它所使用的数据集，公开数据集可以下载。(申请数据集的时候，最好使用学校的邮箱)  
3.论坛或者交流社区，如thinkface

数据规模，越大越好：本案例40000+

#### (2) 二分类数据，第一类人脸，第二类非人脸

人脸数据：路径/xxx.jpg      60，80，280，320  
非人脸数据：只要不是人脸都可以

#### (3) IOU(Intersection of Union)：

IOU < 0.3 : 负样本--非人脸  
IOU > 0.7 : 正样本--人脸(提高泛化能力)  
其他 ：抛弃  

对于正样本：裁剪操作，根据标注把人脸裁剪出来。可以用opencv工具完成制作人脸数据。要检查数据有没有问题。  
对于负样本：进行随机裁剪，通过设置IOU < 0.3认为是负样本，最好使用没有人脸数据的当作负样本。

`第一步结束，我们得到已经准备好的人脸与非人脸图像，准备生成二分类lmdb数据源`

### 二、制作lmdb数据源文件

#### (1)写两个txt文档文件：(将人脸与非人脸图像转换为txt文档保存)

​	Train.txt  
	0/xxx.jpg 0  
	1/xxx.jpg 1  

​	Val.txt  
	xxx.jpg 0  
	xxx.jpg 1

#### (2) face_lmdb.sh:

* [shell脚本启动链接](https://blog.csdn.net/lgk1996/article/details/80081516)

* 修改shell脚本 D:\face_detect-master\face-lmdb.sh   人脸检测06视频

  修改数据集路径；  

  ```
  EXAMPLE=D:/face_detect-master
  DATA=D:/face_detect-master
  TOOLS=D:/caffe-master/Build/x64/Release
  
  TRAIN_DATA_ROOT=D:/face_detect-master/train/
  VAL_DATA_ROOT=D:/face_detect-master/val/
  ```

  Resize图片大小（如AlexNet或者VGG通常都是Resize为227*227）；

  ```
  # Set RESIZE=true to resize the images to 227 x 227. Leave as false if images have
  # already been resized using another tool.
  RESIZE=true
  if $RESIZE; then
    RESIZE_HEIGHT=227
    RESIZE_WIDTH=227
  else
    RESIZE_HEIGHT=0
    RESIZE_WIDTH=0
  fi
  ```

  指定train.txt和val.txt文件位置以及生成的lmdb的路径；

  ```
  echo "Creating train lmdb..."
  
  GLOG_logtostderr=1 $TOOLS/convert_imageset.exe \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      --shuffle \
      $TRAIN_DATA_ROOT \
      $DATA/train.txt \
      $EXAMPLE/face_train_lmdb
  
  echo "Creating val lmdb..."
  
  GLOG_logtostderr=1 $TOOLS/convert_imageset.exe \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      --shuffle \
      $VAL_DATA_ROOT \
      $DATA/val.txt \
      $EXAMPLE/face_val_lmdb
  
  echo "Done."
  #Status API Training Shop Blog About
  ```

* sh  D:\face_detect-master\face-lmdb.sh (shell 运行命令)

  `第二步结束，我们得到原始数据集的lmdb数据源，准备进行网络训练`

## 三、训练Alexnet网络

#### （1）修改train.prototxt模型框架文件

- 开头：该文件定义的是训练和验证时候的网络模型，所以在开始的时候要定义训练集和验证集的来源。  
- 结尾：如果是一般卷积网络的话，最后都使用全连接层，将feature map 转成固定长度的向量，然后输出种类的个数。所以在最后的时候，需要说明输出种类的个数。  因为这里面包含了验证的部分，验证的时候，需要输出结果的准确率，所以需要定义准确率的输出。  因为是训练模型，所以包括forward和backward，所以最后需要定义一个损失函数。  

####   (2) AlexNet-Styled architecture

​	![AlexNet.jpg](https://github.com/ztoString/ImageRepository/raw/master/Face_Detect_caffe/AlexNet.jpg)

####   (3)  修改solver.prototxt模型的超参数文件

- test_iter 一次测试要测试多少个batch 正常设置为测试集个数/batch_size
- 最好使得test_iter * batch_size = 测试样本总个数  
- base_lr: 0.001基础学习率（非常重要！）不能太大

####   (4) 执行train.sh shell脚本文件

```D:/caffe-master/Build/x64/Debug/caffe.exe train --solver=D:/face_detect-master/solver.prototxt #\#--weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \#--gpu 0 
D:/caffe-master/Build/x64/Debug/caffe.exe train --solver=D:/face_detect-master/solver.prototxt #\
#--weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
#--gpu 0 
```

得到训练之后的model

`第三步结束，我们得到训练结束Model,准备利用该Model进行图像测试`

### 四、测试

利用已经训练好的模型，搭建人脸检测算法框架，python脚本

#### (1) 配置deploy.prototxt

该配置文件适用于部署，也就是用于实际场景时候的配置文件

与train_val.prototxt文件的区别在于：

- 去掉了数据层  
- 开头：不必在定义数据集的来源，但是需要定义输入数据的大小格式（在python代码中也进行了相应的修改）。  
- 中间部分：train_val.prototxt 和 deploy.prototxt中间部分一样，定义了一些卷积、激活、池化、Dropout、LRN(local response normalization)、全连接等操作。  
- 结尾：因为只有forward，所以定义的是Softmax，也就是分类器，输出最后各类的概率值。  

#### (2) 编写人脸检测代码(face_detect.py) -- 多尺度的人脸检测：

- model 转换成全卷积  
- 多个scale  
- 前相传播，得到特征图，概率值矩阵  
- 反变换，映射到原图上的坐标值  
- NMS 非极大值抑制  

#### (3) 补充

最后，还想说一个小问题，在运行程序的时候，会发现运行过程非常慢，why?

因为我们在程序中做了多个scale变换，每个scale都要进行一次前向传播

解决方案：采取级联的网络，再加上矫正网络。

推荐一篇经典论文：`A Convolutional Neural Network Cascade for Face Detection`

#### (4)模型准确率影响因素：

- 模型：

当模型准确率达到饱和时，可以尝试使用更深的网络结构，可以进一步提高准确率。（AlexNet --> VGG）

- 数据：（opencv/keras）

对原始数据进行偏移、翻转、镜像变换等数据增强，增大数据量。

- 训练：

例如 -- 当模型训练到第50000次时，得到的准确率最高，在之后的学习过程中，由于过拟合现象导致准确率下降，那么我们可以对第50000次的模型进行fine-tuning(微调)：拿到第50000次生成的模型，在train.sh文件中，指定权重参数为第50000次网络迭代的结果，然后降低solver.prototxt中的学习率。