# face_detect
利用caffe训练人脸检测，并且标记人脸框

##   制作lmdb数据源文件

* [shell脚本启动链接](https://blog.csdn.net/lgk1996/article/details/80081516)
* 修改shell脚本 D:\face_detect-master\face-lmdb.sh   人脸检测06视频
* sh  D:\face_detect-master\face-lmdb.sh (shell 运行命令)

## 网络模型训练

* 修改train.prototxt模型框架文件

* 修改solver.prototxt模型的超参数文件

  ​	test_iter 一次测试要测试多少个batch 正常设置为测试集个数/batch_size

*  新建train.sh文件

* 执行train.sh shell脚本文件

* 利用已经训练好的模型，搭建人脸检测算法框架，python脚本

