# 1 摘要

UWB（Ultra Wide Band，超宽带）信号与其他无线信号相比具有大带宽，功耗低，精度高，穿透能力强等特性，因此其在室内定位领域有着十分广泛的应用。与其他室内定位算法相比，基于UWB的室内定位算法的定位精度可以达到厘米级别，UWB室内定位算法的误差主要来源于NLOS（Non Line Of Sight，非视距）信号。本文采用CNN（Convolutional Neural Network，卷积神经网络）、FCN（Full Convolutional Network，全卷积神经网络）和ResNet（Residual Network，残差网络）三种不同的神经网络识别NLOS信号。为了使神经网络分类模型有更好的鲁棒性，三种神经网络模型直接使用CIR（Channel Impulse Responses，信道冲激响应）原始数据作为神经网络的输入，并且在神经网络模型中加入了Dropout层避免模型过拟合。实验结果表明，在测试集上三个分类模型识别NLOS信号的准确率都达到了80%以上。

## 1.1 NLOS介绍

在基于UWB的室内定位系统中，**UWB信号的传播信道分为两类，一类是LOS信道，另一类是NLOS信道**。其中LOS信道指的是信号在传输过程中没有经过障碍物，无线信号直接从发送器传输到接收器并且在传输过程中几乎不受干扰，传输路径如图1所示。与LOS信道相比，NLOS信道指的是信号在传输过程中由于受到障碍物的阻挡，不能从发射器直线传输到接收器，而是经过周围环境中其他物体的衍射、散射、反射等现象后才能到达接收器。通常NLOS信道的传输路径非常复杂，无线信号在传输的过程中受到干扰并且有严重的衰减情况，其传输路径如图2所示。**由于NLOS信道传输路径复杂，所以基于UWB的室内定位系统使用LOS信号进行定位，在开始定位前需要先对接收器收到的信号进行识别，将NLOS信号剔除，仅使用LOS信号进行定位，这样所获得的定位精度会更好。**因此NLOS信号识别成为许多研究者广泛关注的问题。

<center>
    <img src="https://img-blog.csdnimg.cn/img_convert/d0f96c8f25c9a3d8435d3f990985e976.png" style="margin:0 auto;zoom: 30%;" />
    <br/>
    <font color="black">图1 LOS路径</font>
    <br/>
    <img src="https://img-blog.csdnimg.cn/img_convert/ee04688311b080e31af062b1051f894d.png" style="zoom:30%;" />
    <br/>
    <font color="black">图2 NLOS路径</font>
</center>


## 1.2 本文工作

UWB（Ultra Wide Band，超宽带）信号与其他无线信号相比具有大带宽，功耗低，精度高，穿透能力强等特性，因此其在室内定位领域有着十分广泛的应用，与其他室内定位算法相比基于UWB的室内定位算法的定位精度可以达到厘米级别。UWB室内定位算法的误差主要来源于NLOS（Non Line Of Sight，非视距）信号，针对基于特征的NLOS识别方法需要手动提取特征且识别效果较差这一问题，有学者提出了CNN（Convolutional Neural Network，卷积神经网络）、FCN（Full Convolutional Network，全卷积神经网络）和ResNet（Residual Network，残差网络）等不同的神经网络识别NLOS信号。**本文对三种基于卷积神经网络的NLOS信号分类器进行了分析与评估，实验结果表明当Dropout层参数取0.4到0.6之间时模型的NLOS信号识别率较高；当使用ReLU作为激活函数时模型的NLOS信号识别率更好，其中CNN模型的NLOS信号识别率达到了92.64%。**与其他两种基于特征的机器学习方法相比，基于卷积神经网络的NLOS信号分类器效果更好，且ResNet分类器效果最优。

# 2 数据集介绍

## 2.1 数据集介绍

本文所使用的数据集来自eWINE（Elastic Wireless Networking Experimentation，动态无线网络实验）项目，该项目是欧盟“地平线2020”研究计划的子课题，该数据集内容如表1所示。[数据集 Github 地址](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set)

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>


<center>表1 数据集内容</center>

<div class="center">


| 场景         | LOS样本个数 | NLOS样本个数 |
| ------------ | ----------- | ------------ |
| 办公室1      | 3000        | 3000         |
| 办公室2      | 3000        | 3000         |
| 小型公寓     | 3000        | 3000         |
| 小型车间     | 3000        | 3000         |
| 带客厅的厨房 | 3000        | 3000         |
| 卧室         | 3000        | 3000         |
| 锅炉房       | 3000        | 3000         |

</div>

该数据集使用型号为DWM1000的射频收发器采集室内UWB信号，数据集包括两个办公室场景、小型公寓、小型车间、带客厅的厨房、卧室和锅炉房共7种不同环境下的UWB信号，并且在每个环境选择不同的位置进行采集，每个场景采集了3000条LOS数据和3000条NLOS数据。数据集总共采集了42000条数据，包括21000条LOS数据，21000条NLOS数据。该数据集由7个CSV格式文件组成，每个文件存放不同场景下采集到的6000条UWB数据，每个数据集文件都有1024列数据，其中包括是否为NLOS信号、CIR0到CIR1015共1016个CIR数据、飞行时间、噪声标准差、前导码长度等特征数据。**本文使用1016个CIR数据作为神经网络的输入，并且使用所有场景下采集到的42000条数据。**

## 2.2 数据预处理

根据eWINE项目的提示，使用数据集中的CIR数据需要除以所获取的前导码样本数量，且本文对于每行数据仅使用是否为NLOS信号和CIR数据共1017条数据。**本文将7个数据集文件按照eWINE项目的提示先对CIR数据进行处理，然后将未使用的数据删除只留下需要使用的1017列数据**，为了提高模型训练速度，本文对处理后的CIR数据进行归一化处理。经过以上操作得到本文的原始数据集。**本文按照8:1:1的比例将数据集划分为训练集、验证集和测试集**，并且为了使训练的模型更具泛化性，本文先将所有数据集文件读取到Python二维数组中，接着对数组进行打乱，然后按照比例将数组划分为三个数据集，并且将三个数据集数组分别保存到不同的文件中。

# 3 实验介绍

## 3.1 环境配置

|   实验环境   |        版本        |
| :----------: | :----------------: |
|   操作系统   | Ubuntu 16.04.6 LTS |
|     GPU      |     Tesla V100     |
|     IDE      |    BML Codelab     |
|    Python    |       3.9.16       |
| 深度学习框架 | PaddlePaddle 2.4.1 |

## 3.2 项目结构

```
.
├── README.md                               # 说明文档
├── dataset                        			
│     ├──original                           # 原始数据集
│     ├──test                               # 测试集
|     ├──train                              # 训练集
|     └──val                                # 验证集
|
├── model									
│     ├──CNN.pdarams                        # CNN模型权重文件
|     ├──FCN.pdarams                        # FCN模型权重文件
|     └──ResNet.pdarams                     # ResNet模型权重文件
|
├── activate_function_experiments.ipynb     # 激活函数实验 
├── dataset.py                              # 自定义数据集类
├── divide_dataset.py                       # 划分数据集
├── dropout_experiment.ipynb                # Dropout层实验
├── model.py                                # 模型
├── test.ipynb                              # 模型评估
└── train_and_test.py                       # 训练和测试函数
```


## 3.3 模型架构

<center>
    <img src="https://img-blog.csdnimg.cn/img_convert/749c61deade4419b3e2250a947eb19e8.png" style="margin:0 auto;zoom: 50%;" />
    <br/>
    <font color="black">图3 FCN模型</font>
    <br/>   
    <br/> 
    <img src="https://img-blog.csdnimg.cn/img_convert/12ba8abdccbe455e2e69eca5b8c73319.png" style="margin:0 auto;zoom: 50%;" />
    <br/>
    <font color="black">图4 CNN模型</font>
    <br/> 
    <br/>    <img src="https://img-blog.csdnimg.cn/img_convert/8ecbbceaab94a3bd6aefa5f0691790c6.png" style="margin:0 auto;zoom: 50%;" />
    <br/>
    <font color="black">图5 ResNet模型</font>
    <br/>
</center>


本文共评估了三个卷积神经网络模型，分别是全卷积网络、卷积神经网络和残差网络。三种模型架构如图3到图5所示。





## 3.4 实验步骤

### 3.4.1 划分数据集

运行 `divide_dataset.py` 文件后，会将 `dataset/original` 文件夹下的所有数据集全部加载，打乱后按 `8:1:1` 的比例划分训练集、验证集和测试集。

### 3.4.2 训练模型

`train_and_test.py` 文件中提供了 `train` 函数训练模型。

```python
def train(net, optimizer, epochs, batch_size, train_loader, val_loader, loss_function, save_path)
'''
	net				模型
    optimizer		优化器
    epochs			训练轮次
    batch_size		批大小
    train_loader	训练集加载器
    val_loader		验证集加载器
    loss_function	损失函数
    save_path		模型及评价指标保存路径
'''
```

例如训练模型 `MyResNet_ReLU(dropout_rate=0.5)` 代码如下：

```python
import model
import paddle
from train_and_test import train
from dataset import MyDataset

net = model.MyResNet_ReLU(dropout_rate=0.5)

batch_size = 128

train_dataset = MyDataset('./dataset/train')
eval_dataset = MyDataset('./dataset/val')

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=batch_size)

train(net,
      epochs=300,
      optimizer=paddle.optimizer.Adam(learning_rate=0.0000001, parameters=net.parameters()),
      batch_size=batch_size,
      train_loader=train_loader,
      val_loader=eval_loader,
      loss_function=paddle.nn.CrossEntropyLoss(),
      save_path='./myresnet_relu_128_0.0000001_300/')
```

### 3.4.3 评估

`train_and_test.py` 文件中提供了 `test` 函数训练模型。

def train(net, optimizer, epochs, batch_size, train_loader, val_loader, loss_function, save_path)

```python
def test(dataloader, model):
'''
	dataloader	数据加载器
    model		模型
'''
```

例如测试 `FCN` 模型的代码如下：

```python
import paddle
import model
from dataset import MyDataset
from train_and_test import test

weight = paddle.load('model/FCN.pdparams')

fcn = model.MyFCN_ReLU()

fcn.set_state_dict(weight)

test_dataset = MyDataset('./dataset/test')
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=True)

result = test(test_loader, fcn)
print(result)

```
此项目包含完整的源代码和参考论文，需要的请后台联系。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b2737d4e0c946f088b863cc90882841.png)



我们的网站是[菜码编程](https://www.caima.tech)。
如果你对我们的项目感兴趣可以扫码关注我们的公众号，我们会持续更新深度学习相关项目。
![公众号二维码](https://i-blog.csdnimg.cn/direct/ede922e7158646bcaf95ec7eefc67e49.jpeg#pic_center)

