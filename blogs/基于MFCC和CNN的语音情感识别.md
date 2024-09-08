# 1  介绍

语音情绪识别是音频分类的一个最重要的应用场景，在社会中的很多领域，例如机器人制造，自动化，人机交互、安全、医疗、驾驶和通信等，情绪识别都具有很高的实用价值。

我们今天要讲解的案例——语音情感识别在近年来引起了学术界和工业界的研究热潮。因为情绪作为我们在日常交流中非常重要的表达方式之一，在无法获取说话人面部表情的情况下，音频就成为了理解语言情感含义中不可或缺的一部分。

**本文利用飞桨框架实现的ResNet18模型，实现6种语音情绪的识别**，让大家亲自感受音频特征提取与音频情绪识别的魅力。

# 2  数据集介绍

本文使用[开源数据集](https://aistudio.baidu.com/aistudio/datasetdetail/180007)，该数据集包含生气、恐惧、开心、正常、伤心、惊讶；6类数据，每类数据50个样本，共300条数据。具体情况如表1所示。


<!-- 让表格居中显示的风格 -->

<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>


<font class="center" face="黑体" size=2.>表1 数据集详细信息</font>



| 类别 | 样本个数 |
| :--: | :------: |
| 生气 |    50    |
| 恐惧 |    50    |
| 开心 |    50    |
| 正常 |    50    |
| 伤心 |    50    |
| 惊讶 |    50    |



# 3 数据预处理

数据集中数据为.wav格式的音频文件，本文使用**librosa库（版本0.9.2）**对音频文件进行处理，**使用MFCC算法提取特征并转化为图片形式保存**。

## 3.1 解压数据集

```bash
unzip -o "data/data221024/wav.zip" -d ./wav
```

## 3.2 数据预处理

本文使用的ResNet18模型对转化后的图片进行分类，所以需要对图片尺寸进行处理，本文将生成的图片统一缩放到**224\*224**大小。

### 3.2.1 wav转jpg图片

```python
from data_preprocess import wav2img,image_resize,dataset_partition
# wav转jpg
wav2img()
```

### 3.2.2 图片缩放到224*224

```python
from data_preprocess import image_resize
# 图片缩放
image_resize()
```


## 3.3 数据集划分

按照 8:2 比例划分训练集和验证集，**在主目录下生成train.txt和val.txt**
    

```python
from data_preprocess import dataset_partition
# 按照 8:2 比例划分训练集和验证集
dataset_partition()
```

# 4 加载数据集和网络模型

## 4.1 加载数据集

```python
from mydataset import WAVDataset

# 加载数据集
train_dataset = WAVDataset('train.txt')
val_dataset = WAVDataset('val.txt')
```

## 4.2 加载模型

```python
import warnings
warnings.filterwarnings("ignore")

import paddle
from paddle.vision.models import resnet18
from mydataset import WAVDataset

# 加载预训练的ResNet18模型
net = resnet18(pretrained=True, num_classes=6)

# 模型保存路径
save_path='./model/resnet18/'
```

# 5 模型训练

## 5.1 设置训练超参数

```python
epochs=50

batch_size = 128

optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size)
eval_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size)

## 开始训练
from train_and_test import train
train(
    model=net,
    opt=optim,
    train_loader=train_loader,
    valid_loader=eval_loader,
    epoch_num=epochs,
    save_path=save_path,
    save_freq=20
)
```

```
start training ... 
epoch: 0, accuracy/loss: 0.8571428656578064/0.8857352137565613
[validation] accuracy/loss: 0.1666666716337204/14.263507843017578
epoch: 1, accuracy/loss: 0.7349330186843872/1.2071239948272705
[validation] accuracy/loss: 0.25/6.640896797180176
epoch: 2, accuracy/loss: 0.8638392686843872/0.4662819504737854
[validation] accuracy/loss: 0.18333333730697632/6.875960826873779
...
epoch: 48, accuracy/loss: 1.0/4.671791612054221e-05
[validation] accuracy/loss: 0.8666666746139526/0.5725888609886169
epoch: 49, accuracy/loss: 1.0/4.563992115436122e-05
[validation] accuracy/loss: 0.8666666746139526/0.5731116533279419
```

<center>
<img src="https://img-blog.csdnimg.cn/img_convert/740925687a08680464e09d46d172f978.png" style="margin:0 auto;zoom: 100%;" />


<font class="center" face="黑体" size=4.>图1 训练过程中的准确率</font>
</center>

<center>
<img src="https://img-blog.csdnimg.cn/img_convert/d0f8c24484ef8091c6b01e5fb17071f0.png" style="margin:0 auto;zoom: 100%;" />
<font class="center" face="黑体" size=4.>图2 训练过程中的损失函数</font>
</center>


## 5.2 测试模型

```python
from train_and_test import test

val_dataset = WAVDataset('val.txt')
test(
    model_path=save_path+'model/final.pdparams',
    net=net,
    test_dataloader=paddle.io.DataLoader(val_dataset,batch_size=64),
    save_path=save_path
)
```

```
acc-> 0.8667
precision--> ([1.0, 0.5, 1.0, 1.0, 1.0, 0.7], 0.8666666666666667)
recall--> ([1.0, 1.0, 1.0, 0.7143, 0.7143, 1.0], 0.9047666666666667)
```

<center>
<img src="https://img-blog.csdnimg.cn/img_convert/740106321a43045be62efee6423a68bf.png" style="zoom: 100%" >
<font class="center" face="黑体" size=4.>图3 混淆矩阵</font>
</center>

我们的网站是[菜码编程](https://www.caima.tech)。 [https://www.caima.tech](https://www.caima.tech)
如果你对我们的项目感兴趣可以扫码关注我们的公众号，我们会持续更新深度学习相关项目。
![公众号二维码](https://i-blog.csdnimg.cn/direct/ede922e7158646bcaf95ec7eefc67e49.jpeg#pic_center)
