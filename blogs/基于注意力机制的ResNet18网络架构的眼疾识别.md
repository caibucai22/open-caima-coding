# 1 介绍
眼疾是一种常见的眼部疾病，若不及时发现和治疗，会对视力造成严重影响。而通过机器学习技术，我们可以建立一个眼疾识别系统，帮助医生快速准确地诊断眼部疾病，提高诊断效率和准确性。 本项目旨在通过对眼底图像进行分类，实现眼疾的自动识别。数据集使用[iChallenge-PM](https://aistudio.baidu.com/aistudio/datasetdetail/88464)和[眼病分类数据集](https://aistudio.baidu.com/aistudio/datasetdetail/196662)，本文取上述两个数据集中的部分数据并已整理成**224*224**大小可直接使用。本文提出了**基于注意力机制的ResNet18网络的眼疾识别算法**。主要使用了ResNet18和RenNet18_NAM两种卷积神经模型对患者眼底视网膜图像进行眼底疾病识别，对2种模型的损失函数值、模型参数量和准确率进行对比实验分析。

# 2 加载数据集
```bash
unzip -o -q -d dataset data/data220613/dataset.zip
```
##  2.1 分割数据集
```python
from preproces_data import split_data
split_data(0.8)
```
## 2.2 加载数据到自定义的dataset
```python
from dataset import MyDataset

train_dataset = MyDataset(csv_filepath='train.csv')
test_dataset = MyDataset(csv_filepath='test.csv')
```

# 3 模型构建
**本文使用ResNet18和ResNet18-NAM两个模型进行实验**

ResNet18-NAM是基于归一化的注意力机制的ResNet18模型，模型构建参考了[【AI达人特训营】ResNet50-NAM：一种新的注意力计算方式复现](https://aistudio.baidu.com/aistudio/projectdetail/4204121)
NAM是一种轻量级的高效的注意力机制，采用了CBAM的模块集成方式，重新设计了通道注意力和空间注意力子模块，这样，NAM可以嵌入到每个网络block的最后。对于残差网络，可以嵌入到残差结构的最后。对于通道注意力子模块，使用了Batch Normalization中的缩放因子，如式子（1），缩放因子反映出各个通道的变化的大小，也表示了该通道的重要性。为什么这么说呢，可以这样理解，缩放因子即BN中的方差，方差越大表示该通道变化的越厉害，那么该通道中包含的信息会越丰富，重要性也越大，而那些变化不大的通道，信息单一，重要性小。

<center>
    <img src="https://www.caima.tech/wp-content/uploads/2024/08/post-65-66b8945eb080d." width="100%">
    
</center>

其中 $\mu_B$ 和 $\sigma_B$ 为均值，$B$ 为标准差，$\gamma$ 和 $\beta$ 是可训练的仿射变换参数（尺度和位移）[参考Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf).

通道注意力子模块如图(1)和式(2)所示：
<center>
    <img src="https://www.caima.tech/wp-content/uploads/2024/08/post-65-66b8945edf846." width="100%">
    
</center>

其中$M_c$表示最后得到的输出特征，$\gamma$是每个通道的缩放因子，因此，每个通道的权值可以通过 $W_\gamma =\gamma_i/\sum_{j=0}\gamma_j$ 得到。我们也使用一个缩放因子 $BN$ 来计算注意力权重，称为**像素归一化**。像素注意力如图(2)和式(3)所示：


<center>
    <img src="https://www.caima.tech/wp-content/uploads/2024/08/post-65-66b8945f1b4fa." width="100%">
    
</center>


为了抑制不重要的特征，作者在损失函数中加入了一个正则化项，如式(4)所示。

```python
import paddle
from train_and_test import train
from model import resnet18
from dataset import MyDataset
import warnings
warnings.filterwarnings("ignore")
net = resnet18(num_classes=6)
paddle.summary(net,(64,3,224,224))
```

# 4 模型训练

```python
from train_and_test import train, test

save_path='./google/'

batch_size=32

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size)

eval_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)

optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())


train(
    model=net,
    opt=optim, 
    train_loader=train_loader, 
    valid_loader=eval_loader, 
    epoch_num=100, 
    save_path=save_path, 
    save_freq=20
)
```
**output**

<center>
<img src="https://www.caima.tech/wp-content/uploads/2024/08/post-65-66b8945f5a04c.png" style="margin:0 auto;zoom: 100%;" />

<font class="center" face="黑体" size=4.>图1 训练过程中的准确率</font>
</center>

<center>
<img src="https://www.caima.tech/wp-content/uploads/2024/08/post-65-66b8945f88e73.png" style="margin:0 auto;zoom: 100%;" />

<font class="center" face="黑体" size=4.>图2 训练过程中的损失函数</font>
</center>







# 5 模型评估
``` python
from train_and_test import test
from model import resnet18;
net=resnet18(num_classes=6)
save_path='./resnet18-nam/'

test(
    model_path=save_path+'model/final.pdparams',
    net=net,
    test_dataloader=paddle.io.DataLoader(MyDataset(csv_filepath='test.csv'),
                                         batch_size=32),
    save_path=save_path
)
```

**output**
```
acc-> 0.9528
precision--> ([0.9221, 0.9828, 0.9032, 0.9649, 0.9636, 1.0], 0.9561000000000001)
recall--> ([0.9342, 0.9344, 0.9333, 0.9821, 0.9636, 0.9808], 0.9547333333333334)
```

<center>
<img src="https://www.caima.tech/wp-content/uploads/2024/08/post-65-66b8945fb9440.png" style="zoom: 100%" >
<font class="center" face="黑体" size=4.>图3 混淆矩阵</font>
</center>