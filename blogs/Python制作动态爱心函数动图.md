# Python制作动态爱心函数动图
用Python制作很火的动态爱心动图效果如下
![在这里插入图片描述](https://www.caima.tech/wp-content/uploads/2024/08/post-42-66b3812d323e7.gif)


```python
import numpy as np
from matplotlib import pyplot as plt
import os
import imageio.v2 as imageio

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
x = np.arange(-3, 3, 0.001)  # 生成等差数组
PI = np.pi
alpha = 2.5
filenames = []

while alpha <= 24:
    # 保留小数点后2位
    alpha = np.round(alpha, 2)
    # 设置坐标轴
    plt.axis([-3, 3, -2, 4])
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title("α = " + str(alpha))
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    y = np.power(np.abs(x), 2 / 3) + 0.9 * np.power(3.3 - (x * x), 0.5) * np.sin(alpha * PI * x)
    plt.plot(x, y, color='red')
    filename = str(alpha) + '.png'
    filenames.append(filename)
    # 当前目录保存图片
    plt.savefig(filename)
    # 清除画板
    plt.clf()
    alpha = alpha + 0.01

# 动图的每一帧
frames = []
# 利用方法append把图片挨个存进列表
for image_name in filenames:
    frames.append(imageio.imread(image_name))

# 保存为gif格式的图
imageio.mimsave('heart.gif', frames, 'GIF')
# 删除生成的动图
for image_name in filenames:
    os.remove(image_name)
print("ok")


```
**带进度条版**

```python
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
x = np.arange(-1.8, 1.8, 0.001)  # 生成等差数组
PI = np.pi
alpha = 2.5
max_alpha = 24
filenames = []
gap = 0.1
with tqdm(total=int((max_alpha - alpha) / gap) + 1) as tqdm_obj:
    tqdm_obj.set_description('正在生成每帧图片')
    while alpha <= max_alpha:
        # 保留小数点后2位
        alpha = np.round(alpha, 2)
        # 设置坐标轴
        plt.axis([-3, 3, -2, 4])
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title("α = " + str(alpha))
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        y = np.power(np.abs(x), (2 / 3)) + 0.9 * np.power(3.3 - (x * x), 0.5) * np.sin(alpha * PI * x)
        # print(y)
        plt.plot(x, y, color='red')
        filename = str(alpha) + '.png'
        filenames.append(filename)
        # 当前目录保存图片
        plt.savefig(filename)
        # 清除画板
        plt.clf()
        alpha = alpha + gap
        tqdm_obj.update()

with tqdm(total=filenames.__len__() + 1) as tqdm_obj2:
    tqdm_obj2.set_description('正在合成GIF动图')
    # 动图的每一帧
    frames = []
    # 利用方法append把图片挨个存进列表
    for image_name in filenames:
        frames.append(imageio.imread(image_name))
        tqdm_obj2.update()
    # 保存为gif格式的图
    imageio.mimsave('heart.gif', frames, 'GIF', duration=0.1)
    tqdm_obj2.update()

with tqdm(total=filenames.__len__()) as tqdm_obj3:
    tqdm_obj3.set_description('正在删除每帧图片')
    # 删除生成的动图
    for image_name in filenames:
        os.remove(image_name)
        tqdm_obj3.update()
heart_gif_path = os.path.dirname(os.path.abspath(__file__))
heart_gif_path = os.path.join(heart_gif_path, 'heart.gif')
print("已生成GIF动图路径: " + heart_gif_path)

```

我们的网站是[菜码编程](https://www.caima.tech)。 [https://www.caima.tech](https://www.caima.tech)
如果你对我们的项目感兴趣可以扫码关注我们的公众号，我们会持续更新深度学习相关项目。
![公众号二维码](https://i-blog.csdnimg.cn/direct/ede922e7158646bcaf95ec7eefc67e49.jpeg#pic_center)
