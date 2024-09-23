# 如何用3D折线图直观展示多维数据变化

在数据分析中，我们经常需要展示多个维度的数据变化。最近，我用Python的Matplotlib库绘制了一个非常直观的3D折线图，展示了不同维度的数据随时间的变化。这种图不仅能应用在深度学习模型的训练中，也适用于各种场景下的数据可视化！📊

![3D折线图展示多维数据变化](https://raw.githubusercontent.com/henu77/typoryPic/main/2024/main_0_0.png)

## 1 👩‍💻 主要思路：

1. 数据准备：首先，我根据不同维度的数据生成了一个网格，用来表示每个维度随时间或其他变量的变化。这里我举了深度学习的例子，将批次大小（Batch Size）、训练轮次（Epochs）和准确率（Accuracy）作为变量进行展示，但其实你可以替换为任何数据，比如销售量、时间和收入等。

2. 3D折线图绘制：通过Matplotlib的3D绘图功能，可以展示任意两个维度与第三个维度的关系。每条线代表一个维度（如不同的批次大小、不同的时间点），并随着第三个维度（如准确率、收益）的变化绘制成曲线。

3. 投影和填充：为了增强图形的可读性，我为每条3D曲线添加了投影，让数据在XY平面上也有所展示。通过多边形填充曲线和XY平面之间的区域，使数据的变化更加直观，视觉效果更好。

4. 重点数据标注：在特定的数据点上（如某个时间点或特殊变量值），我用散点和数值标注突出显示这些数据，帮助你更清楚地了解不同维度下数据的具体变化。

## 2 💡 代码亮点：

下面这段代码用于绘制3D折线图，展示不同批次大小和训练轮次下的准确率变化。这里我用了Matplotlib的3D绘图工具包，通过设置不同的颜色映射和线宽，让图形更加美观。同时，我还用虚线连接了特定时间点下不同批次大小的准确率，帮助你更好地理解数据的变化趋势。

```python
# 绘制3D图的核心代码
ax.plot(x_vals, [y]*len(x_vals), z_vals, label=f'Variable {y}', color=cmap(norm(y)), linewidth=line_width)
# 创建投影和填充区域
verts = [(x, y, z) for x, z in zip(x_vals, z_vals)]
verts += [(x, y, 0) for x in x_vals[::-1]]
poly = Poly3DCollection([verts], color=cmap(norm(y)), alpha=0.3)
ax.add_collection3d(poly)
```

## 3 📈 可应用的场景：

- 时间序列数据：比如展示不同时段下某个产品的销售额变化。
- 市场分析：对比不同地区、不同产品线的销售额趋势。
- 金融数据：展示多个股票、基金或其他金融指标随时间的变化。
- 科研分析：比如不同实验条件下测量值的变化。

## 4 ✨ 可视化小技巧：

- 颜色映射（Colormap）可以让你一眼就看出不同维度的数据变化，图形更加清晰易读。
- 可以重点标注某些特定数据点（如峰值、最低点），让数据分析更直观。

如果你正在处理多维度的数据，推荐大家试试这种3D折线图！它能帮你更好地理解数据随不同变量变化的趋势，适合应用于各类数据可视化场景！🧐

## 5 🚀 完整代码：
```python
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘制图形
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具包
import numpy as np  # 导入numpy库用于数组和数值计算
import matplotlib.cm as cm  # 导入matplotlib的颜色映射模块
import matplotlib.colors as mcolors  # 导入颜色规范模块
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 导入3D多边形集合，用于填充区域

# 数据定义
batch_sizes = [5, 10, 15, 20, 25]  # 不同的batch size
epochs = [10, 20, 30, 40, 50, 60, 70, 80]  # 不同的epoch
# 需要展示变化的Epoch
show_epochs = [20, 50, 80]  # 特定的epoch用于重点展示
accuracies = [  # 模拟不同batch size和epoch下的准确率
    [0.21, 0.22, 0.24, 0.34, 0.44, 0.54, 0.61, 0.75],
    [0.23, 0.30, 0.38, 0.40, 0.60, 0.72, 0.84, 0.92],
    [0.14, 0.22, 0.38, 0.44, 0.70, 0.77, 0.85, 0.93],
    [0.15, 0.25, 0.40, 0.43, 0.65, 0.72, 0.84, 0.93],
    [0.16, 0.28, 0.38, 0.44, 0.68, 0.77, 0.84, 0.93]
]

# 创建网格
X, Y = np.meshgrid(epochs, batch_sizes)  # 创建X-Y网格
Z = np.array(accuracies)  # 将准确率数据转化为numpy数组

# 绘制3D图形
fig = plt.figure(figsize=(10, 8))  # 创建一个10x8英寸的图形
ax = fig.add_subplot(111, projection='3d')  # 添加3D坐标轴

# 创建颜色映射
norm = mcolors.Normalize(vmin=min(batch_sizes), vmax=max(batch_sizes))  # 归一化batch size，用于颜色映射
cmap = cm.viridis  # 使用viridis颜色映射

# 设置统一的线宽
line_width = 1.0  # 统一的线宽设置

# 绘制每个batch size的折线
for i, batch_size in enumerate(batch_sizes):  # 循环遍历每个batch size
    # 绘制批次大小对应的3D曲线
    ax.plot(epochs, [batch_size]*len(epochs), accuracies[i], label=f'Batch Size {batch_size}', color=cmap(norm(batch_size)), linewidth=line_width)
    
    # 画出折线在平面上的投影
    ax.plot(epochs, [batch_size]*len(epochs), np.zeros(len(epochs)), color=cmap(norm(batch_size)), linewidth=line_width)
    
    # 创建用于填充的多边形顶点
    verts = [(x, batch_size, z) for x, z in zip(epochs, accuracies[i])]  # 曲线上的顶点坐标
    verts += [(x, batch_size, 0) for x in epochs[::-1]]  # XY平面上的顶点 (Z=0)
    
    # 使用 Poly3DCollection 填充曲线与 XY 平面之间的区域
    poly = Poly3DCollection([verts], color=cmap(norm(batch_size)), alpha=0.3)  # 创建半透明填充区域
    ax.add_collection3d(poly)  # 将填充区域添加到图形中

# 添加颜色条
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)  # 创建颜色映射器
mappable.set_array(batch_sizes)  # 设置颜色映射的数据范围
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)  # 添加颜色条，调整大小和形状

# 在特定的epoch上展示点和数值
for epoch in show_epochs:  # 遍历需要重点展示的epoch
    data_list_x = []  # 存储epoch的数据
    data_list_y = []  # 存储batch size的数据
    data_list_z = []  # 存储准确率数据
    for i, batch_size in enumerate(batch_sizes):  # 遍历每个batch size
        # 绘制重点的散点图，突出显示
        ax.scatter(epoch, batch_size, accuracies[i][epochs.index(epoch)], color=cmap(norm(batch_size)), s=20, edgecolor='black', zorder=10)
        # 在散点旁边显示数值
        ax.text(epoch, batch_size, accuracies[i][epochs.index(epoch)], f'{accuracies[i][epochs.index(epoch)]:.2f}', color='black', zorder=10)
        data_list_x.append(epoch)  # 记录当前的epoch
        data_list_y.append(batch_size)  # 记录当前的batch size
        data_list_z.append(accuracies[i][epochs.index(epoch)])  # 记录当前的准确率
    # 绘制每个特定epoch下的虚线连接不同batch size的准确率
    ax.plot(data_list_x, data_list_y, data_list_z, color='black', linewidth=line_width, linestyle='--')

# 设置坐标轴标签
ax.set_xlabel('Epochs')  # 设置X轴标签
ax.set_ylabel('Batch Size')  # 设置Y轴标签
ax.set_zlabel('Accuracy', labelpad=-2)  # 设置Z轴标签

# 关闭默认的网格
ax.grid(False)  # 关闭网格线

# 调整视角
ax.view_init(elev=30, azim=225)  # 设置视角：俯视角30度，方位角225度

plt.savefig('3d_plot.png', dpi=300, bbox_inches='tight')  # 保存图形为高分辨率的PNG文件
# 显示图形
plt.show()  # 显示图形
```

​    

