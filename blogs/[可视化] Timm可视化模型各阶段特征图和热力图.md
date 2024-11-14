最近在尝试可视化一些特征的工作；其中timm库提供丰富的backbone主干模型，模型接口标准（经常可以看到将timm的backbone用在yolo等模型改进中），帮助很多现有工作减少模型搭建工作量，是一些简单网络搭建的起手式，也非常方便做一些backbone的对比工作。

## 过程说明
部分函数来自于Featup项目，非常感谢他们的工作；

下面的过程结合resnet50说明

1. timm创建模型，和对图像做预处理（调整size等）
> features_only=True 只拿特征

```python
timm_model = timm.create_model(timm_model_name, pretrained=True, features_only=True)

image = Image.open(image_path, mode='r')

data_config = timm.data.resolve_model_data_config(timm_model)
# print(data_config)
timm_transforms = timm.data.create_transform(**data_config, is_training=False)
timm_unnorm = UnNormalize(list(data_config['mean']), list(data_config['std']))
image_transformed = timm_transforms(image).unsqueeze(0)
```

2. 模型做一次前向forward，拿到各个阶段的特征stage_features
```python
timm_model.cuda()
image_cuda = image_transformed.cuda()
stage_features = timm_model(image_cuda)
```

3. 各阶段输出的特征通常具有很多通道数（64,128,256,512,1024 ...），使用PCA进行降维处理（维度数暴露dim参数，可供调整），便于可视化
```python
[stage_feature_pca], _ = pca([stage_feature], 9)
```
![在这里插入图片描述](https://oss.caibucai.top/md/4b5c99435a2a473bb8aa10b52ccc9e97.png)
==如有问题，需要帮助，欢迎留言、私信或加群 交流【群号：392784757】==

4. 热力图基于原始特征，未做PCA，在通道维度进行累加，然后在通道维进行均值化，得到单通道的特征平均图，进一步 /np.max() 归一化到[0,1] 得到热力图
```python
for c in range(feature_map.shape[1]):
    heatmap += feature_map[:, c, :, :]
heatmap_np = np.mean(heatmap.cpu().numpy(), axis=0)
heatmap_np = np.maximum(heatmap_np, 0)
heatmap_np /= np.max(heatmap_np)
```

5. plot_timm_model_stage_features 绘制各阶段的PCA降维特征图
![在这里插入图片描述](https://oss.caibucai.top/md/f01bdb01ff334e7598a1f79a62c2d1d1.png)



6. cv_show_single_feature_heatmap 可视化某一阶段的特征热力图情况，提供 stage 参数指定

![在这里插入图片描述](https://oss.caibucai.top/md/8e63b701960347d3a34ea64a6e9b6ec1.png)
![在这里插入图片描述](https://oss.caibucai.top/md/e4be2a6a59f84f8f8122dcfdab89bc56.png)


7. cv_show_all_feature_heatmap 可视化各个阶段的特征热力图情况
![在这里插入图片描述](https://oss.caibucai.top/md/e9cc6d52b0f643569222501223b71c84.png)

8. 最终运行完，可得到4张图，分别是 所有阶段特征图、某一阶段的热力图，对应某一阶段热力图和原始图像融合图，所有阶段特征图
![在这里插入图片描述](https://oss.caibucai.top/md/b0a31d0be7514b40874db399e27617a9.png)
> 01来自 plot_timm_model_stage_features  ；02 、03来自cv_show_single_feature_heatmap；04来自cv_show_all_feature_heatmap
> 上面对应有附图，这里仅做说明，不再重复展示

如有问题，需要帮助，欢迎留言、私信或加群 交流【群号：392784757】
## 不足之处
当前代码仅在提供的官方模型下进行使用；对于自行训练后的timm模型未做测试（如果只是在timm做微调，应该没有问题），以及在基于timm搭建的网络结构中，这一可视化代码只能做参考，还需要进一步结合网络结构和训练过程做调整修改；
也还算是一个小小的工作，在特征图可视化、热力图可视化等方面希望能对你的工作有所启发，帮助！欢迎一起交流有关这一方面的内容！
## 完整代码
300多行 完整代码（搭配好能使用timm的环境即可使用，下载的模型调整了位置放在了 当前文件夹下，只需提供模型名称和图片）
```python
# -*- coding: UTF-8 -*-
"""
@File    ：timm_feature_visualize.py
@Author  ：cc&uu qq 392784757
@Date    ：2024/11/12 16:19 
@Bref    :
@Ref     : UnNormalize, TorchPCA, pca, remove_axes from featup project
TODO     :目前使用的是timm直接返回的stage_features,然后累加各个通道得到的heatmap
         :
"""
import os
import argparse

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# linux下，原始路径为 ~/.cache/huggingface/hub/
# 使用环境变量的方法，重新设定HF的存储models路径
os.environ['HF_HUB_CACHE'] = '../timm_models/'

# torch 缓存
os.environ['TORCH_HOME'] = '../torch_models/'

import matplotlib.pyplot as plt
import numpy as np

import timm
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
import cv2
from sklearn.decomposition import PCA


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


def remove_axes(axes):
    def _remove_axes(ax):
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        # 无坐标值
        ax.set_xticks([])
        ax.set_yticks([])
        # 无黑框
        ax.set_axis_off()

    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def plot_timm_model_stage_features(timm_model_name, image_path):
    timm_model = timm.create_model(timm_model_name, pretrained=True, features_only=True)
    image = Image.open(image_path, mode='r')

    data_config = timm.data.resolve_model_data_config(timm_model)
    # print(data_config)
    timm_transforms = timm.data.create_transform(**data_config, is_training=False)
    timm_unnorm = UnNormalize(list(data_config['mean']), list(data_config['std']))
    image_transformed = timm_transforms(image).unsqueeze(0)

    timm_model.cuda()
    image_cuda = image_transformed.cuda()
    stage_features = timm_model(image_cuda)
    print('total ', len(stage_features), " stage.")

    for i, stage_feature in enumerate(stage_features):
        shape_ = stage_feature.shape
        print("stage ", i, ' --->', shape_, '---> dim: ', shape_[1])

    # fig, ax = plt.subplots(1, len(stage_features) + 1, figsize=(10, 5))  # w h
    fig, ax = plt.subplots(1, len(stage_features) + 1)  # w h
    image_show = timm_unnorm(image_transformed)[0].permute(1, 2, 0).detach().cpu()
    ax[0].imshow(image_show)
    ax[0].set_title("original", fontsize=15)

    for i, stage_feature in enumerate(stage_features):
        [stage_feature_pca], _ = pca([stage_feature], 9)
        # ax[0, i + 1].imshow(stage_feature_pca[0].permute(1, 2, 0).detach().cpu())
        ax[i + 1].imshow(stage_feature_pca[0, :3].permute(1, 2, 0).detach().cpu())
        ax[i + 1].set_title(f"stage_{i}", fontsize=15)

    remove_axes(ax)
    fig.suptitle("Stage features by pca to 3 components", fontsize=15)
    plt.tight_layout()
    print('save 01-stage_features.png')
    plt.savefig('01-stage_features.png')
    plt.show()
    plt.close(fig)
    return image_show, stage_features


def feature2heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:, 0, :, :] * 0
    heatmap = feature_map[:1, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    # torch
    heatmap_tensor = heatmap.detach().cpu()
    heatmap_tensor = heatmap_tensor.mean(dim=0)
    heatmap_tensor = heatmap_tensor / torch.max(heatmap_tensor)

    # np
    heatmap_np = np.mean(heatmap.cpu().numpy(), axis=0)
    heatmap_np = np.maximum(heatmap_np, 0)
    heatmap_np /= np.max(heatmap_np)

    return heatmap_tensor, heatmap_np


'''
颜色域要统一；cv2 处理 再使用 plt 可视化，颜色会反调 执行bgr2rgb
'''


def cv_show_single_feature_heatmap(image_path, heatmap, stage, save_dir='./', cv_win_show=False):
    """
    显示热力图和原始图像的叠加效果，并保存结果。

    参数：
        image_path (str): 原始图像路径。
        heatmap (ndarray): 热力图，单通道的二维数组。
        save_dir (str): 保存叠加结果的目录，默认当前目录。
    """
    # 读取原始图像并转换为RGB
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像文件未找到：{image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 调整热力图大小与原图一致
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # w,h
    heatmap_uint8 = np.uint8(255 * heatmap_resized)  # w,h uint8

    # 应用伪彩色映射
    heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # w,h,3 uint8
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)

    # 叠加热力图与原始图像
    superimposed_img = cv2.addWeighted(heatmap_jet_rgb, 0.4, img_rgb, 0.6, 0)

    # 使用 Matplotlib 显示结果
    plt.figure(figsize=(10, 4))
    titles = ['Original Image', 'Heatmap', 'Heatmap (Jet)', 'Superimposed']
    images = [img_rgb, heatmap_uint8, heatmap_jet_rgb, superimposed_img]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    print(f'save 02-stage{stage}_feature_visualize.png')
    plt.savefig(f'02-stage{stage}_feature_visualize.png')
    plt.show()

    # 保存叠加结果
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'03-stage{stage}_heatmap_image_fusion.png')
    cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    print(f"save {save_path}")

    # 使用 OpenCV 显示结果
    result = np.hstack([img, heatmap_jet_bgr, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)])
    result_resized = cv2.resize(result, (960, 480))
    if cv_win_show:
        cv2.imshow('Result', result_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cv_show_all_feature_heatmap(image_path, stage_features):
    # 读取原始图像并转换为RGB
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像文件未找到：{image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def process_single_stage_images(heatmap):
        # 调整热力图大小与原图一致
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # w,h
        heatmap_uint8 = np.uint8(255 * heatmap_resized)  # w,h uint8

        # 应用伪彩色映射
        heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # w,h,3 uint8
        heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)

        # 叠加热力图与原始图像
        superimposed_img = cv2.addWeighted(heatmap_jet_rgb, 0.4, img_rgb, 0.6, 0)
        images = [heatmap_uint8, heatmap_jet_rgb, superimposed_img]
        return images

    stage_heatmaps = [feature2heatmap(stage_feature)[1] for stage_feature in stage_features]

    titles = ['Original image', 'Heatmap', 'Heatmap (Jet)', 'Superimposed']

    n_stage = len(stage_features)
    fig, axs = plt.subplots(n_stage, 4)  # w h
    for i_stage in range(n_stage):
        single_stage_images = process_single_stage_images(stage_heatmaps[i_stage])
        for i in range(4):
            if i_stage == 0:
                axs[0, i].set_title(titles[i])

            if i == 0:
                axs[i_stage, i].imshow(img_rgb)
            else:
                axs[i_stage, i].imshow(single_stage_images[i - 1])

    remove_axes(axs)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    print('save 04-all_stage_feature_visualize.png')
    plt.savefig('04-all_stage_feature_visualize.png')
    plt.show()


if __name__ == '__main__':
    # for dev
    timm_model_name = 'resnet50'
    image_path = '../images_dev/cat2.jpg'
    i_stage = 2  # default last
    cv_win_show = False
    image_show, stage_features = plot_timm_model_stage_features(timm_model_name, image_path)

    if i_stage >= len(stage_features):
        raise Exception('i_stage out of range,please reselect one valid stage')
    heatmap_tensor, heatmap_np = feature2heatmap(stage_features[i_stage])
    cv_show_single_feature_heatmap(image_path, heatmap_np, stage=4, cv_win_show=cv_win_show)
    cv_show_all_feature_heatmap(image_path, stage_features)

```
如有问题，需要帮助，欢迎留言、私信或加群 交流【群号：392784757】