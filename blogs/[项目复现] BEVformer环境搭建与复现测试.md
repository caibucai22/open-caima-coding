# BEVFormer 环境搭建与完整复现记录（Tiny + nuScenes v1.0-mini）

> 基于 BEVFormer 官方代码，使用 bevformer_tiny 模型与 nuScenes v1.0-mini 数据集  
> 记录一次从环境搭建到训练、推理、可视化的完整复现过程。

## 一、背景与目标

BEVFormer 是一种基于 Transformer 的多视角 BEV 感知方法，在自动驾驶 3D 目标检测任务中具有较高关注度，但其工程依赖复杂、环境版本要求严格，实际复现过程中容易遇到各种问题。

本文以工程复现为核心，目标是在尽量贴近真实训练环境的前提下，完整跑通 BEVFormer Tiny 模型的训练与推理流程，并整理一套可复现的环境配置与问题处理经验。

本文主要目标包括：

- 跑通完整复现流程（数据 → 训练 → 推理 → 可视化）
- 固定关键依赖版本，降低环境踩坑成本
- 汇总实际复现过程中遇到的问题与解决方案

**适合人群：**

1. 初次复现 BEVFormer 的研究或工程人员  
2. 希望快速验证 BEVFormer pipeline 是否可跑通的读者  
3. 需要参考环境配置与依赖版本的同学  

## 二、复现过程

> 本节为正文核心内容，按真实复现顺序进行组织

### 2.1 软硬件环境（开箱即用）

软件环境如下：

```text
OS: Ubuntu 18.04
Python: 3.8
PyTorch: 1.9.1
CUDA: 11.1
~~~

硬件环境如下：

```text
GPU: RTX 2080 Ti
显存: 22GB（单卡）
```

> bevformer_tiny 模型峰值显存约 13GB，正常训练约 5GB
> 推荐显存 ≥ 24GB，或使用多卡训练
> 系统盘实测占用约 21GB，建议将数据集与实验输出放在数据盘并通过软链接管理

### 2.2 数据集准备与处理

本文使用的数据集为 nuScenes v1.0-mini，主要用于快速验证 BEVFormer 的完整训练与推理流程。

数据下载参考：

- https://aistudio.baidu.com/datasetdetail/244741
- https://aistudio.baidu.com/datasetdetail/57268

数据集组织结构如下：

```text
data/
└── nuscenes/
```

执行数据预处理脚本：

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --canbus ./data v1.0-mini ./data/nuscenes
# 生成 BEVFormer 所需的索引文件与缓存数据
```

关键日志输出如下：

```text
total scene num: 10
train scene: 8, val scene: 2
train sample: 323, val sample: 81
```

> 场景数量与样本划分符合预期，数据索引生成正常

### 2.3 模型训练与推理

#### 模型训练

使用单卡启动训练：

```bash
./tools/dist_train.sh \
  ./projects/configs/bevformer/bevformer_tiny.py \
  1
# 参数 1 表示使用 1 张 GPU
```

支持断点续训：

```bash
./tools/dist_train_resume.sh \
  ./projects/configs/bevformer/bevformer_tiny.py \
  1 \
  权重路径
# 权重路径替换为中断前保存的 checkpoint
```

#### 模型推理与评测

训练完成后执行推理与评测：

```bash
./tools/dist_test.sh \
  ./projects/configs/bevformer/bevformer_tiny.py \
  work_dirs/bevformer_tiny/latest.pth \
  1
```

评测结果示例如下：

```text
mAP: 0.0210
mATE: 1.1116
mASE: 0.6880
mAOE: 1.2319
mAVE: 1.2464
mAAE: 0.7094
NDS: 0.0708
Eval time: 5.8s
```

> 使用的是 v1.0-mini 数据集，指标偏低属于正常现象
> 该结果主要用于验证训练、推理与评测流程是否完整跑通

> 如在复现过程中遇到问题，文章底部 Banner 图找到我们，一起交流讨论、共同进步

### 2.4 结果可视化

执行可视化脚本：

```bash
python tools/analysis_tools/visual.py
# 运行前需在脚本中手动修改检测结果 json 路径
```

生成的可视化结果默认保存在：

```text
result/
```

> 可视化主要用于直观检查模型预测结果与场景是否对齐

### 2.5 最终可复现环境

在完整跑通数据处理、训练与推理流程后，导出最终环境依赖。

```text
torch==1.9.1+cu111
mmcv-full==1.4.0
mmdet==2.14.0
mmdet3d==0.17.1
detectron2==0.6
nuscenes-devkit==1.2.0
yapf==0.30.0
```

> mmcv、detectron2、yapf 等库对版本较为敏感，不建议随意升级
> pip 提示的部分依赖不兼容在实际测试中未影响训练与推理

## 三、踩坑记录与问题处理

> 本节集中整理复现过程中遇到的关键问题，避免干扰主线流程

1. **detectron2 安装失败**

   报错信息如下：

   ```text
   <ATen/cuda/Atomic.cuh> no such file or directory
   ```

   通过手动安装并固定版本解决：

   ```bash
   git clone https://github.com/facebookresearch/detectron2.git
   cd detectron2
   git checkout v0.6
   python setup.py install
   ```

2. **缺少 pyquaternion**

   报错信息：

   ```text
   No module named 'pyquaternion'
   ```

   直接安装即可：

   ```bash
   pip install pyquaternion
   ```

3. **数据集 SDK 依赖冲突**

   安装数据集相关依赖：

   ```bash
   pip install lyft-dataset-sdk==0.0.8
   pip install nuscenes-devkit==1.2.0
   ```

   > pip 会提示部分依赖版本不兼容，实际测试中未影响训练与推理流程

4. **No module named `tools.data_converter`**

   报错来源于：

   ```text
   tools/data_converter/indoor_converter.py
   ```

   将文件中 import 语句里的 `tools.` 前缀去掉即可正常运行。

5. **yapf 版本不兼容**

   报错信息：

   ```text
   FormatCode() got an unexpected keyword argument 'verify'
   ```

   通过固定 yapf 版本解决：

   ```bash
   pip install yapf==0.30.0
   ```

> 如在复现过程中遇到问题，文章底部 Banner 图找到我们，一起交流讨论、共同进步

## 四、参考资料

1. [BEVFormer 官方仓库](https://github.com/fundamentalvision/BEVFormer)
2. [BEVFormer 官方安装文档](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md)
3. [BEVFormer 复现及踩坑全过程记录](https://blog.csdn.net/2301_80879500/article/details/144052908)
4. [mmcv 与 yapf 版本不匹配问题分析](https://blog.csdn.net/weixin_55982578/article/details/137673122)
5. [pyquaternion 报错问题说明](https://blog.csdn.net/qqliuzhitong/article/details/118711495)
6. [官方模型权重与配置文件](https://github.com/fundamentalvision/BEVFormer)