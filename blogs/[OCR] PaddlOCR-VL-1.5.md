



# PaddleOCR-VL-1.5

接上一篇 [OCR] PaddleOCR-VL OCR的新范式

这篇继续PaddleOCR-VL-1.5，与PaddleOCR-VL 进行一下对比学习，基本框架和流程没有大的改变，不会太详细展开，对一些新的技术改进，进行重点学习；还没了解过PaddleOCR-VL，可以找上面文章过一下。


![PaddleOCR-VL-15](https://oss.caibucai.top/md/PaddleOCR-VL-15.png)



PaddleOCR-VL-1.5在保持0.9B参数规模的同时，将 OmniDocBench v1.5 的分数提升至94.5，并引入了不规则形状定位、印章识别、文本检测识别一体化等新能力，同时针对扫描、倾斜、弯折、屏幕拍照、光线变化等五种真实场景构建了 Real5-OmniDocBench 评测基准。



## 架构

![PP-DocLayoutV3](https://oss.caibucai.top/md/PP-DocLayoutV3.png)

### PP-DocLayoutV3

版面分析这里由PP-DocLayoutV2升级到了 PP-DocLayoutV3，

PPDocLayoutV3 专门设计用于处理非平面文档图像。它能直接预测布局元素的多点边界框——而非标准的两点框，在一次前向forward内确定倾斜和变形曲面的逻辑读序。

**多点定位**

为了适应现实场景中复杂的布局（如旋转文本、常见场景或密集形态），采用四点四边形表示，而非传统的两点边界框。

四点格式通过四个顶点定义文本区域：左上顶点（TL）、右上顶点（TR）、右下角（BR）和左下顶点（BL）。

这种格式在定位倾斜和不规则文本形状方面提供了卓越的灵活性，而这些形状使用矩形框无法紧密包裹。

> 这里使用这种四边形四点格式，能覆盖更多现实场景中的特殊情形，也算是一种实例分割了


**端到端统一**

PP-DocLayoutV3 的核心架构创新是将读取顺序预测直接集成到 Transformer 解码器中。具体就是，PaddleOCR团队扩展了RT-DETR框架，同时优化几何定位和逻辑序列。遵循基于查询的范式，解码器迭代细化N个对象查询。读取顺序随后通过全局指针机制，从最终解码层的精炼查询嵌入中推导出来。

> PP-DocLayoutV3使用改进的 RT-DETR 实现了 BBOX和逻辑顺序的一个统一优化；如果还记得的话，PP-DocLayoutV2中的RT-DETR是只输出 BBOX，然后接了个 Pointer Network 来确定阅读逻辑顺序，如下图

![PP-DocLayoutV3-1](https://oss.caibucai.top/md/PP-DocLayoutV3-1.png)

这种端到端范式将检测、分割和读取顺序模块共享统一的特征表示，从而实现空间定位与逻辑序列之间的更好对齐。后处理的逻辑应该都差不多。

### 内容识别模型

内容识别模型也即PaddleOCR-VL-1.5，仍继续使用 vision encoder + ERNIE-4.5-0.3B，增加了 seal recognition 和 text spotting 任务

- seal recognition 即 印章识别

- text spotting 端到段识别，如子图(b)
![](https://oss.caibucai.top/md/v2-60471518a7c488b9990ae51e4af58c71_1440w.jpg)
与PaddleOCR-VL相比，PaddleOCR-VL-1.5在复杂表格和数学公式识别准确率方面有显著提升。此外，模型还对稀有字符、古代中文文本、多语言表格以及下划线和强调符号等文本装饰进行了更细粒度的优化。


## 训练策略
### PP-DocLayoutV3 端到端的联合优化

模型初始化为预训练权重PP-DocLayout_plus-L，训练语料库扩展至超过38k高质量文档样本。每个样本提供真实标注，包括坐标、类别标签和每个布局元素的绝对读数顺序。使用weight-decay为0.0001的AdamW优化器。lr设置为常数 2 × 10⁻4，以确保集成的全局指针和掩码头的稳定收敛。该模型训练了150个epoch，batch-size为32。 

为了实现环境的鲁棒性性，设计了专门的失真感知数据增强流程。与标准增强不同，该pipeline专门模拟现实移动摄影中存在的复杂物理变形。

### PaddleOCR-VL-1.5 渐进式训练

**1. pre-trainging 增强视觉-语言对齐**

与PaddleOCR-VL- 相比，将预训练数据集从2900万对图像-文本对扩展到4600万对。

为了增强视觉骨干的泛化并支持新引入的功能，整合了更广泛的多语言文档和复杂的现实世界场景。此外，在pre-traning注入与seal-recognition和text-spotting相关的大规模预训练数据。

text-spotting任务的最大分辨率提升至2048×28×28像素，使模型能够实现更精确的文本定位和识别。

**2. post-training 针对新任务进行指令微调**

继承了 PaddleOCR-VL 的四个基本指令任务——OCR、表格、公式和图表识别——确保标准文档元素的向后兼容性和高性能。PaddleOCR-VL-1.5-0.9B 的关键创新在于新增了两项专业任务：
1. seal-recognition：引入了专门的官方印章和印章文字，解决了弯曲文字、模糊图像和背景干扰等问题。
2. text-spotting：与仅输出文本内容的标准OCR不同，文本识别任务要求模型同时预测文本及其根据自然阅读顺序的精确空间位置。使用4点坐标表示法统一生成

如

Y = Text ⊕ <LOC_x\_{TL}><LOC_y\_{TL}> . . . <LOC_y_{BL}>

DREAM <LOC_253> <LOC_286> <LOC_346> <LOC_298> <LOC_345> <LOC_339> <LOC_252> <LOC_330>

**3. 强化学习reinforcemnet learning**

不是很懂，直接上原文
1. To enhance generalization and unify diverse label styles, we introduce a Reinforcement Learning stage leveraging Group Relative Policy Optimization(GRPO) . By executing parallel rollouts and calculating relative advantages within each group, GRPO facilitates robust policy updates and mitigates style inconsistency. This process is supported by a dynamic data screening protocol that prioritizes challenging samples with high reward potential and entropy uncertainty, ensuring the model focuses on non-trivial, high-value learning cases.

## 数据构建

PaddleOCR-VL-1.5- 在数据构建上专注提升模型在复杂样本上的稳健性，以及扩展支持能力的广度。
1. 困难样本挖掘，重点识别和加权高不确定性样本以优化模型决策边界;
2. 新能力数据构建，构建专业数据集以解锁文本识别、印章识别和高级多语支持等新技能。


## 性能展示

在omnidocbenchv1.5 刷新了榜单，94.5

![omnidocbenchv1.5](https://oss.caibucai.top/md/omnidocbenchv1.5_metrics.png)

在更难的real5-omnidocbench测试，取得领先

![real5-omnidocbench](https://oss.caibucai.top/md/real5-omnidocbench_metrics.png)

推理性能也有进一步提升

![PaddleOCR-VL-15-infer](https://oss.caibucai.top/md/PaddleOCR-VL-15-infer.png)


## 参考
[PaddleOCR-VL-1.5 huggingface space](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo))
[Paddle0CR-VL-1.5 官方文档](https://www.paddleocr.ai/main/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.html)
[PaddleOCR-VL-1.5 技术报告](https://arxiv.org/pdf/2510.14528)

