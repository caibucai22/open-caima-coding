一直有在关注OCR这个任务，很多项目都会尝试用PaddleOCR去做。2025-10-16百度发布PaddleOCR-VL ，距今也有一段时间了，榜单成绩和实测效果都挺不错，2026-01-29又发布了PaddleOCR-VL-1.5，今天先来了解学习PaddleOCR新范式的开篇之作。

> 很多团队都在尝试结合大模型的范式来继续提升OCR任务的上限，各家的技术路数区别还是较大的。
>
> DeepSeek 发布OCR模型， DeepSeek-OCR V1（2025-10-20） V2（2026-1-28）；里面的概念好像更高级
>
> 智谱 GLM-OCR 2026-02-04 发布，指标也非常能打，在huggingface 上下载量比PaddleOCR系列还高


## PaddleOCR-VL
![](https://oss.caibucai.top/md/PaddleOCR-VL.png)



### 框架与流程

PaddleOCR-VL 大框架由 PP-LayoutV2（RT-DETR + Pointer Network）+ Vision Encoder + LLM Decoder 组成

> 更确切的说 PaddleOCR-VL 是 Vision Encoder + LLM Decoder 这一部分



![PP-DocLayoutV2](https://oss.caibucai.top/md/PP-DocLayoutV2.png)

PP-LayoutV2 使用RT-DETR完成文档版面视觉元素的定位，表格、文本段落、图片、公式、标题、页眉页脚等，得到这些视觉元素的BBox 和 类别，然后由Pointer Network（6个transformer layer） 预测 文档各视觉元素的阅读顺序，这个阅读顺序是以 (N,N)的 realative-order 的logits矩阵展示，最后经过后处理，输出文档上各视觉元素带有顺序的 BBox。至此，PP-LayoutV2 完成了文档布局分析工作。

随后这些文档上的定位到带有顺序的视觉元素以图片形式，进入到视觉语言编解码阶段。在进行视觉编码的过程中 PaddleOCR-VL受到 LLaVA的启发，使用了一个带有 动态分辨率预处理器 dynamic resolution preprocessor 的 视觉 encoder，以及 MLP Projector

> PaddleOCR-VL对输入动态分辨率进行了支持，与固定分辨率或基于切片的传统方法不同，该编码器支持原生动态高分辨率输入，能够以任意分辨率处理图像而无需拉伸或裁剪，避免了图像变形导致的细节丢失。这一特性对于处理密集文本、小字体和复杂符号尤为重要，显著减少了文本密集型任务中的幻觉现象。

视觉编码器采用了NaViT（Native Resolution Vision Transformer）风格设计，从 Keye-VL 的视觉模型初始化。

投影器是一个随机初始化的两层 MLP，采用 GELU 激活函数，合并大小为2。该模块负责将视觉编码器提取的特征高效地映射到语言模型的嵌入空间。

语言解码器部分来自百度自研的 ERNIE-4.5-0.3B，一个轻量级但推理效率极高的开源语言模型。为进一步增强位置表示能力，模型引入了 3D-RoPE（3D Rotary Position Embedding）位置编码。

> ERNIE-4.5-0.3B 与 NaViT 视觉编码器的组合，在显著降低内存占用的同时实现了更快的推理速度。

最终在这一阶段，语言模型以自回归方式生成结构化输出，根据元素类型输出不同格式的内容：文本输出纯文字内容，表格输出 OTSL（Open Table Structure Language）格式，公式输出 LaTeX 格式，图表输出 Markdown 表格格式。

然后由轻量级的后处理模块聚合两阶段的输出结果，将各元素的识别内容按照阅读顺序组织，生成结构化的 Markdown 或 JSON 格式输出。

> 对于跨页表格和段落标题，PaddleOCR-VL-1.5 还支持自动合并处理，有效解决长文档解析中的内容断裂问题。

### 训练策略

PP-DocLayoutV2 的训练分为两个阶段：
1. RT-DETR 检测网络在超过2万个高质量样本的自建数据集上训练100个epoch，使用 PP-DocLayout_plus-L 的预训练权重进行初始化。训练采用标准的目标检测损失函数，优化器使用 AdamW，学习率采用余弦退火策略。
2. 在检测网络训练完成后，固定 RT-DETR 参数，独立训练pointer network 200个epoch。使用恒定学习率 2×10⁻⁴，优化器为 AdamW，损失函数采用广义交叉熵损失（Generalized Cross Entropy Loss）。这种分阶段训练策略确保了两个网络各自收敛到最优状态。

PaddleOCR-VL-0.9B 分为预训练对齐和指令微调两个阶段
1. 模型以 Keye-VL 的视觉编码器和 ERNIE-4.5-0.3B 语言模型的权重初始化，在2900万高质量图像-文本对上训练1个epoch。训练配置：批次大小128，序列长度16384，最大分辨率 1280×28×28，学习率从 5×10⁻⁵ 降至 5×10⁻⁶。该阶段主要目标是实现视觉特征空间与语言特征空间的对齐
2. 在预训练基础上，使用270万精心筛选的样本进行2个epoch的指令微调。训练配置：批次大小128，序列长度16384，最大分辨率提升至 2048×28×28，学习率从 5×10⁻⁶ 降至 5×10⁻⁷。微调任务包括：OCR（文本行、文本块、整页识别）、表格识别（OTSL格式输出）、公式识别（LaTeX格式，区分行内/行间公式）、图表识别（Markdown表格格式）。


### 数据集

来源上：
1. 开源数据集：CASIA-HWDB（手写中文）、UniMER-1M（公式）、ChartQA、PlotQA 等
2. 合成数据：通过 XeLaTeX 和浏览器渲染生成，用于解决类别不平衡问题
3. 网络数据：学术论文、考试试卷、手写笔记、PPT 等可公开获取的文档
4. 内部数据：百度积累的私有 OCR 数据集

构建上进行自动标注 与 困难样本挖掘

- 自动化标注使用双阶段自动标注策略：首先由专家模型 PP-StructureV3 生成初步伪标签，然后通过提示工程引导高级多模态大语言模型（ERNIE-4.5-VL、Qwen2.5-VL）进行精细化修正

- 困难样本挖掘上，构建评估引擎，将元素细分为23种文本类型、20种表格类型、4种公式类型和11种图表类型。使用专业指标（EditDist 评估文本、TEDS 评估表格、RMS-F1 评估图表、BLEU 评估公式）识别模型表现不佳的类别。针对难例类型，利用字体/CSS库和渲染工具合成新的挑战性样本，形成智能反馈循环。


### 性能展示
最终榜单成绩上，

![](https://oss.caibucai.top/md/omnidocbenchv1.5_metrics.png)

在全球权威文档解析评测榜单 OmniDocBench v1.5 上取得了92.86分的综合成绩，超越 GPT-4o、Gemini-2.5 Pro、Qwen2.5-VL-72B 等主流多模态大模型，以及 MonkeyOCR-Pro-3B、MinerU2.5、dots.ocr 等OCR专业模型，在文本识别、公式识别、表格理解和阅读顺序预测四大核心任务上全部刷新 SOTA纪录

推理性能上也很能打

![PaddleOCR-VL-infer](https://oss.caibucai.top/md/PaddleOCR-VL-infer.png)



### 本地测试

![img](https://oss.caibucai.top/md/paddle-or2.png)



![img](https://oss.caibucai.top/md/paddle-or3.png)



![img](https://oss.caibucai.top/md/paddle-or6.png)



![paddle-or7-1](https://oss.caibucai.top/md/paddle-or7-1.jpg)



## 参考

1. [PaddleOCR-VL huggingface space](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)
2. [Paddle0CR-VL 官方文档](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html)
3. [PaddleOCR-VL 技术报告](https://arxiv.org/pdf/2510.14528)



