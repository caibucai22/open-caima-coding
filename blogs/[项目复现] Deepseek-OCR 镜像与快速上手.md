# DeepSeek-OCR（v1 / v2）快速上手

> 本文基于已配置好的 AutoDL 镜像，说明如何在最短时间内运行 DeepSeek-OCR（v1/v2）、使用 GUI 并注意常见问题。

![DeepSeek-OCR GUI 封面](https://com-caibucai-top.oss-cn-hangzhou.aliyuncs.com/md/deepseek-ocr-gui.png)
*DeepSeek-OCR GUI 界面（上传图像输出识别结果）*

## 一、导读
我们在 AutoDL 上搭建了deepseek-ocr的环境，预置模型和测试图像等，本教程演示如何“开箱即用”运行 DeepSeek-OCR，并包含快速运行的配置与命令（适用于 v1 与 v2）。

> AutoDL选择GPU创建实例，选择镜像 deepseek-ai/DeepSeek-OCR-2/DeepSeek-OCR-V2-V1

## 二、DeepSeek-OCR
- DeepSeek-OCR 的核心思想是把图像的视觉特征转换为可被 LLM 处理的视觉 tokens，然后由 LLM 生成文本或结构化输出（例如：OCR 文本、表格化信息）。
- v1（DeepSeek-OCR）为首版实现，聚焦于将视觉编码与 LLM 接合的工程实现；v2（DeepSeek-OCR-2）在模型与流程上有若干改进（如更好的视觉-文本因果流/动态分辨率处理、模型细节与推理优化），在可用性与准确性上通常有提升。具体论文与实现细节可参考官方仓库与论文说明。

## 三、环境与快速上手（开箱即用）

### 软硬件环境（开箱即用）：
**软件环境：**

```text
OS: 镜像内通用 Linux 环境
Python: 3.12
PyTorch: 2.6.0
CUDA: 11.8
vLLM: 0.8.5
flash-attn: 2.7.3
```

**硬件环境：**

```text
GPU: 建议 NVIDIA（3090/4090/A100）
显存: 推荐 >= 24GB（测试中峰值约 24GB）
```

> 注：若目标卡显存较小，建议等待社区量化模型或降低输入分辨率与 batch 大小以避免 OOM。

### 最快执行仅三步
1) 修改配置：在对应仓库中更新 `config.py`（以 v2 为例）

```python
# /root/DeepSeek-OCR2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/config.py
INPUT_PATH = '/root/test_images/img1.jpg'
OUTPUT_PATH = '/root/test_images/'
```

2) 运行推理：

```bash
conda activate deepseek-ocr
cd /root/DeepSeek-OCR2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm
python run_dpsk_ocr_image.py
```

> v1 路径：`/root/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm`（同样修改 `config.py` 并运行）。

3) 查看输出：识别结果会写入 `OUTPUT_PATH`（常见格式：纯文本 / JSON，可能包含坐标 bbox、分块文本等），并在终端打印推理日志。

示例（JSON 摘要）：

```json
{
  "image": "img1.jpg",
  "text": "识别得到的文本...",
  "blocks": [{"bbox": [x,y,w,h], "text": "段1"}, ...]
}
```

### GUI 交互执行
启动方法（目前仅支持v1模型）：

```bash
cd /root/DeepSeek-OCR-GUI
python app.py
# 终端输出: Running on local URL: http://127.0.0.1:7860
```

GUI 适合单张图交互、调参与快速验证；批量处理建议使用命令行脚本。



## 四、踩坑记录与问题处理

暂无

## 五、参考资料

1. [DeepSeek-OCR（v1） GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
2. [DeepSeek-OCR-2（v2） GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
3. [Hugging Face：DeepSeek-OCR 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
4. [vLLM 文档](https://docs.vllm.ai/)
5. [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention)




