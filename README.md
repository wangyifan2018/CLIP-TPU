# CLIP

## 目录 <!-- omit in toc -->

## 1. 简介
CLIP（Contrastive Language-Image Pre-Training）是一个在多种（图像，文本）配对上训练的神经网络。它可以用自然语言进行指导，以预测给定图像最相关的文本片段，而无需直接针对该任务进行优化，这与GPT-2和3的零样本（zero-shot）能力类似。本例程对[CLIP官方开源仓库](https://github.com/openai/CLIP)中的算法进行移植，使之能在SOPHON BM1684X上进行推理。

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)

![CLIP](./CLIP.png)

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16(BM1684X)模型编译和推理
* 支持基于SAIL推理的Python例程


## 3. 准备模型与数据
该模型目前只支持在1684X上运行，已提供编译好的bmodel，​同时，您需要准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
    └── BM1684X
        ├── clip_image_vitb32_bm1684x_f16_16b.bmodel
        ├── clip_image_vitb32_bm1684x_f16_1b.bmodel
        ├── clip_image_vitb32_bm1684x_f16_32b.bmodel
        ├── clip_image_vitb32_bm1684x_f16_8b.bmodel
        ├── clip_text_vitb32_bm1684x_f16_4b.bmodel
        └── text_projection_512_512.npy
```

```
./datasets
    └──
        ├── cifar-100-images
        ├── cifar-100-python
        ├── imagenet_val_1k
        └── test
```

## 4. 模型编译
此部分请参考[Whisper模型的导出与编译](./docs/ChatGLM3_Export_Guide.md)


## 安装环境

### 安装第三方库

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 安装sail

sail安装方法可参考[Sail_Install_Guide](./docs/Sail_Install_Guide.md)


## 使用案例

图像编码
```bash
python embeddings.py
```
```
INFO:root:------------------ Image Encode Time Info ----------------------
INFO:root:Total images: 1000
INFO:root:Total time use: 5387.02ms
INFO:root:Avg time use: 5.39ms
INFO:root:185.63 FPS
```


## 6. 精度测试
### 6.1 测试方法



### 6.2 测试结果
在aishell数据集上，精度测试结果如下：
|   测试平台    |    测试程序   |              测试模型                                 | WER    |
| ------------ | ------------ | ----------------------------------------------------- | ------ |


> **测试说明**：
1. 在使用的模型相同的情况下，wer在不同的测试平台上是相同的。
2. 由于SDK版本之间的差异，实测的wer与本表有1%以内的差值是正常的。

## 7. 性能测试
|    测试平台   |     测试程序      |           测试模型                  |  Preprocess time(ms) |    Inference time(ms)   |
| -----------  | ---------------- | -----------------------------------| --------------------- | ----------------------- |


> **测试说明**：
> 1. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议多次测试取平均值。
> 2. BM1684X SoC的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz。

## See Also

* [OpenCLIP](https://github.com/mlfoundations/open_clip): includes larger and independently trained CLIP models up to ViT-G/14
* [Hugging Face implementation of CLIP](https://huggingface.co/docs/transformers/model_doc/clip): for easier integration with the HF ecosystem
