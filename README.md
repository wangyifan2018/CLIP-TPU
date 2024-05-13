# CLIP <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备模型与数据](#3-准备模型与数据)
- [4. 模型编译](#4-模型编译)
- [5. 安装环境](#5-安装环境)
  - [5.1 安装第三方库](#51-安装第三方库)
  - [5.2 安装sail](#52-安装sail)
- [6. 推理测试](#6-推理测试)
  - [6.1 使用方式](#61-使用方式)
  - [6.2 运行测试](#62-运行测试)
- [7. 精度测试](#7-精度测试)
  - [7.1 测试方法](#71-测试方法)
  - [7.2 测试结果](#72-测试结果)
- [8. 性能测试](#8-性能测试)
  - [8.1 bmrt\_test](#81-bmrt_test)
  - [8.2 程序运行性能](#82-程序运行性能)


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
    ├── BM1684X
    │   ├── clip_image_vitb32_bm1684x_f16_1b.bmodel
    │   ├── clip_image_vitb32_bm1684x_f16_8b.bmodel
    │   ├── clip_image_vitb32_bm1684x_f16_16b.bmodel
    │   ├── clip_image_vitb32_bm1684x_f16_32b.bmodel
    │   ├── clip_text_vitb32_bm1684x_f16_4b.bmodel
    │   └── text_projection_512_512.npy
    └── onnx
        ├── clip_image_vitb32.onnx
        └── clip_text_vitb32.onnx
```

```
./datasets
    ├── cifar-100-images
    ├── cifar-100-python
    ├── imagenet_val_1k
    └── test
```


如果需要尝试自己编译模型，请参考下一节[模型编译](./#4-模型编译)


## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，此部分请参考[Clip模型的导出与编译](./docs/Clip_Export_Guide.md)。如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。


## 5. 安装环境

### 5.1 安装第三方库

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 安装sail

sail安装方法可参考[SAIL安装指南](./docs/Sail_Install_Guide.md)

## 6. 推理测试

### 6.1 使用方式
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

```bash
usage: embeddings.py [--img_dir IMG_DIR] [--image_model IMAGE_MODEL] [--text_model TEXT_MODEL] [--dev_id DEV_ID] [--save_path SAVE_PATH]
    --img_dir: 默认为'./datasets/imagenet_val_1k', help='Directory of input images.'
    --image_model: 默认为'./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel', help='path of image bmodel'
    --text_model: 默认为'./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel'
    --save_path: 默认为'./results/embeddings.pkl', help='Path to save the embeddings.'
    --dev_id: 默认为0, help='dev id'
```

bmcv版本采用硬件编解码，前处理和推理采用多线程并发

```bash
usage: embeddings_bmcv.py [--img_dir IMG_DIR] [--image_model IMAGE_MODEL] [--text_model TEXT_MODEL] [--dev_id DEV_ID] [--save_path SAVE_PATH]
    --img_dir: 默认为'./datasets/imagenet_val_1k', help='Directory of input images.'
    --image_model: 默认为'./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel', help='path of image bmodel'
    --text_model: 默认为'./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel', help='path of text bmodel'
    --save_path: 默认为'./results/embeddings.pkl', help='Path to save the embeddings.'
    --max_que_size: 默认为128, help='Max size of queue.'
    --dev_id: 默认为0, help='dev id'
```


### 6.2 运行测试
编码图片文件夹
```bash
python3 embeddings.py --img_dir ./datasets/imagenet_val_1k --image_model ./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel --text_model ./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel --save_path ./results/embeddings.pkl --dev_id 0
```
bmcv、多线程版本
```bash
python3 embeddings_bmcv.py --img_dir ./datasets/imagenet_val_1k --image_model ./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel --text_model ./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel --save_path ./results/embeddings.pkl --max_que_size 128 --dev_id 0
```

结果存放在 `./results/embeddings.pkl`

> **测试说明**：
1. text_model在本例程中用于初始化CLIP模型，并无实际运行


## 7. 精度测试
### 7.1 测试方法

测试CIFAR100数据集准确率
```bash
python3 acc_eval.py --image_model ./models/BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel --text_model ./models/BM1684X/clip_text_vitb32_bm1684x_f16_4b.bmodel --dev_id 0
```

> **测试说明**：
1. text_model在本例程中用于初始化CLIP模型，并无实际运行

### 7.2 测试结果
在CIFAR100数据集上，精度测试结果如下：
|   测试平台    |    测试程序    |              测试模型                            | ACC(%) |
| ------------ | ------------ | ----------------------------------------------- | ------ |
|   SE7-32     | acc_eval.py  | clip_image_vitb32_bm1684x_f16_16b.bmodel        | 80.080 |


> **测试说明**：
1. 在使用的模型相同的情况下，acc在不同的测试平台上是相同的。
2. 由于SDK版本之间的差异，实测的精度与本表有1%以内的差值是正常的。
3. 不同batch模型精度相同

## 8. 性能测试
### 8.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径和devid参数
bmrt_test --bmodel models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel
```
测试结果中的`calculate time`就是模型推理的时间，多batch size模型应当除以相应的batch size才是每张图片的理论推理时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                        | calculate time(ms)|
| ----------------------------------------------- | ---------------- |
| BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel |          6.54    |
| BM1684X/clip_image_vitb32_bm1684x_f16_8b.bmodel |          2.73    |
| BM1684X/clip_image_vitb32_bm1684x_f16_16b.bmodel|          2.25    |
| BM1684X/clip_image_vitb32_bm1684x_f16_32b.bmodel|          1.66    |


> **测试说明**：
> 1. 性能测试结果具有一定的波动性；
> 2. `calculate time`已折算为平均每张图片的推理时间；
> 3. SoC和PCIe的测试结果基本一致。

### 8.2 程序运行性能
测试数据集为./datasets/imagenet_val_1k

|    测试平台   |     测试程序           |                测试模型                    |     FPS          |
| -----------  | ---------------------| ------------------------------------------| ---------------- |
|   SE7-32     | embeddings.py        | clip_image_vitb32_bm1684x_f16_8b.bmodel   |  53.10           |
|   SE7-32     | embeddings.py        | clip_image_vitb32_bm1684x_f16_16b.bmodel  |  53.98           |
|   SE7-32     | embeddings.py        | clip_image_vitb32_bm1684x_f16_32b.bmodel  |  53.84           |
|   SE7-32     | embeddings_bmcv.py   | clip_image_vitb32_bm1684x_f16_8b.bmodel   | 325.04           |
|   SE7-32     | embeddings_bmcv.py   | clip_image_vitb32_bm1684x_f16_16b.bmodel  | 387.55           |
|   SE7-32     | embeddings_bmcv.py   | clip_image_vitb32_bm1684x_f16_32b.bmodel  | 473.87           |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议多次测试取平均值。
> 2. BM1684X SoC的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz。
> 3. FPS为每秒钟处理的图片数量，处理一张图片的时间包括前处理、推理、后处理时间
> 4. bmcv的前处理resize接口与clip源码有些许差异
> 5. embeddings.py使用官方torch vision前处理，fps受cpu影响较大
