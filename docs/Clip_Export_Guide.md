# Clip模型的导出与编译
可以直接下载我们已经导出的onnx模型，推荐在mlir部分提供的docker中完成转bmodel模型。
**注意**：
- 编译模型需要在x86主机完成。

## 1 TPU-MLIR环境搭建
模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

### 1.1 安装docker
若已安装docker，请跳过本节。
```bash
# 安装docker
sudo apt-get install docker.io
# docker命令免root权限执行
# 创建docker用户组，若已有docker组会报错，没关系可忽略
sudo groupadd docker
# 将当前用户加入docker组
sudo usermod -aG docker $USER
# 切换当前会话到新group或重新登录重启X会话
newgrp docker​
```
> **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

### 1.2 下载并解压TPU-MLIR
从sftp上获取TPU-MLIR压缩包
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/mlir/tpu-mlir_v1.6.135-g12c3f90d8-20240327.tar.gz
```

### 1.3 创建并进入docker
TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:latest
# 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 此时已经进入docker，并在/workspace目录下
# 初始化软件环境
cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
source ./envsetup.sh
```
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考算能官网的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。


## 2 获取onnx

需要通过源码来导出 onnx 文件，clip的变种很多，但是思想类似，以 openai 原始仓库[CLIP官方开源仓库](https://github.com/openai/CLIP)为例。

模型分encode_image和encode_text两部分，以ViT-B/32模型为例，如果需要导出encode_image部分，修改源码 CLIP/clip/model.py:358
```python
    def forward(self, image):
        image_features = self.encode_image(image)
        # text_features = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return image_features
```
然后运行以下代码导出onnx模型

```python
import torch
from clip import *
from PIL import Image
import torch

device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog"] * 256).to(device)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        image,                 # model input (or a tuple for multiple inputs)
        "clip_image_vitb32.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'image': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['image'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

如果需要导出text_image部分，同理
```python
    def forward(self, text):
        # image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return text_features
```

然后运行以下代码导出onnx模型

```python
import torch
from clip import *
from PIL import Image
import torch

device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog"] * 256).to(device)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        text,                 # model input (or a tuple for multiple inputs)
        "clip_text_vitb32.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'text': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['text'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

## 3 bmodel编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。建议使用TPU-MLIR编译BModel。

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](./Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel，具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)相应版本的SDK中获取)。

- 生成FP32 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台（**支持BM1684X），如：

```bash
./scripts/gen_fp32bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684`等文件夹下生成`clip_image_vitb32_bm1684x_f32_1b.bmodel`文件，即转换好的FP32 BModel。

- 生成FP16 BModel

​本例程在`scripts`目录下提供了TPU-MLIR编译FP16 BModel的脚本，请注意修改`gen_fp16bmodel_mlir.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，并在执行时指定BModel运行的目标平台 （**支持BM1684X），如：

```bash
./scripts/gen_fp16bmodel_mlir.sh bm1684x
```

​执行上述命令会在`models/BM1684X/`等文件夹下生成`clip_image_vitb32_bm1684x_f16_1b.bmodel`文件，即转换好的FP16 BModel。