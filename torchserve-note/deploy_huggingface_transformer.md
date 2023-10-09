# Deploy Huggingface Transformers using torchserve

## 目的

调研 torchserve 的使用方法, 如何对服务进行简单的配置.

## 简单上手

参考 [Get started](https://pytorch.org/serve/getting_started.html)

注意事项:
- 先用 CPU 推理进行测试, 可以规避一部分库的兼容问题
- 需要安装 jdk 11, 参考: [how to install java on debian 11](https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-debian-11#step-2-managing-java)

总结一下主要步骤:
- 编写 handler, 定义请求的处理过程, 包括 initialize -> preprocess -> inference -> postprocess. 可以继承 [BaseHandler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py), 复用主体框架, 然后实现本模型的处理逻辑.
- 使用 torch-model-archiver 打包模型
- 使用 torchserve 加载打包的文件, 进行在线推理


## 部署 huggingface transformer

官方例子: [Serving Huggingface Transformers using TorchServe](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers)

### 下载模型

待部署的模型: https://huggingface.co/BAAI/bge-large-zh, 下载时需要安装 git-lfs: ``sudo apt install git-lfs``

### 编写 handler

- 参考: [how to write custom handler](https://pytorch.org/serve/custom_service.html)

```py
#coding=utf-8
import os
import json
import logging
import numpy as np

import torch
import sentence_transformers
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("sentence_transformers version {}".format(sentence_transformers.__version__))


"""
# 打包命令
torch-model-archiver -f --model-name bge_large_zh --version 1.0 --serialized-file bge-large-zh/pytorch_model.bin --handler bge-large-zh/handler.py --extra-files "bge-large-zh/" 

# 为了加速调试, 可以打包格式可以不选 mar
# 打包整个文件, 因为 SentenceTransformer 使用需要使用 pooling 配置
torch-model-archiver -f --model-name=bge_large_zh --version=1.0 --serialized-file=bge-large-zh/pytorch_model.bin --handler=bge-large-zh/handler.py --extra-files="bge-large-zh/" --archive-format=no-archive --export-path=no-archive
# 配合上面的启动命令
torchserve --start --ncs --model-store=no-archive --models bge_large_zh=bge_large_zh --ts-config=torchserve.conf --log-config=log4j2.xml

# 本地调试启动命令
torchserve --start --ncs --model-store=model-store --models bge_large_zh=bge_large_zh.mar --ts-config=torchserve.conf --log-config=log4j2.xml
# 在线服务启动命令
#torchserve --start --ncs --model-store=/remote-model-store --models bge_large_zh=bge_large_zh.mar --ts-config=/remote-model-store/torchserve.conf --log-config=/remote-model-store/log4j2.xml
torchserve --start --ncs --model-store=/remote-model-store --models bge_large_zh=bge_large_zh.mar --ts-config=/remote-model-store/torchserve.conf --log-config=/remote-model-store/log4j2.xml --foreground > /dev/null
# 停止服务. 服务停止后, 后台会自动拉起, 但是所做镜像内所做的改动也会被丢弃, 比如手动安装的 pip 包
torchserve --stop

# request torchserve
curl  http://localhost:8081/models
curl  http://localhost:8081/models/bge_large_zh?customized=true
curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/bge_large_zh --data-raw '{"input":"如何使用torchserve部署模型"}'
# 压测
for i in $(seq 100); do curl -X POST --header 'Content-Type: application/json'  curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/bge_large_zh --data-raw '{"input":["教练, 我想打篮球.", "如何使用torchserve部署模型", "怎么训练bert模型", "怎么使用tensorflow训练bert模型", "怎么使用tfserving部署bert模型"]}'; done
"""


class BGEHandler(BaseHandler):
    def __init__(self):
        super(BGEHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        #  running initialize, 
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_material_list = os.listdir(model_dir)
        logger.info("model_material_list: {}".format(model_material_list))
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # 需要在这里指定 device, 不然只有一个 GPU 有负载
        self.model = sentence_transformers.SentenceTransformer(model_dir, device=self.device)
        # use gpu if gpu is available
        self.model.to(self.device)
        # set the model in the evaluation mode, and it will return the layer structure
        self.model.eval()
        logger.info("running warmup")
        self.inference(["样例数据-1", "样例数据-2"])
        logger.info("Transformer model from path {} loaded successfully".format(model_dir))
        logger.info("running initialize, manifest: {}, properties: {}".format(self.manifest, properties))
        self.initialized = True

    def preprocess(self, requests):
        logger.debug("running preprocess, requests: {}".format(requests))
        input_batch = []
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            if isinstance(input_text, dict):
                format_input = input_text
            else:
                format_input = json.loads(input_text)
            # 注意不要开启 torchserve 的 batch 机制
            if isinstance(format_input["input"], str):
                input_batch.append(format_input["input"])
            elif isinstance(format_input["input"], list):
                input_batch = format_input["input"]
            else:
                raise TypeError("expect input to be either string or list[string]")
            logger.debug("Received {}th input: {}".format(idx, input_text))
        return input_batch

    def inference(self, input_batch):
        logger.debug("running inference, input_batch: {}".format(input_batch))
        #return np.random.rand(len(input_batch), 1024)
        return self.model.encode(input_batch, show_progress_bar=False, normalize_embeddings=True)

    def postprocess(self, inference_output):
        logger.debug("running postprocess, inference_output: {}".format(inference_output))
        #return [[1.0] for i in range(inference_output.shape[0])]
        # convert result to list to solve "Invalid model predict output" error
        resp = {
            "code": 0,
            "type": "success",
            "message": "success",
            "data": inference_output.tolist()
        }
        return [json.dumps(resp)]

    def describe_handle(self):
        logger.debug("running describe_handle")
        output_describe = {
            "model_type": "text embedding",
            "model_source": "https://huggingface.co/BAAI/bge-large-zh",
            "input": {
                "element_type": "string",
                "shape": "[batch_size, -1]",
            },
            "output": {
                "element_type": "float",
                "shape": "[batch_size, 1024]",
            },
        }
        return json.dumps(output_describe)
```


### 打包模型

了解 torch-model-archiver 的使用方法: [introduction to torch-model-archiver](https://github.com/pytorch/serve/blob/master/model-archiver/README.md)

```bash
# tree
.
├── 1_Pooling
│   └── config.json
├── README.md
├── config.json # 模型配置文件
├── config_sentence_transformers.json # 模型训练时的包依赖, 推理环境最好保持一致
├── handler.py  # 上一步生成的请求处理 handler
├── modules.json
├── pytorch_model.bin  # 模型 checkpoint, 包含模型权重, 结构信息
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt # 词表
```


#### 打包命令

```bash
torch-model-archiver -f --model-name bge_large_zh --version 1.0 --serialized-file bge-large-zh/pytorch_model.bin --handler bge-large-zh/handler.py --extra-files "bge-large-zh/" 

# 为了加速调试, 可以打包格式可以不选 mar
torch-model-archiver -f --model-name=bge_large_zh --version=1.0 --serialized-file=bge-large-zh/pytorch_model.bin --handler=bge-large-zh/handler.py --extra-files="bge-large-zh/" --archive-format=no-archive --export-path=no-archive
```


### 部署模型

了解 torchserve:
- [basic usage](https://pytorch.org/serve/server.html)
- [advanced configuration](https://pytorch.org/serve/configuration.html)

配置文件:
- [torchserve.config](https://github.com/pytorch/serve/blob/master/docker/config.properties)
- [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml)

```bash
# cat torchserve.config 
# basic command options: https://pytorch.org/serve/server.html
# how to configure torchserve: https://pytorch.org/serve/configuration.html
# bind inference API to all network interfaces with SSL enabled
inference_address=http://0.0.0.0:8080
management_address=http://127.0.0.1:8081
metrics_address=http://127.0.0.1:8082
# set default_workers_per_model to 1 to prevent server from oom when debugging
default_workers_per_model=32
# Allow model specific custom python packages, Be cautious: it will slow down model loading
#install_py_dep_per_model=true
# log configuration: https://pytorch.org/serve/logging.html#modify-the-behavior-of-the-logs
# config demo: https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml 
async_logging=true
#vmargs=-Dlog4j.configurationFile=file:///volcvikingdb-model-store/log4j2.xml
#vmargs=-Dlog4j.configurationFile=file:///root/code/huggingface_store/log4j2.xml


# 启动 torchserve, 推理接口端口默认是 8080
torchserve --start --ncs --model-store=model-store/ --models bge_large_zh=bge_large_zh.mar --ts-config=torchserve.config
# 停止服务. 服务停止后, 后台会自动拉起
torchserve --stop
# 如果打包时没压缩, 可以使用下面的启动命令
torchserve --start --ncs --model-store=no-archive --models bge_large_zh=bge_large_zh --ts-config=torchserve.config
```


### 在线推理

API 介绍: [torchserve REST API](https://pytorch.org/serve/rest_api.html)

部署镜像:

```Dockerfile
FROM pytorch/torchserve:0.8.1-gpu
USER root
#ENV PIP_INDEX_URL=https://***/pypi/simple/ # switch to private source
RUN apt-get update && apt-get install -yq --no-install-recommends curl wget less
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir sentence_transformers==2.2.2
```


```bash
# 请求方式1: 上传文件
echo '{"input":"教练, 我想打篮球."}' > note.txt
curl -X POST  http://localhost:8080/predictions/bge_large_zh -T note.txt
# 请求方式2
curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/bge_large_zh --data-raw '{"input":"如何使用torchserve部署模型"}'
# 简单压测
for i in $(seq 1000); do curl -X POST --header 'Content-Type: application/json'  curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/bge_large_zh --data-raw '{"input":["教练, 我想打篮球.", "如何使用torchserve部署模型", "怎么训练bert模型", "怎么使用tensorflow训练bert模型", "怎么使用tfserving部署bert模型"]}'; done
```

查看 inference API 定义:

```bash
# https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/inference_open_api.json
# curl -X OPTIONS http://localhost:8080
{
  "openapi": "3.0.1",
  "info": {
    "title": "TorchServe APIs",
    "description": "TorchServe is a flexible and easy to use tool for serving deep learning models",
    "version": "0.8.1"
  },
  "paths": {
    "/predictions/{model_name}": {
      "post": {
        "description": "Predictions entry point to get inference using default model version.",
        "operationId": "predictions",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model.",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "Input data format is defined by each model.",
          "content": {
            "*/*": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Output data format is defined by each model.",
            "content": {
              "*/*": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              }
            }
          },
        ...
```


### 获取当前模型部署的模型列表

参考文档:  [MANAGEMENT API](https://pytorch.org/serve/management_api.html#management-api)

```bash
# curl "http://localhost:8081/models"
{
  "models": [
    {
      "modelName": "bge_large_zh",
      "modelUrl": "bge_large_zh.mar"
    }
  ]
}
```


## 问题记录

- 在线服务上没用没使用 GPU 卡: [how to make huggingface transformer model use gpu](https://github.com/huggingface/transformers/issues/2704)
- [把 huggingface transformer导出成 TorchScript 格式](https://huggingface.co/docs/transformers/torchscript)
- [torch-model-archiver 怎么打包整个文件夹](https://github.com/pytorch/serve/issues/1227)
- [对同一个文本, 使用不同的 batch size, batch_size=1 和 batch_size>1时得到的 embedding 结果有细微差别](https://huggingface.co/BAAI/bge-large-zh/discussions/5)


## 参考文档

- [deploy huggingface bert to production with torchserve](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)
- [BERT TorchServe Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/inference/tutorial-torchserve-neuronx.html)
- [A Quantitative Comparison of Serving Platforms for Neural Networks](https://biano-ai.github.io/research/2021/08/16/quantitative-comparison-of-serving-platforms-for-neural-networks.html)