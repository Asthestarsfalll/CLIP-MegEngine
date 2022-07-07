# CLIP

此仓库包含MegEngine实现的多模态模型`CLIP`，但不包含训练及测试代码。

`models.py`中实现了CLIP的不同配置：`RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`, `ViT-B-32`, `ViT-B-16`, `ViT-L-14`和`ViT-L-14-336px`。

你可以使用以下预训练模型进行体验。

| 模型           | 权重                                                         |
| -------------- | ------------------------------------------------------------ |
| RN50           | [link](https://data.megengine.org.cn/models/weights/RN50.pkl) |
| RN101          | [link](https://data.megengine.org.cn/models/weights/RN101.pkl) |
| RN50x4         | [link](https://data.megengine.org.cn/models/weights/RN50x4.pkl) |
| RN50x16        | [link](https://data.megengine.org.cn/models/weights/RN50x16.pkl) |
| RN50x64        | [link](https://data.megengine.org.cn/models/weights/RN50x64.pkl) |
| ViT-B-32       | [link](https://data.megengine.org.cn/models/weights/ViT-B-32.pkl) |
| ViT-B-16       | [link](https://data.megengine.org.cn/models/weightsViT-B-16.pkl) |
| ViT-L-14       | [link](https://data.megengine.org.cn/models/weights/ViT-L-14.pkl) |
| ViT-L-14-336px | [link](https://data.megengine.org.cn/models/weights/ViT-L-14-336px.pkl) |

## 零样本（zero-shot）分类

用户可以使用以下模板使用`CLIP`进行零样本图像分类。

### 加载网络

```python
from megengine import hub
modelhub = hub.import_module(repo_info='megengine/models', git_host='github.com')

#加载网络结构及预训练模型
clip = hub.load("megengine/models", "rn50", pretrained=True)
clip.eval()

#查看网络配置信息
clip.model_config()
```

### 数据处理

```python
import cv2
from megengine.data.transform import CenterCrop, Compose, Normalize, Resize

#数据处理
image_resolution = clip.image_resolution  # clip需要固定输入图片的大小
transfroms =  Compose([
    Resize(image_resolution, interpolation=cv2.INTER_BICUBI),
    CenterCrop(image_resolution),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

```

数据处理构建完毕后需要用户手动构建`Dataloader`。

### 构建文本模板和类别

`clip`需要一些文本模板来描述某一张图片，比如：`a photo of {}.`，`a photo of many {}.`等，大括号中可以填入各种类别名称。这样为每一个类别都生成n句话，再使用文本编码器和图片编码器的输出向量做相似度计算，得分高者则认为其为该类的概率更高。

`clip`中内置了imagenet的80个文本模板，可通过调用以下代码得到。

```python
imagenet_templates = modelhub.get_imagenet_templates()
```

对于不同的数据集可以采用不同的文本模板，其格式如下：

```
templates: List[str] = [
	'a bad photo of a {}.',
	'a photo of many {}.',
	...
]
```

同时我们需要各个类别的名称，可通过调用以下代码得到imagenet的1000个类别。

```python
imagenet_classes = modelhub.generate_imagenet_classes()
```

对于不同的数据集需要使用对应的类别名称，其格式如下：

```python
classes：List[str] = [
    'tench',
    'goldfish',
    ...
]
```

### 生成零样本分类权重

使用下列代码生成权重。

```python
zeroshot_wieghts = modelhub.zeroshot_classifier(clip, imagenet_classes, imagenet_templates)
```

### 预测

```python
# 传入模型、dataloader和零样本权重即可进行预测
top1, top5 = modelhub.predict(clip, loader, zeroshot_wieghts)
print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")
```

