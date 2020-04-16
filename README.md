# Hit-Detector Code Base

Implementation of our CVPR2020 paper [Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection](https://arxiv.org/pdf/2003.11818.pdf)

We released the searched Hit-Detector Architecture.

### Environments
- Python 3.6
- Pytorch>=1.1.0
- Torchvision == 0.3.0

You can directly run the code ```sh env.sh``` to setup the running environment.

### Data Preparatoin

Your directory tree should be look like this:

````bash
$HitDet.pytorch/data
├── coco
│   ├── annotations
│   ├── train2017
│   └── val2017
│
├── VOCdevkit
│   ├── VOC2007
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   └── SegmentationObject
│   └── VOC2012
│       ├── Annotations
│       ├── ImageSets
│       ├── JPEGImages
│       ├── SegmentationClass
│       └── SegmentationObject
````

### Getting Start

Train the searched model:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --gpus 8 --validate \
--launcher pytorch --work_dir YOURDIR --config configs/nas_trinity/two-stage_hitdet.py
```

### Results on COCO minival

| Model | Params | mAP |
| :---- | :----: | :----:|
| FPN | 41.8M | 36.6 |
| Hit-Det | 27.6M | 41.3 |

## Citation
```
@InProceedings{guo2020hit,
author = {Guo, Jianyuan and Han, Kai and Wang, Yunhe and Zhang, Chao and Yang, Zhaohui and Wu, Han and Chen, Xinghao and Xu, Chang},
title = {Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection},
booktitle = {arXiv preprint arXiv:2003.11818},
year = {2020}
}
```

## Acknowledgement
Our code is based on the open source project [MMDetection](https://github.com/open-mmlab/mmdetection).