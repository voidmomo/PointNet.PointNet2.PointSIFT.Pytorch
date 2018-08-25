# PointNet.PointNet2.PointSIFT.Pytorch

# Abstract

The code include three models:

1. pointnet
2. pointnet++
3. pointsift

The task is classification, maybe I will add the segmentation or detection. 

The dataset include the modelnet40 (avaliable), kitti(future), apolloscape (future)and so on.

# Installation

1. Python3.x
2. pytorch
3. numpy

# Usage

```
python main.py --train=True --save_path=/your/path/to/save --model=pointnet --dataset=modelnet40
```

**Actually, the c extension is useless, so, any grouping or selecting operations is running in the cpu, so the train speed is a little slow.**

the files you need to change are only `main.py`, if you want to add some new datasets, you need add the code into the `dataset.py `and new a `config/your dataset.py` in the config directory. 