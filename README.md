# efficientdet_mmdet
## Overview
This project implements EfficientDet based on mmdetection. To avoid unwanted behaviours caused by modifying the official mmdetection code, 
we separate our customized implementation from the mmdetection code. In other words, although we set the mmdetection in development mode,
we do not modify the mmdetection code in place. 

## Benchmarks on PascalVOC07+12
Model | mAP |
:---:|:---:|
D1 | 0.796|
