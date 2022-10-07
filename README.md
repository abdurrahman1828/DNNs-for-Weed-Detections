# Performance Evaluation of Deep Learning Object Detectors for Weed Detection for Cotton
 
Alternative non-chemical or chemical-reduced weed control tactics are critical for future integrated weed management, especially for herbicide-resistant weeds. Through weed detection and localization, machine vision technology has the potential to enable site- and species-specific treatments targeting individual weed plants. However, due to unstructured field circumstances and the large biological variability of weeds, robust and accurate weed detection remains a challenging endeavor. Deep learning (DL) algorithms, powered by large-scale image data, promise to achieve the weed detection performance required for precision weeding. In this study, a three-class weed dataset with bounding box annotations was curated, consisting of 848 color images collected in cotton fields under variable field conditions. A set of 13 weed detection models were built using DL-based one-stage and two-stage object detectors, including YOLOv5, RetinaNet, EfficientDet, Fast RCNN and Faster RCNN, by transferring pretrained the object detection models to the weed dataset. RetinaNet (R101-FPN), despite its longer inference time, achieved the highest overall detection accuracy with a mean average precision (mAP@0.50) of 79.98%. YOLOv5n showed the potential for real-time deployment in resource-constraint devices because of the smallest number of model parameters (1.8 million) and the fastest inference (17 ms on the Google Colab) while achieving comparable detection accuracy (76.58% mAP@0.50). Data augmentation through geometric and color transformations could improve the accuracy of the weed detection models by a maximum of 4.2%. The software programs and the weed dataset used in this study are made publicly available (https://github.com/abdurrahman1828/DNNs-for-Weed-Detections; www.kaggle.com/yuzhenlu/cottonweeddet3).
## Papers
[Journal Article](https://www.sciencedirect.com/science/article/pii/S2772375522000910)

Rahman, A., Lu, Y., & Wang, H., Performance Evaluation of Deep Learning Object Detectors for Weed Detection for Cotton, Smart Agricultural Technology (2022), DOI: 10.1016/j.atech.2022.100126.

[Conference Proceeding](https://doi.org/10.13031/aim.202200845)

Rahman, A., Lu, Y., & Wang, H. (2022). Deep Neural Networks for Weed Detections Towards Precision Weeding. In 2022 ASABE Annual International Meeting (p. 1). American Society of Agricultural and Biological Engineers.

## Dataset 
Kaggle Link: [Cottonweeddet3](https://www.kaggle.com/datasets/yuzhenlu/cottonweeddet3)

## Annotation Conversions
VIA to COCO JSON `via2coco.py`, and COCO JSON to YOLOv5 format `coco2yolov5.py`
sciprts has been added in the *conversion* folder.

## Data Augmentation
A data augmentation script `augment.py` has been added in the 
*data_augmentation* folder. One can create 2x, 4x, and 8x sized augmented data using this.


## Model Implementations:
### YOLOv5
Yolov5 implementation is based on tutorial notebook provided 
 by [Ultralytics ](https://github.com/ultralytics/yolov5) and 
 [Roboflow](https://models.roboflow.com/).

### Faster RCNN, Fast-RCNN, and RetinaNet 
Faster RCNN, Fast-RCNN, and RetinaNet implementations are based on tutorial 
provided by [Detectron2](https://github.com/facebookresearch/detectron2).

### EfficientDet 
EfficientDet implementation is based on 
[EfficientDet-tensorflow2](https://github.com/wangermeng2021/EfficientDet-tensorflow2) 
project which was developed based on 
[efficientdet](https://github.com/google/automl/tree/master/efficientdet) repository.


## Model Weights
The weights folder contains the trained weights for each model. Separate
[drive link](https://drive.google.com/drive/folders/1myQrC5Wtd87XTkxMsg3DuCVPXDcIyM2x?usp=sharing)
 is provided for larger files.
### Code Ownership
The models and code used for our experiments are not 
maintained by us, please give credit to the respective authors 
of the studied models.
