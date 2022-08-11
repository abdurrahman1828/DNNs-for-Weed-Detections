# Deep Neural Networks for Weed Detections Towards Precision Weeding
 
Alternative non-chemical or chemical-reduced weed control 
methods, especially for herbicide- resistant weeds, 
are critical for long-term and integrated weed management. 
Through weed detection and localization, machine vision 
technology has the potential to enable site- and 
species-specific treatments targeting individual 
weed plants. However, due to unstructured field 
circumstances and large biological variability of weeds, 
robust and accurate weed detection remains a challenging 
endeavor. Deep learning (DL) algorithms, powered by 
large-scale image data, promise to achieve the weed 
detection performance required for precision weeding. 
In this work, a set of DL-based object detectors including 
YOLOv5, RetinaNet, EfficientDet, and Faster RCNN, 
were used for detecting weeds in cotton fields. 
A three-class weed dataset with precise bounding box 
annotations was curated consisting of images collected 
under variable field conditions. Weed detection models 
were built by transferring pretrained object detection 
models to the weed dataset. Impact of eleven data 
augmentation techniques on the weed detection performance 
has been investigated with increasing data size. 
Results demonstrate promising mean average precision 
(mAP) with multi-class cotton weeds and data 
augmentation enhanced the performance of the 
deep object detectors by maximum of 4.2%.
<br />

##Dataset: 
Kaggle link will be provided...
##Model Implementations:
###YOLOv5
Yolov5 implementation is based on tutorial notebook provided 
 by Ultralytics (https://github.com/ultralytics/yolov5) and 
 Roboflow (https://models.roboflow.com/).<br />

###Faster RCNN, Fast-RCNN, and RetinaNet 
Faster RCNN, Fast-RCNN, and RetinaNet implementations are based on tutorial provided by Detectron2 (https://github.com/facebookresearch/detectron2).<br />

###EfficientDet 
EfficientDet implementation is based on https://github.com/wangermeng2021/EfficientDet-tensorflow2 
project which was developed based on 
https://github.com/google/automl/tree/master/efficientdet.
<br />

##Annotation Conversions
VIA to COCO JSON, and COCO JSON to YOLOv5 format 
sciprts has been added in the conversion folder.
##Model Weights
The weights folder contains the trained weights for each model. Separate Google Drive link is provided if the file is large enough.<br />
 
###Code Ownership
The models and code used for our experiments are not 
maintained by us, please give credit to the respective authors 
of the studied models.
