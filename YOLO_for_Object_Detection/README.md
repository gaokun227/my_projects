# Object detection with YOLO (You Only Look Once) 

## Overview

This python script is based on a project assignment in the Convolutional Neural Network course offered by Andrew Ng via Coursera.org

In this assigment, we use a pretrained [YOLO](https://pjreddie.com/darknet/yolo/) model for detecting objects in a sequence of images (./images_in). 

## Running
`python yolo_main.py`

## Key Elements

* yolo_eval: convert the output of YOLO encoding (a lot of boxes) to the predicted boxes along with their scores, box coordinates and classes
   * yolo_filter_boxes: filters YOLO boxes by thresholding on object and class confidence
   * yolo_non_max_suppression: applies non-max suppression to a set of boxes 
 
## Output
![cars.gif](images/cars.gif)

The input images are provided by the Convolutional Neural Network course on Coursera.org
