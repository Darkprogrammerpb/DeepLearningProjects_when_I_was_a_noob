import numpy as np
import time
import cv2
import os
import matplotlib
matplotlib.rcParams['figure.figsize']= (5.0,5.0)
import matplotlib.pyplot as plt
from yolo_wrapper import *
labels_path  = os.getcwd()+'\\yolo-coco\\coco.names'                                     ### load label path
weights_path = os.getcwd()+'\\yolo-coco\\yolov3.weights'                                 ### load weights path
configs_path = os.getcwd()+'\\yolo-coco\\yolov3.cfg'                                     ### Load configuration path
test_image   = os.getcwd()+'\\sample images\\bean and teddy.jpg'                         ### Load test image path
yolo_class = Yolo_Implementation(labels_path,weights_path,configs_path,test_image)       ### call wrapper library created
yolo_class.yolo_non_max_suppress()                                                       ### Image created
