import numpy as np
import time
import cv2
import os
import matplotlib
matplotlib.rcParams['figure.figsize']= (5.0,5.0)
import matplotlib.pyplot as plt


class Yolo_Implementation(object):
    def __init__(self,labels_path,weights_path,config_path,test_image,score_threshold=0.1,nms_threshold=0.2):
        self.score_threshold = score_threshold
        self.nms_threshold   = nms_threshold
        self.weights_path    = weights_path
        self.config_path     = config_path
        self.image           = cv2.imread(test_image)
        (self.H, self.W)     = self.image.shape[:2]
        self.labels          = open(labels_path).read().strip().split("\n")
        self.colors          = np.random.randint(0, 255, size=(len(self.labels), 3),dtype="uint8")
      
    def build_model(self):
        model              = cv2.dnn.readNetFromDarknet(self.config_path,self.weights_path)
        blob               = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (480, 480),swapRB=True, crop=False)
        layers_yolo        = model.getLayerNames()
        yolo_layers_needed = [layers_yolo[i[0]-1] for i in model.getUnconnectedOutLayers()]
        retval             = model.setInput(blob)
        layer_outputs      = model.forward(yolo_layers_needed)
        return layer_outputs
    
    def yolo_filter_boxes(self):
        boxes                = []
        probabilities        = []
        classIDs             = []
        layer_outputs        = self.build_model()
        for output in layer_outputs:
            for detection in output:
                if detection[4]>0.0:                                                ### Detecting the presence of object 
                    scores  = detection[5:]                                         ### Capturing the probabilities of corresponding class ID 
                    classid =  np.argmax(scores)                                    ### Finding the class ID with maximum probability 
                    prob = np.max(scores)                                           ### Finding maximum probability
                    if prob > self.score_threshold:                                      ### Thresholding to filter yolo boxes (score threshold) 
                        box = detection[0:4]*np.array([self.W,self.H,self.W,self.H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        probabilities.append(float(prob))
                        classIDs.append(classid)
        return boxes,probabilities,classIDs
    
    def yolo_non_max_suppress(self):
        boxes,probabilities,classIDs = self.yolo_filter_boxes()
        idxs = cv2.dnn.NMSBoxes(boxes, probabilities,self.score_threshold,self.nms_threshold)
        if len(idxs)>0:
            for i in idxs.flatten():
                (x,y) = (boxes[i][0],boxes[i][1])
                (w,h) = (boxes[i][2],boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], 100*round(probabilities[i],4))
                cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
        plt.figure(figsize=(20,20));
        plt.imshow(self.image[:,:,::-1])
        plt.axis('off');


        

