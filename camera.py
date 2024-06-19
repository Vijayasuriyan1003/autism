# camera.py

import cv2
from random import randint
import PIL.Image
from PIL import Image
import argparse
import shutil

dnn = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

with open('classes.txt') as f:
    classes = f.read().strip().splitlines()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
color_map = {}

class VideoCamera(object):
    def __init__(self):
       
        self.video = cv2.VideoCapture(0)
        
        self.k=1
       
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, frame = self.video.read()

        frame = cv2.flip(frame,1)

        # object detection
        class_ids, confidences, boxes = model.detect(frame)
        for id, confidence, box in zip(class_ids, confidences, boxes):
            x, y, w, h = box
            obj_class = classes[id]

            if obj_class not in color_map:
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                color_map[obj_class] = color
            else:
                color = color_map[obj_class]

            cv2.putText(frame, f'{obj_class.title()} {format(confidence, ".2f")}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            ff=open("get_value.txt","w")
            ff.write(obj_class)
            ff.close()


        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
