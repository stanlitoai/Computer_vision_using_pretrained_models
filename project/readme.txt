############################################
This is object detector/recognition with facial recognition and plate number detector
it can detect and recognize things with high accuracy

Below are the requirement to be able to run the code

################################
Labraries to install are :


import cv2 as cv
import time
import os
import tensorflow as tf
import numpy as np


from matplotlib import pyplot as plt
import imutils
import easyocr


import cv2
import face_recognition

###########################

after that, you can run the code by navigation to #run_me.py

i commented the MODELURL there. you can plate with it.
in the future if you want to try a new model, all you will do is just copy the model url and paste it
lemme give you example...

modelURL = ""
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"

i made the facial recognition in such a way that once it recognize your face, it stores your name and time 
inside #Attendance.csv


