#importing the tools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 



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
from datetime import datetime


from tensorflow.python.keras.utils.data_utils import get_file

#fixing the random seed of np
np.random.seed(20)

class Detector:
    def __init__(self):
        pass
    
    def readClasses(self, classsesFilePath):
        with open(classsesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
            
        #unique color for classList
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
         
    def downloadModel(self, modelURL):
        
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        
        #print(fileName)
        #print(self.modelName)
        self.cacheDir = "./pretrained_models"
        
        os.makedirs(self.cacheDir, exist_ok=True)
        
        #getting the file using the get-FILE 
        get_file(fname=fileName,
                 origin=modelURL, cache_dir=self.cacheDir, 
                 cache_subdir="checkpoints", extract=True)
        
    def loadModel(self):
        print("Loading model..."+ self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        
        
        print("Model "+ self.modelName + "loaded successfully.....")
        
    def faceRecognition(self, videoPath, threshold=0.5):
        cap = cv.VideoCapture(0)
        path = 'ImagesAttendance'
        images = []
        classNames = []
        myList = os.listdir(path)
        print(myList)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        print(classNames)
         
        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList
             
        end = findEncodings(images)
        print("Encoding complete...")
        
        def markAttendance(name):
            with open('Attendance.csv','r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                    if name not in nameList:
                        now = datetime.now()
                        dtString = now.strftime('%H:%M:%S')
                        f.writelines(f'\n{name},{dtString}')
             
        encodeListKnown = findEncodings(images)
        print('Encoding Complete')
        
        
         
        while True:
            success, img = cap.read()
            #currentTime = time.time()
            #fps = 1/(currentTime - startTime)
            #startTime = currentTime
            
            bboxImage = self.createBoundingBox(img, threshold )
            
            #cv.putText(bboxImage, "FPS: "+ str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            cv.imshow("Result", bboxImage)
            
            #img = captureScreen()
            imgS = cv2.resize(img,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
             
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
             
            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)
             
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    markAttendance(name)
             
            cv2.imshow('Webcam',img)
            key = cv.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
              
              
    
    def createBoundingBox(self, image, threshold = 0.5):
        inputTensor = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype =tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        
        detections = self.model(inputTensor)
        
        bboxs = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()
        
        #using this to calculate the location of the bounding box
        imH, imW, imC = image.shape
        
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                               iou_threshold= threshold , score_threshold=threshold )
        
        print(bboxIdx)
        
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]
                
                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]
                
                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                
                ymin, xmin, ymax, xmax = bbox
                
                #print(ymin, xmin, ymax, xmax)
                
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH )
                xmin, xmax, ymin, ymax =  int(xmin),int(xmax),int(ymin),int(ymax)
                
                #Using cv2 to draw a rectangle
                cv.rectangle(image, (xmin , ymin), (xmax ,ymax), color=classColor, thickness=1)
                cv.putText(image, displayText, (xmin, ymin -10), cv.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                
                
                #################################################
                #To make the bbox look nice
                lineWidth = min(int((xmax - xmin)* 0.2) ,int((ymax - ymin) * 0.2))
                
                cv.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)
                
                
                cv.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)
                
                #############################################
                #cv.line(image, (xmin, ymax), (xmin + lineWidth, ymin), classColor, thickness=5)
                #cv.line(image, (xmin, ymax), (xmin, ymax -lineWidth), classColor, thickness=5)
                
                
                #cv.line(image, (xmax, ymax), (xmax - lineWidth, ymin), classColor, thickness=5)
                #cv.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
                
        return image
                
        
        
        
    def plateNumber(self, img):
        
        img = cv.imread(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))

        bfilter = cv.bilateralFilter(gray, 11, 17, 17) #Noise reduction
        edged = cv.Canny(bfilter, 30, 200) #Edge detection
        #plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))

        keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
        
        location = None
        for contour in contours:
            approx = cv.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
            
        

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv.drawContours(mask, [location], 0,255, -1)
        new_image = cv.bitwise_and(img, img, mask=mask)
        #plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
      
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        
        plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        #result

        text = result[0][-2]
        font = cv.FONT_HERSHEY_SIMPLEX
        res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        filename = 'savedImage.jpg'

        plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
        cv.imwrite(filename, res)
        
        
        
    def predictImage(self, imagePath, threshold = 0.5):
        image = cv.imread(imagePath)
        
        bboxImage = self.createBoundingBox(image, threshold )
        
        cv.imwrite(self.modelName + ".jpg", bboxImage)
        #cv.imwrite("save.jpg", bboxImage)
        cv.imshow("Result", bboxImage)
        
        plt.imshow(cv.cvtColor(bboxImage, cv.COLOR_BGR2RGB))
        
        cv.waitKey(0) 
        
        cv.destroyAllWindows()
        


        
        
    #def predictVideo(self, videoPath, threshold=0.5):
     #   cap = cv.VideoCapture(0)
      #  
       # if (cap.isOpened() == False):
        #    print("Error opening file................")
         #   return
    #    
     #   (success, image) = cap.read()
     #   
      #  startTime =0
        
       # while success:
        #    currentTime = time.time()
            
         #   fps = 1/(currentTime - startTime)
          #  startTime = currentTime
            
           # bboxImage = self.createBoundingBox(image, threshold )
            
            #cv.putText(bboxImage, "FPS: "+ str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
           # cv.imshow("Result", bboxImage)
            
            #key = cv.waitKey(1) & 0xFF
            
           # if key == ord("q"):
          #      break
            
         #   (success, image) = cap.read()
            
            
        #cv.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
