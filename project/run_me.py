from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
#testing another model efficientdet_d4_coco17_
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#testing another model
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"
#testing another model
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
#testing with centernet
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz"


classFile = "coco.names"
#imagePath = "test/5.jpg"
#imagePath = 'image3.jpg'
img = 'image1.jpg'
#cap = cv2.VideoCapture(0)

#videoPath = "test/street2.mp4"
videoPath = 0 # for webcam
threshold = 0.5


detect = Detector()
detect.readClasses(classFile)
detect.downloadModel(modelURL)
detect.loadModel()
detect.plateNumber(img)
#detect.predictImage(imagePath, threshold )
#detect.predictVideo(videoPath, threshold)
detect.faceRecognition(videoPath, threshold)
