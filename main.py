from distutils.command.config import config
from email.mime import image
import cv2 as cv
import numpy as np

#abrir el archivo de clases
with open('object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().splitlines()


#cargar el modelo DNN, en este caso se uso mobilnet
model = cv.dnn.readNet(model='frozen_inference_graph.pb',config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')

#leer el video de la camara
cap = cv.VideoCapture(0)

while 1:
    sucess, video = cap.read()
    blob = cv.dnn.blobFromImage(image=video, size=(300, 300), mean=(104, 117, 123), swapRB=True)
    model.setInput(blob)
    output = model.forward()

    for detection in output[0,0, : , :]:
        confidence = detection[2]
        if confidence > .4:
            class_id = detection[1]
            class_name = class_names[int(class_id)-1]
            cv.putText(video, class_name, (50,50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
            cv.imshow("Video", video)
            if cv.waitKey(1) == ord('q'):
                 break
cv.destroyAllWindows()
