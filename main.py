#Importacion de las librerias
import cv2 as cv

#abrir el archivo de clases
with open('object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().splitlines()


#cargar el modelo DNN, en este caso se uso mobilnet
model = cv.dnn.readNet(model='frozen_inference_graph.pb',
                    config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                    framework='TensorFlow')

#leer el video de la camara
cap = cv.VideoCapture(0)

#en un cliclo infinito
while 1:
    #leer la camara
    sucess, video = cap.read()

    #crear un blob de la imagen
    blob = cv.dnn.blobFromImage(image=video, size=(300, 300), mean=(104, 117, 123), swapRB=True)

    #aplicar el modelo al blob
    model.setInput(blob)
    output = model.forward()

    #ciclo para detectar a cual clase pertenece la imagen
    for detection in output[0,0, : , :]:
        confidence = detection[2]
        #si la confianza es arriba de 6 dibujar el nombre de la clase a la que pertenece
        if confidence > .6:
            class_id = detection[1]
            class_name = class_names[int(class_id)-1]
            cv.putText(video, class_name, (50,50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        
        #mostrar la imagen
        cv.imshow("Video", video)

cv.destroyAllWindows()
