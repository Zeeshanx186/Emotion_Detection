from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import csv

videopath = 'sad.mp4'
cap = cv2.VideoCapture(0)

FMD = FaceMeshDetector()
class_name = 'sad'

while cap.isOpened():
    rt,frame = cap.read()
    frame = cv2.resize(frame,(720,480))
    img , faces = FMD.findFaceMesh(frame)

    if faces:
        face = faces[0]
        face_data = list(np.array(face).flatten())
        face_data.insert(0,class_name)

        with open('data.csv','a',newline='') as f:
            csv_writer = csv.writer(f,delimiter = ',')
            csv_writer.writerow(face_data)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
