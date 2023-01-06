import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path='rememberThese'

images=[]
classNames=[]
mylist=os.listdir(path)

for cl in mylist:
    curImage=cv2.imread(f'{path}/{cl}')
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])

def find_encoding(images):
    encodeList=[]
    for image in images:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encoding=face_recognition.face_encodings(image)[0]
        encodeList.append(encoding)
    return encodeList

def takeAttendance(name):
    with open('Attendance.csv','r+') as file:
        myDataList=file.readlines()
        entries=[]

        for line in myDataList:
            entry=line.split(',')
            entries.append(entry[0])
        if name not in entries:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d/%B/%Y')
            file.writelines(f'\n{name},{time},{date}')

encodedFace_train=find_encoding(images)


capture=cv2.VideoCapture(0)

while True:
    successFlag,image=capture.read()
    imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    faceLocations=face_recognition.face_locations(imgS)
    curFaceEncoding=face_recognition.face_encodings(imgS,faceLocations)

    #Drawing a rectangle around every face



    for encodeFace,faceLoc in zip(curFaceEncoding,faceLocations):
        matches=face_recognition.compare_faces(encodedFace_train,encodeFace)
        faceDist=face_recognition.face_distance(encodedFace_train,encodeFace)

        matchIndex=np.argmin(faceDist)

        print(matches[matchIndex])
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceLoc

            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)

            cv2.putText(image, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            takeAttendance(name)
        else:
            y1, x2, y2, x1 = faceLoc

            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)

            cv2.putText(image, "Unknown", (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            takeAttendance("Unknown")

    cv2.imshow("IT FRIKKIN' WORKS",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
