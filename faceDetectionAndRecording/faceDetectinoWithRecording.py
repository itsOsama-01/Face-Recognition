import cv2
import time
import datetime
import json


def doEntry():
    day=datetime.datetime.now().strftime("%d-%m-%Y")
    time=datetime.datetime.now().strftime("%H-%M-%S")
    dicToAppend={"entry":{"date":f"{day}","time":f"{time}"}}

    newJsonEnd=","+json.dumps(dicToAppend)[1:-1]+"}\n"

    with open("log.json","r+") as f:
        f.seek(0,2)
        index=f.tell()

        while not f.read().startswith('}'):
            index-=1
            f.seek(index)
        f.seek(index)
        f.write(newJsonEnd)


#Capturing video from webcam
cap=cv2.VideoCapture(0)

#Loading the cascade classifier for detection of faces
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Set-up variables for recording functionality
detection = False
isTimerStarted=False
noDetectionTime=None

#Recording module of OpenCv
frameSize=(int(cap.get(3)),int(cap.get(4)))
videoFormat=cv2.VideoWriter_fourcc(*"mp4v")


while True:
    #Getting the current frame
    flag,frame=cap.read()

    #Converting to a gray image for processing
    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detecting faces
    faces=faceCascade.detectMultiScale(grayFrame,1.3,5)

    #Recording video when a face is detected
    if len(faces) > 0:
        if detection:
            isTimerStarted=False
        else:
            detection=True
            dayAndTime=datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
            doEntry()
            videoRecord = cv2.VideoWriter(f"{dayAndTime}.mp4", videoFormat, 20, frameSize)
    elif detection:
        if isTimerStarted:
            if time.time()-noDetectionTime > 5:
                detection =False
                isTimerStarted=False
                videoRecord.release()
        else:
            isTimerStarted=True
            noDetectionTime=time.time()
    if detection:
        videoRecord.write(frame)

    #Drawing rectangles around faces
    for (x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),3)

    #Displaying Current image with rectangles around faces
    cv2.imshow("WebCam",frame)

    #Code to quit the program with button q
    if cv2.waitKey(1)==ord('q'):
        break

# Releasing resourses
cap.release()
cv2.destroyAllWindows()