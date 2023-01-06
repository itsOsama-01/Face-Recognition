import face_recognition
import cv2
import numpy as np

image =face_recognition.load_image_file('me.jpg')
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# face = face_recognition.face_locations(image_rgb)[0]
# copy = image_rgb.copy()

#cv2.rectangle(copy,(face[3],face[0]),(face[1],face[2]),(255,255,0),2)

train_encoding=face_recognition.face_encodings(image_rgb)[0]

testImage= face_recognition.load_image_file('me2.jpg')
testImage=cv2.cvtColor(testImage,cv2.COLOR_BGR2RGB)
test_encodings=face_recognition.face_encodings(testImage)[0]

print(face_recognition.compare_faces([train_encoding],test_encodings))
cv2.imshow('first',image_rgb)
cv2.imshow('second',testImage)

cv2.waitKey(0)