import cv2
import numpy as np
import serial
try:
    ser=serial.Serial('COM10',9600)
except:
    print('verifier le port utiliser')
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec= cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingDatas.yml")
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),255,0,0),2
        Id,coef=rec.predict(gray[y:y+h,x:x+w])
        
        if(Id==1):
            
            Id="mourad"
            val='1'
            
        else:
            Id='unknown'
            val='0'
            
        ser.write(val.encode())
        
        #if(Id==9):
            #Id="el kouch"
        #elif(Id==2):
            #Id="nouhaila"
       
       # elif(Id==4444):
           # Id='hajar'
        #else:
           #Id="inconnu"
                       
        cv2.rectangle(img, (x-22,y-90),(x+w+22,y-22),(0,255,0),-1)
        cv2.putText(img, str(Id), (x,y-40), font, 2, (255,255,255), 3)
    cv2.imshow('face',img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
