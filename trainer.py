import os
import cv2
import numpy as np
from PIL import Image
rec= cv2.face.LBPHFaceRecognizer_create()
path='dataset'
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f)for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNP=np.array(faceImg)
        Id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)
        Ids.append(Id)
        cv2.imshow('training',faceNP)
        cv2.waitKey(10)
    return Ids, faces
Ids,faces=getImagesWithID(path)
rec.train(faces,np.array(Ids))
rec.write('recognizer\\trainingDatas.yml')
cv2.destroyAllWindows()
                
                
