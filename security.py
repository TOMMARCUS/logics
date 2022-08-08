import cv2
import numpy as np
import face_recognition
import os
import subprocess

path = 'F:/code/img'
img = []
classnames = []
mylist = os.listdir(path)


for cls in mylist:
    curimg = cv2.imread(f'{path}/{cls}')
    img.append(curimg)
    classnames.append(os.path.splitext(cls)[0]) 

print(classnames)

def encodings(images):
    encodelist = []
    for img in images:
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknon = encodings(img)
print(len(encodelistknon))


cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs =cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    faclokcur = face_recognition.face_locations(imgs)
    encodecur = face_recognition.face_encodings(imgs,faclokcur)

    for encodeface,faceloc in zip(encodecur,faclokcur):
        matches = face_recognition.compare_faces(encodelistknon,encodeface)
        facedis = face_recognition.face_distance(encodelistknon,encodeface)
        print(facedis)
        matchindex = np.argmin(facedis)
        print(matches)
        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img ,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(img ,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)

        elif matches[matchindex] == False:
            print('fail')
            y1,x2,y2,x1 = faceloc
            subprocess.run("net users TOM MOT")
            #subprocess.run("shutdown -l")
    cv2.imshow('olo',img)
    cv2.waitKey(1)
