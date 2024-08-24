#pip install cmake
#pip install facerecognition
#pip install opencv
import facerecognition
import cv2
import csv
import numpy as np
from datetime import datetime

vedio_capture =cv2.VedioCapture(0);#0 is for the zeroth
#Load known faces
megha_image=facerecognition.load_image_file("faces/megha.jpg")
megha_encoding=facerecognition.face_encoding(megha_image)[0]#matching each face image with a number ,here 0 ,makes matching much easier

mahitha_image=facerecognition.load_image_file("faces/mahitha.jpg")
mahitha_encoding=facerecognition.face_encoding(mahitha_image)[0]

known_face_encoding=[megha_encoding,mahitha_encoding]
known_face_names=["megha","mahitha"]

#list of expected students
students=known_face_names.copy()

face_locations=[]
face_encodings=[]

#get the current date and time
now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

#make a csv writer
f=open(f"{current_date}.csv","w+",newline="")
lnwriter=csv.writer(f)#the csv writer is passed with the f file pointer

while True:
    _, frame=vedio_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25 ,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #capture the frame  ,check if the vedio capture was sucess or not
    # #resize the frame to mkae it smaller
    #convert to rgb
    #####

    #now recognize faces
     face_locations=facerecognition.face_locations(rgb_small_frame)
     face_encodings=facerecognition.face_encodings(rgb_small_frame,face_locations)

     for face_encoding in face_encodings:
         matches=facerecognition.compare_faces(known_face_encoding,face_encoding)
         face_distance=facerecognition.face_distance(known_face_encoding,face_encoding)
         best_match_index=np.argmin(face_distance)

         if(matches[best_match_index]):
             name=known_face_names[best_match_index]

         #add text if a person is present
         if name in known_face_names:
             font=cv2.FONT_HERSHEY_SIMPLEX
             bottomLeftCornerOfText=face_locations[best_match_index]
             fontScale=1.5
             fontColor=(255,0,0)
             thickness=3
             lineType=2
              cv2.putText(frame,name+"Present",bottomLeftCornerOfText,font,fontScale,fontColor,thickness)


         if name in students:
             students.remove(name)
             current_time=now.strftime("%H:%M:%S")
             lnwriter.writerow([now,current_time,name])


     cv2.imshow("Attendance",frame)
     if cv2.waitKey(1) & 0xFF == ord("q"):#when i press the q button let the while loop end and break out,& is used cause it is bitwise
       break
