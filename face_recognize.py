
from imutils.video import VideoStream
from imutils.video import FPS
import numpy
import time
import cv2
import sys
import os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'database'
present = []



(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)

        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (112, 92)


(images, lables) = [numpy.array(lis) for lis in [images, lables]]
for abc in range(len(images)):
	print("[INFO] Processing image {}/{}".format(abc+1,len(images)))
	time.sleep(0.03)


print("\n[INFO] Loading Face Recognizer...")
model = cv2.face.FisherFaceRecognizer_create()
print("\n[INFO] Training Model...")
model.train(images,lables)


face_cascade = cv2.CascadeClassifier(haar_file)
print("\n[INFO] Starting Video Stream...")
vs = VideoStream(src=0).start()
fps = FPS().start()
while True:
    im = vs.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if prediction[1]<300:
            cv2.putText(im,'%s - %.2f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.45,(0, 0, 255),2)
        else:
            cv2.putText(im,'Not Recognized',(x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.45,(0, 0, 255),2)
        present.append(names[prediction[0]])
    fps.update()
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
fps.stop()
print("\n[INFO] Elasped Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
myset = set(present)
for x in myset:
    print("Student present : ",format(x))

cv2.destroyAllWindows()
vs.stop()
