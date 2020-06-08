import cv2  
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 


#cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture('./input_video/' + sys.argv[1])
  
# loop runs if capturing has been initialized. 
while 1:  
  
    ret, img = cap.read()  
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
  
    # Display video feed
    cv2.imshow('img',img) 
  
    # Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()