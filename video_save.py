
# coding: utf-8

# In[22]:

# import numpy as np
# import cv2

# detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(1)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# while(True):
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     faces = detector.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
#     cv2.imshow('frame',img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()


# In[21]:

import cv2
import numpy as np

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
# Create a VideoCapture object
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
    print("Unable to read camera feed")
 
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
 
while(True):
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    if ret == True: 
     
    # Write the frame into the file 'output.avi'
        out = cv2.VideoWriter('outpy1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        out.write(frame)

    # Display the resulting frame    
        cv2.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Break the loop
    else:
        break 
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 

