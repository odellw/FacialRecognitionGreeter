# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:03:07 2018

@author: walker the best of the best
"""

#### Import Packages
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import time
import sys

total_pics = 25
pics_to_remove = 3
l_limit = 4.2
f_limit = 450
faces_left = 13

EigenFace = cv2.face.EigenFaceRecognizer_create(8) # createEigenFaceRecognizer(15)      # creating EIGEN FACE RECOGNISER
FisherFace = cv2.face.FisherFaceRecognizer_create(8) #  createFisherFaceRecognizer(2)     # Create FISHER FACE RECOGNISER
FisherFace2 = cv2.face.FisherFaceRecognizer_create(8)
LBPHFace = cv2.face.LBPHFaceRecognizer_create(1,1,7,7) # createLBPHFaceRecognizer(1, 1, 7,7) # Create LBPH FACE RECOGNISER

EigenFace.read('/home/pi/Desktop/EIGEN_Fin_65_10.yml')

FisherFace.read('/home/pi/Desktop/FISHER_Fin_65_10.yml')
FisherFace2.read('/home/pi/Desktop/FISHER_Fin_OTHER_65_10.yml')

LBPHFace.read('/home/pi/Desktop/LBPH69.yml')

print('All of our inputs worked')

#### Face Cascades and Recognition Models
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/haarcascade_frontalface_default.xml')

print('we have loaded all of our files')


### IDs we are using
IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
Names = ['Nick','Walker','Melissa','Peter','Karen','Jon_Marchetti','Harsha_Alturi','Yumel','Leigh',
             'Omar','Abe','Tom','Mitch','Chris_Lally','Chris_st_Jean','Celia','Miguel','Vik',
             'Larisa','Julian','Christian','Emily','Karen','Emma_MacPhail','Gabriella']
    
 
### Make people dataframe   
people = pd.DataFrame({"IDS": IDS,
                       "Names": Names})

people = people.set_index("IDS")
            

### to be used when talking to ppl
def say_something_and_wait(name):
    file_name = name + ".mp3"
    
    command_line_action = 'omxplayer -o local /home/pi/Desktop/greetings/' + file_name
    os.system(command_line_action)
    time.sleep(7)



### Set up video
time.sleep(2) 
cap = cv2.VideoCapture(0)

### Let's start the loop
time.sleep(1)
while True:
    
    ### Assign variables
    count = 0
    ten_faces = []
    
    ### We want 30 photos, 
    while count < total_pics:    
        
        ### We set the time, because if it goes longer then 20s per person then
        # we want to break
        
        if count == 0:
            start_time = time.time()

        # Capture Image save as img   
        success, img = cap.read()
        
        # Convert to grayscale and detect face    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        #Loop over faces    
        for (x,y,w,h) in faces:
            
            ### Color to gray
            destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im = Image.fromarray(destRGB)        
            persons_face = im.crop((x, y, x+w, y+h))
            
            ### Only get big pics
            if persons_face.size[0] > 95:
                count += 1
                
                ### the first 8 images the camera is probably still adjusting
                ### I don't want to write the code to get rid of blurry images
                ### The reason for this is because I found some code online 
                ### That did this, and it didn't do anything other then 
                ### Get rid of all of the images of my dark skinned roomate
                ### Racism is clearly still alive in America
                ### Clearly the robots are taking over and plan to create an al white, sentient society... we should all be greatly concerned
                ### Disregard this, jon was fucking with my code
                
                if count < pics_to_remove:
                    break
                persons_face = persons_face.resize((200, 200))
                
                # if we want to see what faces look like, uncomment this
                faceNP = np.array(persons_face, 'uint8') 
                
                ten_faces.append(faceNP)
                
                print(count)

        elapsed_time = time.time() - start_time
        
        #if we've waited more then 20s let's get out <-- Black People
        if elapsed_time > 25:
            ten_faces = []
            break
   
    ### if we've waited more then 20s then let's go to the beg of the loop
    print(ten_faces)
    if ten_faces == []:
        continue
    
    ### Lets predict some stuff
    Eigen_predictions = [EigenFace.predict(e) for e in ten_faces]
    Fisher_predictions = [FisherFace.predict(e) for e in ten_faces]
    Fisher_predictions2 = [FisherFace2.predict(e) for e in ten_faces]
    LBPH_predictions = [LBPHFace.predict(e) for e in ten_faces]
    
    ### the predictions come ount in a (guess, confidence value format)
    ### So lets split that up and make a DF

    df = pd.DataFrame({"e_guess" : [e[0] for e in Eigen_predictions],
               "e_conf" : [e[1] for e in Eigen_predictions],
               "f_guess" : [e[0] for e in Fisher_predictions],
               "f_conf" : [e[1] for e in Fisher_predictions],
               "f2_guess" : [e[0] for e in Fisher_predictions2],
               "f2_conf" : [e[1] for e in Fisher_predictions2], 
               "l_guess" : [e[0] for e in LBPH_predictions],
               "l_conf" : [e[1] for e in LBPH_predictions]})
    print(df)
    
    ### Get rid of all images we're not sure on
    df = df[df["f_conf"] < f_limit]
    df = df[df['l_conf'] < l_limit]
    
    if len(df) < faces_left:
        print(df)
        say_something_and_wait("cant_recognize")
        
    else:  
        modes = df[["f_guess", "l_guess", 'f2_guess']].mode(1)
        
        # there is obviously a better way to write this, but i don't care
        try:
            for i in range(0, len(modes)):
                if str(modes.iloc[i][1]) != 'nan':
                    modes.iloc[i][0] = df.iloc[i]["f_guess"]
        except:
            print('the model is pretty confident')
                
        
        #IDs_of_person = list(df[["e_guess", "f_guess", "l_guess"]].mode(1)[0])
        IDs_of_person = list(modes[0])
        ID = max(set(IDs_of_person), key=IDs_of_person.count)
        #ID = stats.mode(IDs_of_person)[0][0]

        person = people["Names"].loc[ID]
        
        print(df)
        print(person)
        say_something_and_wait(person)
        


cap.release()
cv2.destroyAllWindows()









