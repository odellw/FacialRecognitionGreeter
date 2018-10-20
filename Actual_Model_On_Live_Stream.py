# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:03:07 2018

@author: walke
"""
#### Import Packages
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from gtts import gTTS
from scipy import stats
import time

#### Face Cascades and Recognition Models
face_cascade = cv2.CascadeClassifier(r'C:\Users\walke\Desktop\Name_find\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\walke\Desktop\Name_find\haarcascade_eye.xml')

EigenFace = cv2.face.EigenFaceRecognizer_create(8) 
EigenFace.read(r'C:\Users\walke\Desktop\Name_find\Recogniser\trainingDataEigan20ppl.xml')

FisherFace = cv2.face.FisherFaceRecognizer_create(8) 
FisherFace.read(r'C:\Users\walke\Desktop\Name_find\Recogniser\trainingDataFisher20ppl.xml')

LBPHFace = cv2.face.LBPHFaceRecognizer_create(1,1,7,7)
LBPHFace.read(r'C:\Users\walke\Desktop\Name_find\Recogniser\trainingDataLBPH20ppl.xml')


### IDs we are using
IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
Names = ['Nick','Walker','Melissa','Peter','Karen','Jon_Marchetti','Harsha_Alturi','Yumel','Leigh',
             'Omar','Abe','Tom','Mitch','Chris_Lally','Chris_st_Jean','Celia','Miguel','Vik',
             'Larisa','Julian','Christian','Emily','Karen','Emma_MacPhail','Gabriella']

### Set up greetings
Greetings = ['Hello President Nick, your penis is so large. oh my god, give it to me', 
             "Hello Walker, you are so amazing. I love you",
             "Hot diggidy dog, it is Melissa", 
             "Hey Peter", 
             "Hey Karen", 
             "Hello you piece of human garbage. You are trash John. I hope you die",
             "Hey Harsha",
             "Hey Yumel",
             "Oh hey it's Leigh. Hi Leigh", 
             "Oh shit hey Omar, you look amazing",
             "Shalom Abe", 
             "Hey Tom", "Hey Mitch", "Hey Chris", "Hey Chris Jean", "Hey Celia",
             "Hey Miguel", "Hey Vik", 
             "Hey Larisa", "Hey Julian", "Hey Christian", "Hey Emily", "Hey Karen", "Hey Emma",
             "Hey Gabriella"]
    
 
### Make people dataframe   
people = pd.DataFrame({"IDS": IDS,
                       "Names": Names,
                       "Greetings": Greetings})

people = people.set_index("IDS")
            

### to be used when talking to ppl
def say_something_and_wait(greet_text):
    tts = gTTS(text=greet_text, lang='en')
    tts.save("greet.mp3")
    os.system("start greet.mp3")
    time.sleep(7)

### Set up video
cap = cv2.VideoCapture(0)

### Let's start the loop
while True:
    
    ### Assign variables
    count = 0
    ten_faces = []
    list_o_faces = []
    
    ### We want 30 photos, 
    while count < 30:    
        
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
                
                if count < 8:
                    break
                persons_face = persons_face.resize((200, 200))
                
                # if we want to see what faces look like, uncomment this
                list_o_faces.append(persons_face)
                
                faceNP = np.array(persons_face, 'uint8') 
                
                ten_faces.append(faceNP)
                
                print(count)

        elapsed_time = time.time() - start_time
        
        #if we've waited more then 20s let's get out 
        if elapsed_time > 20:
            ten_faces = []
            break
   
    ### if we've waited more then 20s then let's go to the beg of the loop
    if ten_faces == []:
        continue
    
    ### Lets predict some stuff
    Eigen_predictions = [EigenFace.predict(e) for e in ten_faces]
    Fisher_predictions = [FisherFace.predict(e) for e in ten_faces]
    LBPH_predictions = [LBPHFace.predict(e) for e in ten_faces]
    
    ### the predictions come ount in a (guess, confidence value format)
    ### So lets split that up and make a DF

    df = pd.DataFrame({"e_guess" : [e[0] for e in Eigen_predictions],
               "e_conf" : [e[1] for e in Eigen_predictions],
               "f_guess" : [e[0] for e in Fisher_predictions],
               "f_conf" : [e[1] for e in Fisher_predictions],
               "l_guess" : [e[0] for e in LBPH_predictions],
               "l_conf" : [e[1] for e in LBPH_predictions]})
    
    ### Get rid of all images we're not sure on
    df = df[df["l_conf"] < 2]
    
    if len(df) < 13:
        print(df)
        say_something_and_wait("I can't recognize who you are")
        
    else:  
        modes = df[["e_guess", "f_guess", "l_guess"]].mode(1)
        
        # there is obviously a better way to write this, but i don't care
        try:
            for i in range(0, len(modes)):
                if str(modes.iloc[i][1]) != 'nan':
                    modes.iloc[i][0] = df.iloc[i]["f_guess"]
        except:
            print('the model is pretty confident')
                
        
        IDs_of_person = list(df[["e_guess", "f_guess", "l_guess"]].mode(1)[0])
        ID = stats.mode(IDs_of_person)[0][0]

        person = people["Names"].loc[ID]
        
        greet_text = people["Greetings"].loc[ID]
        
        print(df)
        say_something_and_wait(greet_text)
        
    


cap.release()
cv2.destroyAllWindows()

        