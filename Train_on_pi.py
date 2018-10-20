# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:45:47 2018

@author: walke
"""


import os                                               # importing the OS for path
import cv2 as cv2                                           # importing the OpenCV library
import numpy as np                                      # importing Numpy library
from PIL import Image                                   # importing Image library
import pandas as pd

EigenFace = cv2.face.EigenFaceRecognizer_create(8) # createEigenFaceRecognizer(15)      # creating EIGEN FACE RECOGNISER
FisherFace = cv2.face.FisherFaceRecognizer_create(8) #  createFisherFaceRecognizer(2)     # Create FISHER FACE RECOGNISER
LBPHFace = cv2.face.LBPHFaceRecognizer_create(1,1,7,7) # createLBPHFaceRecognizer(1, 1, 7,7) # Create LBPH FACE RECOGNISER

#EigenFace = cv2.createEigenFaceRecognizer(8)      # creating EIGEN FACE RECOGNISER
#FisherFace = cv2.createFisherFaceRecognizer(8)     # Create FISHER FACE RECOGNISER
#LBPHFace = cv2.createLBPHFaceRecognizer(1, 1, 7,7) # Create LBPH FACE RECOGNISER




path = '/home/pi/Desktop/Best_Images'

#path = 'dataSet'                                        # path to the photos
def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  # Open image and convert to gray
        faceImage = faceImage.resize((200,200))         # resize the image so the EIGEN recogniser can be trained
        faceNP = np.array(faceImage, 'uint8')           # convert the image to Numpy array
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Retreave the ID of the array
        FaceList.append(faceNP)                         # Append the Numpy Array to the list
        IDs.append(ID)                                  # Append the ID to the IDs list
        cv2.imshow('Training Set', faceNP)              # Show the images in the list
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    return np.array(IDs), FaceList                      # The IDs are converted in to a Numpy array

IDs, FaceList = getImageWithID(path)



### Get info on the training set
def train_set_data(path, IDs):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    df = pd.DataFrame({"IDS_Used" : list(set(IDs)),
                       "Number_of_Images": 0})
    index = pd.Index(list(df["IDS_Used"]))
    for imagePath in imagePaths:
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        location = index.get_loc(ID)
        df.loc[location]["Number_of_Images"] += 1
    print(df)
    print(str(np.mean(df["Number_of_Images"])) + "  <-- Average Pics per ID")
    print(str(len(df["IDS_Used"])) + "  <-- number of IDs")
    num_ids = len(df["IDS_Used"])
    return df, num_ids
        
df, num_ids = train_set_data(path, IDs)

#split the images for eigen_faces
other_half_FaceList = []
for i in range(0, len(FaceList)):
    if i % 20 < 13:
        other_half_FaceList.append(FaceList[i])
        
other_half_IDs = []
for i in range(0, len(FaceList)):
    if i % 20 < 13:
        other_half_IDs.append(IDs[i])
    



# ------------------------------------ TRAING THE RECOGNISER ----------------------------------------
print('TRAINING......')
#EigenFace.train(np.array(other_half_FaceList), np.array(other_half_IDs))                          # The recongniser is trained using the images
#print('EIGEN FACE RECOGNISER COMPLETE...')
#EigenFace.write('EIGEN_Fin_65_10.yml')
#print('FILE SAVED..')
FisherFace.train(np.array(other_half_FaceList), np.array(other_half_IDs))  
print('FISHER FACE RECOGNISER COMPLETE...')
FisherFace.write('FISHER_Fin_OTHER_65_10.yml')
print('Fisher Face XML saved... ')
#LBPHFace.train(FaceList, IDs)
#print('LBPH FACE RECOGNISER COMPLETE...')
#LBPHFace.write('LBPH_Fin.yml')
#print ('ALL XML FILES SAVED...')

cv2.destroyAllWindows()



