# FacialRecognitionGreeter
A Facial Recognition Greeter for Code Tower at Babson College

See this video for it in action:
https://youtu.be/Fgctv-_GrGM

TL-DR:
We duct-taped a Raspberry Pi to the wall, used a combination of OpenCV's built in facial recognition tools (EigenFace, FisherFace, etc...) and a ConvNet (not included in this repository) to recognize the members of our living community and greet them with funny messages according to who they are.


The original accuracy was terrible when we had the algorithms run on their own, so we had to get a little clever to get our goal accuracy of over 80%. The machine learning concept of "a bunch of different weak predictors become a strong predictor" was sort of the idea here.

The final output took around 20 pictures of each person's face in rapid succession, had each Facial Recognition software vote for which person they thought it was, and then tallied up the results. 

The current state of the program is not remotely stable, as the project was developed quite some time ago and maintains weak documentation, I thought I would leave it here for future members of Code Tower in case they wished to rebuild it with the recognizer trained on the new memebers, and for potential employers.



