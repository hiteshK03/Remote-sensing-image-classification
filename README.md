# image-classification
Transfer Learning is used to build this classifier
The data is classified into 3 folders - Train, Validation and Test.
The train and validation images is classified into 3 classes - aircrafts, none and ships.
The test folder contains a folder ts which contains all the test images which are unclassified.
The train images are randomly selected and feeded into the validation folder.
The best model after training is selected for testing. This needs to be changed in the test.py code (line 43) used for testing.
The train.py code is used for training the model used for testing.

Instructions for compiling the code using terminal -
for training - python3 train.py
for testing - python3 test.py
