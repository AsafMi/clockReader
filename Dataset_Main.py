# This file describes how I created the database
import os
import cv2 as cv
import pickle
import numpy as np
import time as t
import sys
from tqdm import trange

# Workflow & steps legend:
# Step 1 : Creation of the dataset folder structure.
#                          V
# Manual work of "undressing" the clock into 3 images (using GIMP):
# "Naked_Clock" , "Hour_Hand" , "Minute_Hand" + extract the center of the clock.
#                          V
# Step 2 : Extracting the time origin from the raw images.
# Step 3 : Use Raw_Data images for the creation of the Labeled_Data. (WARNING: !!!Expensive!!!)

steps_activation = [3]

# Choose a folder in which I'll locate all the training data
main_path = r"Clock_Dataset"

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 1: Create the folder structure as follows: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~~~~~~~ 2 sub-folders: "Raw_Data" and "Labeled_Data"
# ~~~~~~~ Inside "Raw_Data": 50 folders for 100 clocks images ("clock1" ... "clock50")
# ~~~~~~~ Inside "Labels_Data: 720 folders for 720 labels (60 minutes X 12 Hours)
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
rawDirectory = os.path.join(main_path, "Raw_Data")
labeledDirectory = os.path.join(main_path, "Labeled_Data")
if 1 in steps_activation or 'All' in steps_activation:  # Enter only when the step is activated by the user
    for clockNo in range(50):  # Raw_Data structure creation
        try:
            rawClockPath = os.path.join(rawDirectory, f"clock{clockNo + 1}")
            os.makedirs(rawClockPath)
            print(f"Directory {rawClockPath} created")
        except OSError:
            print(f"Directory {rawClockPath} is already exist")

    for clockNo in range(720):  # Raw_Data structure creation
        try:
            labeledClockPath = os.path.join(labeledDirectory, f"{clockNo}")
            os.makedirs(labeledClockPath)
            print(f"Directory {labeledClockPath} created")
        except OSError:
            print(f"Directory {labeledClockPath} is already exist")

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 2: Extracting the time origin from the raw images~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
def nothing(x):
    pass
if 2 in steps_activation or 'All' in steps_activation:  # Enter only when the step is activated by the user
    clockNo = 19
    rawClockPath = os.path.join(rawDirectory, f"clock{clockNo}")

    images = {'Hr': cv.imread(os.path.join(rawClockPath, 'Hr.png'), -1),
              'Min': cv.imread(os.path.join(rawClockPath, 'Min.png'), -1)}

    cv.namedWindow('control', 1)
    cv.createTrackbar('Angle_Hr', 'control', 0, 360, lambda x: x)
    cv.createTrackbar('Angle_Min', 'control', 0, 360, lambda x: x)
    Angle_Hr, Angle_Min = 0, 0
    center = tuple(np.genfromtxt(os.path.join(rawClockPath, "center.txt")))
    while ((cv.waitKey(1) & 0xFF) != 27):
        Angle_Hr = cv.getTrackbarPos('Angle_Hr', 'control')
        Angle_Min = cv.getTrackbarPos('Angle_Min', 'control')
        Angle = {'Naked_Clock': 0, 'Hr': Angle_Hr, 'Min': Angle_Min}
        targetImage = cv.imread(os.path.join(rawClockPath, 'Naked Clock.png'), -1)
        for i in images:
            rot_mat = cv.getRotationMatrix2D(center, Angle[i], 1.0)
            image = cv.warpAffine(images[i], rot_mat, images[i].shape[1::-1], flags=cv.INTER_LINEAR)
            alpha_s = image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                targetImage[:, :, c] = (alpha_s * image[:, :, c] +
                                        alpha_l * targetImage[:, :, c])

        cv.imshow('image', targetImage)
    cv.destroyAllWindows()
    time_origin = {'Hr': Angle_Hr, 'Min': Angle_Min}
    with open('time origin.pkl', 'wb') as f:
        pickle.dump(time_origin, f, pickle.HIGHEST_PROTOCOL)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 3: Create the Labeled_Data ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

if 3 in steps_activation or 'All' in steps_activation:  # Enter only when the step is activated by the user
    startTime = t.time()
    for clockNo in trange(50, file=sys.stdout, desc='Clock Number'):
        rawClockPath = os.path.join(rawDirectory, f"clock{clockNo + 1}")

        images = {'Hr': cv.imread(os.path.join(rawClockPath, 'Hr.png'), -1),
                  'Min': cv.imread(os.path.join(rawClockPath, 'Min.png'), -1)}

        center = tuple(np.genfromtxt(os.path.join(rawClockPath, "center.txt")))
        with open(os.path.join(rawClockPath, 'time origin.pkl'), 'rb') as f:
            time_origin = pickle.load(f)
        nakedClockImage = cv.imread(os.path.join(rawClockPath, 'Naked Clock.png'), -1)
        for Hour in trange(12, file=sys.stdout, desc='Hour'):
            for Minute in range(60):
                targetImage = np.copy(nakedClockImage)
                labeledClockPath = os.path.join(labeledDirectory, f"{Hour * 60 + Minute}")
                Angle_Min = time_origin['Min'] - (360 / 60) * Minute
                Angle_Hr = time_origin['Hr'] - (360 / 12) * Hour - (360 / 12) * (Minute / 60)
                Angle = {'Hr': Angle_Hr, 'Min': Angle_Min}
                for i in images:
                    rot_mat = cv.getRotationMatrix2D(center, Angle[i], 1.0)
                    image = cv.warpAffine(images[i], rot_mat, images[i].shape[1::-1], flags=cv.INTER_LINEAR)
                    alpha_s = image[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        targetImage[:, :, c] = (alpha_s * image[:, :, c] +
                                                alpha_l * targetImage[:, :, c])

                cv.imwrite(os.path.join(labeledClockPath, f"clock{clockNo}.jpg"), targetImage)

        elapsedTime = (t.time() - startTime)
        print("Elapsed time: {0:.1f} Minutes".format(elapsedTime / 60))
