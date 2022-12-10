# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:53:47 2020

@author: Asaf Mizrahi
"""

import os
import cv2 as cv
import pickle
import numpy as np

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def rotate_image(image, center, angle):
  image_center = tuple(center)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

def nothing(x):
    pass

path = "C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock3"
os.chdir(path)

var = {'Naked_Clock' : cv.imread('Naked Clock.png', -1),
       'Hr'          : cv.imread('Hr.png', -1),
       'Min'         : cv.imread('Min.png', -1)}

center = load_obj('center')
time_origin = load_obj('time origin')
clock = []
time = []
for Hour in range(0, 12, 1):
    for Minute in range(0, 60, 1):
        var['Naked_Clock'] = cv.imread('Naked Clock.png', -1)
        Angle_Min = time_origin['Min'] - (360/60)*Minute
        Angle_Hr = time_origin['Hr'] - (360/12)*Hour - (360/12)*(Minute/60)
        Angle = {'Naked_Clock':0, 'Hr':Angle_Hr, 'Min':Angle_Min}
        for i in center:
            image = rotate_image(var[i], center[i], Angle[i])
            center[i] = np.array(center[i])
            offset = np.abs(center['Naked_Clock'] - center[i])
            y1, y2 = offset[1], offset[1] + image.shape[0]
            x1, x2 = offset[0], offset[0] + image.shape[1]
            alpha_s = image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                var['Naked_Clock'][y1:y2, x1:x2, c] = (alpha_s * image[:, :, c] +
                                               alpha_l * var['Naked_Clock'][y1:y2, x1:x2, c])
        clock.append(var['Naked_Clock'])
        time.append(Hour*60 + Minute)
    #cv.imshow('image', var['Naked_Clock'])
    var['Naked_Clock'] = cv.imread('Naked Clock.png', -1)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
save_obj(clock, 'clock')
save_obj(time, 'time')