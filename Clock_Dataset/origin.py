# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:55:51 2020

@author: Asaf Mizrahi
"""


#try:
#    from IPython import get_ipython
#    get_ipython().magic('clear')
#    get_ipython().magic('reset -f')
#except:
#    pass


import cv2 as cv
import os
import pickle

def nothing(x):
    pass

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = "C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock2"
os.chdir(path)

var = {'Naked_Clock' : cv.imread('Naked Clock.png', -1),
       'Hr'          : cv.imread('Hr.png', -1),
       'Min'         : cv.imread('Min.png', -1)}

center = {}
for i in var:
    cv.namedWindow('control')
    cv.createTrackbar('X_center', 'control', 0, var[i].shape[1], nothing)
    cv.createTrackbar('Y_center', 'control', 0, var[i].shape[0], nothing)
    cv.createTrackbar('R_center', 'control', 0, var[i].shape[0], nothing)
    X_center, Y_center, R_center = 0, 0, 0
    while((cv.waitKey(1) & 0xFF) != 27):
        X_center = cv.getTrackbarPos('X_center', 'control')
        Y_center = cv.getTrackbarPos('Y_center', 'control')
        R_center = cv.getTrackbarPos('R_center', 'control') 
        pic = cv.circle(var[i], (X_center, Y_center), R_center, (147, 20, 255))
        cv.imshow('image', pic)
        var = {'Naked_Clock' : cv.imread('Naked Clock.png', 0),
               'Hr'          : cv.imread('Hr.png', 0),
               'Min'         : cv.imread('Min.png', 0)}

    center.update({i: [X_center, Y_center]})
    cv.destroyAllWindows()
save_obj(center, 'center')