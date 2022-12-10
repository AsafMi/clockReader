import os
import cv2 as cv
import pickle
import numpy as np
import time as t
import sys
from tqdm import trange


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


start = t.time()

DATA = 19
path = "C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock{}".format(DATA)
os.chdir(path)

var = {'Naked_Clock' : cv.imread('Naked Clock.png', -1),
       'Hr'          : cv.imread('Hr.png', -1),
       'Min'         : cv.imread('Min.png', -1)}

center = open("center.txt","r")
center = np.loadtxt(center)
time_origin = load_obj('time origin')
clock = []
time = []
j=0
for Hour in trange(12, file=sys.stdout, desc='Hour'):
    for Minute in trange(60, file=sys.stdout, leave=False, unit_scale=True, desc='Min'):
        var['Naked_Clock'] = cv.imread('Naked Clock.png', -1)
        Angle_Min = time_origin['Min'] - (360/60)*Minute
        Angle_Hr = time_origin['Hr'] - (360/12)*Hour - (360/12)*(Minute/60)
        Angle = {'Naked_Clock':0, 'Hr':Angle_Hr, 'Min':Angle_Min}
        for i in var:
            image = rotate_image(var[i], center, Angle[i])
            alpha_s = image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                var['Naked_Clock'][:, :, c] = (alpha_s * image[:, :, c] +
                                                   alpha_l * var['Naked_Clock'][:, :, c])
    
        
        clock.append(var['Naked_Clock'])
        time.append(Hour*60 + Minute)
        newpath = r"C:\Users\Asaf Mizrahi\Desktop\Asaf\ML\Projects\Clock reader\Training data\train\{}".format(j)
        j += 1
        os.chdir(newpath)
        filename = 'clock{}.jpg'.format(DATA)
        cv.imwrite(filename, var['Naked_Clock'])
        os.chdir(path)
    #cv.imshow('image', var['Naked_Clock'])
    var['Naked_Clock'] = cv.imread('Naked Clock.png', -1)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
save_obj(clock, 'clock')
save_obj(time, 'time')

end = t.time()
timetime = (end - start)
print("It took: {0:.1f} Minutes".format(timetime/60))