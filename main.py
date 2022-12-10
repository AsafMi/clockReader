# This is the main file which assemble all the major processes

import shutil
for i in range(50):
    base1 = f"C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock{i+1}\\center.txt"
    base2 = f"C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock{i+1}\\Naked Clock.png"
    base3 = f"C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock{i+1}\\Min.png"
    base4 = f"C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock{i+1}\\Hr.png"
    base4 = f"C:\\Users\\Asaf Mizrahi\\Desktop\\Asaf\\ML\\Projects\\Clock reader\\Training data\\clock{i+1}\\time origin.pkl"
    dst1 = f"C:\\Users\\Asaf Mizrahi\\PycharmProjects\\clockReader\\Clock_Dataset\\Raw_Data\\clock{i+1}\\center.txt"
    dst2 = f"C:\\Users\\Asaf Mizrahi\\PycharmProjects\\clockReader\\Clock_Dataset\\Raw_Data\\clock{i+1}\\Naked Clock.png"
    dst3 = f"C:\\Users\\Asaf Mizrahi\\PycharmProjects\\clockReader\\Clock_Dataset\\Raw_Data\\clock{i+1}\\Min.png"
    dst4 = f"C:\\Users\\Asaf Mizrahi\\PycharmProjects\\clockReader\\Clock_Dataset\\Raw_Data\\clock{i+1}\\Hr.png"
    dst4 = f"C:\\Users\\Asaf Mizrahi\\PycharmProjects\\clockReader\\Clock_Dataset\\Raw_Data\\clock{i+1}\\time origin.pkl"
    #shutil.copyfile(base1, dst1)
    #shutil.copyfile(base2, dst2)
    #shutil.copyfile(base3, dst3)
    #shutil.copyfile(base4, dst4)
    shutil.copyfile(base4, dst4)


