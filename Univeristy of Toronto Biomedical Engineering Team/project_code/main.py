#import cv2 as cv #OpenCV (not needed)
#import numpy as np # Not needed
import time
import os
#import datetime as dt   # not needed
#import tkinter          # not needed
#import customtkinter    # not needed
from GUI import *
from Loadcell import *
from VideoProcess import *
#import threading        # not needed
try:
    from picamera import PiCamera #not needed
except:
    pass

##features to add
#   -> make it so that camera playback is not always shown on UI window
#   -> improve admin user login/logout functionality
#   -> change tare buttons to ones that show bag status (options are full, empty, in use, and one day include leaky)

if __name__ == "__main__":
    print('\n\n'+'='*50+'\n\n'+os.getcwd()+'\n\n'+'='*50+'\n')
    loadcell1 = Loadcell_Module(5,6,22,23,17,18,True)   # 5=pin1, 6 = pin2, 22 = pin3, 23 = pin4, 17 = pin5, 18 = pin6
    camera = EffluentMonitor(True)
    time.sleep(10)
    app = App(camera, loadcell1)
    Update(app, loadcell1, camera)
    time.sleep(1)
    app.start_display()
