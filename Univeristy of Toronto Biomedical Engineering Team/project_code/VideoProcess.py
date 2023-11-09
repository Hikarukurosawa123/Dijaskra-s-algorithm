import cv2
import time
import threading
import sys

class EffluentMonitor(threading.Thread):
    def __init__(self, start_monitor = True):
        self.keep_running = True            
        self.start_monitor = start_monitor          # flag to know to start monitoring (always True currently). Would be nice to differentiate patient to patient trials.
        threading.Thread.__init__(self)             # start a new thread for video processing
        self.start()                                # run this new thread (will jump to run method)

    # mean calculated red channel mean value
    def calc_mean(self, arr: list): 
        '''
        Given a list of red channel values, returns
        the mean of this list.

        Parameters
        ----------

        self : EffluentMonitor Object
            Takes self Effluent Monitor parameters.

        arr :  list
            A list of integers or floats containing red
            channel values.

        Returns
        -------
        
        mean : float
            A float mean value calculated from the inputted array.
        '''
        return sum(arr) / len(arr)                  # calculate the mean red channel values and return it

    def save_data(lst_of_data, lst_of_red, lst_of_green, lst_of_blue): #looks like this is not doing much as 
        print('empty')                                                  # currently constructed

    def crop(self, arr):
        '''
        Given an input array, an array of reduced size is returned
        whereby the first and last 119 columns of pixels are removed,
        and the first and last 159 rows of pixels are removed.
        '''
        buffer = arr[219:-219,159:-159]             # crop the input array
        return buffer                               # return the result array

    def start_record(self): #needs to be tested
        '''
        Taking no inputs, this function initializes the writing
        of an avi file that we will add frames to for future reference. can also explore saving video as a .h264 (low storage size)
        '''
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        f_width = int(self.root.get(3)*0.5)     # assess frame width
        f_height = int(self.root.get(4)*0.5)    # assess frame height
        # self.out = cv2.VideoWriter('monitor.avi',
        #                       cv2.VideoWriter_fourcc('M','J','P','G'), 
        #                       10, (f_width,f_height))   # start writing an avi
                                            # file that saves sequences of images

    def run(self):
        while self.keep_running: # double check that this while loop is necessary!!
            lst_of_red = []
            lst_of_green = []
            lst_of_blue = []
            self.lst_of_data = []                           # list of data to use for monitoring purposes
            self.process_time = 0                           #a counter for image processing (processes video every second)
            if self.start_monitor:                          # if we are starting to monitor
                if sys.platform == "darwin":                # if mac. Delete in final code
                    b_end = cv2.CAP_AVFOUNDATION            # set 1 and different backend 
                    self.root =cv2.VideoCapture(1, b_end)   # to bypass continuity camera
                else:                                       # if not a mac (no issue of continuity camera)
                    self.root = cv2.VideoCapture(0)         # input 0 is default (builtin camera), 1 and 2 could be inputted if. KEEP IN FINAL CODE
                                                    # if different cameras are attached
                self.start_record()                 # initialize recording avi file
                #self.lst_of_data = [] #average red values
                fps = self.root.get(cv2.CAP_PROP_FPS)
                load_time = 0 #allows camera to load

                # get the start time
                start_time = time.time() #<--currently not in use

                #checks if camera is open
                if self.root.isOpened():        # try to get the first frame
                    self.return_val, self.frame = self.root.read()
                else:                           # otherwise
                    self.return_val = False     # do not enter the upcoming loop
            else:                               # and if we haven't started monitoring
                self.return_val = False         # do not enter the upcoming loop (this never seems to be True however)

            #print("Load Time: ", time.time() - start_time, "seconds")

            while self.return_val and self.keep_running:        # if the camera is on and so is the UI ==> pute everything under line 109 and eliminate redundacncy (ie. mean_red shouldnt be appended to a list, just evaluate prop_red)
                self.return_val, self.frame = self.root.read()  # update video frames
                                                                #
                if load_time >= 1 or not self.return_val:       # if the loading time is greater than a second, suggest removing "or not self.return_val"
                    #self.out.write(self.frame)                  # write frame data to our avi file
                    b, g, r = cv2.split(self.frame)             # split image data into blue, green and red channels
                    r = self.crop(r)                        # crop the red channel data
                    b = self.crop(b)                        # and the blue channel data
                    g = self.crop(g)                        # and the green channel data
                    mean_red = r.sum()/r.size               # calculate mean red channel value
                    mean_green = g.sum()/g.size             # repeat for green
                    mean_blue = b.sum()/b.size              # and for blue and add them all to lists
                    lst_of_red.append(mean_red)             #
                    lst_of_blue.append(mean_blue)           #
                    lst_of_green.append(mean_green)         #
                    if self.process_time >= 1:                  # if a second has passed, process the images
                        #print(process_time)                    # determine the image processing time
                        mean_red = self.calc_mean(lst_of_red)   # calculate the mean red channel value over the past second
                        mean_blue= self.calc_mean(lst_of_blue)  # do the same for blue
                        mean_green= self.calc_mean(lst_of_green)# and the same for green
                        prop_red = (mean_red/(mean_red + mean_blue + mean_green))*100   # calculate and print the relative
                        #print(prop_red, "%")                    # amount of red intensity compared to the total image
                        self.lst_of_data.append([prop_red])     # add this proportion result to our list of saved data
                        self.process_time = 0                   # reset our process time variable
                        lst_of_red = []                         # raw red values reset
                        lst_of_green = []                       # raw green values reset
                        lst_of_blue = []                        # raw blue values reset

                load_time += 1 / fps                                # update our load time
                self.process_time += 1 / fps                        # update our processing time

                #closes camera when esc key struck
                #key = cv2.waitKey(20)
                #if key == 27: # exit on esc
                    #print('Camera Stop: ', time.time() - start_time, 'seconds')
               #    break

            """if not self.keep_running:
                if self.root:
                    self.root.save_data()
                    self.root.release()
                    self.out.release()
                    #cv2.destroyWindow("preview")
            """    


            #print(lst_of_data) #prints data

            # get the end time
            #end_time = time.time()

            # get the execution time
            #elapsed_time = end_time - start_time
            #print('Execution time:', elapsed_time, 'seconds')
        sys.exit()