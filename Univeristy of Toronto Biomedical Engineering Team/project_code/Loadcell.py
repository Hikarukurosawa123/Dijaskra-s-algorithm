import time
import sys
try: #remove in final code
    from hx711 import HX711
    import RPi.GPIO as GPIO
except:
    from emulated_hx711 import HX711
import threading
import csv
import datetime

class Loadcell_Module(threading.Thread):
    def __init__(self, pin1, pin2, pin3, pin4, pin5, pin6, keep_running):
        self.pin1a = pin1 #pin 1 on GPIO of Raspberry Pi (for loadcell 1)
        self.pin2a = pin2 #pin 2 on GPIO of Raspberry Pi (for loadcell 1)
        self.pin1b = pin3   # pin 3 on GPIO of raspberry pi (for loadcell 2)
        self.pin2b = pin4   # pin 4 on GPIO of raspberry pi (for loadcell 2)
        self.pin1c = pin5   # pin 5 on GPIO of raspberry pi (for loadcell 3)
        self.pin2c = pin6   # pin 6 on GPIO of raspberry pi (for loadcell 3)

        #----------initialize parameters for data recording--------#
        self.keep_running = keep_running #status bit for UI running period (ie. yes/no).
        self.list_no_noise1 = [] #noise-reduced mass list (IMPORTANT: it caps out at len == 20)
        self.list_flow1 = [] #list of instantaneous flowrate (IMPORTANT: it caps out at len == 10)
        self.average1 = 0.0 #to display on GUI, average flwo rate
        self.ttbr1 = 'N/A' #linked to GUI, see flowratew display in GUI.py. Time to bag replacement
        self.list_no_noise2 = [] #noise-reduced mass list (IMPORTANT: it caps out at len == 20)
        self.list_flow2 = [] #list of instantaneous flowrate (IMPORTANT: it caps out at len == 10)
        self.average2 = 0.0 #to display on GUI, average flwo rate
        self.ttbr2 = 'N/A' #linked to GUI, see flowratew display in GUI.py. Time to bag replacement
        self.list_no_noise3 = [] #noise-reduced mass list (IMPORTANT: it caps out at len == 20)
        self.list_flow3 = [] #list of instantaneous flowrate (IMPORTANT: it caps out at len == 10)
        self.average3 = 0.0 #to display on GUI, average flwo rate
        self.ttbr3 = 'N/A' #linked to GUI, see flowratew display in GUI.py. Time to bag replacement
        self.volume1 = None #set initial total saline volume
        threading.Thread.__init__(self)
        self.start()

    #----------save recorded data to CSV---------------------------#
    def save_data(list_raw_mass, list_mass, list_flowrate): 
        '''THE FUNCTION BELOW HAS TO BE REDONE.
        -> The goal of this function is to save the average flowrate, the noise-reduced mass, and the raw mass along with their
        repective timestamps to a csv file. This will run whenever the user has stoped a monitoring period
        
        how this function should work--> during a monitoring period, data should be appended over time to an array
        when the monitoring period stops, this array should be saved to a CSV to save all the measurements over that time period
        '''
        pass
        # location ="test"
        
        # # field names 
        # fields = ['Raw Mass', 'Run Time', '\t', 'Noise-Reduced Mass', 'Run Time', '\t', 'Instantaneous Flowrate', 'Run Time'] 
  
        # with open(location, 'w') as f:
      
        #     # using csv.writer method from CSV package
        #     write = csv.writer(f)
      
        #     write.writerows(fields)
        #     write.writerows(list_raw_mass, list_mass, list_flowrate)

    '''Runs when monitoring is stopped'''
    
    def cleanAndExit(self,list_raw_masses, list_masses, list_flowrates):
        '''
        Clean up fields and exit running. Also save recorded data
        Triggered when user stops a monitoring period
        '''
        try: #get rid of in final code
            GPIO.cleanup()
        except:
            pass
        self.save_data(list_raw_masses[0], list_masses[0], list_flowrates[0])    # save data for loadcell 1
        self.save_data(list_raw_masses[1], list_masses[1], list_flowrates[1])    # save data for loadcell 2
        self.save_data(list_raw_masses[2], list_masses[2], list_flowrates[2])    # save data for loadcell 3
        # sys.exit() #we dont want this long term, we would like the code to just place the loadcell in an idle state (ie. still running, but not monitoring)


    def calc_flowrate(self, num=1):
        #----determine which loadcell to calculate flowrate for---#
        if num == 1: ls_no_noise = self.list_no_noise1          # loadcell 1
        elif num == 2: ls_no_noise = self.list_no_noise2        # loadcell 2
        else: ls_no_noise = self.list_no_noise3                 # loadcell 3

        #---calculate flowrate----------------------------------#
        if len(ls_no_noise) == 20:
            flowrate=(ls_no_noise[0][0]-ls_no_noise[-1][0])/(ls_no_noise[-1][1]-ls_no_noise[0][1]) #semi-instanteous flowrate
            if flowrate < 0.4: #low flowrate case
                flowrate=0
            # print("Instantaneous Flowrate: ", flowrate, " mL/s")
            if num == 1: 
                self.list_flow1.append(flowrate)    # if loadcell 1 add to loadcell 1 data
                ls_flow = self.list_flow1
            elif num==2: 
                self.list_flow2.append(flowrate)    # if loadcell 2 add to loadcell 2 data
                ls_flow = self.list_flow2
            else: 
                self.list_flow3.append(flowrate)    # if loadcell 3 add to loadcell 3 data
                ls_flow = self.list_flow3
            
            
            if len(ls_flow)>=10:
                #print(list_flow[:][1], " and ", len(list_flow))
                average = round(sum(ls_flow[-10:])/len(ls_flow[-10:]),1)
                if num == 1: self.average1 = average    # update loadcell 1 average if applicable
                elif num == 2: self.average2 = average  # update loadcell 2 average if applicable
                else: self.average3 = average           # update loadcell 3 average if applicable

                #print("Average Flowrate: ", self.average, " mL/s")     
                #-----time to bag replacement calculation-------#
                try:
                    seconds = round(ls_no_noise[-1][0]/ls_no_noise[-1],2)
                    minutes = int(seconds//60)
                    seconds = round(seconds % 60,1)
                    if num == 1: self.ttbr1 =  f'{minutes}m : {seconds}s'   # update TTBR for loadcell 1 if applicable
                    elif num == 2: self.ttbr2 =  f'{minutes}m : {seconds}s' # update TTBR for loadcell 2 if applicable
                    else: self.ttbr3 =  f'{minutes}m : {seconds}s'          # update TTBR for loadcell 3 if applicable
                except: self.ttbr = 'N/A'
        else:
            print("Loading...") #delete in final product

    def calc_volume1(self):
        #calculate the volume based on the latest mass value 
        self.volume1 = self.list_no_noise1[-1][0]

    def setup_loadcell(self):
        # HOW TO CALCULATE THE REFFERENCE UNIT
        # To set the reference unit to 1. Put 1kg on your sensor or anything you have and know exactly how much it weights.
        # In this case, 92 is 1 gram because, with 1 as a reference unit I got numbers near 0 without any weight
        # and I got numbers around 184000 when I added 2kg. So, according to the rule of thirds:
        # If 2000 grams is 184000 then 1000 grams is 184000 / 2000 = 92.
        # hx.set_reference_unit(113)
        referenceUnit = 400
        
        #------set up first loadcell---------------#
        self.cell1 = HX711(self.pin1a, self.pin2a)
        self.cell1.set_reference_unit(referenceUnit)
        self.cell1.set_reading_format("MSB", "MSB")
        self.tare_loadcell(1) #required prior to use

        #-------set up second loadcell-------------#
        self.cell2 = HX711(self.pin1b, self.pin2b)
        self.cell2.set_reference_unit(referenceUnit)
        self.cell2.set_reading_format("MSB", "MSB")
        self.tare_loadcell(2) #required prior to use

        #-------set up second loadcell-------------#
        self.cell3 = HX711(self.pin1c, self.pin2c)
        self.cell3.set_reference_unit(referenceUnit)
        self.cell3.set_reading_format("MSB", "MSB")
        self.tare_loadcell(3) #required prior to use

        # print("Add Bag Now!") #start to add weight
        time.sleep(10) #10 second timmer
        # print("Done!") #done adding weight

    def tare_loadcell(self, bagnum = 1):
        #------ tare differently depending on which bag is tared-----#
        if bagnum == 1: root = self.cell1
        elif bagnum == 2: root = self.cell2
        else: root = self.cell3
        
        root.reset()
        root.tare()

    def clean_up(self, bagnum = 1):            #clears lists for analysis. need to be implemented
        if bagnum == 1:                         # clear bag 1
            self.list_no_noise1.clear()
            self.list_flow1.clear()
        elif bagnum == 2:                       # clear bag 2
            self.list_no_noise2.clear()
            self.list_flow2.clear()
        else:                                   # clear bag 3
            self.list_no_noise3.clear()
            self.list_flow3.clear()
        print("clear complete")

    def run(self):
        self.setup_loadcell()
        #----setup data for loadcell 1-------#
        self.list_no_noise1=[]
        list_mass1 = []         #to be saved when collecting data. Stores every noise-cancelled mass recorded
        list_raw_mass1 = []     #to be saved when collecting data. Every mass obtained, not noise cancelled
        self.list_flow1=[]      # to be saved when collecting data. Stores all instantanous flowrates
        self.ttbr1 = 'N/A'      # assume infinite time to bag replacement
        load_bag1 = True        #push button connected to UI, botton for opporater to press when bag loaded default false
        self.average1 = 0       # to be updated
        
        #----setup data for loadcell 2-------#
        self.list_no_noise2=[]
        list_mass2 = []         #to be saved when collecting data. Stores every noise-cancelled mass recorded
        list_raw_mass2 = []     #to be saved when collecting data. Every mass obtained, not noise cancelled
        self.list_flow2=[]      # to be saved when collecting data. Stores all instantanous flowrates
        self.ttbr2 = 'N/A'      # assume infinite time to bag replacement
        load_bag2 = True        #push button connected to UI, botton for opporater to press when bag loaded default false
        self.average2 = 0       # to be updated

        #----setup data for loadcell 3-------#
        self.list_no_noise3=[]
        list_mass3 = []         #to be saved when collecting data. Stores every noise-cancelled mass recorded
        list_raw_mass3 = []     #to be saved when collecting data. Every mass obtained, not noise cancelled
        self.list_flow3=[]      # to be saved when collecting data. Stores all instantanous flowrates
        self.ttbr3= 'N/A'      # assume infinite time to bag replacement
        load_bag3 = True        #push button connected to UI, botton for opporater to press when bag loaded default false
        self.average3 = 0       # to be updated

        #------global parameters-------#
        tolerance = 5.0 #tolerance to reduce loadcell noise
        start_time = time.time()

        while self.keep_running:
            try:   
                elapsed = time.time() - start_time      # get time elapsed 

                #-----identify loadcell 1 data----------#
                val1 = self.cell1.get_weight(5)         # weight from loadcell1
                list_raw_mass1.append([val1, elapsed])  # save raw mass from loadcell 1
                
                #-----identify loadcell 2 data----------#
                val2 = self.cell2.get_weight(5)         # weight from loadcell2
                list_raw_mass2.append([val2, elapsed])   # save raw mass from loadcell 2

                #-----identify loadcell 3 data----------#
                val3 = self.cell3.get_weight(5)         # weight from loadcell2
                list_raw_mass3.append([val3, elapsed])   # save raw mass from loadcell 2
        
                ''' IGNORE FOR NOW
                #to allow for reseting of parameters once the bag is replaced
                while load_bag == False:
                    if load_bag == True:
                        self.tare_loadcell()
                        self.list_no_noise.clear
                        self.list_flow.clear
                '''
                        
                # Determining flowrate, put the last 20 mass readings into a list and calculate the difference between the first and last value.
                # determine average flowrate, put the last 10 flowrate readings into a list and calculate the average.
                
                #----update looadcell 1 data----------#
                if len(self.list_no_noise1) == 0: #adding the first item to the list (tare to be done WITHOUT bag on)
                    self.list_no_noise1.append([val1, elapsed])
                    #print("List size 0: ", list_no_noise[0])
                    self.calc_flowrate(1)
                    self.calc_volume1()

                elif (abs(self.list_no_noise1[-1][0] - val1)) < tolerance: #TO EXPLORE: lower tolerance to avoid small fluctuations when bag is closed, upper bound to avoid noise when bag is swinging
                    if len(self.list_no_noise1)==20: del self.list_no_noise1[0]
                    self.list_no_noise1.append([val1, elapsed])
                    list_mass1.append([val1, elapsed])
                    self.calc_flowrate(1)
                    self.calc_volume1()

                #----update looadcell 2 data----------#
                if len(self.list_no_noise2) == 0: #adding the first item to the list (tare to be done WITHOUT bag on)
                    self.list_no_noise2.append([val2, elapsed])
                    #print("List size 0: ", list_no_noise[0])
                    self.calc_flowrate(2)

                elif (abs(self.list_no_noise2[-1][0] - val2)) < tolerance: #TO EXPLORE: lower tolerance to avoid small fluctuations when bag is closed, upper bound to avoid noise when bag is swinging
                    if len(self.list_no_noise2)==20: del self.list_no_noise2[0]
                    self.list_no_noise2.append([val2, elapsed])
                    list_mass2.append([val2, elapsed])
                    self.calc_flowrate(2)
                
                #----update looadcell 3 data----------#
                if len(self.list_no_noise3) == 0: #adding the first item to the list (tare to be done WITHOUT bag on)
                    self.list_no_noise3.append([val3, elapsed])
                    #print("List size 0: ", list_no_noise[0])
                    self.calc_flowrate(3)

                elif (abs(self.list_no_noise3[-1][0] - val3)) < tolerance: #TO EXPLORE: lower tolerance to avoid small fluctuations when bag is closed, upper bound to avoid noise when bag is swinging
                    if len(self.list_no_noise3)==20: del self.list_no_noise3[0]
                    self.list_no_noise3.append([val3, elapsed])
                    list_mass3.append([val3, elapsed])
                    self.calc_flowrate(3)
                
                #if len(self.list_flow) > 0: list_flowrate.append(self.list_flow[-1])
                
                #-----power down and back up all loadcells----#
                self.cell1.power_down()                      # start with loadcell 1
                self.cell1.power_up()
                time.sleep(0.1)
                self.cell2.power_down()                      # then update loadcell 2
                self.cell2.power_up()
                time.sleep(0.1)
                self.cell3.power_down()                      # then update loadcell 3
                self.cell3.power_up()
                time.sleep(0.1)
                #save all calculated data and raw data

            except (KeyboardInterrupt, SystemExit):
                list_raw_masses = [list_raw_mass1,list_raw_mass2,list_raw_mass3]    # list of all raw masses
                list_masses = [list_mass1, list_mass2, list_mass3]                  # list of all noise reduced masses
                list_flows = [self.list_flow1, self.list_flow2, self.list_flow3]    # list of all instantaneous flowrates
                self.cleanAndExit(list_raw_masses, list_masses, list_flows)

        # -----final saving and cleanup-------------------------------------#
        list_raw_masses = [list_raw_mass1,list_raw_mass2,list_raw_mass3]    # list of all raw masses
        list_masses = [list_mass1, list_mass2, list_mass3]                  # list of all noise reduced masses
        list_flows = [self.list_flow1, self.list_flow2, self.list_flow3]    # list of all instantaneous flowrates
        self.cleanAndExit(list_raw_masses, list_masses, list_flows)         
