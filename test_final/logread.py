# -*- coding: utf-8 -*-

# Log reading class - 24/09/2019
# Julius Andretti

from numpy import loadtxt,transpose,array,asanyarray
from datetime import datetime, timedelta,timezone
import matplotlib.pyplot as plt

class LogRead:
    def __init__(self,file,header=1):
        # file is a string containing the path to the log file
        # header indicates if the log file has a header

        if not isinstance(file, str):
            raise Exception('Log path must be a string!')


        ignoreRows = 0    
        if header: # If the log has a header, it means we can ignore its first lines
            log = open(file)
            line = log.readline()
            line = log.readline()
            ignoreRows=2           
            while (line[0:3] != '+--'):
                line = log.readline()
                ignoreRows += 1
            log.close()

        data = loadtxt(file,skiprows=ignoreRows,delimiter=';',dtype='str') # Data read line by line
        
        names = data[0] # First line contains the names

        data = transpose(data[1:len(data)]) # We transpose our data so that each column will correspond to an element of the array

        numStart = 0
        if names[0] == 'DATE/TIME': # If our data contains a DATE/TIME entry, we store its inputs first
            numStart = 1

        numData = [] # Now we'll start formatting the numeric data
        for i in range(numStart,len(data)):
            if (names[i] in ['STATE','ORIENTATION']):
                numData.append([names[i],asanyarray(data[i],dtype='int16')]) # Some entries are integers
            else:
                numData.append([names[i],asanyarray(data[i],dtype='float64')]) # Most are real 

        self.dateTime = []
        self.ms = []
        self.event = []
        self.temperature= []
        self.extTemperature = []
        self.orientation = []
        self.pim = []
        self.pimN = []
        self.tat = []
        self.tatN = []
        self.zcm = []
        self.zcmN = []
        self.light = []
        self.ambLight = []
        self.redLight = []
        self.greenLight = []
        self.blueLight = []
        self.irLight = []
        self.uvaLight = []
        self.uvbLight = []
        self.state = []
        for i in range(len(numData)):
            if names[0] == 'DATE/TIME':
                self.dateTime = data[0]
            if numData[i][0] == 'MS':
                self.ms = numData[i][1] 
            elif numData[i][0] == 'EVENT':
                self.event = numData[i][1]
            elif numData[i][0] == 'TEMPERATURE':
                self.temperature = numData[i][1]
            elif numData[i][0] == 'EXT TEMPERATURE':
                self.extTemperature = numData[i][1]
            elif numData[i][0] == 'ORIENTATION':
                self.orientation = numData[i][1]
            elif numData[i][0] == 'PIM':
                self.pim = numData[i][1]
            elif numData[i][0] == 'PIMn':
                self.pimN = numData[i][1]
            elif numData[i][0] == 'TAT':
                self.tat = numData[i][1]
            elif numData[i][0] == 'TATn':
                self.tatN = numData[i][1]
            elif numData[i][0] == 'ZCM':
                self.zcm = numData[i][1]
            elif numData[i][0] == 'ZCMn':
                self.zcmN = numData[i][1]
            elif numData[i][0] == 'LIGHT':
                self.light = numData[i][1]
            elif numData[i][0] == 'AMB LIGHT':
                self.ambLight = numData[i][1]
            elif numData[i][0] == 'RED LIGHT':
                self.redLight = numData[i][1]
            elif numData[i][0] == 'GREEN LIGHT':
                self.greenLight = numData[i][1]
            elif numData[i][0] == 'BLUE LIGHT':
                self.blueLight = numData[i][1]
            elif numData[i][0] == 'IR LIGHT':
                self.irLight = numData[i][1]
            elif numData[i][0] == 'UVA LIGHT':
                self.uvaLight = numData[i][1]
            elif numData[i][0] == 'UVB LIGHT':
                self.uvbLight = numData[i][1]
            elif numData[i][0] == 'STATE':
                self.state = numData[i][1]
        
        self.names = names        
        self.numData = numData
        
    def plotter(self):
        for i in range(len(self.numData)):
            print(str(i)+'. '+self.numData[i][0])
        plot = int(input('Enter the number assigned to the variable you want to plot: '))
        plt.figure()
        plt.plot(self.numData[plot][1])
        plt.show() 
    
    def listAll(self):
        print('List of variables read from log file:')
        for i in range(len(self.names)):
            print('- '+self.names[i])
    
    def makeT(self):
        time = []
        # absoluto
        if len(self.dateTime) > 0:
            # day = int(self.dateTime[0][0:2])
            # month = int(self.dateTime[0][3:5])
            # year = int(self.dateTime[0][6:10])
            # hour = int(self.dateTime[0][11:13])
            # minute = int(self.dateTime[0][14:16])
            # second = int(self.dateTime[0][17:19])
            # time.append([datetime(year,month,day,hour,minute,second,tzinfo=datetime.timezone.utc),0])
            for i in range(0,len(self.dateTime)):
                day = int(self.dateTime[i][0:2])
                month = int(self.dateTime[i][3:5])
                year = int(self.dateTime[i][6:10])
                hour = int(self.dateTime[i][11:13])
                minute = int(self.dateTime[i][14:16])
                second = int(self.dateTime[i][17:19])
                stamp = datetime(year,month,day,hour,minute,second,tzinfo=timezone.utc)
                # delta = stamp - time[0][0]
                time.append([stamp,int(stamp.timestamp())])

        # relativo
        # if len(self.dateTime) > 0:
        #     day = int(self.dateTime[0][0:2])
        #     month = int(self.dateTime[0][3:5])
        #     year = int(self.dateTime[0][6:10])
        #     hour = int(self.dateTime[0][11:13])
        #     minute = int(self.dateTime[0][14:16])
        #     second = int(self.dateTime[0][17:19])
        #     time.append([datetime(year,month,day,hour,minute,second,tzinfo=datetime.timezone.utc),0])
        #     for i in range(1,len(self.dateTime)):
        #         day = int(self.dateTime[i][0:2])
        #         month = int(self.dateTime[i][3:5])
        #         year = int(self.dateTime[i][6:10])
        #         hour = int(self.dateTime[i][11:13])
        #         minute = int(self.dateTime[i][14:16])
        #         second = int(self.dateTime[i][17:19])
        #         stamp = datetime(year,month,day,hour,minute,second,tzinfo=timezone.utc)
        #         delta = stamp - time[0][0]
        #         time.append([stamp,int(delta.total_seconds)])

        return transpose(array(time))