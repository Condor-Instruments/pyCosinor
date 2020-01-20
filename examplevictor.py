# -*- coding: utf-8 -*-
# Cosinor fitting example - 24/10/2019
# Julius Andretti

from numpy import array, random, arange
from math import pi,cos
import cosinor as cs

# Arbitrary inputs
y = [102,96.8,97,92.5,95,93,99.4,99.8,105.5]
t = array([97,130,167.5,187.5,218,247.5,285,315,337.5])*24/360
T = 24
alpha = 0.05

# Class instantiation
# ini = cs.Cosinor(t,array(y),T,alpha)

# # Calculations
# ini.fit()

# # Prints the estimated parameters
# ini.printParam()

# # Prints the confidence interval for the Mesor and the confidence pĺot for Amplitude and Acrophase (if the Zero Amplitude Test fails)
# ini.printParamCI()


def tester(): # Creates test vectors with a period of 24 hours
    Mesor = float(input('Enter the mesor value: '))
    Amplitude = float(input('Enter the amplitude value: '))
    Acrophase = float(input('Enter the acrophase value (hours): '))
    Error = float(input('Enter the error percentage: '))/100
    # T = int(input('Enter the period:'))
    k = float(input('Enter number of cycles: '))
    n = int(input('Enter the number of points: '))

    t = arange(n)*((24*60*60*k)/n)
    w = 2*pi/(24*60*60)
    #y = array([(Mesor+Amplitude*cos(w*t[i] - Acrophase*w)) for i in range(n)]) + random.uniform(-Error*Amplitude, Error*Amplitude, n)  

    y.clear()
    for i in range(n):
        if(i%2 == 0):
            Er = -Error*Amplitude
        else:
            Er = +Error*Amplitude
        y.append((Mesor+Amplitude*cos(w*t[i] - Acrophase*w)) + Er)

    test = cs.Cosinor(t,y,24*60*60,0.05)
    test.fit()
    test.printParam()
    test.printParamCI()

# To test with user made inputs
tester()