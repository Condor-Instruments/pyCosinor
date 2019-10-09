from numpy import array,arange,ones
from math import pi,cos
import pytest
import sys,inspect,os
#print(inspect.getfile(inspect.currentframe()))
#print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))  
filepath = inspect.getfile(inspect.currentframe())
backtodir = '/test/test_cosinor_single_cosine.py'
sys.path.insert(1, filepath[0:(len(filepath)-len(backtodir))])

import cosinor as csnr

tol = 1e-10

Mesor = 100
Amplitude = 50
Acrophase = pi*0.6
k = 3
n = 600
t = arange(n)*((24*k)/n)
w = 2*pi/24
y = array([(Mesor+Amplitude*cos(w*t[i] - Acrophase)) for i in range(n)])
ini = csnr.Cosinor(t,y,24,0.05)
ini.fit()

def test_Mesor_ini():
    dif = abs(ini.M - Mesor)
    assert ini.M == Mesor or dif < tol

def test_Amp_ini():
    dif = abs(ini.Amp - Amplitude)
    assert ini.Amp == Amplitude or dif < tol
    
def test_Acrophase_ini():
    dif = abs(ini.phi - (-Acrophase))
    assert ini.phi == -Acrophase or dif < tol
    
def test_Zero_Amplitude_Test_ini_False():   
    assert ini.zeroAmp == False

def test_Zero_Amplitude_Test_ini_True():   
    tru = csnr.Cosinor(t,Mesor*ones(n),24,0.05)
    tru.fit()
    assert tru.zeroAmp == True
        
        
wdif = 2*pi/25
ydif = array([(Mesor+Amplitude*cos(w*t[i] - Acrophase)) for i in range(n)])
inidif = csnr.Cosinor(t,ydif,24,0.05)
inidif.fit()

def test_Mesor_ini_T25():
    dif = abs(ini.M - Mesor)
    assert inidif.M == Mesor or dif < tol

def test_Amp_ini_T25():
    dif = abs(ini.Amp - Amplitude)
    assert inidif.Amp == Amplitude or dif < tol
    
def test_Acrophase_ini_T25():
    dif = abs(ini.phi - (-Acrophase))
    assert inidif.phi == -Acrophase or dif < tol
    
def test_Zero_Amplitude_Test_False_T25():   
    assert inidif.zeroAmp == False
