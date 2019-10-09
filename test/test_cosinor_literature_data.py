from numpy import array,arange,ones
from math import pi,cos
import pytest
import sys,inspect,os
#print(inspect.getfile(inspect.currentframe()))
#print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))  
filepath = inspect.getfile(inspect.currentframe())
backtodir = '/test/test_cosinor_literature_data.py'
sys.path.insert(1, filepath[0:(len(filepath)-len(backtodir))])

import cosinor as csnr

tol = 1e-10

a = csnr.Cosinor(array([97,130,167.5,187.5,218,247.5,285,315,337.5])*24/360,[102,96.8,97,92.5,95,93,99.4,99.8,105.5],24,0.05)
a.fit()

def test_Mesor():
    dif = abs(a.M - 99.7269745887)
    assert a.M == 99.7269745887 or dif < tol

def test_Amp():
    dif = abs(a.Amp - 6.38383296016)
    assert a.Amp == 6.38383296016 or dif < tol
    
def test_Acrophase():
    dif = abs(a.phi - (-0.4754187912802497))
    assert a.phi == -0.4754187912802497 or dif < tol
    
def test_Gamma():
    dif = abs(a.gamma - 2.92194951534)
    assert a.gamma == 2.92194951534 or dif < tol
    
def test_Beta():
    dif = abs(a.beta - 5.67587299832)
    assert a.beta == 5.67587299832 or dif < tol

def test_Zero_Amplitude_Test():   
    assert a.zeroAmp == False
    
def test_Zero_Amplitude_Test_p_value():   
    dif = abs(a.p_3a - 0.000511378765161)
    assert a.p_3a == 0.000511378765161 or dif < tol
    

