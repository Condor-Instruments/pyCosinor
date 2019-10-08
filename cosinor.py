# -*- coding: utf-8 -*-

# Cosinor fitting function - 11/09/2019
# Julius Andretti
# References:
# [1] BINGHAM, G.; ARGOBAST, B.; CORNÉLISSEN, G; LEE, J.; HALBERG, F.: Inferential statistical methods for estimating 
# and comparing Cosinor parameters 


from numpy import array, dot, hypot, linspace, real, delete, arange, zeros, random
from numpy import linalg as LA
from math import pi, cos, sin, atan, sqrt
from cmath import sqrt as csqrt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from statistics import mean
import scipy.stats as scist

#def isBalanced(t,T):
#    n = len(t)
#    dist = (t[n-1] - t[0])/n

def confidencePlot(Amp,beta,gamma,posGamma,posBeta1,posBeta2): # Plots the joint confidence region
    fig = plt.figure() # Initializing the figure
    ax_carthesian  = fig.add_axes([0.1, 0.1, 1, 1], frameon=False, aspect='equal') # Carthesian axis:
    plt.ylabel('Beta')
    plt.xlabel('Gamma')
    ax_carthesian.axis([-3.133*Amp, 3.133*Amp, -3.133*Amp, 3.133*Amp]) 
    ax_carthesian.plot(posGamma, posBeta1, 'g', posGamma, posBeta2, 'g')
    ax_carthesian.plot(linspace(0,gamma,4000), linspace(0,beta,4000), 'r', linewidth=2) 
    ax_carthesian.plot(linspace(gamma*1.2,(1.167*(2/3))*3.1*gamma,30), linspace(beta*1.2,(1.167*(2/3))*3.1*beta,30), 'r', ls='--', linewidth=2)
    ax_carthesian.plot(linspace(-((1.167*(2/3))*3.1*Amp),((1.167*(2/3))*3.1*Amp),10), zeros(10), ls='--', linewidth=0.5, color='black')
    ax_carthesian.plot(zeros(10), linspace(-((1.167*(2/3))*3.1*Amp),((1.167*(2/3))*3.1*Amp),10), ls='--', linewidth=0.5, color='black')
    ax_carthesian.grid('off')
    
    # The clock drawing
    circle1 = plt.Circle((0,0),radius=(1.167*(2/3))*3.1*Amp,ls='-',lw=1.3,fill=False)
    circle2 = plt.Circle((0,0),radius= 3.1*Amp,ls='-',lw=1.2,fill=False)
    ax_carthesian.add_artist(circle1)
    ax_carthesian.add_artist(circle2)     
    ax_polar = fig.add_axes([0.2, 0.2, 0.8, 0.8], polar=True, frameon=False) # Polar axis:
    ax_polar.axis([0, 10, 0, 10])
    plt.setp(ax_polar.get_yticklabels(), visible=False)
    ax_polar.set_xticks(linspace(0, 2*pi, 24, endpoint=False))
    ax_polar.set_xticklabels([str(i)+':00' for i in range(24)])
    ax_polar.set_theta_offset(pi/2.0) # 0:00 on top 
    ax_polar.set_theta_direction(-1) # Clockwise orientation
    ax_polar.grid('off')

    plt.draw()

def acrophaseDet(beta, gamma):
    def sgn(k): # 1 if k >= 0, 0 else
            sgn = 1.
            if k < 0.0:
                sgn = -1.
            return sgn
            
    theta = atan(abs(gamma/beta)) # Based on [1] (page 12)
    a = sgn(beta)
    b = sgn(gamma)
    if a == 1 and b == 1:
        phi = -theta
    elif a == -1 and b == 1:
        phi = -pi + theta;
    elif a == -1 and b == -1:
        phi = -pi - theta;
    elif a == 1 and b == -1:
        phi = -2*pi + theta   
    return phi

def cosinorFit(t,y,T,alpha):
    # t is the time vector
    # y is the vector containing the values at time t
    # T is the assumed period of the rhytm
    # alpha is the statistic significance needed

    print('---Parameter estimation---')  
    n = len(t)  # Number of points
    w = 2*pi/T  # Cicle length 
    
    # Our goal is to find an approximation to our y function in the form y(t) = M + Acos(wt + phi) 
    # Now we define transformations so that we have a linear model y(t) = M + beta*x(t) + gamma*z(t) with 
    # A = sqrt(beta^2 + gamma^2) and phi = (tan^(-1))(gamma/beta)
    x = array([cos(w*t[i]) for i in range(n)]) 
    z = array([sin(w*t[i]) for i in range(n)])  
    
    # Parameters are estimated using least squares method with normal equations in the form d = Su, where u is the solution vector
    S = array([[  n   ,  sum(x) ,   sum(z)], 
               [sum(x), dot(x,x), dot(x,z)],
               [sum(z), dot(x,z), dot(z,z)]])
               
    d = array( [sum(y), dot(y,x), dot(y,z)])
    M, beta, gamma = LA.solve(S,d) # The parameters are estimated
    # Now we transform our parameters to obtain the cosinor form
    Amp = hypot(beta,gamma)
    phi = acrophaseDet(beta,gamma)

    print('Mesor:',M)
    print('Amplitude:',Amp)
    print('Acrophase:',phi,'('+str((-1)*phi/w)+' hours)') # Considering a period T = 24h
    print('(Gamma: '+str(gamma)+'  Beta: '+str(beta)+')\n')
    
    print('Cosinor fit x Original data:') 
    aproxF = array([(M + Amp*cos(w*t[i] + phi)) for i in range(n)]) # aproxF is the "curve" that better fits our points
    plt.figure()
    plt.plot(t, y, 'g', t, aproxF, 'b')
    l1 = mlines.Line2D([], [], color='blue', label='Cosinor')
    l2 = mlines.Line2D([], [], color='green', label='Original')
    plt.legend(handles=[l2,l1])
    plt.xlabel('Time')
    plt.show()
    
    print('---Confidence Intervals---')  
    RSS = dot((y-(M + beta*x + gamma*z)),(y-(M + beta*x + gamma*z))) # The residual sum of squares
    sigma = sqrt(RSS/(n-3)) # Sample standard deviation
    
    # These next constants are the variances with respect to our variables
    X = dot((x - mean(x)),(x - mean(x)))
    Z = dot((z - mean(z)),(z - mean(z)))
    T = dot((x - mean(x)),(z - mean(z)))

    C = LA.inv(S) # The covariance matrix that describes our sample is E = (sigma^2)*C
    std_M = sigma*sqrt(C[0,0]) # The Mesor's standard deviation
    CI_M = scist.t.interval(1-alpha/2,n-3,M,std_M) # Now we can estimate a confidence interval for the Mesor
    print('Mesor CI:',CI_M)
    
    covM = C[1:3,1:3]*sigma**2 # This is the covariance matrix for beta and gamma
    invCovM = LA.inv(covM) # Its inverse 
    eigen = LA.eig(covM) # Its eigenvalues and eigenvectors are calculated
    
    # Now we will build the conic section representing the joint confidence region for beta and gamma
    # Its equation will be in the form:   a*beta^2 + b*beta*gamma + c*gamma^2 + d*beta + e*gamma + f <= 0
    F_distr = scist.f.ppf(1-alpha,2,n-3) # Percent point function to our confidence level
#    print('\nF value (aplha = '+str(alpha)+'): '+str(F_distr))
    
    a = X
    b = 2*T
    c = Z
    d = -2*X*beta - 2*T*gamma
    e = -2*T*beta - 2*Z*gamma
    f = X*(beta**2) + 2*T*beta*gamma + Z*(gamma**2) - 2*(sigma**2)*F_distr
#    fConservative = 
#    discriminant = b**2 - 4*a*c       
    
    zeroAmp = f<=0 # This constant tells us if our confidence region admits beta = gamma = 0 as a solution
    print('\nZero Amplitude Test Result:',zeroAmp)
    p_3a = scist.f.pdf(abs((X*(beta**2) + 2*T*beta*gamma + Z*(gamma**2))/(2*(sigma**2))),2,n-3) # p-value for the zero amplitude test
    print('Associated p-value: ', p_3a)
    
    if ~zeroAmp: # If the zero amplitude is rejected we can estimate the joint confidence region    
        dist = 2*sqrt(eigen[0][0]*F_distr*sigma) # This is an estimate of the amplitude of the interval containing statistic significant gamma values
            
        posGamma = linspace(gamma-2*dist,gamma+2*dist,4000) # A vector containing possible gamma values
        # Each gamma value has 2 possible beta values
        posBeta1 = array([ ( (-(b*posGamma[i] + d) + csqrt( (b*posGamma[i] + d)**2 - 4*a*(c*(posGamma[i]**2) + e*posGamma[i] + f) ))/(2*a)) for i in range(4000)])
        posBeta2 = array([ ( (-(b*posGamma[i] + d) - csqrt( (b*posGamma[i] + d)**2 - 4*a*(c*(posGamma[i]**2) + e*posGamma[i] + f) ))/(2*a)) for i in range(4000)])
        
        # If a posGamma value belongs to a point on the ellipse, the real part of the posBeta vectors must be different
        toDelete = []
        for i in range(4000):
            if real(posBeta1[i]) == real(posBeta2[i]):
                toDelete.append(i)
        # Exceptions are the outermost points of the ellipse in the x axis (gamma)
        for i in range(len(toDelete)):
            if i < (len(toDelete)-1):
                if (toDelete[i+1]-toDelete[i]) > 1:
                    del toDelete[i]
                    del toDelete[i+1]
                    break
            
        posGamma = delete(posGamma,toDelete)
        posBeta1 = real(delete(posBeta1,toDelete))
        posBeta2 = real(delete(posBeta2,toDelete))
        
        CI_Amp = [Amp,Amp]
        intGamma = [posGamma[0],posGamma[len(posGamma) - 1]]
        intBeta = [beta,beta]
        CI_phi = [phi,phi]    
        
        for i in range(len(posGamma)):
            # Finding the maximum and minimum values for beta
            if posBeta1[i] > posBeta2[i]:
                if posBeta1[i] > intBeta[1]:
                    intBeta[1] = posBeta1[i]
                if posBeta2[i] < intBeta[0]:
                    intBeta[0] = posBeta2[i]
            else:
                if posBeta2[i] > intBeta[1]:
                    intBeta[1] = posBeta2[i]
                if posBeta1[i] < intBeta[0]:
                    intBeta[0] = posBeta1[i]
            
            # Finding maximum and minimum values for the Amplitude
            Amp1 = sqrt(posGamma[i]**2 + posBeta1[i]**2)
            Amp2 = sqrt(posGamma[i]**2 + posBeta2[i]**2)
            if Amp1 > Amp2:
                if Amp1 > CI_Amp[1]:
                    CI_Amp[1] = Amp1 
                if Amp2 < CI_Amp[0]:
                    CI_Amp[0] = Amp2
            else:
                if Amp2 > CI_Amp[1]:
                    CI_Amp[1] = Amp2
                if Amp1 < CI_Amp[0]:
                    CI_Amp[0] = Amp1
                    
            # Finding maximum and minimum values for the Acrophase      
            phi1 = acrophaseDet(posBeta1[i],posGamma[i])
            phi2 = acrophaseDet(posBeta2[i],posGamma[i])
            if phi1 > phi2:
                if phi1 > CI_phi[1]:
                    CI_phi[1] = phi1
                if phi2 < CI_phi[0]:
                    CI_phi[0] = phi2
            else:
                if phi2 > CI_phi[1]:
                    CI_phi[1] = phi2
                if phi1 < CI_phi[0]:
                    CI_phi[0] = phi1
                    
        print('\nAmplitude CI:', CI_Amp)
        print('Acrophase CI:', CI_phi)
        print('(Gamma: ['+str(intGamma[0])+', '+str(intGamma[1])+'])')
        print('(Beta: ['+str(intBeta[0])+', '+str(intBeta[1])+'])')
    
        print('\nJoint Confidence Region:')
        confidencePlot(Amp,beta,gamma,posGamma,posBeta1,posBeta2) # Calls a function that prepares a Cosinor plot
    else:
        print('Since the Zero Amplitude Test couldnt be rejected, a statistic significant confidence region cant be determined')
    
def ini(): # Initial test function
    y = [102,96.8,97,92.5,95,93,99.4,99.8,105.5]
    t = array([97,130,167.5,187.5,218,247.5,285,315,337.5])*24/360
    cosinorFit(t,array(y),24,0.05)
    
def tester(): # Creates test vectors with a period of 24 hours
    Mesor = float(input('Enter the mesor value: '))
    Amplitude = float(input('Enter the amplitude value: '))
    Acrophase = float(input('Enter the acrophase value (hours): '))
    Error = float(input('Enter the error percentage: '))/100
    k = float(input('Enter number of cycles: '))
    n = int(input('Enter the number of points: '))
    
    t = arange(n)*((24*k)/n)
    w = 2*pi/24
    y = array([(Mesor+Amplitude*cos(w*t[i] - Acrophase*w)) for i in range(n)])
    y = y + random.uniform(-Error*Amplitude, Error*Amplitude, n)
    print('\n\nResults:\n')    
    cosinorFit(t,y,24,0.05)    
    
#tester()
ini()    