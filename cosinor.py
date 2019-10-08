# Cosinor fitting function - 11/09/2019
# Julius Andretti

from numpy import array, dot, hypot, linspace, real, delete, arange
from numpy import linalg as LA
from math import pi, cos, sin, atan, sqrt
from cmath import sqrt as csqrt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as ppt
from statistics import mean
import scipy.stats as scist

#
#def isBalanced(t,T):
#    n = len(t)
#    dist = (t[n-1] - t[0])/n

def acrophaseDet(beta, gamma):
    def sgn(k): #Dá o sinal de um número 
            sgn = 1.
            if k < 0.0:
                sgn = -1.
            return sgn
    theta = atan(abs(gamma/beta))
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

    n = len(t)  # Number of points
    w = 2*pi/T  # Cicle length 
    
    x = array([cos(w*t[i]) for i in range(n)])
    z = array([sin(w*t[i]) for i in range(n)])  
    
    S = array([[  n   ,  sum(x) ,   sum(z)],
               [sum(x), dot(x,x), dot(x,z)],
               [sum(z), dot(x,z), dot(z,z)]])
    d = array( [sum(y), dot(y,x), dot(y,z)])
    
    M, beta, gamma = LA.solve(S,d)
    
    Amp = hypot(beta,gamma)
    phi = acrophaseDet(beta,gamma)
        
    print('Mesor:',M)
    print('Amplitude:',Amp)
    print('Acrophase:',phi,'\n')
    
    # Plot 1: Cosinor fit x Original data 
#    aproxF = array([(M + A*cos(w*t[i] + phi)) for i in range(n)])
#    plt.figure()
#    plt.plot(t, y, 'g', t, aproxF, 'b')
#    l1 = mlines.Line2D([], [], color='blue', label='Cosinor')
#    l2 = mlines.Line2D([], [], color='green', label='Original')
#    plt.legend(handles=[l2,l1])
#    plt.xlabel('Time')
#    plt.show()
    
    RSS = dot((y-(M + beta*x + gamma*z)),(y-(M + beta*x + gamma*z)))
    sigma = sqrt(RSS/(n-3))
    
    X = dot((x - mean(x)),(x - mean(x)))
    Z = dot((z - mean(z)),(z - mean(z)))
    T = dot((x - mean(x)),(z - mean(z)))

    C = LA.inv(S)
    scale = sigma*sqrt(C[0,0])    
    CI_M = scist.t.interval(1-alpha/2,n-3,M,scale)
    print('Mesor CI:',CI_M)
    
#    covM = C[1:3,1:3]*sigma**2
#    invCovM = LA.inv(covM)
#    eigen = LA.eig(covM)    
    
    F_distr = scist.f.ppf(1-alpha,2,n-3)
    print('F value (aplha = '+str(alpha)+'): '+str(F_distr))
    a = X
    b = 2*T
    c = Z
    d = -2*X*beta - 2*T*gamma
    e = -2*T*beta - 2*Z*gamma
    f = X*(beta**2) + 2*T*beta*gamma + Z*(gamma**2) - 2*(sigma**2)*F_distr
#    fConservative = 
    discriminant = b**2 - 4*a*c       
    
    zeroAmp = f<=0
    print('Zero Amplitude Test Result:',zeroAmp)
    if ~zeroAmp:
        lado1 = sqrt(2*(a*(e**2)+ c*(d**2) - b*d*e + discriminant*f)*(a+c+hypot((a-c),b)))/discriminant
    #    l1 = 2*sqrt(eigen[0][0]*F_distr)
            
        posGamma = linspace(gamma-2*lado1,gamma+2*lado1,4000)
        posBeta1 = array([ ( (-(b*posGamma[i] + d) + csqrt( (b*posGamma[i] + d)**2 - 4*a*(c*(posGamma[i]**2) + e*posGamma[i] + f) ))/(2*a)) for i in range(4000)])
        posBeta2 = array([ ( (-(b*posGamma[i] + d) - csqrt( (b*posGamma[i] + d)**2 - 4*a*(c*(posGamma[i]**2) + e*posGamma[i] + f) ))/(2*a)) for i in range(4000)])
        
        toDelete = []
        for i in range(4000):
            if real(posBeta1[i]) == real(posBeta2[i]):
                toDelete.append(i)
        
        for i in range(len(toDelete)):
            if (toDelete[i+1]-toDelete[i]) > 1:
                del toDelete[i]
                del toDelete[i+1]
                break
            
        posGamma = delete(posGamma,toDelete)
        posBeta1 = delete(posBeta1,toDelete)
        posBeta2 = delete(posBeta2,toDelete)
                
    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
                
    
    plt.plot(posGamma, posBeta1, 'g', posGamma, posBeta2, 'g')
    plt.axis([-20, 20, -20, 20])
    plt.ylabel('Beta')
    plt.xlabel('Gamma')
    ax.set_aspect('equal')
    circle1 = plt.Circle((0,0),radius=17,ls='-',lw=2,fill=False)
    circle2 = plt.Circle((0,0),radius=19,ls='-',lw=2,fill=False)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    plt.show()  
    
    CI_Amp = [Amp,Amp]
    CI_phi = [phi,phi]    
    
    for i in range(len(posGamma)):
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
    
    print('Amplitude CI:', CI_Amp)
    print('Acrophase CI:', CI_phi)
    
    
    
y = [102,96.8,97,92.5,95,93,99.4,99.8,105.5]
t = array([97,130,167.5,187.5,218,247.5,285,315,337.5])/360

cosinorFit(t,array(y),1,0.05)


## BRISAS PRO PLOT

#    plt.figure()
#    ax = plt.subplot(111, polar=True)
#    #ax.scatter(angles, np.ones(100)*1)
#    
#    # suppress the radial labels
#    plt.setp(ax.get_yticklabels(), visible=False)
#    
#    # set the circumference labels
#    ax.set_xticks(linspace(0, 2*pi, 8, endpoint=False))
#    ax.set_xticklabels(arange(8)*3)
#    
#    # make the labels go clockwise
#    ax.set_theta_direction(-1)
#    
#    # place 0 at the top
#    ax.set_theta_offset(pi/2.0)    
#    
#    # plt.grid('off')
#    
#    # put the points on the circumference
    #plt.ylim(0,1)
#    color=['w','w','w','w']
#    theta = np.linspace(-np.pi, np.pi, 100)  
#    fig = plt.figure()# initializing the figure
#    rect = [0.1, 0.1, 0.8, 0.8]# setting the axis limits in [left, bottom, width, height]
#    ax_carthesian  = fig.add_axes(rect)# the carthesian axis:
#    ax_polar = fig.add_axes([0.1, 0.2, 0.8, 0.6], polar=True, frameon=False)# the polar axis:
#    plt.setp(ax_polar.get_yticklabels(), visible=False)
#    ax_polar.set_xticks(linspace(0, 2*pi, 8, endpoint=False))
#    ax_polar.set_xticklabels(['0:00','3:00','6:00','9:00','12:00','15:00','18:00','21:00',])
#    ax_polar.set_theta_offset(pi/2.0)   
#    ax_polar.set_theta_direction(-1)
#    ax_carthesian.plot(posGamma, posBeta1, 'g', posGamma, posBeta2, 'g')
#    plt.show()    