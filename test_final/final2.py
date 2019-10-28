from numpy import cos, arange
from math import pi
import cosinor as cs
import logread as lr

alpha = 0.05
T = 24*60*60
log = lr.LogRead('exlog.txt')
t1 = log.makeT()[1][0:48*60]
t2 = log.makeT(a=0)[1][0:48*60]
activity = [(123 + 45*cos((2*pi/T)*i - pi/3)) for i in range(48*60)]

for i in range(24):
	test1 = cs.Cosinor(t1[i*60:i*60+24*60],activity[i*60:i*60+24*60],T,alpha)
	test1.fit()
	test2 = cs.Cosinor(t2[i*60:i*60+24*60],activity[i*60:i*60+24*60],T,alpha)
	test2.fit()
	print('acroA='+str(test1.phi)+'; acroR='+str(test2.phi)+'; dif='+str(abs(test2.phi-test1.phi)))



