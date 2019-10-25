import logread as lr
import cosinor as cs

log = lr.LogRead('exlog.txt')

ini = cs.Cosinor((log.makeT()[1]),log.pim,24*60*60,0.05)

# Calculations
ini.fit()

# Prints the estimated parameters
ini.printParam()

# # Prints the confidence interval for the Mesor and the confidence pĺot for Amplitude and Acrophase (if the Zero Amplitude Test fails)
# ini.printParamCI()