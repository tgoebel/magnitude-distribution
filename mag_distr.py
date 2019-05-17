#python2.7
"""
    - test FMD fitting by creating a synthetic exponential distribution
      and fitting it using the ML estimator

      Functionality:
        1) create synthetic distribution
           - select Mc, b and addNoise (Gaussian)
           
           @author tgoebel U of Memphis, 5/15/2019
"""
from __future__ import division

# some changes here check !!

import numpy as np
import matplotlib.pyplot as plt
iRanSeed = 1234
np.random.seed( iRanSeed)
#-------fct. / module def-----------------------------------------------
from src.FMD_GR import *
oFMD = FMD()

#============================================================================================================
#                             variables and files
#============================================================================================================
N       = int(5* 1e3)
b       = 1.2
xmin    = 2.2
f_sigma = .15 #uncertain: .15 * magBinsize

aMc     = np.arange(1.5, 6, .05)  #'maxCurvature' #np.arange(1.5, 4, .05) #'KS' #'MC'#2.5



#=========================================1==================================================================
#                             synthetic data
#============================================================================================================
oFMD.randomFMD( N, b, xmin)
#Gaussian uncertainty
oFMD.data['mag'] += np.random.randn( N)*f_sigma


#=========================================2==================================================================
#                             determine Mc and b usign KS-test
#============================================================================================================
oFMD.mag_dist()
oFMD.get_Mc( aMc)

print( 'completeness', round( oFMD.par['Mc'], 1))
oFMD.fit_GR( binCorrection = 0)
print( oFMD.par)

#=========================================3==================================================================
#                                plot FMD and KS statistic as fct of Mc
#============================================================================================================
#oFMD.plotDistr()
plt.figure(1, figsize = (6,10))
ax1 = plt.subplot(211)
oFMD.plotFit(ax1)
ax2 = plt.subplot( 212)
oFMD.plotKS( ax2)
ax2.set_xlim( ax1.get_xlim())
plt.savefig( 'FMD_test_seed_%i.png'%( iRanSeed))
plt.show()