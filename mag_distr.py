# python3.7
"""
    - test FMD fitting by creating a synthetic exponential distribution
      and fitting it using the ML estimator

      Functionality:
        1) create synthetic distribution
           - select Mc, b and addNoise (Gaussian), important for Mc from KS-test for binned data
           
           @author tgoebel U of Memphis, 5/15/2019
"""
import numpy as np
import matplotlib.pyplot as plt
iRanSeed = 1234
np.random.seed( iRanSeed)
#-------fct. / module def-----------------------------------------------
from src.FMD_GR import FMD
oFMD = FMD()

#============================================================================================================
#                             variables and files
#============================================================================================================
N       = 5000
b       = 1.1
xmin    = 2.2
f_sigma = .15

# specify how completeness should be compute
mc_type = np.arange(1.5, 6, .05)
#1_ give range for misfit calculation:  np.arange(1.5, 6, .05)
#2  histogram bin with most events: 'maxCurvature'
#3  min. misfit calculated over whole data range 'KS'
#=========================================1==================================================================
#                             synthetic data
#============================================================================================================
oFMD.randomFMD( N, b, xmin)
#Gaussian uncertainty
oFMD.data['mag'] += np.random.randn( N)*f_sigma

#=========================================2==================================================================
#                             determine Mc and b usign KS-test
#============================================================================================================
oFMD.mag_dist( )
oFMD.get_Mc(  mc_type = mc_type)#'maxCurvature')

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