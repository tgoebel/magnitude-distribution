"""
    - magnitude frequency analysis fcts.

      -determine completeness from misfit between observation and model suign KS-test
      - computer b-value (analytical solution to ML estimator)
      - a-value
      - std error
      - distribution plots


           @author tgoebel U of Memphis, 5/15/2019
"""
from __future__ import division # include true division 
import numpy as np
import matplotlib.pyplot as plt
import os
#import scipy.stats




class FMD:
    """
        magnitude data, distribution and analysis functions
        - determine Mc, b and a value
        - plot magnitude histogram and cumulative distributions
        - create synthetic distributions

    example usage:
            self.loadASCII
            self.determineMC
            self.fit_GR
            self.plotDist


    TODO:
    benchmark example:


    TODO: synthetic data examples:
            np.random.seed( 123456)
            self.synthMags( N = 1e3, b=1.1, Mc=2)
            self.loadASCII
            self.determineMC
            self.fit_GR
            print( len(self.data['mag], self.par['b'], self.data['a']))


    """

    def __init__(self):
        self.data = { 'mag'     : np.array([]),
                      # hist and cumul distribution
                      'magBins' : np.array([]),
                      'magHist' : np.array([]),
                      'cumul'   : np.array([]),
                      }
        self.par  = {
                        'b'        : None, # float()
                        'a'        : None, # float()
                        'Mc'       : None, # float()
                        'stdDev'   : None, # uncertainty of b exponent based on N and b
                        'binsize'  : 0.1, # default value
                        }

    def error_check(self):
        if len( self.data['mag']) == 0:
            error_str = "no magnitudes assigned to self.data['mag']"
            raise ValueError( error_str)

        if self.par['Mc'] == None:
            error_str = 'Mc not found, run function getMc() first'
            raise ValueError( error_str)



    def get_Mc(self, mc_type='Mc', **kwargs):
        """ if mc_type is not specified, default behavior is max. curvature
                   mc_type = 'maxCurvature' - max. event bin of histogram #default
                             'KS'           - use whole magnitude range to compute min. KS distance
                             'Std'          - deviation from linearity from moving std-window

                             float          - set Mc
                             list or array  - give range or list of Mcs then use min KS-distance to get best fit in that range
                                            example: mc_type = [2.0, 4.0] - determine Mc between 2 and 4
                                                     mc_type = np.arange( 2, 4.1, .1) # find Mc between 2 and 4 using data spacing
                             """
        N = len( self.data['mag'])
        if isinstance( mc_type, (str)):
            if mc_type == 'maxCurvature' or mc_type == 'MC':
                # MC from maximum curvature
                self.mag_dist()
                sel = ( self.data['magHist'] == self.data['magHist'].max() )
                self.par['Mc'] = self.data['magBins'][sel.T].max()
            elif mc_type ==  'KS':# use KS stats to find best-fit xmin
                self.par['Mc'] = self.Mc_KS( np.unique( self.data['mag']),  **kwargs)
            else:
                error_str = "completeness type unknown:, %s, use 'KS, float, maxCurvature (or MC) or array for KS test" %( mc_type)
                raise ValueError(  error_str)

        elif isinstance( mc_type, (float, int) ):
            self.par['Mc']      = float( mc_type )

        elif isinstance( mc_type, (np.ndarray, list) ):#use KS stats for data within min. max. values
            if np.array( mc_type).shape[0] > 2:
                self.par['Mc'] = self.Mc_KS( mc_type,  **kwargs)
            else:
                sel_mag = np.logical_and( self.data['mag']>=np.array( mc_type).min(),
                                          self.data['mag']<=np.array( mc_type).max() )
                aMc_sim = self.data['mag'][sel_mag]
                self.par['Mc'] = self.Mc_KS(  np.unique( aMc_sim),  **kwargs)
        else:
            error_str = "completeness type unknown:, %s, use 'KS, float, maxCurvature (or MC) or array for KS test" %( mc_type)
            raise ValueError( error_str)

    def mag_dist(self, **kwargs):
        """
           - mag histogram and cumulative distr.
        :return
            self.data['magBins'], self.data['magHist']
            self.data['cumul']
        """
        if 'binsize' in kwargs.keys() and kwargs['binsize'] is not None:
            self.par['binsize'] = kwargs['binsize']
        # sort magnitudes
        self.data['mag']   = np.array( sorted( self.data['mag']))
        # cumulative distribution
        self.data['cumul'] = np.cumsum( np.ones( len( self.data['mag'])))[::-1]

        self.data['magBins'] = np.arange( round( self.data['mag'].min(), 1), self.data['mag'].max()+self.par['binsize'] , self.par['binsize'])
        self.data['magHist'], __ = np.histogram( self.data['mag'], self.data['magBins'])
        #switch to central bin instead of min, max from numpy
        self.data['magBins'] = self.data['magBins'][0:-1]+self.par['binsize']*.5

    def fit_GR(self, **kwargs):
        """  - basic MLE solution for b-value, compute a and stddev
        Parameters
        ----------
            kwargs: 'binCorrection' - adjust bin correction or set to zero
                                default: 0.5*Mc

        Returns: b, a,  stdDev (of b-value) as self.par[<tag>]

        see:
        Aki (1965) Max likelihood estimate of b in the formila log N = a-bM and its confidence limits, Bull. Earthquake Res. Inst. Univ. Tokyo,
        Shi, Y., and Bolt, B.A., 1982, The standard error of the magnitude-frequency b-value: Bull. Seismol. Soc. Am., v. 72, p. 1677-1687.
        -------

        """
        self.error_check()
        if 'binCorrection' in kwargs.keys() and kwargs['binCorrection'] is not None:
            binCorrection      = kwargs['binCorrection']
        else:
            binCorrection      = self.par['binsize']*0.5

        sel_Mc = self.data['mag']>=self.par['Mc']
        N      = sel_Mc.sum()
        meanMag            = self.data['mag'][sel_Mc].mean()
        self.par['b']      = ( 1 / ( self.data['mag'][sel_Mc].mean()- (self.par['Mc'] - binCorrection)) ) * np.log10( np.e )
        #                     ( 2.3 * np.sqrt((sum((vMag-meanMag)**2) ) / ( N * (N-1))) ) * bValue**2
        self.par['stdDev'] = ( 2.3 * np.sqrt((sum((self.data['mag'][sel_Mc]-meanMag)**2) ) / ( N * (N-1))) ) * self.par['b']**2
        self.par['a']      = np.log10( N) + self.par['b'] * self.par['Mc']

    def MLE_b(self, Mc, **kwargs):
        if 'binCorrection' in kwargs.keys() and kwargs['binCorrection'] is not None:
            binCorrection      = kwargs['binCorrection']
        else:
            binCorrection      = self.par['binsize']*0.5
        sel_Mc = self.data['mag'] >= Mc
        return ( 1. / ( self.data['mag'][sel_Mc].mean() - ( Mc - binCorrection)) ) * np.log10( np.e )

#    def std_b(self):
#        return ( 2.3 * np.sqrt((sum((self.data['mag']-meanMag)**2) ) / ( N * (N-1))) ) * self.par['b']**2

    def Mc_KS(self,  vMag_sim, **kwargs):
        """

        :param vMag_sim: - potential completeness values - find Mc with smallest misfit to G+R
        :param kwargs:
                maxErr_b  - float #Default = 0.25
                            threshold value for stdDev of b-value to avoid choosing Mc in the distrib. tail
                useAllMc  = True, #default = False
                            use the entire mag. range for potential completeness estimates
                            even if stdDev of b-value is bigger than maxErr_b
        :return:
        """
        if 'binCorrection' in kwargs.keys() and kwargs['binCorrection'] is not None:
            binCorrection      = kwargs['binCorrection']
        else:
            binCorrection      = self.par['binsize']*.5
        if 'maxErr_b' in kwargs.keys() and kwargs['maxErr_b'] is not None:
            maxErr_b  = kwargs['maxErr_b']
        else:
            maxErr_b  = .25
        #------------------------------------------------------------------------------

        sorted_Mag  = np.sort( self.data['mag'])
        vMag_sim    = np.sort( vMag_sim)
        vKS_stats   = np.zeros( vMag_sim.shape[0])
        vStd_err    = np.zeros( vMag_sim.shape[0])
        vB_sim      = np.zeros( vMag_sim.shape[0])
        i = 0

        for curr_Mc in vMag_sim:
            sel_Mc        = self.data['mag'] >= curr_Mc
            meanMag       = self.data['mag'][sel_Mc].mean()
            curr_b        = ( 1. / ( meanMag - ( curr_Mc - binCorrection)) ) * np.log10( np.e )#self.MLE_b( curr_Mc)
            vKS_stats[i]  = self.KS_D_value_PL( curr_Mc)
            sel = sorted_Mag >= curr_Mc
            N_aboveMc = sel.sum()
            if N_aboveMc > 1:
                vStd_err[i]   = ( 2.3 * np.sqrt((sum((self.data['mag'][sel_Mc]-meanMag)**2) ) / ( N_aboveMc* (N_aboveMc-1))) ) * curr_b**2
            else:
                vStd_err[i]   = 999
            vB_sim[i]     = curr_b
            i = i + 1
        # store in data dictionary to be called with self.plotKS
        self.data['a_KS']     = vKS_stats
        self.data['a_MagSim'] = vMag_sim

        if 'useAllMCs' in kwargs.keys() and kwargs['useAllMCs'] == True:
            return vMag_sim[vKS_stats == vKS_stats.min()][0]
        else:        
            # select xmins alpha values below corresponding to alpha value below a certain error threshold
            sel = vStd_err < maxErr_b
            if sel.sum() > maxErr_b:
                sel2 = vKS_stats[sel] == vKS_stats[sel].min()
                return vMag_sim[sel][sel2][0]
            else:
                return np.nan


    def KS_D_value_PL(self, Mc):
        """ compute KS statistic

        :input   Mc - completeness (Mc is updated to get best-fit if called from within Mc_KS() )

        :return ks_D - d statistic which is hte max. distance between modeled and observed cumul.

        see: Clauset, A., Shalizi, C.R., and Newmann, M.E.J., 2009,
        Power-law distributions in empirical data: SIAM review, v. 51, no. 4, p. 661-703.
        """

        aMag_tmp = np.sort( self.data['mag'][self.data['mag']>=Mc] )
        # convert Mc and Mag to use power-law scaling
        vX_tmp   = 10**aMag_tmp
        xmin     = 10**Mc
        n        = aMag_tmp.shape[0]
        if n == 0:
            return np.inf
        else:
            alpha    = float(n) / ( np.log(vX_tmp/xmin)).sum()
            obsCumul = np.arange(n, dtype='float')/n #from 0 to ; Fn = i/n; where i=1, ..., n see e.g. wikipedia Anderson-Darling
            modCumul = 1-(xmin/vX_tmp)**alpha
            ks = (abs(obsCumul - modCumul)).max()
            return ks


    def UTSU_test(self, n1, n2, b1, b2):
        """
        determine if b1 and b2 between two distributions 1 and 2
        are significantly different
        :input
                n1, n2 - number of events in distribution 1 and 2
                b1, b2 - b-values
        :return
                probability that b1 and b2 stem from the same population of magnitudes
        """
        n = n1+n2
        if b1 > 0:
            b1 = -b1
        if b2 > 0:
            b2 = -b2
        dA = -2*n*np.log( n) + 2*n1*np.log( (n2*b1)/b2 + n1) + 2*n2*np.log( (n1*b2)/b1 + n2) - 2
        return np.exp(  -(dA/2) - 2)

    #==================================================================================================
    #                         randomized data
    #==================================================================================================
    def randomFMD(self, N, b, xmin, **kwargs):
        """ create distrib. that follows powerlaw above Mc 
        input:
               N      - number of samples
               b      - G+R b-value
               xmin   - specify the range of the power law
                             #default - .1

        kwargs:  'binsize'  - for Mc binsize correction??
                 'addNoise' - True or float if True use
        ref: Clauest et al 2009, equ. D.4 - appendix
             Felzer el al. 2002, Triggering of the 1991 7.1 Hector Mine
        """
        if 'binsize' in  kwargs.keys() and kwargs['binsize'] is not None:
            self.par['binsize'] = kwargs['binsize']
        #return (.5*xmin)*(1.-numpy.random.ranf(N))**( -1./(alpha-1.))+0.5
        #self.data['mag'] = (xmin - self.par['binsize']*.5) - np.log10( np.random.ranf(N))/b
        self.data['mag'] = (xmin) - np.log10( np.random.ranf(N))/b
    #==================================================================================================
    #                         plots
    #==================================================================================================
    def plotDistr(self, ax = None):
        """
        - plot histogram, cumul. distrib, fit, and Mc
        :return:  axis object
        """
        if len( self.data['magBins']) == 0 or len( self.data['cumul']) == 0:
            error_str = 'mag bins and/or cumul empty run : self.mag_dist() first'
            raise ValueError( error_str)

        N   = len( self.data['mag'])
        if ax == None:
            plt.figure(1, figsize=(8,7))
            ax = plt.axes([.12,.126,.83,.83]) #(111)

        #___________________________ distribution___________________________________________
        ax.semilogy( self.data['magBins'], self.data['magHist'], 'ks', ms = 5, mew = 1, label = 'histogram')
        ax.semilogy( self.data['mag'], self.data['cumul'], 'bo', ms = 2, label='cumulative' )

        #___________________________ labels and limits------------------------------
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Number of Events')
        ax.legend( shadow = False, numpoints=1, loc = 'upper right')#, frameon = False)
        #ax.set_xlim( self.par['Mc']-2, self.data['mag'].max()+.5)#ax.get_xlim()[1])
        ax.set_ylim(1, N*1.2)
        ax.grid('on')
        return ax

    def plotFit(self, ax):
        """
        - plot histogram, cumul. distrib, fit, and Mc
        :return:  axis object
        """
        self.error_check()
        N   = len( self.data['mag'][self.data['mag'] >= self.par['Mc']])

        if ax == None:
            plt.figure(1, figsize=(8,7))
            ax = plt.axes([.12,.126,.83,.83]) #(111)

        #___________________________ distribution___________________________________________
        ax.semilogy( self.data['magBins'], self.data['magHist'], 'ks', ms = 5, mew = 1, label = 'histogram')
        sel  = self.data['mag'] > self.par['Mc'] - 1 # just one below completeness for fast plotting
        ax.semilogy( self.data['mag'][sel], self.data['cumul'][sel], 'bo', ms = 2, label='cumulative' )
        #___________________________ plot completeness magnitude___________________________________________
        #get mag. bin corresponding to Mc
        sel = abs( self.data['mag'] - self.par['Mc']) == abs( self.data['mag'] - self.par['Mc']).min()
        #print( len(sel), sel.sum(), len( self.data['cumul']), len( self.data['mag'])
        ax.plot( [self.par['Mc']], [self.data['cumul'][sel][0]], 'rv', ms = 10, label='$M_c = %.1f$' % (self.par['Mc']) )

        mag_hat = np.linspace( self.data['mag'].min()-2*self.par['binsize'],
                               self.data['mag'].max()+2*self.par['binsize'], 10)
        N_hat  = 10**( (-self.par['b']*mag_hat) + self.par['a'])
        #ax.semilogy( subEventMag, b_fit, 'r', label='$log(N) = -%.2f \cdot M_{\mathsf{rel}} + %.2f$' %(self.BValue, self.AValue) )
        ax.semilogy( mag_hat, N_hat, 'r--',
                     label='$log(N) = -%.1f \cdot M + %.1f$' %( round(self.par['b'],1), round( self.par['a'],1)) )

        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Number of Events')
        ax.set_title('$N (M>M_c) = %.0f ; \; \sigma_b = %.3f $' % (N, self.par['stdDev']))
        ax.legend( shadow = False, numpoints=1, loc = 'upper right')#, frameon = False)
        #ax.set_xlim( self.par['Mc']-2, self.data['mag'].max()+.5)#ax.get_xlim()[1])
        ax.set_ylim(1, len( self.data['mag'])*1.2)
        ax.grid('on')
        return ax
            
            
            
    def plotKS(self, ax = None):
        """
        plot results of KS test to find Mc and b-value
        :param ax:
        :return:
        """
        if 'a_KS' in self.data.keys():
            if ax == None:
                plt.figure()
                ax = plt.subplot(111)

            ax.plot( self.data['a_MagSim'], self.data['a_KS'], 'b-')
            ax.set_xlabel( 'Magnitude')
            ax.set_ylabel( 'KS-D')
            
            
            
            
            
       