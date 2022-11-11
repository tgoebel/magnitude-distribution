# magnitude-distribution
FMD analysis module in python

created: April 22 2019

- analyze magnitude distribution that follow Gutenberg_Richter relationship:
log(N) = a - bM

- find MC
- plot distribution
- fit distribution 
- create random data

To get started run the two sample scripts:

      mag_distr.py          # create synthetic G+R distribution with noise and uses MLE to fit the distribution
      
      mag_distr_hs_2011.py  # fit Southern California seimicity catalogs fron Hauksson & Shearer, 2011
                            # with b = 1.00445, a = 7.09266617; Mc = 2.5000, and std_dev = 0.0050418
