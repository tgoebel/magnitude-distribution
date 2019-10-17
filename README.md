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

Use the following references when using this code:
Aki, K., 1965, Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits: Bull. Earthquake Res. Inst., Tokyo Univ., v. 43, p. 237–239.

Clauset, A., Shalizi, C.R., and Newmann, M.E.J., 2009, Power-law distributions in empirical data: SIAM review, v. 51, no. 4, p. 661–703.

Goebel, T.H.W., Kwiatek, G., Becker, T.W., Brodsky, E.E., and Dresen, G., 2017, What allows seismic events to grow big?: Insights from b-value and fault roughness analysis in laboratory stick-slip experiments: Geology, v. 45, no. 9, p. 815–818, doi: 10.1130/G39147.1.
