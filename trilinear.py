import numpy as np
    
def trilinear_density(self,coords):
       """
       This method creates a trilinearly interpolated density map out of the 
       corner density values provided by the user.
       
       :param coords: corner density values, in the form [d_000,d_001, ...] 
       where values are in order xyz and 0 or 1s determine which face the 
       values refer to.
       :type coords: dict
       :return: D, (trilinear) density matrix
       :rtype: np array

       """
       D = np.zeros((self.L[0],self.L[1],self.L[2]))
       for i in range(self.L[1]):
              for j in range(self.L[2]):
                     for k in range(self.L[0]):
                            x_d = (i/(self.L[1]-1))
                            y_d = (j/(self.L[2]-1))
                            z_d = (k/(self.L[0]-1))
                            ## First interpolation
                            d_00 = coords['d_000']*(1-x_d) + \
                                   coords['d_100']*x_d
                            d_01 = coords['d_001']*(1-x_d) + \
                                   coords['d_101']*x_d
                            d_10 = coords['d_010']*(1-x_d) + \
                                   coords['d_110']*x_d
                            d_11 = coords['d_011']*(1-x_d) + \
                                   coords['d_111']*x_d
                            ## Second interpolation
                            d_0 = d_00*(1-y_d) + d_10*y_d
                            d_1 = d_01*(1-y_d) + d_11*y_d
                            ## Third interpolation
                            d = d_0*(1-z_d)+d_1*z_d
                            D[k,i,j] = d
              
       return D