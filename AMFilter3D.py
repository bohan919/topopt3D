# TRANSLATED PYTHON FUNCTION FOR AMFilter.m by LANGELAAR
import numpy as np
import numpy.matlib
import scipy.sparse as sps
from copy import deepcopy

def AMFilter(x, baseplate, *args):
    #   Possible uses:
    #   xi = AMfilter(x, baseplate)   idem, with baseplate orientation specified
    #   [xi, df1dx, df2dx,...] = AMfilter(x, baseplate, df1dxi, df2dxi, ...)
    #       This includes also the transformation of design sensitivities
    # where
    #   x : blueprint design (2D array), 0 <= x(i,j) <= 1
    #   xi: printed design (2D array)
    #   baseplate: character indicating baseplate orientation: 'N','E','S','W'
    #              default orientation is 'S'
    #              for 'X', the filter is inactive and just returns the input.
    #   df1dx, df1dxi etc.:  design sensitivity (2D arrays)
    #INTERNAL SETTINGS
    P = 40
    ep = 1e-4 
    xi_0 = 0.5 # parameters for smooth max/min functions
    # INPUT CHECKS
    if baseplate=='X':
    # bypass option: filter does not modify the blueprint design
        xi = x
        varargout = args
        return xi, varargout
    baseplateUpper = baseplate.upper()
    orientation = "SWNE"
    nRot = orientation.find(baseplateUpper) 
    nSens = max(0, len(args))

    # Orientation
    x = np.rot90(x, nRot, axes=(0,2))
    xi = np.zeros(np.shape(x))
    lstArgs = list(args)
    i = 0
    for arg in lstArgs:
        arg = np.rot90(arg, nRot,axes=(0,2))   
        lstArgs[i] = arg
        i = i+1
    args = tuple(lstArgs)
    nelz, nely, nelx = np.shape(x)

    #AM Filter
    Ns = 5
    Q = P + np.log(Ns)/np.log(xi_0)             
    SHIFT = 100*np.finfo(float).tiny **(1/P)
#     SHIFT = 100*np.finfo(float).eps **(1/P)
    BACKSHIFT = 0.95*Ns**(1/Q)*SHIFT**(P/Q)     
    Xi = np.zeros((nelz,nely,nelx))
    keep = np.zeros((nelz, nely, nelx))
    sq = np.zeros((nelz, nely, nelx)) 

    # Baseline: identity
    xi[:,nely-1,:] = x[:,nely-1,:]
    for i in range(nely - 2, -1, -1): #nely-2,nely-3,  0
###
#     xi[:,0,:] = x[:,0,:]
#     for i in range(1, nely+1, 1): #1,2,  nely
###
        # compute maxima of current base row
        cbr = np.zeros((nelz + 2, 1, nelx + 2))+ SHIFT
        cbr[1:nelz+1,0,1:nelx+1] = xi[:,i+1,:] + SHIFT
                        
###
        keep[:, i,:] = (cbr[1:(nelz+1), 0, 0:nelx]**P + cbr[1:(nelz+1), 0, 1:(nelx+1)]**P + cbr[1:(nelz+1), 0, 2:(nelx+2)]**P +
                        cbr[0:nelz, 0, 1:(nelx+1)]**P + cbr[2:(nelz+2), 0, 1:(nelx+1)]**P)
        Xi[:, i,:] = keep[:, i,:]**(1 / Q) - BACKSHIFT
        sq[:, i,:] = np.sqrt((x[:, i,:] - Xi[:, i,:])** 2 + ep)
        xi[:,i,:] = 0.5*((x[:,i,:]+Xi[:,i,:])-sq[:,i,:]+np.sqrt(ep))
            
    # SENSITIVITIES
    if nSens != 0:
        dfxi = deepcopy(list(args))
        dfx = deepcopy(list(args))
        varLambda = np.zeros((nelz, nSens, nelx))
        
        # from top to base layer
        for i in range(nely - 1): # 0, ,nely-2
            # smin sensitivity terms
            dsmindx = 0.5 * (1 - (x[:, i,:] - Xi[:, i,:]) / sq[:, i,:])
            dsmindXi = 1 - dsmindx
            
            # smax sensitivity terms
            cbr = np.zeros((nelz + 2, 1, nelx + 2)) + SHIFT
            #+ SHIFT 看看能否避免nan
            cbr[1:nelz + 1, 0, 1:nelx + 1] = xi[:, i + 1,:] + SHIFT            

     
            dmx = np.zeros((nelz, Ns, nelx))
            for j in range(5):
                if j <= 2:
                    dmx[:, j,:] = (P / Q) * keep[:, i,:]**(1 / Q - 1) * cbr[1:(nelz+1), 0, j:(nelx+j)]**(P - 1)
                elif j == 3:
                    dmx[:, j,:] = (P / Q) * keep[:, i,:]**(1 / Q - 1) * cbr[0:nelz, 0, 1:(nelx+1)]**(P - 1)
                elif j == 4:
                    dmx[:, j,:] = (P / Q) * keep[:, i,:]**(1 / Q - 1) * cbr[2:(nelz+2), 0, 1:(nelx+1)]**(P - 1)

### padding for dmx
#             dmx_padding=mp.pad(dmx,(1,1),'constant')
            dmx_padding=np.zeros((nelz+2, Ns, nelx+2))
            dmx_padding[1:(nelz+1),:,1:(nelx+1)]=dmx[:,:,:]

###
        
            for k in range(nSens):
                dfx[k][:, i,:] = dsmindx * (dfxi[k][:, i,:] + varLambda[:, k,:])
                preLambda = np.zeros((nelz + 2, 1, nelx + 2))
                preLambda[1:(nelz + 1), 0, 1:(nelx + 1)] = (dfxi[k][:, i,:] + varLambda[:, k,:]) * dsmindXi
                for nz in range(nelz):
                    for nx in range(nelx):
                        varLambda[nz, k, nx] = np.sum(
                            
                                      np.array([  preLambda[nz + 1, 0, nx ]*dmx_padding[nz+ 1, 2, nx],
                                               preLambda[nz + 1, 0, nx + 1]*dmx_padding[nz + 1, 1, nx + 1],
                                               preLambda[nz + 1, 0, nx + 2]*dmx_padding[nz + 1, 0, nx + 2],
                                               preLambda[nz , 0, nx + 1]*dmx_padding[nz, 4, nx+ 1],
                                               preLambda[nz + 2, 0, nx + 1]*dmx_padding[nz+2 , 3, nx + 1]    ])
                                             )
                                      
        i = nely - 1
        for k in range(nSens):
                dfx[k][:, i,:] = dfxi[k][:, i,:] + varLambda[:, k,:]

    # ORIENTATION
    xi = np.rot90(xi,-nRot, axes=(0,2))
    varargout = ()
    for s in range(nSens):
        varargout = varargout + (np.rot90(dfx[s],-nRot,axes=(0,2)) ,)

    return xi, varargout