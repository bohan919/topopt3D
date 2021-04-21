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
    BACKSHIFT = 0.95*Ns**(1/Q)*SHIFT**(P/Q)     
    Xi = np.zeros((nelz,nely,nelx))
    keep = np.zeros((nelz, nely, nelx))
    sq = np.zeros((nelz, nely, nelx)) 
    
    # XiZ = np.zeros((nelz, nely, nelx))
    # XiX = np.zeros((nelz, nely, nelx))
    # keepZ = np.zeros((nelz, nely, nelx))
    # keepX = np.zeros((nelz, nely, nelx))
    # sqZ = np.zeros((nelz, nely, nelx))
    # sqX = np.zeros((nelz, nely, nelx))

    # Baseline: identity
    xi[:,nely-1,:] = x[:,nely-1,:]
    # xiZ = np.zeros((nelz, nely, nelx))
    # xiX = np.zeros((nelz, nely, nelx))
    for i in range(nely - 2, -1, -1):
        # compute maxima of current base row
        cbr = np.zeros((nelz + 2, 1, nelx + 2))
        cbr[1:nelz+1,0,1:nelx+1] = xi[:,i+1,:] + SHIFT
        keep[:, i,:] = (cbr[1:nelz+1, 0, 0:nelx]**P + cbr[1:nelz+1, 0, 1:nelx+1]**P + cbr[1:nelz+1, 0, 2:nelx+2]**P +
                        cbr[0:nelz, 0, 1:nelx+1]**P + cbr[2:nelz+2, 0, 1:nelx+1]**P)
        Xi[:, i,:] = keep[:, i,:]**(1 / Q) - BACKSHIFT
        sq[:, i,:] = np.sqrt((x[:, i,:] - Xi[:, i,:])** 2 + ep)
        xi[:,i,:] = 0.5*((x[:,i,:]+Xi[:,i,:])-sq[:,i,:]+np.sqrt(ep))
        # for j in range(nelz):
        #     # compute maxima of current base row
        #     cbr = np.pad(xi[j,i+1,:],(1,1),'constant') + SHIFT
        #     keepZ[j,i,:] = cbr[0:nelx]**P + cbr[1:nelx+1]**P + cbr[2:]**P
        #     XiZ[j,i,:] = keepZ[j,i,:]**(1/Q) - BACKSHIFT
        #     sqZ[j,i,:] = np.sqrt( (x[j,i,:]-XiZ[j,i,:])**2 + ep )
        #     # set row above to supported value using smooth minimum
        #     xiZ[j,i,:] = 0.5 * ((x[j,i,:] + XiZ[j,i,:]) - sqZ[j,i,:] + np.sqrt(ep))
        # for j in range(nelx):
        #     # compute maxima of current base column
        #     cbr = np.pad(xi[:,i+1,j],(1,1),'constant') + SHIFT
        #     keepX[:,i,j] = cbr[0:nelz]**P + cbr[1:nelz+1]**P + cbr[2:]**P
        #     XiX[:,i,j] = keepX[:,i,j]**(1/Q) - BACKSHIFT
        #     sqX[:,i,j] = np.sqrt( (x[:,i,j]-XiX[:,i,j])**2 + ep )
        #     # set row above to supported value using smooth minimum
        #     xiX[:,i,j] = 0.5 * ((x[:,i,j] + XiX[:,i,j]) - sqX[:,i,j] + np.sqrt(ep))
        # xi[:,i,:] = np.maximum(xiZ[:,i,:], xiX[:,i,:])
    # keep = np.maximum(keepZ, keepX)
    # sq = np.maximum(sqZ, sqX)
    # Xi = np.maximum(XiZ, XiX)
            
    # SENSITIVITIES
    if nSens != 0:
        # dfxi = ()
        # for arg in args:
        #     dfxi = dfxi + (copy.deepcopy(np.reshape(arg,(nelz, nely, nelx))),)
        # #dfxi = args
        # #dfx = args
        # dfx = ()
        # for arg in args:
        #     dfx = dfx + (copy.deepcopy(np.reshape(arg,(nelz, nely, nelx))),)
        # dfxCol = copy.deepcopy(dfx)
        # dfxRow = copy.deepcopy(dfx)
        # dfxiCol = copy.deepcopy(dfxi)
        # dfxiRow = copy.deepcopy(dfxi)
        # dfxiCol = deepcopy(list(args))
        # dfxiRow = deepcopy(list(args))
        # dfxCol = deepcopy(list(args))
        # dfxRow = deepcopy(list(args))

        dfxi = deepcopy(list(args))
        dfx = deepcopy(list(args))
        varLambda = np.zeros((nelz, nSens, nelx))

        # # from top to base layer:
        # for j in range(nelx):
        #     varLambdaRow = np.zeros((nSens, nelz))

        #     for i in range(nely-1):
        #         # smin sensitivity terms
        #         dsmindx = 0.5*( 1-(x[:,i,j]-XiX[:,i,j])/sqX[:,i,j] )
        #         dsmindXi = 1-dsmindx
        #         # smax sensitivity terms
        #         cbr = np.pad(xi[:,i,j],(1,1),'constant') + SHIFT
        #         dmx = np.zeros((Ns,nelz))
        #         for s in range(Ns):
        #             dmx[s,:] = (P/Q)*keepX[:,i,j]**(1/Q-1)*cbr[0+s:nelz+s:1]**(P-1)
        #         # rearrange data for quick multiplication
        #         qj = np.matlib.repmat([[-1],[0],[1]],nelz,1)
        #         qi = np.matlib.repmat(np.arange(nelz)+1,3,1)
        #         qi = np.ravel(qi, order='F')
        #         qi = np.reshape(qi, (3*nelz,1))

        #         qj = qj + qi
        #         qs = np.ravel(dmx, order='F')[np.newaxis]
        #         qsX, qsY = np.shape(qs)
        #         qs = np.reshape(qs, (qsX*qsY,1))
        #         dsmaxdxi = sps.csr_matrix( (np.squeeze(qs[1:len(qs)-1]), (np.squeeze(qi[1:len(qi)-1])-1,np.squeeze(qj[1:len(qj)-1])-1) ),dtype=np.float )
        #         dsmaxdxi.eliminate_zeros()
        #         for k in range(nSens):
        #             dfxRow[k][:,i,j] = (dsmindx*( dfxiRow[k][:,i,j]+varLambdaRow[k,:] ))[np.newaxis]
        #             varLambdaRow[k,:] = ( (dfxiRow[k][:,i,j]+varLambdaRow[k,:])*dsmindXi ) @ dsmaxdxi
        #     # base layer
        #     i = nely
        #     for k in range(nSens):
        #         dfxRow[k][:, i-1, j] = (dfxiRow[k][:, i-1, j] + varLambdaRow[k,:])[np.newaxis]
        
        # from top to base layer
        for i in range(nely - 1):
            # smin sensitivity terms
            dsmindx = 0.5 * (1 - (x[:, i,:] - Xi[:, i,:]) / sq[:, i,:])
            dsmindXi = 1 - dsmindx
            
            # smax sensitivity terms
            cbr = np.zeros((nelz + 2, 1, nelx + 2))
            cbr[1:nelz + 1, 0, 1:nelx + 1] = xi[:, i + 1,:] + SHIFT
            
            dmx = np.zeros((nelz, Ns, nelx))
            for j in range(3):
                if j <= 2:
                    dmx[:, j,:] = (P / Q) * keep[:, i,:]**(1 / Q - 1) * cbr[1:nelz+1, 0, j:(nelx+j)]**(P - 1)
                elif j == 3:
                    dmx[:, j,:] = (P / Q) * keep[:, i,:]**(1 / Q - 1) * cbr[0:nelz, 0, 1:nelx+1]**(P - 1)
                elif j == 4:
                    dmx[:, j,:] = (P / Q) * keep[:, i,:]**(1 / Q - 1) * cbr[2:nelz+2, 0, 1:nelx+1]**(P - 1)

            for k in range(nSens):
                dfx[k][:, i,:] = dsmindx * (dfxi[k][:, i,:] + varLambda[:, k,:])
                preLambda = np.zeros((nelz + 2, 1, nelx + 2))
                preLambda[1:nelz + 1, 0, 1:nelx + 1] = (dfxi[k][:, i,:] + varLambda[:, k,:]) * dsmindXi
                for nz in range(nelz):
                    for nx in range(nelx):
                        varLambda[nz, k, nx] = np.sum(
                            np.array([preLambda[nz + 1, 0, nx - 1 + 1],
                                      preLambda[nz + 1, 0, nx + 1],
                                      preLambda[nz + 1, 0, nx + 1 + 1],
                                      preLambda[nz - 1 + 1, 0, nx + 1],
                                      preLambda[nz + 1 + 1, 0, nx + 1]])*dmx[nz, 1, nx])
                                      
            i = nely - 1
            for k in range(nSens):
                dfx[k][:, i,:] = dfxi[k][:, i,:] + varLambda[:, k,:]

        


            # varLambdaCol = np.zeros((nSens, nelx))
            # for i in range(nely-1):
            #     # smin sensitivity terms
            #     dsmindx = 0.5*( 1-(x[j,i,:]-XiZ[j,i,:])/sqZ[j,i,:] )
            #     dsmindXi = 1-dsmindx
            #     # smax sensitivity terms
            #     cbr = np.pad(xi[j,i+1,:],(1,1),'constant') + SHIFT
            #     dmx = np.zeros((Ns,nelx))
            #     for s in range(Ns):
            #         dmx[s,:] = (P/Q)*keepZ[j,i,:]**(1/Q-1)*cbr[0+s:nelx+s:1]**(P-1)
            #     # rearrange data for quick multiplication
            #     qj = np.matlib.repmat([[-1],[0],[1]],nelx,1)
            #     qi = np.matlib.repmat(np.arange(nelx)+1,3,1)
            #     qi = np.ravel(qi, order='F')
            #     qi = np.reshape(qi, (3*nelx,1))

            #     qj = qj + qi
            #     qs = np.ravel(dmx, order='F')[np.newaxis]
            #     qsX, qsY = np.shape(qs)
            #     qs = np.reshape(qs, (qsX*qsY,1))
            #     dsmaxdxi = sps.csr_matrix( (np.squeeze(qs[1:len(qs)-1]), (np.squeeze(qi[1:len(qi)-1])-1,np.squeeze(qj[1:len(qj)-1])-1) ),dtype=np.float )
            #     dsmaxdxi.eliminate_zeros()
            #     for k in range(nSens):
            #         dfxCol[k][j,i,:] = (dsmindx*( dfxiCol[k][j,i,:]+varLambdaCol[k,:] ))[np.newaxis]
            #         varLambdaCol[k,:] = ( (dfxiCol[k][j,i,:]+varLambdaCol[k,:])*dsmindXi ) @ dsmaxdxi
            # base layer
            # i = nely
            # for k in range(nSens):
            #     dfxCol[k][j,i - 1,:] = (dfxiCol[k][j,i - 1,:] + varLambdaCol[k,:])[np.newaxis]
       
        # dfx = [0.5*(dfxCol[i]+dfxRow[i]) for i in range(nSens)]
                
    
    # ORIENTATION
    xi = np.rot90(xi,-nRot, axes=(0,2))
    varargout = ()
    for s in range(nSens):
        varargout = varargout + (np.rot90(dfx[s],-nRot,axes=(0,2)) ,)

    return xi, varargout