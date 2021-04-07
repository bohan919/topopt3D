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
    Ns = 3
    Q = P + np.log(Ns)/np.log(xi_0)             
    SHIFT = 100*np.finfo(float).tiny **(1/P)
    BACKSHIFT = 0.95*Ns**(1/Q)*SHIFT**(P/Q)     
    XiY = np.zeros((nelz, nely, nelx))
    XiX = np.zeros((nelz, nely, nelx))
    keepY = np.zeros((nelz, nely, nelx))
    keepX = np.zeros((nelz, nely, nelx))
    sqY = np.zeros((nelz, nely, nelx))
    sqX = np.zeros((nelz, nely, nelx))

    # Baseline: identity
    xi[nelz - 1,:,:] = x[nelz - 1, :,:]
    xiY = np.zeros((nelz, nely, nelx))
    xiX = np.zeros((nelz, nely, nelx))
    
    for i in range(nelz-2, -1, -1):
        for j in range(nely):
            # compute maxima of current base row
            cbr = np.pad(xi[i+1,j,:],(1,1),'constant') + SHIFT
            keepY[i,j,:] = cbr[0:nelx]**P + cbr[1:nelx+1]**P + cbr[2:]**P
            XiY[i,j,:] = keepY[i,j,:]**(1/Q) - BACKSHIFT
            sqY[i,j,:] = np.sqrt( (x[i,j,:]-XiY[i,j,:])**2 + ep )
            # set row above to supported value using smooth minimum
            xiY[i,j,:] = 0.5 * ((x[i,j,:] + XiY[i,j,:]) - sqY[i,j,:] + np.sqrt(ep))
        for j in range(nelx):
            # compute maxima of current base column
            cbr = np.pad(xi[i+1,:,j],(1,1),'constant') + SHIFT
            keepX[i,:,j] = cbr[0:nely]**P + cbr[1:nely+1]**P + cbr[2:]**P
            XiX[i,:,j] = keepX[i,:,j]**(1/Q) - BACKSHIFT
            sqX[i,:,j] = np.sqrt( (x[i,:,j]-XiX[i,:,j])**2 + ep )
            # set row above to supported value using smooth minimum
            xiX[i,:, j] = 0.5 * ((x[i, :, j] + XiX[i, :, j]) - sqX[i, :, j] + np.sqrt(ep))
        xi[i,:,:] = np.maximum(xiY[i,:,:], xiX[i,:,:])
    # keep = np.maximum(keepY, keepX)
    # sq = np.maximum(sqY, sqX)
    # Xi = np.maximum(XiY, XiX)
            
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
        dfxiCol = deepcopy(list(args))
        dfxiRow = deepcopy(list(args))
        dfxCol = deepcopy(list(args))
        dfxRow = deepcopy(list(args))

        # # from top to base layer:
        # for j in range(nelx):
        #     varLambdaRow = np.zeros((nSens, nely))

        #     for i in range(nelz-1):
        #         # smin sensitivity terms
        #         dsmindx = 0.5*( 1-(x[i,:,j]-XiX[i,:,j])/sqX[i,:,j] )
        #         dsmindXi = 1-dsmindx
        #         # smax sensitivity terms
        #         cbr = np.pad(xi[i,:,j],(1,1),'constant') + SHIFT
        #         dmx = np.zeros((Ns,nely))
        #         for s in range(Ns):
        #             dmx[s,:] = (P/Q)*keepX[i,:,j]**(1/Q-1)*cbr[0+s:nely+s:1]**(P-1)
        #         # rearrange data for quick multiplication
        #         qj = np.matlib.repmat([[-1],[0],[1]],nely,1)
        #         qi = np.matlib.repmat(np.arange(nely)+1,3,1)
        #         qi = np.ravel(qi, order='F')
        #         qi = np.reshape(qi, (3*nely,1))

        #         qj = qj + qi
        #         qs = np.ravel(dmx, order='F')[np.newaxis]
        #         qsX, qsY = np.shape(qs)
        #         qs = np.reshape(qs, (qsX*qsY,1))
        #         dsmaxdxi = sps.csr_matrix( (np.squeeze(qs[1:len(qs)-1]), (np.squeeze(qi[1:len(qi)-1])-1,np.squeeze(qj[1:len(qj)-1])-1) ),dtype=np.float )
        #         dsmaxdxi.eliminate_zeros()
        #         for k in range(nSens):
        #             dfxRow[k][i,:,j] = (dsmindx*( dfxiRow[k][i,:,j]+varLambdaRow[k,:] ))[np.newaxis]
        #             varLambdaRow[k,:] = ( (dfxiRow[k][i,:,j]+varLambdaRow[k,:])*dsmindXi ) @ dsmaxdxi
        #     # base layer
        #     i = nelz
        #     for k in range(nSens):
        #         dfxRow[k][i - 1,:, j] = (dfxiRow[k][i - 1,:, j] + varLambdaRow[k,:])[np.newaxis]

        for j in range(nely):
            varLambdaCol = np.zeros((nSens, nelx))
            for i in range(nelz-1):
                # smin sensitivity terms
                dsmindx = 0.5*( 1-(x[i,j,:]-XiY[i,j,:])/sqY[i,j,:] )
                dsmindXi = 1-dsmindx
                # smax sensitivity terms
                cbr = np.pad(xi[i+1,j,:],(1,1),'constant') + SHIFT
                dmx = np.zeros((Ns,nelx))
                for s in range(Ns):
                    dmx[s,:] = (P/Q)*keepY[i,j,:]**(1/Q-1)*cbr[0+s:nelx+s:1]**(P-1)
                # rearrange data for quick multiplication
                qj = np.matlib.repmat([[-1],[0],[1]],nelx,1)
                qi = np.matlib.repmat(np.arange(nelx)+1,3,1)
                qi = np.ravel(qi, order='F')
                qi = np.reshape(qi, (3*nelx,1))

                qj = qj + qi
                qs = np.ravel(dmx, order='F')[np.newaxis]
                qsX, qsY = np.shape(qs)
                qs = np.reshape(qs, (qsX*qsY,1))
                dsmaxdxi = sps.csr_matrix( (np.squeeze(qs[1:len(qs)-1]), (np.squeeze(qi[1:len(qi)-1])-1,np.squeeze(qj[1:len(qj)-1])-1) ),dtype=np.float )
                dsmaxdxi.eliminate_zeros()
                for k in range(nSens):
                    dfxCol[k][i,j,:] = (dsmindx*( dfxiCol[k][i,j,:]+varLambdaCol[k,:] ))[np.newaxis]
                    varLambdaCol[k,:] = ( (dfxiCol[k][i,j,:]+varLambdaCol[k,:])*dsmindXi ) @ dsmaxdxi
            # base layer
            i = nelz
            for k in range(nSens):
                dfxCol[k][i - 1, j,:] = (dfxiCol[k][i - 1, j,:] + varLambdaCol[k,:])[np.newaxis]
       
        # dfx = [0.5*(dfxCol[i]+dfxRow[i]) for i in range(nSens)]
                
    
    # ORIENTATION
    xi = np.rot90(xi,-nRot, axes=(0,2))
    varargout = ()
    for s in range(nSens):
        varargout = varargout + (np.rot90(dfxCol[s],-nRot,axes=(0,2)) ,)

    return xi, varargout