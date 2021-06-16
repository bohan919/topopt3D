# TOPOLOGY OPTIMISATION BASED ON top88.mat WITH LANGELAAR'S AMFILTER FOR 2D BY BOHAN PENG - IMPERIAL COLLEGE LONDON 2021
# AMFILTER CALL TYPE 1 - OBTAIN xPrint only
# AMFILTER CALL TYPE 2 - OBTAIN xPrint and Sensitivities
# heaviside - 0: density filter only
#             1: density filter and heaviside filter
# DISCLAIMER -                                                             #
# The author reserves all rights but does not guaranty that the code is    #
# free from errors. Furthermore, he shall not be liable in any event       #
# caused by the use of the program.                                        #

import numpy as np
from scipy.sparse import csr_matrix
from pypardiso import spsolve

# GENERATE ELEMENT STIFFNESS MATRIX 
def lk_H8(nu):
    A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                  [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]])
    k = 1 / 144 * A.T @ np.array([[1], [nu]])
    
    K1 = np.array([[k[0], k[1], k[1], k[2], k[4], k[4]],
                [k[1], k[0], k[1], k[3], k[5], k[6]],
                [k[1], k[1], k[0], k[3], k[6], k[5]],
                [k[2], k[3], k[3], k[0], k[7], k[7]],
                [k[4], k[5], k[6], k[7], k[0], k[1]],
                [k[4], k[6], k[5], k[7], k[1], k[0]]])
    K1 = K1.squeeze()
    K2 = np.array([[k[8], k[7], k[11], k[5], k[3], k[6]],
                    [k[7], k[8], k[11], k[4], k[2], k[4]],
                    [k[9], k[9], k[12], k[6], k[3], k[5]],
                    [k[5], k[4], k[10], k[8], k[1], k[9]],
                    [k[3], k[2], k[4], k[1], k[8], k[11]],
                    [k[10], k[3], k[5], k[11], k[9], k[12]]])
    K2 = K2.squeeze()
    K3 = np.array([[k[5], k[6], k[3], k[8], k[11], k[7]],
                    [k[6], k[5], k[3], k[9], k[12], k[9]],
                    [k[4], k[4], k[2], k[7], k[11], k[8]],
                    [k[8], k[9], k[1], k[5], k[10], k[4]],
                    [k[11], k[12], k[9], k[10], k[5], k[3]],
                    [k[1], k[11], k[8], k[3], k[4], k[2]]])
    K3 = K3.squeeze()
    K4 = np.array([[k[13], k[10], k[10], k[12], k[9], k[9]],
                    [k[10], k[13], k[10], k[11], k[8], k[7]],
                    [k[10], k[10], k[13], k[11], k[7], k[8]],
                    [k[12], k[11], k[11], k[13], k[6], k[6]],
                    [k[9], k[8], k[7], k[6], k[13], k[10]],
                    [k[9], k[7], k[8], k[6], k[10], k[13]]])
    K4 = K4.squeeze()
    K5 = np.array([[k[0], k[1], k[7], k[2], k[4], k[3]],
                    [k[1], k[0], k[7], k[3], k[5], k[10]],
                    [k[7], k[7], k[0], k[4], k[10], k[5]],
                    [k[2], k[3], k[4], k[0], k[7], k[1]],
                    [k[4], k[5], k[10], k[7], k[0], k[7]],
                    [k[3], k[10], k[5], k[1], k[7], k[0]]])
    K5 = K5.squeeze()
    K6 = np.array([[k[13], k[10], k[6], k[12], k[9], k[11]],
                    [k[10], k[13], k[6], k[11], k[8], k[1]],
                    [k[6], k[6], k[13], k[9], k[1], k[8]],
                    [k[12], k[11], k[9], k[13], k[6], k[10]],
                    [k[9], k[8], k[1], k[6], k[13], k[6]],
                    [k[11], k[1], k[8], k[10], k[6], k[13]]])
    K6 = K6.squeeze()
    K1st = np.concatenate((K1, K2, K3, K4), axis=1)
    K2nd = np.concatenate((K2.T, K5, K6, K3.T), axis=1)
    K3rd = np.concatenate((K3.T, K6, K5.T, K2.T), axis=1)
    K4th = np.concatenate((K4, K3, K2, K1.T), axis=1)
    Ktot = np.concatenate((K1st, K2nd, K3rd, K4th), axis=0)

    KE = 1/((nu+1)*(1-2*nu))*Ktot 
    return KE

def main(nelx, nely, nelz, volfrac, penal, rmin):
    # USER DEFINED LOOP PARAMETERS
    maxloop = 1000
    tolx = 0.01
    displayflag = 0
    # USER DEFINED MATERIAL PROPERTIES
    E0 = 1
    Emin = 1e-9
    nu = 0.3
    # USER DEFINED LOAD DoFs
    il, jl, kl = np.meshgrid(nelx, 0, np.arange(nelz + 1))
    loadnid = kl * (nelx + 1) * (nely + 1) + il * (nely + 1) + (nely + 1 - jl)
    loaddof = 3 * np.ravel(loadnid, order='F') - 1  #CURRENTLY A 1D ARRAY (used for sparse later)
    # USER DEFINED SUPPORT FIXED DOFS
    iif, jf, kf = np.meshgrid(0, np.arange(nely + 1), np.arange(nelz + 1))
    fixednid = kf * (nelx + 1) * (nely + 1) + iif * (nely + 1) + (nely + 1 - jf)
    fixeddof = np.concatenate((3 * np.ravel(fixednid, order='F'), 3*np.ravel(fixednid, order='F')-1,
                           3*np.ravel(fixednid, order='F') - 2)) #CURRENTLY A 1D ARRAY (used for sparse later)

    # PREPARE FE ANALYSIS
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    F = csr_matrix((-1 * np.ones(np.shape(loaddof)), (loaddof-1, np.ones(np.shape(loaddof))-1)),
                    shape=(ndof, 1))
    U = np.zeros((ndof, 1))
    freedofs = np.setdiff1d(np.arange(ndof) + 1, fixeddof)
    KE = lk_H8(nu)
    nodegrd = np.reshape(np.arange((nely + 1) * (nelx + 1)) + 1, (nely + 1, nelx + 1), order = 'F')
    nodeids = np.reshape(nodegrd[0:-1, 0:-1], (nely * nelx, 1), order='F')
    nodeidz = np.arange(0, (nelz - 1) * (nely + 1) * (nelx + 1) + 1, (nely + 1) * (nelx + 1))[np.newaxis]
    nodeids = (np.matlib.repmat(nodeids, np.shape(nodeidz)[0], np.shape(nodeidz)[1])
                    + np.matlib.repmat(nodeidz, np.shape(nodeids)[0], np.shape(nodeids)[1]))
    edofVec = (3 * np.ravel(nodeids, order='F') + 1)[np.newaxis]
    edofMat = (np.matlib.repmat(edofVec.T, 1, 24) +
                np.matlib.repmat(np.concatenate((
                    np.array([0, 1, 2]), 3*nely + np.array([3, 4, 5, 0, 1, 2]), np.array([-3, -2, -1]),
                    3*(nely + 1)*(nelx + 1) + np.concatenate((
                        np.array([0, 1, 2]), 3*nely+np.array([3, 4, 5, 0, 1, 2]), np.array([-3, -2, -1])
                        ))
                    )), nele, 1))
    iK = np.reshape(np.kron(edofMat, np.ones((24, 1))).T, (24 * 24 * nele, 1), order='F')
    jK = np.reshape(np.kron(edofMat, np.ones((1, 24))).T, (24 * 24 * nele, 1), order='F')

    # PREPARE FILTER
    iH = np.ones((int(nele * (2 * (np.ceil(rmin) - 1) + 1)** 2), 1))
    iHdummy = []
    jH = np.ones(np.shape(iH))
    jHdummy = []
    sH = np.zeros(np.shape(iH))
    sHdummy = []
    k = 0
    #####################
    for k1 in np.arange(nelz)+1:
        for i1 in np.arange(nelx)+1:
            for j1 in np.arange(nely)+1:
                e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1
                for k2 in np.arange(max(k1 - (np.ceil(rmin) - 1), 1), min(k1 + (np.ceil(rmin) - 1), nelz) + 1):
                    for i2 in np.arange(max(i1 - (np.ceil(rmin) - 1), 1), min(i1 + (np.ceil(rmin) - 1), nelx) + 1):
                        for j2 in np.arange(max(j1 - (np.ceil(rmin) - 1), 1), min(j1 + (np.ceil(rmin) - 1), nely) + 1):
                            e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2
                            if k < np.size(iH):
                                iH[k] = e1
                                jH[k] = e2
                                sH[k] = max(0, rmin - np.sqrt((i1 - i2)** 2 + (j1 - j2)** 2 + (k1 - k2)** 2))
                            else:
                                # iH = np.append(iH, [[e1]], 0)
                                iHdummy.append(e1)
                                # jH = np.append(jH, [[e2]], 0)
                                jHdummy.append(e2)
                                # sH = np.append(sH, [[max(0, rmin - np.sqrt((i1 - i2)** 2 + (j1 - j2)** 2 + (k1 - k2)** 2))]], 0)
                                sHdummy.append(max(0, rmin - np.sqrt((i1 - i2)** 2 + (j1 - j2)** 2 + (k1 - k2)** 2)))
                            k = k + 1
    #####################
    iH = np.concatenate((iH, np.array(iHdummy).reshape((len(iHdummy), 1))))
    jH = np.concatenate((jH, np.array(jHdummy).reshape((len(jHdummy), 1))))
    sH = np.concatenate((sH, np.array(sHdummy).reshape((len(sHdummy), 1))))

    H = csr_matrix((np.squeeze(sH), (np.squeeze(iH.astype(int)) - 1, np.squeeze(jH.astype(int)) - 1)))
    Hs = csr_matrix.sum(H,axis=0).T

    # INITIALIZE ITERATION
    x = np.tile(volfrac, [nelz, nely, nelx])
    xPhys = x
    loop = 0
    change = 1
    # START ITERATION
    while change > tolx and loop < maxloop:
        loop = loop + 1
        # FE ANALYSIS
        sK = np.reshape(np.ravel(KE, order='F')[np.newaxis].T @ (Emin+xPhys.transpose(0,2,1).ravel(order='C')[np.newaxis]**penal*(E0-Emin)),(24*24*nele,1),order='F')
        K = csr_matrix((np.squeeze(sK), (np.squeeze(iK.astype(int)) - 1, np.squeeze(jK.astype(int)) - 1)))
        K = (K + K.T) / 2
        U[freedofs - 1,:] = spsolve(K[freedofs - 1,:][:, freedofs - 1], F[freedofs - 1,:])[np.newaxis].T 

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce = np.reshape(np.sum((U[edofMat - 1].squeeze() @ KE) * U[edofMat - 1].squeeze(), axis=1), (nelz, nelx, nely), order = 'C').transpose(0,2,1)
        c = np.sum(np.sum(np.sum(Emin + xPhys ** penal * (E0 - Emin) * ce)))
        dc = -penal * (E0 - Emin) * (xPhys ** (penal - 1)) * ce
        dv = np.ones((nelz, nely, nelx))
        # FILTERING AND MODIFICATION OF SENSITIVITIES
        dc = np.array((H @ (dc.transpose(0,2,1).ravel(order='C')[np.newaxis].T/Hs))).reshape((nelz, nelx, nely), order = 'C').transpose(0,2,1)
        dv = np.array((H @ (dv.transpose(0,2,1).ravel(order='C')[np.newaxis].T/Hs))).reshape((nelz, nelx, nely), order = 'C').transpose(0,2,1)

        # OPTIMALITY CRITERIA UPDATE
        l1 = 0
        l2 = 1e9
        move = 0.05
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xPhys = np.array((H @ (xnew.transpose(0,2,1).ravel(order='C')[np.newaxis].T)/Hs)).reshape((nelz, nelx, nely), order = 'C').transpose(0,2,1)
            if np.sum(xPhys.transpose(0,2,1).ravel(order='C')) > volfrac * nele:
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.absolute(np.ravel(xnew, order='F') - np.ravel(x, order='F')))
        x = xnew
        print("it.: {0} , ch.: {1:.3f}, obj.: {2:.4f}, Vol.: {3:.3f}".format(
            loop, change, c, np.mean(xPhys.transpose(0,2,1).ravel(order='C'))))
    
    return xPhys




    
    
