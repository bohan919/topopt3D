# import numpy as np
# import numpy.matlib
# import scipy.sparse as sps
# from scipy.sparse.linalg import spsolve

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import colors

# np.savetxt("output.csv", a_np_array, delimiter=",")

# SAVE AND READ NUMPY ARRAYS
# np.save('xPrint.npy', xPrint) # save
# new_num_arr = np.load('data.npy') # load

# from pyvista import examples
# examples.plot_wave()

import top3D
import top3DAM
import cProfile
import pstats
from pstats import SortKey
nelx = 40
nely = 20
nelz = 10
volfrac = 0.4
penal = 2
rmin = 2
heaviside = 0
# top3D.main(nelx, nely, nelz, volfrac, penal, rmin)
# top3DAM.main(nelx, nely, nelz, volfrac, penal, rmin, heaviside)
cProfile.run('top3D.main(nelx, nely, nelz, volfrac, penal, rmin)','cprofilestats3D.txt')
p = pstats.Stats('cprofilestats3D.txt')
p.sort_stats('tottime').print_stats(50)



# import AMFilter3D
# #x = np.random.randint(0, 2, size = (5, 10, 20))
# ### IMPORTANT NOTES:
# # for x-input to AMFilter3D, the layers are inputed from the top layer to the bottom layer
# ###################
# from scipy.io import loadmat
# fromPy = loadmat('dc_dv.mat')
# x2D = fromPy['xPhys']
# x = np.dstack((x2D, x2D))
# dc2D = fromPy['dc']
# dc = np.dstack((dc2D,dc2D))
# dv2D = fromPy['dv']
# dv = np.dstack((dv2D,dv2D))
# xprint = AMFilter3D.AMFilter(x, 'S',dc.astype('float64'), dv.astype('float64'))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid(np.linspace(0, 2, len(x)), np.linspace(0, 2, len(x)))
# plot = ax.plot_surface(X=X, Y=Y, Z=x, cmap='YlGnBu_r', vmin=0, vmax=200)