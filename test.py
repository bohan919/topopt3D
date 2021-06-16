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
import numpy as np
import top3D
import top3DAM
import cProfile
import pstats
from pstats import SortKey
nelx = 60
nely = 30
nelz = 20
volfrac = 0.4
penal = 2
rmin = 2
heaviside = 0
# xPrint = top3D.main(nelx, nely, nelz, volfrac, penal, rmin)
xPrintAM = top3DAM.main(nelx, nely, nelz, volfrac, penal, rmin, heaviside)
# np.save('xPrint_60_30_1.npy', xPrint) # save

# Profiling
# cProfile.run('top3D.main(nelx, nely, nelz, volfrac, penal, rmin)','cprofilestats3D.txt')
# p = pstats.Stats('cprofilestats3D.txt')
# p.sort_stats('tottime').print_stats(50)
