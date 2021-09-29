# METHOD TO CREATE THE 3D VOXEL PLOTS 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# EXAMPLE ARRAY TO PLOT
# a = np.load('xPrintNoAM.npy') # load the example output from top3D
# aplot = a.transpose(2,0,1)      # transpose from top3D or top3DAM 3D numpy array to plot in the intuitive manner
# aplot = np.where(aplot > 0.7, aplot, 0)   # Filter to prevent elements of density less than the threshold from plotting 
                                            # (PURELY FOR THE PURPOSE OF VISUALISATION)

a = np.array([[[0.1,0.2,0.3,0.4],[0.4,0.5,0.6,0.7],[0.7,0.3,0.2,1]],[[0.7,0.8,0.9,1],[1,0.1, 0,0.3],[0.4,0.3,0.2,0.1]]])
aplot = a.transpose(2,1,0)      # transpose from standard 3D numpy array to plot in the intuitive manner
                                        # z - height - 0th at the bottom, going upwards (e.g. a[n,:,:])
                                        # y - depth (into the page) - 0th at the front (e.g. a[:,n,:])
                                        # x - width - 0th at the left (e.g. a[:,:,n])


# Plotting with colour mapping
fig = plt.figure()
ax = fig.gca(projection='3d')
cmap = plt.get_cmap("Greys")
norm= plt.Normalize(aplot.min(), aplot.max())
# ax.voxels(np.ones_like(a), facecolors=cmap(norm(a)), edgecolor="black")   # plotting all elements including the 0 ones [Not desirable for us]
ax.voxels(aplot, facecolors=cmap(norm(aplot)))                              # not plotting the element if the value is 0
ax.set_box_aspect(aplot.shape)                                              # Plot in uniform aspect ratio (shape convention x,y,z)

plt.show()