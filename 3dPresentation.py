import pyvista as pv
import numpy as np

xPrint = np.load('xPrint_40_20_10.npy')  # load
xPrintNoAM = np.load('xPrintNoAM.npy') # load
# 3D PLOT
# convert numpy array to what pyvista wants
data = pv.wrap(xPrint)
dataNoAM = pv.wrap(xPrintNoAM)
# data.set_active_scalars("Spatial Cell Data")
# create plot
p = pv.Plotter()
p.add_volume(dataNoAM, cmap="turbo")
# p.add_volume(data, cmap="turbo", opacity="sigmoid_5")
# p.add_mesh(data,cmap="viridis",show_edges=True)
# data.plot(opacity='linear')
p.show()