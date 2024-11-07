# importing
import numpy as np
import matplotlib.pyplot as plt
import scipy as ss
import h5py
import matplotlib.colors as colors
import os
import Toolbox as tb

# pre allocation
cell_pos = []

# hardcoded stuff
frame_rate = 2.18      # in Hz

# define path: rn only one recording, later then more general
# in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
# to access other working directories: os.chdir("")
data_path = "D:\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"

# get vxpy stuff
#
#camera = tb.load_hdf5(data_path + "Camera.hdf5", name="")      # doesnt work, but in carina's code wasnt used
# camera command is still mentioned in carina's script so I didnt delete it yet
display = tb.load_hdf5(data_path + "Display.hdf5", name="phase")
io = h5py.File(data_path + "Io.hdf5")

# get suite2p stuff
F = np.load(data_path + "suite2p\\plane0\\F.npy")
ops = np.load(data_path + "suite2p\\plane0\\ops.npy", allow_pickle=True).item()
stat = np.load(data_path + "suite2p\\plane0\\stat.npy", allow_pickle=True)

# extract cell positions
for i in range(len(stat)):
    cell_pos.append(tuple(stat[i]["med"]))      # med is list of position of cell
