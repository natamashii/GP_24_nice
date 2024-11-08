import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import h5py
import Toolbox as tb


# hardcoded stuff
frame_rate = 2.18      # in Hz

# define path: rn only one recording, later then more general
# in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
# to access other working directories: os.chdir("")
data_path = "C:/Users/samue/Master/3_Semester/GP_Ari/GP24_fish1_rec1_05112024/"

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

#get all moving dot phases and their breaks
data_movdot = tb.extract_mov_dot(display)


# get phases of big dots/small dots
#big
data_big_dot = tb.extract_dot_ds(data_movdot, 30)
#small
data_small_dot = tb.extract_dot_ds(data_movdot, 5)

#get the phases of the windows 
#big
left_window_big = tb.extract_dot_window(data_big_dot, "left")
front_window_big = tb.extract_dot_window(data_big_dot, "front")
right_window_big = tb.extract_dot_window(data_big_dot, "right")
back_window_big = tb.extract_dot_window(data_big_dot, "back")

#small
left_window_small = tb.extract_dot_window(data_small_dot, "left")
front_window_small = tb.extract_dot_window(data_small_dot, "front")
right_window_small = tb.extract_dot_window(data_small_dot, "right")
back_window_small = tb.extract_dot_window(data_small_dot, "back")

#offset_angle = elevation
