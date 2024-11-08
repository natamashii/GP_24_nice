__author__ = "Natalie_Fischer"
# importing
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import scipy as ss
import h5py
import matplotlib.colors as colors
import os
import Toolbox as tt

# pre allocation
cell_pos = []

# hardcoded stuff
frame_rate = 2.18  # in Hz

# define path: rn only one recording, later then more general
# in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
# to access other working directories: os.chdir("")
data_path = "E:\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"

# get vxpy stuff
#
# camera = tb.load_hdf5(data_path + "Camera.hdf5", name="")      # doesnt work, but in carina's code wasnt used
# camera command is still mentioned in carina's script so I didnt delete it yet
display = tt.load_hdf5(data_path + "Display.hdf5", name="phase")
io = h5py.File(data_path + "Io.hdf5")

# get suite2p stuff
F = np.load(data_path + "suite2p\\plane0\\F.npy")  # intensity trace for each detected cell
ops = np.load(data_path + "suite2p\\plane0\\ops.npy", allow_pickle=True).item()
stat = np.load(data_path + "suite2p\\plane0\\stat.npy", allow_pickle=True)

# extract cell positions
for i in range(len(stat)):
    cell_pos.append(tuple(stat[i]["med"]))  # med is list of position of cell


# extract time points of calcium trace frames (left PC) within behaviour trace frames (right PC)
# box function + corresponding times: box function is recorded by right PC when left PC is doing a frame (switch to True)
# identify indices where left PC scan a frame
box_trace = np.diff(io["di_frame_sync"][:].flatten())
frametimes = io["di_frame_sync_time"][:-1][box_trace].flatten()  #identify time points of frame of left PC


########################### DFF ###########################
# raw case
dff_1 = tt.calc_dff(F=F)

fig1, axs1 = plt.subplots(4, 1, sharex=True, sharey=True)

# window of 3 min
des_min = 3
wind = int(des_min * 60 * frame_rate) + 1
dff_wind_1 = tt.calc_dff_wind(F=F, window=wind)
# plot it first cell's dff trace
axs1[0].plot(dff_wind_1[0, :], color="magenta")
axs1[0].set_title(str(des_min))

# window of 4 min
des_min = 4
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=F, window=wind)
# plot it first cell's dff trace
axs1[1].plot(dff_wind_1[0, :], color="magenta")
axs1[1].set_title(str(des_min))

# window of 5 min
des_min = 5
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=F, window=wind)
# plot it first cell's dff trace
axs1[2].plot(dff_wind_1[0, :], color="magenta")
axs1[2].set_title(str(des_min))

# window of 6 min
des_min = 6
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=F, window=wind)
# plot it first cell's dff trace
axs1[3].plot(dff_wind_1[0, :], color="magenta")
axs1[3].set_title(str(des_min))

fig1.suptitle("Raw Case")
# log case

log_f = np.log(F)
fig2, axs2 = plt.subplots(4, 1, sharex=True, sharey=True)

# window of 3 min
des_min = 3
wind = int(des_min * 60 * frame_rate) + 1
dff_wind_1 = tt.calc_dff_wind(F=log_f, window=wind)
# plot it first cell's dff trace
axs2[0].plot(dff_wind_1[0, :], color="magenta")
axs2[0].set_title(str(des_min))

# window of 4 min
des_min = 4
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=log_f, window=wind)
# plot it first cell's dff trace
axs2[1].plot(dff_wind_1[0, :], color="magenta")
axs2[1].set_title(str(des_min))

# window of 5 min
des_min = 5
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=log_f, window=wind)
# plot it first cell's dff trace
axs2[2].plot(dff_wind_1[0, :], color="magenta")
axs2[2].set_title(str(des_min))

# window of 6 min
des_min = 6
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=log_f, window=wind)
# plot it first cell's dff trace
axs2[3].plot(dff_wind_1[0, :], color="magenta")
axs2[3].set_title(str(des_min))

fig2.suptitle("Log Case")
# Smooth Case

smooth_f = tt.avg_smooth(data=F, window=3)
fig3, axs3 = plt.subplots(4, 1, sharex=True, sharey=True)

# window of 3 min
des_min = 3
wind = int(des_min * 60 * frame_rate) + 1
dff_wind_1 = tt.calc_dff_wind(F=smooth_f, window=wind)
# plot it first cell's dff trace
axs3[0].plot(dff_wind_1[0, :], color="magenta")
axs3[0].set_title(str(des_min))

# window of 4 min
des_min = 4
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=smooth_f, window=wind)
# plot it first cell's dff trace
axs3[1].plot(dff_wind_1[0, :], color="magenta")
axs3[1].set_title(str(des_min))

# window of 5 min
des_min = 5
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=smooth_f, window=wind)
# plot it first cell's dff trace
axs3[2].plot(dff_wind_1[0, :], color="magenta")
axs3[2].set_title(str(des_min))

# window of 6 min
des_min = 6
wind = int(des_min * 60 * frame_rate)
dff_wind_1 = tt.calc_dff_wind(F=smooth_f, window=wind)
# plot it first cell's dff trace
axs3[3].plot(dff_wind_1[0, :], color="magenta")
axs3[3].set_title(str(des_min))

fig3.suptitle("Smooth Case")
plt.axis("off")
plt.show()

#%%

# naturally, communication between PCs is not flawless...
# last frame scanned by left PC will not be finished, therefore num of frames defined by both PCs must be compared
if not len(frametimes) == np.shape(F)[1]:
    # since last frame never finished by left PC, must be discarded
    if np.shape(F)[1] - len(frametimes) == -1:
        frametimes = frametimes[:-1]
    # in case of imperfect communication...
    elif np.shape(F)[1] >= len(frametimes):
        frametimes = np.linspace(frametimes[0], frametimes[-1], np.shape(F)[1] + 1)
        frametimes = frametimes[:-1]
    # wtf
    else:
        print("Error, CHECK DATA")

# for visualizing this shit
# line is the behaviour recording of right PC
# dots are were left PC started to scan a frame
plt.figure()
plt.plot(io["di_frame_sync_time"][:], io["di_frame_sync"][:], color="green")
plt.scatter(frametimes, np.zeros(np.shape(frametimes)), color="magenta")
plt.show()

# extract relevant phases
valid_data = tt.extract_mov_dot(display)
imp_times = []
break_times = []
phase_times = []
phase_names = list(valid_data.keys())
for phase in range(len(valid_data)):
    imp_times[phase] = valid_data[phase_names[phase]]["__start_time"]
    visualname = valid_data[phase_names[phase]]["__visual_name"]
    # if phase was a break
    if visualname == "SphereUniformBackground":
        # sort start points and end points
        # endpoint: startpoint + duration
        start = valid_data[phase_names[phase]]["__start_time"]
        duration = valid_data[phase_names[phase]]["__target_duration"]
        end = start + duration
        # identify index within F trace: min difference between start time and time point in frametimes
        # (cuz it never aligns perfectly)

        break_times.append((start, end))
    # if phase was moving dot
    elif visualname == "SingleDotRotatingBackAndForth":
        # sort start points and end points
        # endpoint: startpoint + duration
        start = valid_data[phase_names[phase]]["__start_time"]
        duration = valid_data[phase_names[phase]]["__target_duration"]
        end = start + duration
        phase_times.append((start, end))

# get start times & end times of phases
# then identify indices in calcium trace
# same for breaks
# yeah