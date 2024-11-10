__author__ = "Natalie_Fischer"
# importing
import numpy as np
import matplotlib as mpl

from Preprocessing_test import regressor_thisCon

mpl.use("TkAgg")
import matplotlib.pyplot as plt
import scipy as ss
import h5py
import matplotlib.colors as colors
import os
import Toolbox as tt

# pre allocation
cell_pos = []
phase_data = {}

# hardcoded stuff
frame_rate = 2.18  # in Hz
des_wind = 5    # window size for sliding window median (DFF), in min
tau = 1.6
ds = [5, 30]    # dot sizes used in recordings
dot_winds = ["left", "right", "back", "front"]  # locations of moving dot stimulus

# define path: rn only one recording, later then more general
# in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
# to access other working directories: os.chdir("")
data_path = "E:\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"

# get vxpy stuff
display = tt.load_hdf5(data_path + "Display.hdf5", name="phase")
io = h5py.File(data_path + "Io.hdf5")

# get suite2p stuff
F = np.load(data_path + "suite2p\\plane0\\F.npy")  # intensity trace for each detected cell
ops = np.load(data_path + "suite2p\\plane0\\ops.npy", allow_pickle=True).item()
stat = np.load(data_path + "suite2p\\plane0\\stat.npy", allow_pickle=True)

# %% Calculate DFF
smooth_f = tt.avg_smooth(data=F, window=3)

# plot multiple cells
dff = tt.calc_dff_wind(F=smooth_f, window=des_wind, frame_rate=frame_rate)
fig, axs = plt.subplots(15, 1, sharex=True, sharey=True, constrained_layout=True)
for c in range(15):
    axs[c].plot(dff[c, :], color="magenta")
fig.suptitle(str(des_wind))
plt.show()
# %% Identify Phase Time Intervals

# align frames between both PCs
frame_times = tt.adjust_frames(io=io, F=F)

# find phases of Moving Dot & corresponding break phases
valid_data = tt.extract_mov_dot(display)

# get start + end times of all relevant phases
phase_names = list(valid_data.keys())

phase_data["visualname"] = []
phase_data["index"] = []
phase_data["start_time"] = []
phase_data["end_time"] = []
phase_data["dff_start_idx"] = []
phase_data["dff_end_idx"] = []

for idx, phase in enumerate(phase_names):
    phase_data["visualname"].append(valid_data[phase]["__visual_name"])
    phase_data["index"].append(idx)
    start = valid_data[phase]["__start_time"]
    phase_data["start_time"].append(start)
    duration = valid_data[phase]["__target_duration"]
    end = start + duration
    phase_data["end_time"].append(end)

    # identify corresponding indices in dff trace: start of phase
    diff_time = np.abs(frame_times - start)
    phase_data["dff_start_idx"].append(np.argmin(diff_time))
    # identify corresponding indices in dff trace: end of phase
    diff_time = np.abs(frame_times - end)
    phase_data["dff_end_idx"].append(np.argmin(diff_time))

dff_indices = np.array([phase_data["dff_start_idx"][:], phase_data["dff_end_idx"][:]])

# cut out dff trace with relevant phases
dff_interest = dff[:, np.min(dff_indices):np.max(dff_indices)]

# %% Regressor Part
print("to be continued")
# split dff trace in attributes (written by samu)

move_dot_5 = tt.extract_dot_ds(data_movdot=valid_data, dot_size=ds[0])
move_dot_30 = tt.extract_dot_ds(data_movdot=valid_data, dot_size=ds[1])

dot_left_5 = tt.extract_dot_window(data_dot_ds=move_dot_5, window=dot_winds[0])
dot_right_5 = tt.extract_dot_window(data_dot_ds=move_dot_5, window=dot_winds[1])
dot_back_5 = tt.extract_dot_window(data_dot_ds=move_dot_5, window=dot_winds[2])
dot_front_5 = tt.extract_dot_window(data_dot_ds=move_dot_5, window=dot_winds[3])

dot_left_30 = tt.extract_dot_window(data_dot_ds=move_dot_30, window=dot_winds[0])
dot_right_30 = tt.extract_dot_window(data_dot_ds=move_dot_30, window=dot_winds[1])
dot_back_30 = tt.extract_dot_window(data_dot_ds=move_dot_30, window=dot_winds[2])
dot_front_30 = tt.extract_dot_window(data_dot_ds=move_dot_30, window=dot_winds[3])

# %%
# TODO: ask carina to explain her code once more cuz im too dumb for this shit again...
# get amount of all phases now
amount_phases = len(move_dot_5) + len(move_dot_30)

regressor_win_buffer = [1, 10]
all_regressors = np.zeros([amount_phases, np.shape(F)[1]])
all_regressors_conv = np.zeros(np.shape(all_regressors))

for idx in phase_data["index"]:
    current_phase = phase_data["visualname"][idx]
    if current_phase == "SingleDotRotatingBackAndForth":
        start = phase_data["dff_start_idx"][idx]
        end = phase_data["dff_end_idx"][idx]
        all_regressors[idx, start:end] = 1
        all_regressors_conv[idx, start:end] = tt.CIRF(regressor=all_regressors[idx, start:end], n_ca_frames=len(frame_times), tau=tau)
    elif current_phase == "SphereUniformBackground":
        # breaks needed for later z score i guess and/or correlation


# extract cell positions
for i in range(len(stat)):
    cell_pos.append(tuple(stat[i]["med"]))  # med is list of position of cell


