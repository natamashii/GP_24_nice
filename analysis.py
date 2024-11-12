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
import pandas as pd
import Toolbox as tt

# pre allocation
cell_pos = []
phase_data = {}
data_elevations_list_big = []
tp_windows_big = []
indices_windows_big = []
data_elevations_list_small = []
tp_windows_small = []
indices_windows_small = []

# hardcoded stuff
frame_rate = 2.18  # in Hz
des_wind = 5    # window size for sliding window median (DFF), in min
tau = 1.6
ds = [5, 30]    # dot sizes used in recordings
dot_winds = ["left", "right", "back", "front"]  # locations of moving dot stimulus

# offset_angle = elevation
left_elevations_big = np.arange(15, -16, -15)
front_elevation_big = np.arange(45, 14, -15)
right_elevation_big = np.arange(15, -16, -15)
back_elevation_big = np.arange(-15, -46, -15)

left_elevations_small = np.arange(15, -16, -5)
front_elevation_small = np.arange(45, 14, -5)
right_elevation_small = np.arange(15, -16, -5)
back_elevation_small = np.arange(-15, -46, -5)

# define path: rn only one recording, later then more general
# in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
# to access other working directories: os.chdir("")
# data_path for at home
data_path = "E:\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"

# data_path for at lab
#data_path = "Z:\\shared\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"

# data_path for jupyter

# get vxpy stuff
display = tt.load_hdf5(data_path + "Display.hdf5", name="phase")
io = h5py.File(data_path + "Io.hdf5")

# get suite2p stuff
F = np.load(data_path + "suite2p\\plane0\\F.npy")  # intensity trace for each detected cell
ops = np.load(data_path + "suite2p\\plane0\\ops.npy", allow_pickle=True).item()
stat = np.load(data_path + "suite2p\\plane0\\stat.npy", allow_pickle=True)
# data for when run in jupyter lal
# F = np.load(data_path + "suite2p/F.npy")  # intensity trace for each detected cell
# ops = np.load(data_path + "suite2p/ops.npy", allow_pickle=True).item()
# stat = np.load(data_path + "suite2p/stat.npy", allow_pickle=True)

print("Done")

# %% Calculate DFF
# smooth traces with average in sliding window
smooth_f = tt.avg_smooth(data=F, window=3)

# calculate DFF with median in sliding window as F0
dff = tt.calc_dff_wind(F=smooth_f, window=des_wind, frame_rate=frame_rate)

print("Done")

# %%
# align frames between both PCs
frame_times = tt.adjust_frames(io=io, F=F)

# find phases of Moving Dot & corresponding break phases
valid_data = tt.extract_mov_dot(display)

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

elevations_list_big = [left_elevations_big, front_elevation_big, right_elevation_big,
                       back_elevation_big]
elevations_list_small = [left_elevations_small, front_elevation_small,
                         right_elevation_small, back_elevation_small]
data_windows_list_big = [dot_left_30, dot_front_30, dot_right_30,
                         dot_back_30]
data_windows_list_small = [dot_left_5, dot_front_5, dot_right_5,
                           dot_back_5]

# split into stuff: 30 dot
# iterate over windows of dot presentation
for j in range(len(elevations_list_big)):
    # pre allocation
    data_elevations_window = []
    tp_elevations_window_big = []
    indices_elevations_window_big = []
    # iterate over elevation levels used in this stimulus protocol: absolute position of dot elevation-wise
    for elevation in elevations_list_big[j]:
        # pre allocation
        valid_phases = []
        # iterate over all phases for this dot size
        for curr_phase, i in zip(data_windows_list_big[j].keys(),
                                 range(len(data_windows_list_big[j]))):
            # if current phase meets current absolute elevation position
            if data_windows_list_big[j][curr_phase]['dot_offset_angle'] == elevation:
                valid_phases.append(curr_phase)
        # pre allocation
        data_elevation = {}
        tp_elevation_big = []
        indices_elevation_big = []
        # iterate over currently relevant phases
        for i in valid_phases:
            # add keys and the data behind to the new dictionary
            data_elevation[i] = data_windows_list_big[j][i]
            st_elevation = data_windows_list_big[j][i]["__start_time"]
            tp_switch_big = data_windows_list_big[j][i]["__start_time"] + data_windows_list_big[j][i]["t_switch"]
            et_elevation = st_elevation + data_windows_list_big[j][i]["__target_duration"]
            # find corresponding start frame in calcium trace
            start_index = min(enumerate(frame_times), key=lambda x: abs(x[1] - st_elevation))[0]
            # find corresponding direction switch frame in calcium trace
            switch_index = min(enumerate(frame_times), key=lambda x: abs(x[1] - tp_switch_big))[0]
            # find corresponding end frame in calcium trace
            end_index = min(enumerate(frame_times), key=lambda x: abs(x[1] - et_elevation))[0]

            tp_elevation_big.append(st_elevation)
            tp_elevation_big.append(tp_switch_big[0])
            tp_elevation_big.append(et_elevation)

            indices_elevation_big.append(start_index)
            indices_elevation_big.append(switch_index)
            indices_elevation_big.append(end_index)

        data_elevations_window.append(data_elevation)
        tp_elevations_window_big.append(tp_elevation_big)
        indices_elevations_window_big.append(indices_elevation_big)
    data_elevations_list_big.append(data_elevations_window)
    tp_windows_big.append(tp_elevations_window_big)
    indices_windows_big.append(indices_elevations_window_big)

# split stuff: 5 dot
# iterate over windows of dot presentation
for k in range(len(elevations_list_small)):
    # pre allocation
    data_elevations_window = []
    tp_elevations_window_small = []
    indices_elevations_window_small = []
    # iterate over elevation levels used in this stimulus protocol: absolute position of dot elevation-wise
    for elevation in elevations_list_small[k]:
        # pre allocation
        valid_phases = []
        # iterate over all phases for this dot size
        for curr_phase, i in zip(data_windows_list_small[k].keys(),
                                 range(len(data_windows_list_small[k]))):
            # if current phase meets current absolute elevation position
            if data_windows_list_small[k][curr_phase]['dot_offset_angle'] == elevation:
                valid_phases.append(curr_phase)
        # pre allocation
        data_elevation = {}
        tp_elevation_small = []
        indices_elevation_small = []
        # iterate over currently relevant phases
        for i in valid_phases:
            # add keys and the data behind to the new dictionary
            data_elevation[i] = data_windows_list_small[k][i]
            st_elevation = data_windows_list_small[k][i]["__start_time"]
            et_elevation = st_elevation + data_windows_list_small[k][i]["__target_duration"]
            tp_switch = data_windows_list_small[k][i]["__start_time"] + data_windows_list_small[k][i]["t_switch"]
            # find corresponding start frame in calcium trace
            start_index = min(enumerate(frame_times), key=lambda x: abs(x[1] - st_elevation))[0]
            # find corresponding direction switch frame in calcium trace
            switch_index = min(enumerate(frame_times), key=lambda x: abs(x[1] - tp_switch))[0]
            # find corresponding end frame in calcium trace
            end_index = min(enumerate(frame_times), key=lambda x: abs(x[1] - et_elevation))[0]

            tp_elevation_small.append(st_elevation)
            tp_elevation_small.append(tp_switch[0])
            tp_elevation_small.append(et_elevation)

            indices_elevation_small.append(start_index)
            indices_elevation_small.append(switch_index)
            indices_elevation_small.append(end_index)

        data_elevations_window.append(data_elevation)
        tp_elevations_window_small.append(tp_elevation_small)
        indices_elevations_window_small.append(indices_elevation_small)
    data_elevations_list_small.append(data_elevations_window)
    tp_windows_small.append(tp_elevations_window_small)
    indices_windows_small.append(indices_elevations_window_small)

data_elevations_list = [data_elevations_list_big, data_elevations_list_small]
tp_windows = [tp_windows_big, tp_windows_small]
indices_windows = [indices_windows_big, indices_windows_small]

print("Done")

# %% Regressor stuff
# first do regressor over all dot phases, then later maybe also on each direction within a dot phase
regressor_win_buffer = [1, 10]
all_regressors = []
all_regressors_conv = []
all_regressors_phase = []
all_regressors_phase_stp = []
all_regressors_phase_etp = []


# BUILD the regressor
# iterate over dot sizes
for idx_ds in range(len(data_elevations_list)):
    # iterate over stimulus window
    for idx_wind in range(len(data_elevations_list[idx_ds])):
        # iterate over elevation level
        for idx_el in range(len(data_elevations_list[idx_ds][idx_wind])):
            current_cond = data_elevations_list[idx_ds][idx_wind][idx_el]
            phase_names = list(current_cond.keys())
            start_ind = indices_windows[idx_ds][idx_wind][idx_el][0:-1:3]
            switch_ind = indices_windows[idx_ds][idx_wind][idx_el][1:-1:3]
            end_ind = indices_windows[idx_ds][idx_wind][idx_el][2:-1:3]
            end_ind.append(indices_windows[idx_ds][idx_wind][idx_el][-1])
            # build the expressor
            regressor_trace = np.zeros((np.shape(F)[1]))
            for idx_rep in range(len(data_elevations_list[idx_ds][idx_wind][idx_el])):
                regressor_trace[start_ind[idx_rep]:end_ind[idx_rep]] = 1
            all_regressors.append(regressor_trace)
            all_regressors_phase.append(phase_names)
            all_regressors_phase_stp.append(start_ind)
            all_regressors_phase_etp.append(end_ind)
            # Convolution: Build regressor at relevant time points of current stimulus version (these are nonzero)
            regressor_trace_conv = tt.CIRF(regressor=regressor_trace, n_ca_frames=len(frame_times), tau=tau)
            all_regressors_conv.append(regressor_trace_conv)

print("Done")

# %% Correlation: Find Correlation of cells to Moving Dot
# pre allocation
corr_array = np.zeros((np.shape(dff)[0], len(all_regressors_conv)))

# iterate over all cells
for cell, trace in enumerate(dff):
    # iterate over all conditions
    for cond, reg_trace in enumerate(all_regressors_conv):
        current_phases = all_regressors_phase[cond]
        # find corresponding indices in dff
        ultimate_start = np.min(all_regressors_phase_stp[cond])
        ultimate_end = np.max(all_regressors_phase_etp[cond])
        corr_array[cell, cond] = np.corrcoef(trace[ultimate_start:ultimate_end+1], reg_trace[ultimate_start:ultimate_end+1])[0, 1]

# select only good cells
gc_ind = np.where(corr_array > .3)
good_cells, good_cells_idx = np.unique(gc_ind[0], return_index=True)
gc_phase = gc_ind[1][good_cells_idx]

#get dff-traces containing only good cells
dff_good = dff[good_cells, :]

print("Done")

# %% Autocorrelation: Yeet cells that do not react to Moving Dot
# kick out badly autocorrelated cells
indices_all = [indices_windows_big, indices_windows_small]

cell_size = np.shape(dff)[0]
ds_size = len(indices_all)
window_size = len(indices_windows_small)



# Define avg_corr_coefs_rec with the calculated dimensions
avg_corr_coefs_rec = np.zeros((cell_size, window_size * 3 + window_size * 7))


for cell in range(cell_size):
    for ds in range(ds_size):
        for window in range(window_size):
            elevation_size = len(indices_windows_small[window])
            for elevation in range(elevation_size):
                #get subsets for the repetitions
                subset1 = dff[cell, indices_windows_small[window][elevation][0]-1 : indices_windows_small[window][elevation][2]+10]
                subset2 = dff[cell, indices_windows_small[window][elevation][3]-1 : indices_windows_small[window][elevation][5]+10]
                subset3 = dff[cell, indices_windows_small[window][elevation][6]-1 : indices_windows_small[window][elevation][8]+10]
                # Determine the maximum length of the arrays
                max_length = max(len(subset1), len(subset2), len(subset3))
                # Pad each array with NaNs to match the maximum length
                if max_length - len(subset1) != 0:
                    subset1 = np.append(subset1, [np.nan] * (max_length - len(subset1)))
                if max_length - len(subset2) != 0:
                    subset2 = np.append(subset2, [np.nan] * (max_length - len(subset2)))
                if max_length - len(subset3) != 0:
                    subset3 = np.append(subset3, [np.nan] * (max_length - len(subset3)))
                #create dataframe of the subsets
                df = pd.DataFrame({'Array1': subset1,
                                  'Array2': subset2,
                                  'Array3': subset3})
                #calculate the pearson correlation scores
                corr_coef = df.corr()
                #get the needed correlation scores
                corr_coefs_phase = (corr_coef.iloc[1, 0], corr_coef.iloc[2, 0], corr_coef.iloc[2, 1])
                #transform them into an array to use np.nanmean
                array_corr_coefs = np.array(corr_coefs_phase, dtype = np.float64)
                #average correlation scores of repetitions
                avg_corr_coef = np.nanmean(corr_coefs_phase)
                if ds == 0:
                    # Handling for indices_windows_big
                    index = window * 3 + elevation
                elif ds == 1:
                    # Handling for indices_windows_small
                    index = window_size * 3 + window * 7 + elevation
                avg_corr_coefs_rec[cell, index] = avg_corr_coef
# %% check if it worked

indices = np.where(avg_corr_coefs_rec > 0.4)
good_cells = np.unique(indices[0])
len(good_cells)

# %%
np.random.seed(1)
rdm_good_cells = np.random.choice(good_cells, size = 100, replace = False)
fig, axs = plt.subplots(nrows=100, ncols=1, figsize=(15, 150), constrained_layout = True)
for i in range(len(rdm_good_cells)):
    axs[i].plot(dff_good[i])
# look at final product with pixel plot


# %%

