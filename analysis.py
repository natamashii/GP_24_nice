__author__ = "Natalie_Fischer"
# importing
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import scipy as ss
import h5py
import matplotlib.colors as colors
import os
import Toolbox_2p0 as tt
import analyse_moving_dot_gp24 as tb


#%%
# pre allocation
cell_pos = []

# hardcoded stuff
frame_rate = 2.18  # in Hz
des_wind = 5    # window size for sliding window median (DFF), in min
tau = 1.6
ds = [5, 30]    # dot sizes used in recordings
all_dot_sizes = np.array([[30, 5], [-15, -5]], dtype="int32")
all_windows = np.array([[-180, -90, 0, 90],
                        [15, 45, 15, -15], [-15, 15, -15, -45]], dtype="int32")
win_buffer = [-1, 10]
conds_dotsizes = ["big", "small"]
conds_windows = ["left" "front", "right", "back"]
conds_elevations = [["1, 2, 3"], ["1", "2", "3", "4", "5", "6", "7"]]

# define path: rn only one recording, later then more general
# in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
# to access other working directories: os.chdir("")
# data_path for at home
data_path = "C:/Users/samue/Master/3_Semester/GP_Ari/GP24_fish1_rec1_05112024/"

# data_path for at lab
# data_path = "Z:\\shared\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"

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

# %% Calculate DFF
# smooth traces with average in sliding window
smooth_f = tt.avg_smooth(data=F, window=3)

# calculate DFF with median in sliding window as F0
dff = tt.calc_dff_wind(F=smooth_f, window=des_wind, frame_rate=frame_rate)

# %% Split Data into stimulus conditions
# align frames between both PCs
frame_times = tt.adjust_frames(io=io, F=F)

# find phases of Moving Dot & corresponding break phases
valid_data = tt.extract_mov_dot(display)

time_points, phase_names, indices, break_phases = (
    tt.extract_version2(valid_data=valid_data, all_dot_sizes=all_dot_sizes,
                        all_windows=all_windows, frame_times=frame_times))

# dimension 0: dot size: 0 = 30 ; 1 = 5
# dimension 1: dot window: left, front, right, back
# dimension 2: dot elevation level
# dimension 3: number of repetition
# dimension 4: start, switch, end

# %% Build Regressor

all_regressors, all_regressors_conv, all_regressors_phase_stp, all_regressors_phase_etp =\
    tt.build_regressor(indices=indices, dff=dff, frame_times=frame_times, tau=tau)

# %% Correlation: Find Correlation of cells to Moving Dot Phases
# pre allocation
corr_array = tt.corr(dff=dff, regressors=all_regressors_conv, regressor_phase_stp=all_regressors_phase_stp, regressor_phase_etp=all_regressors_phase_etp)

# select only good cells
good_cells, good_cells_idx = np.unique(np.where(corr_array > .3)[0], return_index=True)

# %% Autocorrelation: Yeet Cells that fluctuate in their responses to stimulus repetitions

auto_corrs = tt.autocorr(dff=dff, indices=indices, win_buffer=win_buffer, regressors=all_regressors_conv)

really_good_cells, really_good_cells_idx = np.unique(np.where(auto_corrs > .4)[0], return_index=True)

# find intersection between correlation result and autocorrelation result
best_cells = tt.compare(corr_cells=good_cells, autocorr_cells=really_good_cells)


# %% sort cells

chosen_cells = dff[best_cells.astype("int64"), :]

# Carina's suggestion: split time into stimulus condition, take average value over all repeated phases & sort this

avg_values = np.zeros((np.shape(indices)[0]))
split_cells = []
split_counter = 0
for_sorting = np.zeros((np.shape(chosen_cells)[0], 3, 40))

# iterate over dot sizes
for ds in range(np.shape(indices)[0]):
    # iterate over windows
    for wind in range(np.shape(indices)[1]):
        # iterate over elevations
        for el in range(np.shape(indices)[2]):
            if not np.isnan(indices[ds, wind, el, :, :]).any():
                # iterate over repetitions
                for rep in range(np.shape(indices)[3]):
                    cond_start = np.nanmin(indices[ds, wind, el, rep, 0]).astype("int64")
                    cond_end = np.nanmax(indices[ds, wind, el, rep, 2]).astype("int64")
                    split_cells.append(chosen_cells[:, cond_start:cond_end])
                    # get average of cell's dff in this phase
                    for cell, trace in enumerate(chosen_cells):
                        avg = np.nanmean(trace[cond_start:cond_end])
                        # average dff across repetition
                        for_sorting[cell, rep, split_counter] = avg
                split_counter += 1

# peak of average dff over condition: 40 values per cell
for_sorting_el = np.nanmax(for_sorting, axis=1)
# arg peak of peak mean dff: indices of condition the cell correlates best to
sorting = np.nanargmax(for_sorting_el, axis=1)


t_min = np.nanmin(indices).astype("int64")
t_max = np.nanmax(indices).astype("int64")

sorted_cells = list([] * len(all_regressors))
amount_cells = np.zeros((len(all_regressors)))
idx_cells = list([] * len(all_regressors))

# iterate over all possible conditions of the stimulus
for cond in range(len(all_regressors)):
    # find how many cells were sorted to this condition
    amount_cells[cond] = np.shape(np.where(sorting == cond))[1]
    # get indices in sorting of cells sorted to this condition
    ind_cells = np.where(sorting == cond)[0].astype("int64")
    idx_cells.append(ind_cells)
    # only continue if there are cells sorted to this condition
    if amount_cells[cond] > 0:
        # iterate over relevant cells
        for cell_idx in ind_cells:
            get_cell = chosen_cells[cell_idx, t_min:t_max]
            sorted_cells.append(get_cell)
sorted_cells = np.array(sorted_cells)
"""
# now split time axis and sort after this
# this should set an order of numbers for conditions, after this the cells must be sorted to...
# order: big dot left, right, front, back ; small dot left, right, front, back
# maybe put them in dictionary...
sorted_all = {}
sort_counter = 0
# iterate over dot sizes
for ds_idx, ds in enumerate(conds_dotsizes):
    # iterate over windows
    for wind_idx, wind in enumerate(conds_windows):
        # iterate over elevation levels
        for el_idx, el in enumerate(conds_elevations[ds_idx]):
            phase_name = list(phase_names[ds_idx, wind_idx, el_idx, :])
            start_ind = list(time_points[ds_idx, wind_idx, el_idx, :, 0])
            end_ind = list(time_points[ds_idx, wind_idx, el_idx, :, 2])
            sorted_all[f"{ds}_{wind}_{el}"] = list(sort_counter, phase_name, start_ind, end_ind)

# COMBINE IT WITH UPPER NESTED FOR LOOP
"""

# %% Z-Score
tail_length = 5

# pre allocation
mean_std = np.zeros((2, np.shape(dff)[0]))
cell_breaks = []
z_score = np.zeros(np.shape(chosen_cells))
break_start = np.zeros((len(break_phases)))
break_end = np.zeros((len(break_phases)))


# iterate over all cells
for cell, trace in enumerate(chosen_cells):
    # get pauses of each cell
    current_cell_break = []
    # iterate over all break phases
    for b_phase in range(len(break_phases)):
        start_ind = break_phases[b_phase][2]
        break_start[b_phase] = start_ind
        end_ind = break_phases[b_phase][4]
        break_end[b_phase] = end_ind
        current_cell_break.extend(chosen_cells[cell, start_ind+tail_length:end_ind])
    # save all break dff frames for current cell
    cell_breaks.append(current_cell_break)
    # get mean for this cell
    cell_mean = np.mean(current_cell_break)
    mean_std[0, cell] = cell_mean
    # get std for this cell
    cell_std = np.std(current_cell_break)
    mean_std[1, cell] = cell_std

    # get z score for this cell
    for frame_idx, frame in enumerate(trace):
        z_score[cell, frame_idx] = (frame - cell_mean) / cell_std

# identify total begin and end of Moving Dot interval
t_min = np.nanmin([np.nanmin(indices), np.nanmin(break_start)]).astype("int64")
t_max = np.nanmax([np.nanmax(indices), np.nanmax(break_end)]).astype("int64")

z_scores_cells = z_score[:, t_min:t_max]

# %%
fig, axs = plt.subplots()
pixelplot = axs.imshow(z_scores_cells, cmap="PiYG")
#axs.set_xticks(np.linspace(time_points[np.where(indices == np.nanmin(indices))], time_points[np.where(indices == np.nanmax(indices))], np.shape(chosen_cells)[1]).flatten())
axs.set_xlabel("Time [s]")
axs.set_ylabel("Cells")
axs.set_title("Cells of Rec 1 Fish 1 Day 1, Only DFF, attempted sort")
fig.colorbar(pixelplot)

plt.show(block=False)

#%%
split_counter = 0
time_conds = np.full((len(all_regressors), 2, 3), np.nan, dtype="int64")
# iterate over dot sizes
for ds in range(np.shape(indices)[0]):
    # iterate over windows
    for wind in range(np.shape(indices)[1]):
        # iterate over elevations
        for el in range(np.shape(indices)[2]):
            if not np.isnan(indices[ds, wind, el, :, :]).any():
                for rep in range(np.shape(indices)[3]):
                    # find start & end
                    cond_start = indices[ds, wind, el, rep, 0].astype("int64")
                    time_conds[split_counter, 0, rep] = cond_start
                    cond_end = indices[ds, wind, el, rep, 2].astype("int64")
                    time_conds[split_counter, 1, rep] = cond_end

#%% get average dffover repetitions for each condition and cell
mean_dff_bcs, new_inds = tt.get_mean_dff_trace(chosen_cells, indices)

#%%get the AUCs
num_stim = len(all_regressors)

AUCs_cells = tb.get_AUCs(mean_dff_bcs, new_inds, num_stim)












