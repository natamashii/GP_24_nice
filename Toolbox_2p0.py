# importing
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import numpy as np
import scipy as ss
import pandas as pd
import math
from colorama import Fore
import h5py

# import seaborn as sns
# import skimage
# import statsmodels

# function for saving data as HDF5 (stolen from David)
def save_hdf5(list_of_data_arrays, list_of_labels, new_directory: str, export_path: str, permission='a'):
    print('save data to ' + export_path)
    permission = permission
    with h5py.File(export_path, permission) as f:
        gr = f.create_group(new_directory)
        for i in range(len(list_of_data_arrays)):
            gr[list_of_labels[i]] = list_of_data_arrays[i]


# function to load an HDF5 file (stolen from David)
def load_hdf5(import_path: str, name: str):
    print("load from HDF5...")
    data_dict = {}
    f = h5py.File(import_path, "r")
    names = list(f.keys())
    names_idx = [i for i, s in enumerate(f.keys()) if name in s]
    for g_idx in names_idx:
        group = names[g_idx]
        current_group = f[group]
        # get current group's attribute names
        cg_names = list(current_group.attrs.keys())
        # get current group's attribute values
        cg_values = list(current_group.attrs.values())
        data_dict[group] = {}
        for subgrp in current_group.keys():
            data_dict[group][subgrp] = f[group][subgrp][()]
        for att in range(len(cg_names)):
            data_dict[group][cg_names[att]] = cg_values[att]
    f.close()
    return data_dict

# function to align time traces bc communication between PCs isn't flawless...
def adjust_frames(io, F):
    # io["di_frame_sync"] is box function within behaviour recording (right PC)
    # switch to 1 = left PC started a frame scan
    # switch to 0 = left PC ended the frame scan
    box_trace = np.diff(io["di_frame_sync"][:].flatten())   # indices of switches
    frame_times = io["di_frame_sync_time"][:-1][box_trace].flatten()    # time points of switches
    calcium_frames = np.shape(F)[1]     # amount of frames of calcium trace
    # control for errors by comparing sizes of calcium trace frames & detected switches
    if not len(frame_times) == calcium_frames:
        # last started frame scan of left PC will never be finished, therefore must be discarded
        if len(frame_times) - calcium_frames == 1:
            frame_times = frame_times[:-1]
        # case of miscommunication of PCs: less detected switches by right PC than performed by left PC
        elif calcium_frames >= len(frame_times):
            # inform abt miscommunication
            print("WARNING: MISCOMMUNICATION BETWEEN PC OCCURRED! LESS SCAN SWITCHES DETECTED BY RIGHT PC THAN PERFORMED BY LEFT PC!")
            # "interpolate" frame_times
            frame_times = np.linspace(frame_times[0], frame_times[-1], calcium_frames + 1)
            # exclude last detected switch cuz never completed
            frame_times = frame_times[:-1]
        # case of total chaotic miscommunication
        else:
            print("ERROR!! TOTAL MISCOMMUNICATION BETWEEN PC OCCURRED! CHECK TRACES INDIVIDUALLY!")
            fig, axs = plt.subplots()
            axs.plot(io["di_frame_sync_time"][:], io["di_frame_sync"][:], color="green", label="Box Trace, io['di_frame_sync_time']")
            axs.scatter(frame_times, np.zeros(np.shape(frame_times)), color="magenta", label="Detected Switches by Right PC")
            plt.show(block=False)
    return frame_times

def extract_mov_dot(datatable):
    """
    extracts all phases and their data containing a visual name 'SingleDotRotatingBackAndForth'
    and creates a new dictionary only containing these

    Parameters
    ----------
    datatable : dict
        display file after loading using natalies load_hdf5 function

    Returns
    -------
    valid_data : dict
        contains all phases with moving dot trials

    """
    print("extracting moving dot phases")
    # get phases of interest (remove all lines that are interesting)
    valid_phases = np.ones(len(datatable))
    for i in range(len(datatable)):  # loop over all phases
        if datatable["phase" + str(i)]['__visual_name'] != 'SingleDotRotatingBackAndForth':
            valid_phases[i] = 0  # get the indices of all phases
    valid_indices = np.where(valid_phases == True)[0]  # get the indices of the phases with moving dots

    valid_data = {}
    for i in range(min(valid_indices) - 1, max(valid_indices) + 1):  # loop over valid phases
        valid_data[f"phase{i}"] = datatable[f"phase{i}"]  # add keys and the data behind to the new dictionary
    return valid_data

# function to split Display.hdf5 data down to switching directions
def extract_version2(valid_data, all_dot_sizes, all_windows, frame_times):
    valid_phase_names = list(valid_data.keys())
    # pre allocation
    time_points = np.full((2, 4, 7, 3, 3), np.nan)
    phase_names = np.full((2, 4, 7, 3), np.nan, dtype=object)
    indices = np.full((2, 4, 7, 3, 3), np.nan)
    break_phases = []
    # iterate over valid_data
    for idx in range(len(valid_phase_names)):
        # get data from current phase
        phase = valid_data[valid_phase_names[idx]]
        current_phase_name = valid_phase_names[idx]
        start_time = np.min(phase["__time"])
        start_idx = convert_time_frame(frame_times=frame_times, time_point_variable=start_time)
        end_time = np.max(phase["__time"])
        end_idx = convert_time_frame(frame_times=frame_times, time_point_variable=end_time)
        # identify if Moving Dot Phase or Break Phase
        visual_name = phase["__visual_name"]
        # in case of Moving Dot Phase:
        if visual_name == "SingleDotRotatingBackAndForth":
            # get dot specific conditions
            dot_size = phase["dot_angular_diameter"]
            dot_window = phase["dot_start_angle"]
            dot_elevation = phase["dot_offset_angle"].astype("int32")
            dot_t_switch = start_time + phase["t_switch"]
            dot_t_switch_idx = convert_time_frame(frame_times=frame_times, time_point_variable=dot_t_switch)
            # set dot size dimension
            dim_0 = np.where(all_dot_sizes == dot_size)[1]
            # set dot window dimension
            dim_1 = np.where(all_windows == dot_window)[1]
            # set dot elevation dimension
            elevations = np.arange(all_windows[1, dim_1], all_windows[2, dim_1] - 1, all_dot_sizes[1, dim_0])
            dim_2 = np.where(elevations == dot_elevation)
            # find number of repetition
            phase_reps = time_points[dim_0, dim_1, dim_2, :, 0]
            dim_3 = np.sum(~np.isnan(phase_reps[:]))
            # put information into monster arrays
            phase_names[dim_0, dim_1, dim_2, dim_3] = current_phase_name
            time_points[dim_0, dim_1, dim_2, dim_3, 0] = start_time
            time_points[dim_0, dim_1, dim_2, dim_3, 1] = dot_t_switch
            time_points[dim_0, dim_1, dim_2, dim_3, 2] = end_time
            indices[dim_0, dim_1, dim_2, dim_3, 0] = int(start_idx)
            indices[dim_0, dim_1, dim_2, dim_3, 1] = int(dot_t_switch_idx)
            indices[dim_0, dim_1, dim_2, dim_3, 2] = int(end_idx)
        elif visual_name == "SphereUniformBackground":
            break_phases.append([current_phase_name, start_time, start_idx, end_time, end_idx])
    return time_points, phase_names, indices, break_phases

# function to get sorting indices for cells in their best response to stimuli
def get_sort_cells(indices, chosen_cells, num_conds, conds_dotsizes, conds_windows, conds_elevations):
    # pre allocation
    conditions = []
    for_sorting = np.zeros((np.shape(chosen_cells)[0], 3, num_conds))
    time_sorting = np.full((num_conds, 2, 3), np.nan)
    split_counter = 0
    # iterate over dot sizes
    for ds in range(np.shape(indices)[0]):
        # iterate over windows
        for wind in range(np.shape(indices)[1]):
            # iterate over elevations
            for el in range(np.shape(indices)[2]):
                # only continue if repetition exists
                if not np.isnan(indices[ds, wind, el, :, :]).any():
                    # iterate over repetitions
                    # set condition as string
                    conditions.append(f"{conds_dotsizes[ds]}_{conds_windows[wind]}_{conds_elevations[ds][el]}")
                    for rep in range(np.shape(indices)[3]):
                        # find start & end
                        cond_start = indices[ds, wind, el, rep, 0].astype("int64")
                        time_sorting[split_counter, 0, rep] = cond_start
                        cond_end = indices[ds, wind, el, rep, 2].astype("int64")
                        time_sorting[split_counter, 1, rep] = cond_end
                        # get average of cell's dff in this phase
                        for cell, trace in enumerate(chosen_cells):
                            avg = np.nanmean(trace[cond_start:cond_end])
                            # average dff across repetition
                            for_sorting[cell, rep, split_counter] = avg
                    split_counter += 1
    # peak of average dff over condition: 40 values per cell
    for_sorting_el = np.nanmax(for_sorting, axis=1)
    # arg peak of peak mean dff: indices of condition the cell correlates best to
    cell_sorting = np.nanargmax(for_sorting_el, axis=1)
    return cell_sorting, time_sorting, conditions

# function to sort cells in their best response to stimuli
def sort_cells(sorting, num_conds, chosen_cells):
    # pre allocation
    sorted_cells = list([] * num_conds)
    amount_cells = np.zeros((num_conds))
    sorted_cells_idx = list([] * num_conds)

    # iterate over all possible conditions of the stimulus
    for cond in range(num_conds):
        # find how many cells were sorted to this condition
        amount_cells[cond] = np.shape(np.where(sorting == cond))[1]
        # get indices in sorting of cells sorted to this condition
        ind_cells = np.where(sorting == cond)[0].astype("int64")
        sorted_cells_idx.append(ind_cells)
        # only continue if there are cells sorted to this condition
        if amount_cells[cond] > 0:
            # iterate over relevant cells
            for cell_idx in ind_cells:
                get_cell = chosen_cells[cell_idx, :]
                sorted_cells.append(get_cell)
    sorted_cells = np.array(sorted_cells)
    return sorted_cells, sorted_cells_idx

# function to sort time dimension into stimuli
def sort_times(sorted_cells, time_sorting):
    # pre allocation
    time_sorted_cells = np.zeros((np.shape(sorted_cells)[0], 1))
    time_counter = 0
    line_counter = np.zeros(np.shape(time_sorting)[0]).astype("int64")
    # iterate over conditions
    for cond in range(np.shape(time_sorting)[0]):
        # iterate over repetitions of this condition
        for rep in range(np.shape(time_sorting)[2]):
            start = int(time_sorting[cond, 0, rep])
            end = int(time_sorting[cond, 1, rep])
            # get amount of frames for this repetition
            interval = end - start
            # append to pre allocated array
            time_sorted_cells = np.concatenate((time_sorted_cells, sorted_cells[:, start:end]), axis=1)
            time_counter += interval
        line_counter[cond] = time_counter
    # exclude first col (placeholder col)
    time_sorted_cells = time_sorted_cells[:, 1:]
    return time_sorted_cells, line_counter

# function to convert time point into calcium frame indices
def convert_time_frame(frame_times, time_point_variable):
    return min(enumerate(frame_times), key=lambda x: abs(x[1] - time_point_variable))[0]

# function to get t_min & t_max of Dot Stimulus Interval
def t_min_max(indices):
    t_min = np.nanmin(indices).astype("int64")
    t_max = np.nanmax(indices).astype("int64")
    return t_min, t_max

# function to smooth data via sliding window
def avg_smooth(data, window):
    low_lim = int(window / 2)
    upp_lim = int(data.shape[1] + window / 2)
    # pre allocation
    smooth_data = np.zeros(np.shape(data))
    # add nan values before and after data trace
    nan_data = np.empty((data.shape[0], int(data.shape[1] + window)))
    nan_data[:] = np.nan
    nan_data[:, low_lim:upp_lim] = data
    print("smooth data")
    l = data.shape[0]
    __printProgressBar(0, l, prefix="progress:", suffix="complete", length=50)
    progress_counter = 1
    for cell, trace in enumerate(nan_data):
        for t in range(low_lim, upp_lim):
            smooth_data[cell, t - low_lim] = np.nanmean(trace[t - low_lim:t + low_lim])
        __printProgressBar(progress_counter, l, prefix="progress:", suffix="complete", length=50)
        progress_counter = progress_counter + 1
    return smooth_data

# function to calculate dff via sliding window approach
def calc_dff_wind(F, window, frame_rate=2.18):
    # control that window size is optimal
    window = int(window * 60 * frame_rate)  # unit to frames
    if not window % 2:
        window += 1
    upp_lim = int(F.shape[1] + window / 2)
    low_lim = int(window / 2)
    # pre allocation
    dff_wind = np.zeros(np.shape(F))
    nan_F = np.empty((F.shape[0], int(F.shape[1] + window)))
    nan_F[:] = np.nan
    nan_F[:, low_lim:upp_lim] = F
    print("calculate dff via sliding window approach")
    l = dff_wind.shape[0]
    __printProgressBar(0, l, prefix="progress:", suffix="complete", length=50)
    progress_counter = 1
    # iterate over each cell
    for cell, trace in enumerate(nan_F):
        # window approach for F0
        F0 = np.zeros((np.shape(nan_F)[1]))
        for t in range(low_lim, upp_lim):
            # get median within window
            F0[t] = np.nanmedian(trace[t - low_lim:t + low_lim])
            dff_wind[cell, t - low_lim] = (trace[t] - F0[t]) / F0[t]
        __printProgressBar(progress_counter, l, prefix="progress:", suffix="complete", length=50)
        progress_counter = progress_counter + 1
    return dff_wind

# function to calculate dff via global median
def calc_dff(F):
    print("Calculate dff")
    dff = np.zeros(np.shape(F))
    l = dff.shape[0]
    __printProgressBar(0, l, prefix="progress:", suffix="complete", length=50)
    progress_counter = 1
    for cell, trace in enumerate(F):
        F0 = np.median(trace)
        dff[cell, :] = [(x - F0) / F0 for x in trace]
        __printProgressBar(progress_counter, l, prefix="progress:", suffix="complete", length=50)
        progress_counter = progress_counter + 1
    return dff

# function to calculate dff via z-score
def calc_dff_zscore(F):
    print("Calculate dff with z-score")
    dff = np.zeros(np.shape(F))
    l = dff.shape[0]
    __printProgressBar(0, l, prefix="progress:", suffix="complete", length=50)
    progress_counter = 1
    for cell, trace in enumerate(F):
        dff[cell, :] = ss.stats.score(trace, axis=0)
        __printProgressBar(progress_counter, l, prefix="progress:", suffix="complete", length=50)
        progress_counter = progress_counter + 1
    return dff

def get_mean_dff_trace(chosen_cells, indices):
    print("Compute average dFF curve over all 3 repetitions")
    # pre allocation
    best_cells_avg = []
    # iterate over cells
    indices_mean_dff = np.full((len(indices), len(indices[0]), len(indices[0, 0]), 2), np.nan)
    counter = 0
    # iterate over all cells
    for cell in range(np.shape(chosen_cells)[0]):
        # iterate over dot sizes
        avg_dff_cell = []
        for d in range(len(indices)):
            # iterate over windows
            for window in range(len(indices[d])):
                # identify amount of elevation levels used for this stimulus
                for elevation in range(len(indices[d, window])):
                    if not np.isnan(indices[d, window, elevation]).any():
                        counter += 1
                        # identify phases of repetitions
                        rep1 = chosen_cells[cell, int(indices[d, window, elevation, 0, 0]):
                                         int(indices[d, window, elevation, 0, 2])]
                        rep2 = chosen_cells[cell, int(indices[d, window, elevation, 1, 0]):
                                         int(indices[d, window, elevation, 1, 2])]
                        rep3 = chosen_cells[cell, int(indices[d, window, elevation, 2, 0]):
                                         int(indices[d, window, elevation, 2, 2])]
                        rep_matrix = np.full((3, max([len(rep1), len(rep2), len(rep3)])), np.nan)
                        rep_matrix[0, 0:len(rep1)] = rep1
                        rep_matrix[1, 0:len(rep2)] = rep2
                        rep_matrix[2, 0:len(rep3)] = rep3
                        # calc average across all repetitions
                        avg_dff = np.nanmean(rep_matrix, axis=0)
                        #get new start-indices
                        indices_mean_dff[d, window, elevation, 0] = len(avg_dff_cell)
                        #extend average dff values
                        avg_dff_cell.extend(avg_dff)
                        #get new end-indices
                        indices_mean_dff[d, window, elevation, 1] = len(avg_dff_cell)
        best_cells_avg.append(avg_dff_cell)
    mean_dff_best_cells = np.array(best_cells_avg)
    return mean_dff_best_cells, indices_mean_dff

# function to get average dff of all repetitions for each condition & cell
def mean_dff(chosen_cells, indices):
    print("Get Mean DFF for each cell & condition")
    # pre allocation
    mean_best_cells = []
    mean_best_cells_ind = np.full((np.shape(indices)[0], np.shape(indices)[1], np.shape(indices)[2], 2), np.nan)
    save_idx = True

    # iterate over all cells
    for cell, trace in enumerate(chosen_cells):
        cell_mean = []
        # iterate over dot sizes
        for ds in range(np.shape(indices)[0]):
            # iterate over windows
            for wind in range(np.shape(indices)[1]):
                # iterate over elevations
                for el in range(np.shape(indices)[2]):
                    if not np.isnan(indices[ds, wind, el]).any():
                        # iterate over repetitions
                        max_len = [indices[ds, wind, el, rep, 2] - indices[ds, wind, el, rep, 0] for rep in
                                   range(np.shape(indices)[3])]
                        repetitions = np.full((np.shape(indices)[3], np.max(max_len).astype("int64")), np.nan)
                        for rep in range(np.shape(indices)[3]):
                            repetitions[rep, :int(max_len[rep])] = trace[int(indices[ds, wind, el, rep, 0]):int(
                                indices[ds, wind, el, rep, 2])]
                        mean_rep = np.nanmean(repetitions, axis=0)
                        cell_mean.extend(mean_rep)
                        # save indices
                        if save_idx:
                            mean_best_cells_ind[ds, wind, el, 0] = len(cell_mean) - len(mean_rep)
                            mean_best_cells_ind[ds, wind, el, 1] = len(cell_mean)
        mean_best_cells.append(cell_mean)
        save_idx = False
    mean_best_cells = np.array(mean_best_cells)
    return mean_best_cells, mean_best_cells_ind

# function of regressor curve (for convolution), Credits to Carina
def CIRF(regressor, n_ca_frames, tau=1.6):
    time = np.arange(0, n_ca_frames)
    exp = np.exp(-time / tau)
    reg_conv = np.convolve(regressor, v=exp)
    reg_conv = reg_conv[:n_ca_frames]
    return reg_conv

# function to build a regressor
def build_regressor(indices, dff, frame_times, tau):
    print("Build Regressor for each relevant phase")
    # pre allocation
    all_regressors = []
    all_regressors_conv = []
    all_regressors_phase = []
    all_regressors_phase_stp = []
    all_regressors_phase_etp = []
    # BUILD the regressor
    # iterate over dot sizes
    for idx_ds in range(np.shape(indices)[0]):
        # iterate over stimulus window
        for idx_wind in range(np.shape(indices)[1]):
            # iterate over elevation level
            for idx_el in range(np.shape(indices)[2]):
                if not np.isnan(indices[idx_ds, idx_wind, idx_el, :, :]).any():
                    start_ind = indices[idx_ds, idx_wind, idx_el, :, 0].astype("int64")
                    switch_ind = indices[idx_ds, idx_wind, idx_el, :, 1].astype("int64")
                    end_ind = indices[idx_ds, idx_wind, idx_el, :, 2].astype("int64")

                    # build the expressor
                    regressor_trace = np.zeros((np.shape(dff)[1]))
                    for idx_rep in range(np.shape(indices)[3]):
                        regressor_trace[int(start_ind[idx_rep]):int(end_ind[idx_rep])] = 1
                    all_regressors.append(regressor_trace)
                    all_regressors_phase_stp.append(start_ind)
                    all_regressors_phase_etp.append(end_ind)
                    # Convolution: Build regressor at relevant time points of current stimulus version (these are nonzero)
                    regressor_trace_conv = CIRF(regressor=regressor_trace, n_ca_frames=len(frame_times), tau=tau)
                    all_regressors_conv.append(regressor_trace_conv)
    return all_regressors, all_regressors_conv, all_regressors_phase_stp, all_regressors_phase_etp

# function to calculate correlation of dff with regressor trace
def corr(dff, regressors, regressor_phase_stp, regressor_phase_etp):
    print("Compute correlation of each cell with each stimulus condition: ")
    # pre allocation
    corr_array = np.zeros((np.shape(dff)[0], len(regressors)))
    l = dff.shape[0]
    __printProgressBar(0, l, prefix="progress:", suffix="complete", length=50)
    progress_counter = 1
    # iterate over all cells
    for cell, trace in enumerate(dff):
        # iterate over all conditions
        for cond, reg_trace in enumerate(regressors):
            ultimate_start = np.min(regressor_phase_stp[cond])
            ultimate_end = np.max(regressor_phase_etp[cond])
            corr_array[cell, cond] = np.corrcoef(trace[ultimate_start:ultimate_end+1], reg_trace[ultimate_start:ultimate_end+1])[0, 1]
        __printProgressBar(progress_counter, l, prefix="progress:", suffix="complete", length=50)
        progress_counter = progress_counter + 1
    return corr_array

# function to calculate autocorrelation of cells over repetitions
def autocorr(dff, indices, win_buffer, regressors):
    print("Compute autocorrelation of each cell over all repetitions of each stimulus condition: ")
    # pre allocation
    auto_corrs = np.zeros((np.shape(dff)[0], len(regressors)))
    l = dff.shape[0]
    __printProgressBar(0, l, prefix="progress:", suffix="complete", length=50)
    progress_counter = 1
    # iterate over cells
    for cell in range(np.shape(dff)[0]):
        # iterate over dot sizes
        for ds in range(len(indices)):
            # iterate over windows
            for window in range(len(indices[ds])):
                # identify amount of elevation levels used for this stimulus
                for elevation in range(len(indices[ds][window])):
                    if not np.isnan(indices[ds][window]).any():
                        # identify phases of repetitions
                        rep1 = dff[cell, int(indices[ds, window, elevation, 0, 0]) + win_buffer[0]:
                                         int(indices[ds, window, elevation, 0, 2]) + win_buffer[1]]
                        rep2 = dff[cell, int(indices[ds, window, elevation, 1, 0]) + win_buffer[0]:
                                         int(indices[ds, window, elevation, 1, 2]) + win_buffer[1]]
                        rep3 = dff[cell, int(indices[ds, window, elevation, 2, 0]) + win_buffer[0]:
                                         int(indices[ds, window, elevation, 2, 2]) + win_buffer[1]]

                        # identify maximum length of these
                        max_size = max((len(rep1), len(rep2), len(rep3)))
                        # adjust sizes to be equal (necessary for proper autocorrelation computation)
                        if max_size - len(rep1) > 0:
                            rep1 = np.append(rep1, [np.nan] * (max_size - len(rep1)))
                        if max_size - len(rep2) > 0:
                            rep2 = np.append(rep2, [np.nan] * (max_size - len(rep2)))
                        if max_size - len(rep3) > 0:
                            rep3 = np.append(rep3, [np.nan] * (max_size - len(rep3)))

                        # put repetitions into dataframe
                        dataframe = pd.DataFrame({"Repetition_1": rep1, "Repetition_2": rep2, "Repetition_3": rep3})
                        # calculate Pearson Correlation
                        reps_corr = np.array([dataframe.corr().iloc[1, 0], dataframe.corr().iloc[2, 0], dataframe.corr().iloc[2, 1]],
                                             dtype="float64")
                        # average take average of these scores
                        auto_corrcoef = np.nanmean(reps_corr)

                        # put into auto_corr array
                        if ds == 0:
                            # Handling for indices_windows_big
                            index = window * 3 + elevation
                        elif ds == 1:
                            # Handling for indices_windows_small
                            index = len(indices[0]) * 3 + window * 7 + elevation
                        auto_corrs[cell, index] = auto_corrcoef
        __printProgressBar(progress_counter, l, prefix="progress:", suffix="complete", length=50)
        progress_counter = progress_counter + 1
    return auto_corrs

# function to get intersection between correlation result and autocorrelation result
def compare(corr_cells, autocorr_cells):
    valid_cells = [corr_cells, autocorr_cells]
    # identify which is smaller/which is larger
    larger = np.argmax([len(corr_cells), len(autocorr_cells)])
    smaller = np.argmin([len(corr_cells), len(autocorr_cells)])

    # pad nans to smaller array to make it comparable
    placeholder = np.full(np.shape(valid_cells[larger]), np.nan)
    placeholder[0:len(valid_cells[smaller])] = valid_cells[smaller]

    comparecells = np.array([cell in valid_cells[larger] for cell in placeholder])

    best_cells = placeholder[comparecells]
    return best_cells

# function to convert time seconds to calcium trace indices
def convert_time_frame(frame_times, time_point_variable):
    return min(enumerate(frame_times), key=lambda x: abs(x[1] - time_point_variable))[0]

# function to compute z score of chosen cells
def z_score_comp(chosen_cells, break_phases, tail_length=5):
    print("Compute Z Score of Chosen Cells")
    # pre allocation
    mean_std = np.zeros((2, np.shape(chosen_cells)[0]))
    cell_breaks = []
    z_score_cells = np.zeros(np.shape(chosen_cells))
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
            current_cell_break.extend(chosen_cells[cell, start_ind + tail_length:end_ind])
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
            z_score_cells[cell, frame_idx] = (frame - cell_mean) / cell_std
    return z_score_cells, mean_std

# function to calculate Area under the curve for all cells in each condition
def get_AUCs(mean_best_cells, mean_best_cells_ind, num_stim):
    # pre allocation
    aucs_best_cells = np.empty((np.shape(mean_best_cells)[0], num_stim))

    # iterate over cells
    for cell in range(np.shape(aucs_best_cells)[0]):
        cond_counter = 0
        # iterate over all dot sizes
        for ds in range(np.shape(mean_best_cells_ind)[0]):
            for wind in range(np.shape(mean_best_cells_ind)[1]):
                for el in range(np.shape(mean_best_cells_ind)[2]):
                    if not np.isnan(mean_best_cells_ind[ds, wind, el]).any():
                        aucs_best_cells[cell, cond_counter] = np.trapz(mean_best_cells[cell,
                                                                       int(mean_best_cells_ind[ds, wind, el, 0]):
                                                                       int(mean_best_cells_ind[ds, wind, el, 1])])
                        cond_counter += 1
    return aucs_best_cells

# function to generate stimulus masks for each condition
def stimulus_mask(indices, all_windows, all_dot_sizes, window_width, num_el):
    # pre allocation
    masks = []
    el_zero = 90  # identify where in mask is 0° elevation

    # iterate over dot sizes
    for ds in range(np.shape(indices)[0]):
        ds_masks = np.zeros((np.shape(indices)[1], num_el[ds], np.nanmax(all_windows[1:, :]) * 4, window_width * 2))
        # iterate over windows
        for window in range(np.shape(indices)[1]):
            # iterate over elevations
            elevations_wind = np.linspace(all_windows[1, window], all_windows[2, window], num_el[ds], endpoint=True).astype("int64")
            for el in range(np.shape(elevations_wind)[0]):
                # set azimuth range
                left_lim = all_windows[0, window] + window_width - int(all_dot_sizes[0, ds] / 2)
                right_lim = all_windows[0, window] + window_width + window_width + int(all_dot_sizes[0, ds] / 2)
                azimuth = np.arange(left_lim, right_lim).astype("int64")
                # adjust negative indices
                azimuth[azimuth < 0] = 360 + azimuth[azimuth < 0]
                azimuth[azimuth >= 360] = azimuth[azimuth >= 360] - 360

                # set elevation range
                upp_lim = el_zero - elevations_wind[el] - int(all_dot_sizes[0, ds] / 2)
                low_lim = el_zero - elevations_wind[el] + int(all_dot_sizes[0, ds] / 2)
                elevation = np.arange(upp_lim, low_lim, 1)
                # iterate over elevations and set window to 1
                for ell in elevation:
                    ds_masks[window, el, ell, azimuth] = 1
        masks.append(ds_masks)
    return masks

# function to weight the generated masks for all cells
def generate_rf(masks, aucs_best_cells, best_cells):
    print("weight the computed stimuli masks with AUC of cells")
    # pre allocation
    rf_matrices = np.zeros((np.shape(best_cells)[0], 2, np.shape(masks[0])[2], np.shape(masks[0])[3]))

    # iterate over all cells
    for cell in range(np.shape(best_cells)[0]):
        rf_cells = [[], []]
        cond_counter = 0
        # iterate over dot sizes
        for ds in range(len(masks)):
            mask_ds = masks[ds]
            # iterate over windows
            for window in range(np.shape(mask_ds)[0]):
                # iterate over elevations
                for el in range(np.shape(mask_ds)[1]):

                    current_mask = mask_ds[window, el, :, :]

                    current_auc = aucs_best_cells[cell, cond_counter]
                    rf_cond = current_mask * current_auc
                    rf_cells[ds].append(rf_cond)
                    cond_counter += 1
        rf_matrices[cell, 0, :, :] = np.mean(np.dstack(rf_cells[0]), axis=2)
        rf_matrices[cell, 1, :, :] = np.mean(np.dstack(rf_cells[1]), axis=2)
    return rf_matrices

# function to get RF center (defined as maximum)
def get_rf_center(rf_matrices):
    # pre allocation
    rf_centers = np.zeros((2, np.shape(rf_matrices)[1], np.shape(rf_matrices)[0]))
    # iterate over cells
    for cell in range(np.shape(rf_matrices)[0]):
        # iterate over dot sizes
        for ds in range(np.shape(rf_matrices)[1]):
            # get index (as x & y coordinates) of maximum entry = RF center
            current_matrix = rf_matrices[cell, ds, :, :]
            rf_centers[0, ds, cell], rf_centers[1, ds, cell] = np.unravel_index(np.argmax(current_matrix),
                                                                                np.shape(current_matrix))
    return rf_centers

# function to make plots more aesthetic
def plot_beautiful(ax, xmin=None, xmax=None, ymin=None, ymax=None, step=None,
                   xlabel="", ylabel="", title="", legend=True):
    ax.spines[["top", "right"]].set_visible(False)  # toggle off top & right ax spines
    if not xmin == xmax:
        ax.set_xticks(np.linspace(xmin, xmax, step))  # if values given, adjust x ticks
    if not ymin == ymax:
        ax.set_yticks(np.linspace(ymin, ymax, step))  # if values given, adjust x ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        # move legend to right & outside of plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=5)

# function to make stimulus plot for pixel plot
def extract_reg(time_sorting, time_sorted_cells, all_regressors_conv, regbuffer=0):
    # pre allocation
    extracted_regressors = np.zeros((np.shape(time_sorting)[0], np.shape(time_sorted_cells)[1]))
    idx_counter = 0
    # iterate over stimulus conditions
    for reg in range(np.shape(time_sorting)[0]):
        regressor = all_regressors_conv[reg]
        # iterate over repetitions
        for rep in range(np.shape(time_sorting)[2]):
            start = time_sorting[reg, 0, rep].astype("int64")
            end = time_sorting[reg, 1, rep].astype("int64")
            interval = end - start
            regressor_a = regressor[start - regbuffer:end + regbuffer]
            # if at end of cell trace
            if np.shape(extracted_regressors)[1] < (idx_counter - regbuffer + len(regressor_a)):
                limit = (idx_counter - regbuffer + len(regressor_a)) - np.shape(extracted_regressors)[1]
                placeholder = len(extracted_regressors[reg, idx_counter - regbuffer:idx_counter + limit])
                extracted_regressors[reg, idx_counter - regbuffer:idx_counter + limit] = regressor_a[0:placeholder]
            else:
                extracted_regressors[reg, idx_counter - regbuffer:idx_counter + interval + regbuffer] = regressor_a
                idx_counter += len(regressor_a) - regbuffer
    return extracted_regressors

# function for pixel plot
def pixel_plot(time_sorted_cells, frame_rate, conditions, time_sorting, line_counter, extracted_regressors,
               c, cmap="hot", clim_factor=0.5):
    # pre allocation
    labels = []
    show_line = False
    pixel_ticks = []
    pixel_labels = []
    trace_label = []
    # redefine time axis
    time = (np.linspace(0, np.shape(time_sorted_cells)[1] / frame_rate)).astype("int64")
    x = np.arange(0, np.shape(time_sorted_cells)[1])
    dot_interval = [[0], []]
    # get list of conditions
    split_all = [i.split("_") for i in conditions]

    # create figure
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(8)
    # create grid for different subplots
    spec = grid.GridSpec(ncols=2, nrows=2, width_ratios=[3, .05], wspace=.2, hspace=0.00, height_ratios=[1, .2])
    pixelplot = fig.add_subplot(spec[0])
    box_trace = fig.add_subplot(spec[2])

    # plot pixelplot
    im = pixelplot.imshow(time_sorted_cells, cmap="hot")
    im.set_clim(vmin=np.nanmin(time_sorted_cells), vmax=np.nanmax(time_sorted_cells) * clim_factor)

    # iterate over all Conditions
    for line in range(len(conditions)):
        split_conds = split_all[line]
        color = c[line]

        # get start and end indices of all reps for this condition
        starts = time_sorting[line, 0, :].astype("int64")
        ends = time_sorting[line, 1, :].astype("int64")

        if split_conds[0] == "big":
            if split_conds[2] == "3":
                show_line = True
                dot_interval[0].append(line_counter[line])
        elif split_conds[0] == "small":
            if split_conds[2] == "7":
                show_line = True
                dot_interval[1].append(line_counter[line])
        if show_line:
            pixelplot.axvline(line_counter[line], ymin=0, ymax=np.shape(time_sorted_cells)[0], linestyle="--",
                              color=color,
                              linewidth=2)
            pixel_ticks.append(line_counter[line].astype("int64"))
            pixel_labels.append(f"{split_conds[0]} {split_conds[1]}")
            trace_label.append((line_counter[line] / frame_rate).astype("int64"))

            box_trace.axvline(line_counter[line], ymin=0, ymax=np.nanmax(extracted_regressors) + 1, linestyle="--",
                              color=color,
                              linewidth=2)
        box_trace.plot(x, extracted_regressors[line, :], color=color)
        show_line = False

    box_trace.axhline(0, xmin=0, xmax=np.shape(extracted_regressors)[1], color="black", linewidth=1.5)

    # shade areas: big dot
    box_trace.fill_between(x=[min(dot_interval[0]), max(dot_interval[0])],
                           y1=[np.nanmax(extracted_regressors) + 1, np.nanmax(extracted_regressors) + 1],
                           color=[.7, .3, .69, .2])
    # shade areas: smol dot
    box_trace.fill_between(x=[max(dot_interval[0]), max(dot_interval[1])],
                           y1=[np.nanmax(extracted_regressors) + 1, np.nanmax(extracted_regressors) + 1],
                           color=[.7, .48, .3, .2])

    # adjust plots
    pixelplot.set_ylabel("Cells", labelpad=20)
    pixelplot.set_xlim([time[0], time[-1]])
    pixelplot.set_xticks(ticks=pixel_ticks[:], labels=pixel_labels[:])
    pixelplot.spines[["top", "right"]].set_visible(False)  # toggle off top & right ax spines

    box_trace.set_xlim([time[0], time[-1]])
    box_trace.set_xlabel("Relative Time [s]", labelpad=20)
    box_trace.spines[["top", "right"]].set_visible(False)  # toggle off top & right ax spines
    box_trace.set_xticks(ticks=pixel_ticks[:], labels=trace_label[:])

    # cbar
    cax = fig.add_subplot(spec[1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Z-Score", labelpad=25)
    return fig

# Function for Progressbar, Credits to David
def __printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█',
                       printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(Fore.GREEN + f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print(Fore.RESET)
