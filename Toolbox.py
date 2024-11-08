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


def extract_dot_ds(data_movdot, dot_size):
    """
    Extracting the phases containing moving dot depending on dot size

    Parameters
    ----------
    data_movdot : dict
        dictionary containing all stimulus phases with a moving dot and the
        breaks inbetween (should also work with the dictionary of the entire
                          stimulus - not tested)
    dot_size : int
        dot angular diameter - either 5 or 30

    Returns
    -------
    move_dot_ds : dict
        dictionary containing all phases of a moving dot of one specific size.
        - ds: dot_size

    """
    print("getting phases depending on your dot sizes")
    # get phases of interest (remove all lines that are interesting)
    valid_phases = []
    for curr_phase, i in zip(data_movdot.keys(), range(len(data_movdot))):           #loop over all phases
        if data_movdot[curr_phase]['__visual_name'] == 'SingleDotRotatingBackAndForth':
            if data_movdot[curr_phase]['dot_angular_diameter'] == dot_size:
                valid_phases.append(curr_phase)             #get the indices of all phases
        else:
            continue                            #skip phases without a moving dot
    move_dot_ds = {}
    for i in valid_phases:     #loop over valid phases
        move_dot_ds[i] = data_movdot[i]   #add keys and the data behind to the new dictionary
    return move_dot_ds

def extract_dot_window(data_dot_ds, window):
    """
    extract phases of moving dots within one of the tested windows. To use after
    extracting phases of one dot size.

    Parameters
    ----------
    data_dot_ds : dict
        Containing all phases with a moving dot of one size.
    window : string
        determining which window should be extracted.
        Possible windows: left, front, right, back

    Returns
    -------
    data_window : dict
        containing the phases with all their information that are presented in one
        window.

    """
    if window == "left":
        valid_phases = []
        for curr_phase, i in zip(data_dot_ds.keys(), range(len(data_dot_ds))):           #loop over all phases
            if data_dot_ds[curr_phase]['__visual_name'] == 'SingleDotRotatingBackAndForth':
                if data_dot_ds[curr_phase]['dot_start_angle'] == -180:
                    valid_phases.append(curr_phase)             #get the indices of all phases
            else:
                continue                            #skip phases without a moving dot
        data_window = {}
        for i in valid_phases:     #loop over valid phases
            data_window[i] = data_dot_ds[i]   #add keys and the data behind to the new dictionary
    elif window == "front":
        valid_phases = []
        for curr_phase, i in zip(data_dot_ds.keys(), range(len(data_dot_ds))):           #loop over all phases
            if data_dot_ds[curr_phase]['__visual_name'] == 'SingleDotRotatingBackAndForth':
                if data_dot_ds[curr_phase]['dot_start_angle'] == -90:
                    valid_phases.append(curr_phase)             #get the indices of all phases
            else:
                continue                            #skip phases without a moving dot
        data_window = {}
        for i in valid_phases:     #loop over valid phases
            data_window[i] = data_dot_ds[i]   #add keys and the data behind to the new dictionary
    elif window == "right":
        valid_phases = []
        for curr_phase, i in zip(data_dot_ds.keys(), range(len(data_dot_ds))):           #loop over all phases
            if data_dot_ds[curr_phase]['__visual_name'] == 'SingleDotRotatingBackAndForth':
                if data_dot_ds[curr_phase]['dot_start_angle'] == 0:
                    valid_phases.append(curr_phase)             #get the indices of all phases
            else:
                continue                            #skip phases without a moving dot
        data_window = {}
        for i in valid_phases:     #loop over valid phases
            data_window[i] = data_dot_ds[i]   #add keys and the data behind to the new dictionary
    elif window == "back":
        valid_phases = []
        for curr_phase, i in zip(data_dot_ds.keys(), range(len(data_dot_ds))):           #loop over all phases
            if data_dot_ds[curr_phase]['__visual_name'] == 'SingleDotRotatingBackAndForth':
                if data_dot_ds[curr_phase]['dot_start_angle'] == 90:
                    valid_phases.append(curr_phase)             #get the indices of all phases
            else:
                continue                            #skip phases without a moving dot
        data_window = {}
        for i in valid_phases:     #loop over valid phases
            data_window[i] = data_dot_ds[i]   #add keys and the data behind to the new dictionary
    return data_window

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

# function of regressor curve (for convolution), Credits to Carina
def CIRF(regressor, n_ca_frames, tau=1.6):
    time = np.arange(0, n_ca_frames)
    exp = np.exp(-time / tau)
    reg_conv = np.convolve(regressor, exp)
    reg_conv = reg_conv[:n_ca_frames]
    return reg_conv

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
