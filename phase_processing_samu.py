import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import h5py
import Toolbox as tb

# hardcoded stuff
frame_rate = 2.18  # in Hz
des_wind = 5    # window size for sliding window median (DFF), in min
tau = 1.6
ds = [5, 30]    # dot sizes used in recordings
dot_winds = ["left", "right", "back", "front"]  # locations of moving dot stimulus

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

# %%
"""
# Calculate DFF
smooth_f = tb.avg_smooth(data=F, window=3)

# plot multiple cells
dff = tb.calc_dff_wind(F=smooth_f, window=des_wind, frame_rate=frame_rate)
"""
# align frames between both PCs
frame_times = tb.adjust_frames(io=io, F=F)

# %%

#get all moving dot phases and their breaks
data_movdot = tb.extract_mov_dot(display)

# get phases of big dots/small dots
#big
data_big_dot = tb.extract_dot_ds(data_movdot, 30)
#small
data_small_dot = tb.extract_dot_ds(data_movdot, 5)

#get the phases of the windows 
#big
data_left_window_big = tb.extract_dot_window(data_big_dot, "left")
data_front_window_big = tb.extract_dot_window(data_big_dot, "front")
data_right_window_big = tb.extract_dot_window(data_big_dot, "right")
data_back_window_big = tb.extract_dot_window(data_big_dot, "back")

#small
data_left_window_small = tb.extract_dot_window(data_small_dot, "left")
data_front_window_small = tb.extract_dot_window(data_small_dot, "front")
data_right_window_small = tb.extract_dot_window(data_small_dot, "right")
data_back_window_small = tb.extract_dot_window(data_small_dot, "back")

#offset_angle = elevation
left_elevations_big = np.arange(15, -16, -15)
front_elevation_big = np.arange(45, 14, -15)
right_elevation_big = np.arange(15, -16, -15)
back_elevation_big = np.arange(-15, -46, -15)

left_elevations_small = np.arange(15, -16, -5)
front_elevation_small = np.arange(45, 14, -5)
right_elevation_small = np.arange(15, -16, -5)
back_elevation_small = np.arange(-15, -46, -5)

elevations_list_big = [left_elevations_big, front_elevation_big, right_elevation_big,
                       back_elevation_big]
elevations_list_small = [left_elevations_small, front_elevation_small,
                         right_elevation_small, back_elevation_small]
data_windows_list_big = [data_left_window_big, data_front_window_big, data_right_window_big,
                         data_back_window_big]
data_windows_list_small= [data_left_window_small, data_front_window_small, data_right_window_small,
                         data_back_window_small]

data_elevations_list_big = []
tp_windows_big = []
indices_windows_big = []
for j in range(len(elevations_list_big)):
    data_elevations_window = []
    tp_elevations_window_big = []
    indices_elevations_window_big = []
    for elevation in elevations_list_big[j]:
        valid_phases = []    
        for curr_phase, i in zip(data_windows_list_big[j].keys(), range(len(data_windows_list_big[j]))):           #loop over all phases
            if data_windows_list_big[j][curr_phase]['dot_offset_angle'] == elevation:
                valid_phases.append(curr_phase)    
        data_elevation = {}
        tp_elevation_big = []
        indices_elevation_big = []
        for i in valid_phases:     #loop over valid phases
            data_elevation[i] = data_windows_list_big[j][i]   #add keys and the data behind to the new dictionary
            st_elevation = data_windows_list_big[j][i]["__start_time"]
            tp_switch_big = data_windows_list_big[j][i]["__start_time"] + data_windows_list_big[j][i]["t_switch"]
            et_elevation = st_elevation + data_windows_list_big[j][i]["__target_duration"]
            start_index = min(enumerate(frame_times), key=lambda x: abs(x[1]-st_elevation))[0]
            switch_index = min(enumerate(frame_times), key=lambda x: abs(x[1]-tp_switch_big))[0]
            end_index = min(enumerate(frame_times), key=lambda x: abs(x[1]-et_elevation))[0]
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
    
    
data_elevations_list_small = []
tp_windows_small = []
indices_windows_small = []
for k in range(len(elevations_list_small)):
    data_elevations_window = []
    tp_elevations_window_small = []
    indices_elevations_window_small = []
    for elevation in elevations_list_small[k]:
        valid_phases = []    
        for curr_phase, i in zip(data_windows_list_small[k].keys(), range(len(data_windows_list_small[k]))):           #loop over all phases
            if data_windows_list_small[k][curr_phase]['dot_offset_angle'] == elevation:
                valid_phases.append(curr_phase)    
        data_elevation = {}
        tp_elevation_small = []
        indices_elevation_small = []
        for i in valid_phases:     #loop over valid phases
            data_elevation[i] = data_windows_list_small[k][i]   #add keys and the data behind to the new dictionary
            st_elevation = data_windows_list_small[k][i]["__start_time"]
            et_elevation = st_elevation + data_windows_list_small[k][i]["__target_duration"]
            tp_switch = data_windows_list_small[k][i]["__start_time"] + data_windows_list_small[k][i]["t_switch"]
            start_index = min(enumerate(frame_times), key=lambda x: abs(x[1]-st_elevation))[0]
            switch_index = min(enumerate(frame_times), key=lambda x: abs(x[1]-tp_switch))[0]
            end_index = min(enumerate(frame_times), key=lambda x: abs(x[1]-et_elevation))[0]
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
    

