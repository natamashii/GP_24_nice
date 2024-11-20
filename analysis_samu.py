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
import statistics

def analysis_complete(data_path, plot_name):
    #%%
    # pre allocation
    cell_pos = []
    
    # hardcoded stuff
    frame_rate = 2.18  # in Hz
    des_wind = 5    # window size for sliding window median (DFF), in min
    tau = 1.6
    
    all_dot_sizes = np.array([[30, 5], [-15, -5]], dtype="int32")
    all_windows = np.array([[-180, -90, 0, 90],
                            [15, 45, 15, -15], [-15, 15, -15, -45]], dtype="int32")
    win_buffer = [-1, 10]
    conds_dotsizes = ["big", "small"]
    conds_windows = ["left", "front", "right", "back"]
    conds_elevations = [["1", "2", "3"], ["1", "2", "3", "4", "5", "6", "7"]]
    
    # define path: rn only one recording, later then more general
    # in jupyter notebook: default working directory is location of this file (can be seen with "print(os.getcwd())"  )
    # to access other working directories: os.chdir("")
    # data_path for at home
    #data_path = "E:\\GP_24\\05112024\\GP24_fish1_rec1_05112024\\"
    
    
    # data_path for jupyter
    # data_path = "/home/jovyan/data/fish_1_05112024/"
    
    # get vxpy stuff
    display = tt.load_hdf5(data_path + "Display.hdf5", name="phase")
    io = h5py.File(data_path + "Io.hdf5")
    
    # get suite2p stuff
    F = np.load(data_path + "suite2p\\plane0\\F.npy")
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
    corr_array = tt.corr(dff=dff, regressors=all_regressors_conv, regressor_phase_stp=all_regressors_phase_stp,
                         regressor_phase_etp=all_regressors_phase_etp)
    
    # select only good cells
    good_cells, good_cells_idx = np.unique(np.where(corr_array > .3)[0], return_index=True)
    
    # %% Autocorrelation: Yeet Cells that fluctuate in their responses to stimulus repetitions
    
    auto_corrs = tt.autocorr(dff=dff, indices=indices, win_buffer=win_buffer, regressors=all_regressors_conv)
    
    really_good_cells, really_good_cells_idx = np.unique(np.where(auto_corrs > .4)[0], return_index=True)
    
    # find intersection between correlation result and autocorrelation result
    best_cells = tt.compare(corr_cells=good_cells, autocorr_cells=really_good_cells)
    
    # %% Z-Score of Cells
    # get absolute start and end of dot stimulus interval
    t_min, t_max = tt.t_min_max(indices=indices)
    
    z_scores_cells, mean_std = tt.z_score_comp(chosen_cells=dff, break_phases=break_phases, tail_length=3)
    
    # %% sort cells
    
    # chosen_cells = dff[best_cells.astype("int64"), :]
    chosen_cells = z_scores_cells[best_cells.astype("int64"), :]
    
    # get indices of sorting cells from top to bottom
    cell_sorting, time_sorting, conditions = tt.get_sort_cells(indices=indices, chosen_cells=chosen_cells,
                                                               num_conds=len(all_regressors), conds_dotsizes=conds_dotsizes,
                                                               conds_windows=conds_windows, conds_elevations=conds_elevations)
    
    # sort cells from top to bottom
    sorted_cells = tt.sort_cells(sorting=cell_sorting, num_conds=len(all_regressors),
                                 chosen_cells=chosen_cells, indices=indices)
    
    # sort time dimension of cells to stimulus conditions
    time_sorted_cells = tt.sort_times(sorted_cells=sorted_cells, time_sorting=time_sorting, t_min=t_min)
    
    # %% Pixel Plot
    # TODO: adjust time axis
    # TODO: decide on beautiful cbar
    # TODO: label of cbar
    # TODO: choose right cell shit
    # TODO: rewrite as function
    # TODO: adjust font sizes
    # TODO: adjust Title
    # TODO: normalize cbar to values
    # TODO: resize cbar relative to image
    
    # interpolation makes it kinda blurry
    
    # fig, axs = plt.subplots()
    # pixelplot = axs.imshow(time_sorted_cells, cmap="PiYG")
    # pixelplot.set_clim(np.nanmin(time_sorted_cells), np.nanmax(time_sorted_cells))
    
    # axs.set_xlabel("Time [s]")
    # axs.set_ylabel("Cells")
    # axs.set_title("Cells of Rec 1 Fish 1 Day 1, Only DFF, attempted sort")
    # fig.colorbar(pixelplot)
    
    # plt.show(block=False)
    
    #%% get average dffover repetitions for each condition and cell
    mean_dff_bcs, new_inds = tt.get_mean_dff_trace(chosen_cells, indices)
    
    #%%get the AUCs
    num_stim = len(all_regressors)
    
    AUCs_all_cells = tb.get_AUCs(mean_dff_bcs, new_inds, num_stim)
    
    
    #%% get the stimulus trace
    conds_ds = [30, 5]
    conds_winds = [-180, -90, 0, 90]
    conds_elevs = [[45, 15], [15, -15], [-15, -45], [15, -15]]
    
    
    stims = np.full((2, 4, 7, 3), np.nan)
    stims_list = []
    for ds in range(np.shape(indices)[0]):
        #stims[ds] = conds_ds[ds]
        # iterate over windows
        for wind in range(np.shape(indices)[1]):
            #stims[ds, wind] = conds_winds[wind]
            # iterate over elevations
            for el in range(np.shape(indices)[2]):
                # only continue if repetition exists
                if not np.isnan(indices[ds, wind, el]).any():
                    if ds == 0:
                        offsets = np.arange(conds_elevs[wind][0], conds_elevs[wind][1]-1, -15)
                        #stims[ds, wind, el] = offsets[el]
                        stims[ds, wind, el, 0] = conds_ds[ds]
                        stims[ds, wind, el, 1] = conds_winds[wind]
                        stims[ds, wind, el, 2] = offsets[el]
                        stims_list.append([conds_ds[ds], conds_winds[wind], offsets[el]])
                    if ds == 1:
                        offsets = np.arange(conds_elevs[wind][0], conds_elevs[wind][1]-1, -5)
                        #stims[ds, wind, el] = offsets[el]
                        stims[ds, wind, el, 0] = conds_ds[ds]
                        stims[ds, wind, el, 1] = conds_winds[wind]
                        stims[ds, wind, el, 2] = offsets[el]
                        stims_list.append([conds_ds[ds], conds_winds[wind], offsets[el]])
                        
    #%% analysis masks 
    list_receptive_fields_biig = []
    list_receptive_fields_smol = []
    #for cell in range(np.shape(AUCs_all_cells)[0]):
    num_rf_plot = 10
    counter = 0
    if np.shape(AUCs_all_cells)[0] < 10:
        num_rf_plot = np.shape(AUCs_all_cells)[0]
    for cell in range(np.shape(AUCs_all_cells)[0]):
        rf_matrix_total_avg_biig, rf_matrix_total_avg_smol = tb.create_stimulus_mask(stims_list, AUCs_all_cells[cell])
        list_receptive_fields_biig.append(rf_matrix_total_avg_biig)
        list_receptive_fields_smol.append(rf_matrix_total_avg_smol)
    for plot in range(num_rf_plot):
        counter += 1
        fig = plt.figure()
        ax = fig.add_subplot(111)  # Add a single subplot to the figure
        tb.plot_rf(fig, ax, list_receptive_fields_biig[plot])
        fig.savefig(f'rf_big_{plot_name}{counter}.svg')
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)  # Add a single subplot to the figure
        tb.plot_rf(fig, ax, list_receptive_fields_smol[plot])
        fig.savefig(f'rf_small_{plot_name}{counter}.svg')
        plt.close()

        
    #%%
    cells_index_big_dot = []
    cells_index_small_dot = []
    azimuths = np.arange(0,361, 45)
    center_rf_cells_bd = []
    center_rf_cells_sd = []
    for cell in range(np.shape(list_receptive_fields_biig)[0]):
        max_bd = np.max(list_receptive_fields_biig[cell])
        max_sd = np.max(list_receptive_fields_smol[cell])    
        if max_bd > max_sd:
            threshold_cell = np.mean(list_receptive_fields_biig[cell]) + 2*np.std(list_receptive_fields_biig[cell])
            azims = np.where(list_receptive_fields_biig[cell] > threshold_cell)[1]
            # Compute distances for each target
            distances = np.abs(azimuths[:, None] - azims)  # Shape: (len(azimuths), len(azims))   
            # Find the index of the minimum distance for each target
            closest_indices = np.argmin(distances, axis=0)       
            # Get the closest azimuths
            closest_azimuths = azimuths[closest_indices]
            #get the azimuth at which the most closest azimuths lay as best azimuth of the cell
            if len(closest_azimuths) != 0:
                best_azim = statistics.mode(closest_azimuths)
            else:
                continue
            #get the elevation and an alternative azimuth value using only the information contained in the max
            best_elev = np.unravel_index(np.argmax(list_receptive_fields_biig[cell]), np.shape(list_receptive_fields_biig[cell]))[0]
            center_rf_cells_bd.append([best_azim, best_elev])
            #get the ROI index of suite2p
            cells_index_big_dot.append(best_cells[cell].astype(int))
        else:
            threshold_cell = np.mean(list_receptive_fields_smol[cell]) + 2*np.std(list_receptive_fields_smol[cell])
            azims = np.where(list_receptive_fields_smol[cell] > threshold_cell)[1]
            # Compute distances for each target
            distances = np.abs(azimuths[:, None] - azims)  # Shape: (len(azimuths), len(azims))   
            # Find the index of the minimum distance for each target
            closest_indices = np.argmin(distances, axis=0)       
            # Get the closest azimuths
            closest_azimuths = azimuths[closest_indices]
            #get the azimuth at which the most closest azimuths lay as best azimuth of the cell
            if len(closest_azimuths) != 0:
                best_azim = statistics.mode(closest_azimuths)
            else:
                continue
            #get the elevation and an alternative azimuth value using only the information contained in the max
            best_elev = np.unravel_index(np.argmax(list_receptive_fields_smol[cell]), np.shape(list_receptive_fields_smol[cell]))[0]
            center_rf_cells_sd.append([best_azim, best_elev])
            #get the ROI index of suite2p
            cells_index_small_dot.append(best_cells[cell].astype(int))
    
    return center_rf_cells_bd, cells_index_big_dot, center_rf_cells_sd, cells_index_small_dot

  
#%% müll
# fig = plt.figure()
# plt.hist(best_azims_cells_bd, bins = azimuths)
# plt.xticks(azimuths)
# plt.show()


#%% müll2.0
"""
best_azim_cells_bd = []
for cell in range(np.shape(center_cells_big_dot)[0]):
    best_azim = min(azimuths, key=lambda x: abs(x - center_cells_big_dot[cell][1]))
    best_azim_cells_bd.append(best_azim)
    #center_cells_big_dot[cell][1] = best_azim 
best_azim_cells_sd = []
for cell in range(np.shape(center_cells_small_dot)[0]):
    best_azim = min(azimuths, key=lambda x: abs(x - center_cells_small_dot[cell][1]))
    best_azim_cells_sd.append(best_azim)
 """   

