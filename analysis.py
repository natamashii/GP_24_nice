__author__ = "Natalie"
# importing
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import h5py
import os
import Toolbox_2p0_aaaaaaaaaaaaaaaaa as tt

# TODO: improve code
# TODO: cbar for pixelplot regressot part
# TODO: functions in class
# TODO: rewrite receptive fields
# TODO: rewrite anaotmical registration

# pre allocation
layer_cells = []
layer_sizes = []

# hardcoded stuff
frame_rate = 2.18  # in Hz
des_wind = 5    # window size for sliding window median (DFF), in min
tau = 1.6

all_dot_sizes = np.array([[30, 5], [-15, -5]], dtype="int32")
all_windows = np.array([[-180, -90, 0, 90],
                        [15, 45, 15, -15], [-15, 15, -15, -45]], dtype="int32")
# Note: i could also calculate the other end of the azimuth level with angle velocity & duration...
# measurement goes clockwise
window_width = 180
num_el = [3, 7]

win_buffer = [-1, 10]
conds_dotsizes = ["big", "small"]
conds_windows = ["left", "front", "right", "back"]
conds_elevations = [["1", "2", "3"], ["1", "2", "3", "4", "5", "6", "7"]]

# indices for recordings, sorted after recorded fish layer (+60um, +40um, -20um, (+0um) relative to AC)
files = [[0, 4, 7, 11, 14, 17],
         [1, 5, 8, 12, 15, 18],
         [3, 6, 9, 13, 16, 19],
         [2, 10, 20]]

sort_mechanism = 0
sort_as_string = ["max_sort", "corr_sort"]
to_save = True
pre_process = False
save_path = "E:\\GP_Ari\\analysis\\"
roottpath = "E:\\GP_Ari\\recordings\\"

rec_folders = os.listdir(roottpath)
sorted_recordings_folder = [next(os.walk(roottpath + day + "\\"))[1] for day in rec_folders]
# get amount of different layers that were recorded
layers = max([len(sorted_recordings_folder[day]) for day in range(len(sorted_recordings_folder))])

# Preprocessing

if pre_process:
    for day in range(len(sorted_recordings_folder)):
        recording_day = sorted_recordings_folder[day]
        for rec in range(len(recording_day)):
            data_path = roottpath + f"{rec_folders[day]}\\{recording_day[rec]}\\"
            print(f"Perform preprocessing for rec {rec + 1} of day {rec_folders[day]}")

            # get vxpy stuff
            display = tt.load_hdf5(data_path + "Display.hdf5", name="phase")
            io = h5py.File(data_path + "Io.hdf5")

            # get suite2p stuff
            F = np.load(data_path + "suite2p\\plane0\\F.npy")  # intensity trace for each detected cell
            ops = np.load(data_path + "suite2p\\plane0\\ops.npy", allow_pickle=True).item()
            stat = np.load(data_path + "suite2p\\plane0\\stat.npy", allow_pickle=True)

            # Calculate DFF
            # smooth traces with average in sliding window
            smooth_f = tt.avg_smooth(data=F, window=3)

            # calculate DFF with median in sliding window as F0
            dff = tt.calc_dff_wind(F=smooth_f, window=des_wind, frame_rate=frame_rate)

            # Split Data into stimulus conditions
            # align frames between both PCs
            frame_times = tt.adjust_frames(io=io, F=F)

            # find phases of Moving Dot & corresponding break phases
            valid_data = tt.extract_mov_dot(display, desired_phase="SingleDotRotatingBackAndForth")
            time_points, phase_names, indices, break_phases = (
                tt.extract_data(valid_data=valid_data, all_dot_sizes=all_dot_sizes,
                                    all_windows=all_windows, frame_times=frame_times))

            # dimension 0: dot size: 0 = 30 ; 1 = 5
            # dimension 1: dot window: left, front, right, back
            # dimension 2: dot elevation level
            # dimension 3: number of repetition
            # dimension 4: start, switch, end

            # Build Regressor
            all_regressors, all_regressors_conv, all_regressors_phase_stp, all_regressors_phase_etp =\
                tt.build_regressor(indices=indices, dff=dff, frame_times=frame_times, tau=tau)

            # Correlation: Find Correlation of cells to Moving Dot Phases
            # pre allocation
            corr_array = tt.corr(dff=dff, regressors=all_regressors_conv, regressor_phase_stp=all_regressors_phase_stp,
                                 regressor_phase_etp=all_regressors_phase_etp)

            # select only good cells
            good_cells, good_cells_idx = np.unique(np.where(corr_array > .3)[0], return_index=True)

            # Autocorrelation: Yeet Cells that fluctuate in their responses to stimulus repetitions
            auto_corrs = tt.autocorr(dff=dff, indices=indices, win_buffer=win_buffer, regressors=all_regressors_conv)

            really_good_cells, really_good_cells_idx = np.unique(np.where(auto_corrs > .4)[0], return_index=True)

            # find intersection between correlation result and autocorrelation result
            best_cells = tt.compare(corr_cells=good_cells, autocorr_cells=really_good_cells)
            best_cells = best_cells.astype("int64")

            # Z-Score of Cells
            # get absolute start and end of dot stimulus interval
            t_min, t_max = tt.t_min_max(indices=indices)

            z_scores_cells, _ = tt.z_score_comp(chosen_cells=dff, break_phases=break_phases, tail_length=3)

            # sort cells
            chosen_cells = dff[best_cells.astype("int64"), :]

            # get indices of sorting cells from top to bottom: maximum absolute mean activity
            max_sorting, time_sorting, conditions = tt.get_sort_cells_max(indices=indices, chosen_cells=chosen_cells,
                                                                       num_conds=len(all_regressors),
                                                                       conds_dotsizes=conds_dotsizes,
                                                                       conds_windows=conds_windows,
                                                                       conds_elevations=conds_elevations)
            # get indices of sorting cells from top to bottom: maximum absolute correlation
            corr_sorting = tt.get_sort_cells_corr(chosen_cells=chosen_cells, best_cells=best_cells, corr_array=corr_array)

            # decide which cell sorting to choose based on sorting mechanism
            sorting = [max_sorting, corr_sorting]
            cell_sorting = sorting[sort_mechanism]

            # sort cells from top to bottom
            sorted_cells, sorted_cells_idx = tt.sort_cells(sorting=cell_sorting, num_conds=len(all_regressors),
                                         chosen_cells=chosen_cells)

            # sort time dimension of cells to stimulus conditions
            time_sorted_cells, line_counter = tt.sort_times(sorted_cells=sorted_cells, time_sorting=time_sorting)
            extracted_regressors = tt.extract_reg(time_sorting=time_sorting, time_sorted_cells=time_sorted_cells,
                                                  all_regressors_conv=all_regressors_conv, regbuffer=0)

            all_regressors_conv = np.array(all_regressors_conv)

            # save data of each recording to HDF5
            if to_save:
                tt.save_hdf5(list_of_data_arrays=[chosen_cells, time_sorted_cells, time_sorting, line_counter,
                                                  conditions, best_cells, sorted_cells, sorted_cells_idx, indices,
                                                  t_min, dff, all_regressors_conv, extracted_regressors],
                             list_of_labels=["chosen_cells", "time_sorted_cells", "time_sorting", "line_counter",
                                             "conditions", "best_cells", "sorted_cells", "sorted_cells_idx", "indices",
                                             "t_min", "dff", "all_regressors_conv", "extracted_regressors"],
                             new_directory=f"fish_1_rec{rec}",
                             export_path=save_path + f"rec_folders_{rec_folders[day]}\\"
                                                     f"fish_1_rec_{rec}_{sort_as_string[sort_mechanism]}.HDF5",
                             permission="a")

# load data
all_chosen_cells = [[] for layer in range(layers)]
all_best_cells = [[] for layer in range(layers)]
all_indices = [[] for layer in range(layers)]
all_t_min = [[] for layer in range(layers)]
all_dff = [[] for layer in range(layers)]
all_all_regressors_conv = [[] for layer in range(layers)]

# hardcoded toggle variables to not having to scroll all way up ._.
sort_mechanism = 0
sort_as_string = ["max_sort", "corr_sort"]
to_save = True

# iterate over all recordings: sorted in list (fish layer) of lists (corr recording)
for day in range(len(sorted_recordings_folder)):
    recording_day = sorted_recordings_folder[day]
    for rec in range(len(recording_day)):
        data = tt.load_hdf5(import_path=save_path + f"rec_folders_{rec_folders[day]}\\"
                                                    f"fish_1_rec_{rec}_{sort_as_string[sort_mechanism]}.HDF5",
                            name=f"fish_1_rec{rec}")
        data = data[f"fish_1_rec{rec}"]
        all_all_regressors_conv[rec].append(data["all_regressors_conv"])
        all_best_cells[rec].append(data["best_cells"])
        all_chosen_cells[rec].append(data["chosen_cells"])
        all_dff[rec].append(data["dff"])
        all_indices[rec].append(data["indices"])
        all_t_min[rec].append(data["t_min"])

# Preparation for Pixelplot
# pre allocation
layer_specifics = {"cells": [],
                   "conditions": [],
                   "time_sorted_cells": [],
                   "line_counter": [],
                   "sorted_cells_idx": [],
                   "extracted_regressors": []}

# stack all cells from same fish layer to one array
for layer in range(layers):
    # get amount of cells
    sizes = max([np.shape(i)[1] for i in all_chosen_cells[layer]])
    layer_sizes.append(sizes)
    cells = []
    cell_counter = 0
    # iterate over each recording performed for current fish layer
    for r in range(len(all_chosen_cells[layer])):
        rec_cells = all_chosen_cells[layer][r]
        # pad rec_cells with zeros
        placeholder = np.zeros((np.shape(rec_cells)[0], sizes))
        placeholder[0:np.shape(rec_cells)[0], 0:np.shape(rec_cells)[1]] = rec_cells
        cells.append(placeholder)
    cells = np.vstack(cells)
    layer_cells.append(cells)

    # sort cells for entire fish layer
    cell_sorting, time_sorting, conditions = tt.get_sort_cells_max(indices=all_indices[layer][0], chosen_cells=cells,
                                                                    num_conds=40, conds_dotsizes=conds_dotsizes,
                                                                    conds_windows=conds_windows,
                                                                    conds_elevations=conds_elevations)
    # sort cells from top to bottom
    sorted_cells, sorted_cells_idx = tt.sort_cells(sorting=cell_sorting, num_conds=40, chosen_cells=cells)

    # sort time dimension of cells to stimulus conditions
    time_sorted_cells, line_counter = tt.sort_times(sorted_cells=sorted_cells, time_sorting=time_sorting)

    # extract regressor
    extracted_regressors = tt.extract_reg(time_sorting=time_sorting, time_sorted_cells=time_sorted_cells,
                                          all_regressors_conv=all_all_regressors_conv[layer][0], regbuffer=0)
    the_one_regressor = extracted_regressors

    # store necessary stuff in dictionary
    layer_specifics["cells"].append(cells)
    layer_specifics["conditions"].append(conditions)
    layer_specifics["time_sorted_cells"].append(time_sorted_cells)
    layer_specifics["line_counter"].append(line_counter)
    layer_specifics["sorted_cells_idx"].append(sorted_cells_idx)
    layer_specifics["extracted_regressors"].append(extracted_regressors)

# Pixel Plot
# plot specific hardcoded variables
# list of figure & cbar factors that are layer specific
heights = [10, 20, 15, 10]
widths = [40, 20, 30, 40]
fig_sizes = [[30, 30],
             [33, 48],
             [37, 47],
             [30, 30]]
sizes = [220, 220, 220, 220]
tick_sizes = [170, 170, 170, 170]
lgd_distance = [4.5, 3.5, 3.5, 5.5]
cax_factor = [.75, .15, .27, .55]
cax_height = [.78, .965, .925, .82]
cax_x = [-0.064, -0.05, -0.05, -0.049]
cax_y2 = [-.05, -.1, -.1, -.06]
hspace = [.0, .001, .01, .0]
cax_width = [.01, .015, .015, .01]
y2_dist = [10, 10, 10, 10]

axs_width = 5.5
tick_pad = 20
ticks = 20
tick_width = 3
clim_factor = .5
labelpad = 100
cmap = "hot"
font_family = ["sans-serif", "cursive"] # second entry is for debugging
font = ["Arial", "Comic Sans MS"]   # second entry is for debugging
dot_area_colours = [[.788, .706, .831, 1], [.973, .819, .996, 1]]
linewidth = 8
fig_save = True
# list of figure & cbar factors that are layer specific

# define cmap for regressor_plot
h_colours = np.repeat([310/360], 40)
s_colours = np.repeat([89/100], 40)
v_colours = np.append([np.tile(np.linspace(50.4/100, 90.6/100, 3, endpoint=True), 4)],
                     [np.tile(np.linspace(50.4/100, 90.6/100, 7, endpoint=True), 4)])
colour = mpl.colors.hsv_to_rgb(np.transpose(np.array([h_colours[:], s_colours[:], v_colours[:]])))
colour = np.hstack((colour, np.ones((40, 1))))
nice_cmap = mpl.colors.LinearSegmentedColormap.from_list(name="nice_pink", colors=[colour[0, :], colour[7, :]])
cc = mpl.cm.ScalarMappable(cmap=nice_cmap)

# iterate over all fish layers
for layer in range(layers):
    # plot-specific pre allocation
    dot_interval = [[0], []]
    pixel_ticks = [0]
    pixel_yticks = []
    pixel_labels = []
    trace_ticks = []
    trace_labels = []

    # add separation lines & add lines to regressor_plot at same time
    show_line = False  # if True, plots line of separating conditions, toggles itself off
    split_conds = [cond.split("_") for cond in layer_specifics["conditions"][layer]]  # split name of conditions

    # set amount of cells for each condition in one array (needed for drawing separation lines in cell dimension)
    cell_interval = np.zeros((np.shape(layer_specifics["sorted_cells_idx"][layer])[0] + 1), dtype="int64")
    amount_cells = int(np.shape(layer_specifics["time_sorted_cells"][layer])[0])
    for i in range(np.shape(layer_specifics["sorted_cells_idx"][layer])[0]):
        cell_interval[i] = amount_cells
        amount_cells -= np.sum(~np.isnan(layer_specifics["sorted_cells_idx"][layer][i, :]))

    # set x values for regressor_plot
    x = np.arange(0, np.shape(layer_specifics["extracted_regressors"][layer])[1])

    with (plt.rc_context({"font.size": sizes[layer], "axes.titlesize": sizes[layer], "axes.labelsize": sizes[layer],
                         "ytick.labelsize": tick_sizes[layer], "xtick.labelsize": tick_sizes[layer],
                         "legend.fontsize": tick_sizes[layer], "figure.titlesize": sizes[layer],
                          "figure.labelsize": sizes[layer],
                         "axes.linewidth": axs_width, "font.weight": "bold", "xtick.major.pad": tick_pad,
                          "ytick.major.pad": tick_pad, "xtick.major.size": ticks,
                          "ytick.major.size": ticks, "xtick.major.width": tick_width,
                          "ytick.major.width": tick_width, "axes.labelpad": labelpad, "axes.labelweight": "bold",
                          "font.family": font_family[0], "font.sans-serif": font[0],
                          "figure.subplot.hspace": hspace[layer],
                          "figure.subplot.left": .1, "figure.subplot.bottom": .1})):

        fig, axs = plt.subplots(2, 2, constrained_layout=False,
                                figsize=(mpl.rcParams["figure.figsize"][0] * fig_sizes[layer][0],
                                         mpl.rcParams["figure.figsize"][1] * fig_sizes[layer][1]),
                                height_ratios=(1, heights[layer]), width_ratios=(widths[layer], 1))
        pixelplot = axs[1, 0]
        regressor_plot = axs[0, 0]

        # plot pixelplot
        im = pixelplot.imshow(layer_specifics["time_sorted_cells"][layer][::-1], cmap=cmap, origin="lower")
        im.set_clim(vmin=np.nanmin(layer_specifics["cells"][layer]),
                    vmax=np.nanmax(layer_specifics["cells"][layer]) * clim_factor)

        # iterate over all conditions
        for cond in range(len(layer_specifics["conditions"][layer])):
            # if at end of elevation levels for big dot and window
            if split_conds[cond][0] == "big" and split_conds[cond][2] == "3":
                show_line = True
                dot_interval[0].append(layer_specifics["line_counter"][layer][cond])
                line_colour = dot_area_colours[0]
            # if at end of elevation levels for small dot and window
            if split_conds[cond][0] == "small" and split_conds[cond][2] == "7":
                show_line = True
                dot_interval[1].append(layer_specifics["line_counter"][layer][cond])
                line_colour = dot_area_colours[1]

            # plot separation line
            if show_line:
                # vertical line of separation in time
                pixelplot.axvline(layer_specifics["line_counter"][layer][cond], ymin=0,
                                  ymax=np.shape(layer_specifics["time_sorted_cells"][layer])[0], linestyle="--",
                                  color=line_colour, linewidth=linewidth+3)
                # horizontal line of separation in cells
                pixelplot.axhline(cell_interval[cond + 1], xmin=0,
                                  xmax=np.shape(layer_specifics["time_sorted_cells"][layer])[1], linestyle="--",
                                  color=line_colour, linewidth=linewidth+3)
                # vertical line of separation in time for regressor_plot
                regressor_plot.axvline(layer_specifics["line_counter"][layer][cond], ymin=0,
                                       ymax=np.nanmax(layer_specifics["extracted_regressors"][layer][cond]) + 1,
                                       linestyle="--", color="white", linewidth=linewidth)
                # save position of vline
                pixel_ticks.append(layer_specifics["line_counter"][layer][cond].astype("int64"))
                trace_ticks.append(layer_specifics["line_counter"][layer][cond].astype("int64"))
                # save position of hline
                pixel_yticks.append(cell_interval[cond + 1])
                pixel_labels.append(f"{split_conds[cond][0]}  \n{split_conds[cond][1]}")
                trace_labels.append((layer_specifics["line_counter"][layer][cond] / frame_rate).astype("int64"))

            # plot regressor trace of current condition
            regressor_plot.plot(x, layer_specifics["extracted_regressors"][layer][cond, :], linewidth=linewidth,
                                color=colour[cond, :])

            show_line = False   # toggle off to display line only once per condition

        # placeholder subplot
        axs[0, 1].axis("off")
        axs[1, 1].axis("off")

        # adjust pixelplot
        # add cbar
        ax = axs[1, 1]
        factor = ax.get_position().y0
        cax = fig.add_axes([ax.get_position().x1 + .05, ax.get_position().y0 + cax_factor[layer]*factor,
                            cax_width[layer],
                            ax.get_position().height * cax_height[layer]])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Z-Score", labelpad=labelpad, weight="bold", size=sizes[layer])

        # adjust x-axis
        # move ticks to center of window (Credits to Carina)
        pixel_ticks.append(
            int(np.shape(layer_specifics["time_sorted_cells"][layer])[1]) +
            (int(np.shape(layer_specifics["time_sorted_cells"][layer])[1]) - pixel_ticks[-1]) * 2)
        pixel_labels.append(" ")
        pixelplot.set_xticks(ticks=np.array(pixel_ticks[1:]) - np.diff([0] + pixel_ticks[1:]) / 2,
                             labels=pixel_labels[:])
        pixelplot.set_xlim([0, np.shape(layer_specifics["time_sorted_cells"][layer])[1]])
        # adjust y-axis
        pixelplot.set_ylabel("Cells")
        pixel_yticks.insert(0, int(np.shape(layer_specifics["time_sorted_cells"][layer])[0]))
        pixelplot.set_ylim([0, np.shape(layer_specifics["time_sorted_cells"][layer])[0]])
        # add secondary y-axis for labeling hlines
        y2 = pixelplot.secondary_yaxis("right")
        y2.set_yticks(ticks=np.array(np.array(pixel_yticks[:]) - np.diff([0] + pixel_yticks[:]) / 2)[1:],
                      labels=pixel_labels[:-1])
        y2.set_ylabel("Cell Categories", labelpad=labelpad * y2_dist[layer])

        pixelplot.spines[["top", "right"]].set_visible(False)  # toggle off top & right ax spines

        # adjust regressor_plot
        # shade areas
        # big dot
        area_bd = regressor_plot.fill_between(x=[min(dot_interval[0]), max(dot_interval[0])],
                                              y1=[np.nanmax(layer_specifics["extracted_regressors"][layer]) + 1,
                                                  np.nanmax(layer_specifics["extracted_regressors"][layer]) + 1],
                                              color=dot_area_colours[0], label="Dot Size = 30 °")
        # smol dot
        area_sd = regressor_plot.fill_between(x=[max(dot_interval[0]), max(dot_interval[1])],
                                              y1=[np.nanmax(layer_specifics["extracted_regressors"][layer]) + 1,
                                                  np.nanmax(layer_specifics["extracted_regressors"][layer]) + 1],
                                              color=dot_area_colours[1], label="Dot Size = 5 °")
        # add legend
        legend = axs[0, 1].legend(handles=[area_bd, area_sd], frameon=False, loc="center right",
                                       prop={"weight": "bold"}, bbox_to_anchor=(lgd_distance[layer], 1))

        # add pink cbar
        ax = axs[1, 1]
        factor = ax.get_position().y1
        cax = fig.add_axes([ax.get_position().x1 + cax_x[layer], ax.get_position().y1 + cax_y2[layer] * factor,
                            cax_width[layer],
                            ax.get_position().height * .1])
        cbar = plt.colorbar(cc, cax=cax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "30"])
        cbar.set_label("\n Elevation rel. to \n stimulus window [°]",
                       labelpad=labelpad*16, weight="bold", size=tick_sizes[layer], rotation=0, loc="top")

        regressor_plot.set_xlabel("Time [s]")
        regressor_plot.set_xticks(ticks=trace_ticks[:], labels=trace_labels[:])
        regressor_plot.set_xlim([0, np.shape(layer_specifics["time_sorted_cells"][layer])[1]])
        #adjust y-axis
        regressor_plot.set_ylabel("Regressor")
        regressor_plot.set_yticks(ticks=[0, 1, 2, 3], labels=["0", "1", "2", "3"])
        regressor_plot.set_ylim([0, 2.2])

        regressor_plot.spines[["top", "right"]].set_visible(False)  # toggle off top & right ax spines

        fig.suptitle("Stimulus Conditions", weight="bold")

        #plt.show(block=False)

        if fig_save:
            plt.savefig(f"E:\\GP_Ari\\figures\\"
                        f"pixelplot_fish_level_{layer}_{sort_as_string[sort_mechanism]}.svg", format="svg")
            plt.savefig(f"E:\\GP_Ari\\figures\\"
                        f"pixelplot_fish_level_{layer}_{sort_as_string[sort_mechanism]}.pdf", format="pdf")
            #plt.savefig(f"E:\\GP_Ari\\figures\\"
                        #f"pixelplot_fish_level_{layer}_{sort_as_string[sort_mechanism]}.jpg", format="jpg")
            plt.savefig(f"E:\\GP_Ari\\figures\\"
                        f"pixelplot_fish_level_{layer}_{sort_as_string[sort_mechanism]}.png", format="png")

# %% Trace Plots
np.random.seed(0)
random_day = np.random.randint(0, len(rec_folders)+1)
recording_day = sorted_recordings_folder[random_day]
np.random.seed(1)
random_rec = np.random.randint(0, len(recording_day)+1)
data = tt.load_hdf5(import_path=save_path + f"rec_folders_{rec_folders[random_day]}\\"
                                            f"fish_1_rec_{random_rec}_{sort_as_string[sort_mechanism]}.HDF5",
                    name=f"fish_1_rec{random_rec}")
data = data[f"fish_1_rec{random_rec}"]
dff = data["dff"]
t_min = data["t_min"]
all_regressors_conv = data["all_regressors_conv"]
time_ax = np.linspace(0, (np.shape(dff)[1] - t_min) / frame_rate, np.shape(dff)[1] - t_min, endpoint=True).astype(
    "int64")

#%% DEBUG ZONE

# %% Trace Plots
data_path = "E:\\GP_24\\recordings\\05112024\\GP24_fish1_rec1_05112024\\"
# get vxpy stuff
display = tt.load_hdf5(data_path + "Display.hdf5", name="phase")
io = h5py.File(data_path + "Io.hdf5")
# get suite2p stuff
F = np.load(data_path + "suite2p\\plane0\\F.npy")  # intensity trace for each detected cell
ops = np.load(data_path + "suite2p\\plane0\\ops.npy", allow_pickle=True).item()
stat = np.load(data_path + "suite2p\\plane0\\stat.npy", allow_pickle=True)

# Calculate DFF
# smooth traces with average in sliding window
smooth_f = tt.avg_smooth(data=F, window=3)
# calculate DFF with median in sliding window as F0
dff = tt.calc_dff_wind(F=smooth_f, window=des_wind, frame_rate=frame_rate)

# Split Data into stimulus conditions
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

# Build Regressor
all_regressors, all_regressors_conv, all_regressors_phase_stp, all_regressors_phase_etp =\
    tt.build_regressor(indices=indices, dff=dff, frame_times=frame_times, tau=tau)
# Correlation: Find Correlation of cells to Moving Dot Phases
# pre allocation
corr_array = tt.corr(dff=dff, regressors=all_regressors_conv, regressor_phase_stp=all_regressors_phase_stp,
                     regressor_phase_etp=all_regressors_phase_etp)
# select only good cells
good_cells, good_cells_idx = np.unique(np.where(corr_array > .3)[0], return_index=True)
# Autocorrelation: Yeet Cells that fluctuate in their responses to stimulus repetitions
auto_corrs = tt.autocorr(dff=dff, indices=indices, win_buffer=win_buffer, regressors=all_regressors_conv)
really_good_cells, really_good_cells_idx = np.unique(np.where(auto_corrs > .4)[0], return_index=True)
# find intersection between correlation result and autocorrelation result
best_cells = tt.compare(corr_cells=good_cells, autocorr_cells=really_good_cells)

t_min, t_max = tt.t_min_max(indices=indices)
z_scores_cells, mean_std = tt.z_score_comp(chosen_cells=dff, break_phases=break_phases, tail_length=3)
# sort cells
chosen_cells = z_scores_cells[best_cells.astype("int64"), :]
# get indices of sorting cells from top to bottom
cell_sorting, time_sorting, conditions = tt.get_sort_cells(indices=indices, chosen_cells=chosen_cells,
                                                           num_conds=len(all_regressors), conds_dotsizes=conds_dotsizes,
                                                           conds_windows=conds_windows, conds_elevations=conds_elevations)

amount_plots = 10
chosen_cells = dff[best_cells.astype("int64"), :]
random_cells = [1, 12, 2, 17, 13, 15, 7, 8, 20, 10]
fig_trace = tt.plot_regressor(dff=chosen_cells, t_min=t_min, frame_rate=frame_rate, random_cells=random_cells,
                              all_regressors_conv=all_regressors_conv, cell_sorting=cell_sorting,
                              amount_plots=amount_plots, size=28, linewidth=3)
fig_trace.suptitle("Placeholder: different cells with high correlation & autocorrealtion")
plt.show(block=False)
fig_save = True
if fig_save:
    plt.savefig(f"D:\\GP_24\\figures\\dff_regressor_trace_high_corr.svg", format="svg")

fig_trace = tt.plot_regressor(dff=dff, t_min=t_min, frame_rate=frame_rate, random_cells=random_cells,
                              all_regressors_conv=all_regressors_conv, cell_sorting=cell_sorting,
                              amount_plots=amount_plots, size=28, linewidth=3)
fig_trace.suptitle("Placeholder: different cells with trashy correlation & autocorrealtion")
plt.show(block=False)
fig_save = True
if fig_save:
    plt.savefig(f"E:\\GP_24\\figures\\dff_regressor_trace_trash_corr.svg", format="svg")


# %% plot normal complete trace and one complete regressor

# for cell 3 and 7
size = 28
with ((plt.rc_context({"font.size": size, "axes.titlesize": size, "axes.labelsize": size + size*.2, "ytick.labelsize": size,
                       "xtick.labelsize": size, "legend.fontsize": size + size*.2, "figure.titlesize": size,
                       "figure.labelsize": size + size*.2}))):
    fig, axs = plt.subplots(4, 1, constrained_layout=False, sharex=True, sharey=True)
    celltrace = axs[0]
    regtrace = axs[1]
    chosen_cells = dff[best_cells.astype("int64"), :]
    time = np.linspace(0, np.shape(chosen_cells)[1]/frame_rate, np.shape(chosen_cells)[1], endpoint=True).astype("int64")
    cell = chosen_cells[3]
    cell_pain = celltrace.plot(time, cell, color="tab:blue", linewidth=3, label="dFF")
    reg_pain = regtrace.plot(time, all_regressors_conv[cell_sorting[3]], color="tab:pink", linewidth=3, label="Regressor")
    celltrace.spines[["top", "right"]].set_visible(False)
    regtrace.spines[["top", "right"]].set_visible(False)

    celltrace = axs[2]
    regtrace = axs[3]
    cell = chosen_cells[7]
    cell_pain = celltrace.plot(time, cell, color="tab:blue", linewidth=3, label="dFF")
    reg_pain = regtrace.plot(time, all_regressors_conv[cell_sorting[7]], color="tab:pink", linewidth=3, label="Regressor")
    celltrace.spines[["top", "right"]].set_visible(False)
    regtrace.spines[["top", "right"]].set_visible(False)

    regtrace.set_xlabel("Time [s]", labelpad=20, weight="bold")
    plt.figlegend(["Regressor", "dFF"], frameon=False, prop={"weight": "bold"})
    fig.supylabel("dFF", weight="bold")

    plt.show(block=False)

    plt.savefig(f"E:\\GP_24\\figures\\cell_traces_reg_only.svg", format="svg")

# %% RECEPTIVE FIELD MAPPING: average dff of all repetitions for each condition & cell
mean_best_cells, mean_best_cells_ind = tt.mean_dff(chosen_cells=chosen_cells, indices=indices)

# get AUCs for each cell and each condition
aucs_best_cells = tt.get_AUCs(mean_best_cells=mean_best_cells, mean_best_cells_ind=mean_best_cells_ind,
                              num_stim=len(all_regressors_conv))

# create mask: one dot size & one window one elevation level
masks = tt.stimulus_mask(indices=indices, all_windows=all_windows, all_dot_sizes=all_dot_sizes, window_width=window_width, num_el=num_el)

# Weight the masks with AUC
rf_matrices = tt.generate_rf(masks=masks, aucs_best_cells=aucs_best_cells, best_cells=best_cells)

# dimension 1: 0 = big dot, 1 = small dot

# find RF center
rf_centers = tt.get_rf_center(rf_matrices=rf_matrices)

# dim 0: 0 = elevation ; 1 = azimuth
# dim 1: 0 = big dot ; 1 = small dot

if to_save:
    tt.save_hdf5(list_of_data_arrays=[aucs_best_cells, rf_matrices, rf_centers],
                 list_of_labels=["aucs_best_cells", "rf_matrices", "rf_centers"], new_directory=f"fish_1_rec{rec}",
                 export_path=save_path + f"rec_folders_{rec_folders[day]}\\fish_1_rec_{rec}_rf_matrices.HDF5",
                 permission="a")

# %% Plot RFs
# TODO: histogram
#limit = np.shape(rf_matrices)[0]
limit = 1
cmap = "hot"
with ((plt.rc_context({"font.size": size, "axes.titlesize": size, "axes.labelsize": size, "ytick.labelsize": size,
                     "xtick.labelsize": size, "legend.fontsize": size, "figure.titlesize": size,
                     "figure.labelsize": size}))):
    for rf in range(limit):
        fig_rf_big, ax_rf_big = plt.subplots()
        im = ax_rf_big.imshow(rf_matrices[rf, 0, :, :], cmap=cmap)
        cbar = plt.colorbar(im, ax=ax_rf_big, label="RF Intensity")
        cbar.set_label("RF Intensity", labelpad=20)
        ax_rf_big.set_xlabel("Azimuth [deg]")
        ax_rf_big.set_ylabel("Elevation [deg]")
        ax_rf_big.set_title("Receptive Field for Big Dot")
        ax_rf_big.set_xticks(ticks=np.linspace(0, 360, 13, endpoint=True),
                             labels=np.linspace(-180, 180, 13, endpoint=True).astype("int64"))
        ax_rf_big.set_yticks(ticks=np.linspace(0, 180, 13, endpoint=True),
                             labels=np.linspace(90, -90, 13, endpoint=True).astype("int64"))
        ax_rf_big.axvline(180, ymin=0, ymax=180, color="white", linestyle="--", linewidth=2)
        ax_rf_big.axhline(90, xmin=0, xmax=360, color="white", linestyle="--", linewidth=2)

        fig_rf_small, ax_rf_small = plt.subplots()
        im = ax_rf_small.imshow(rf_matrices[rf, 1, :, :], cmap=cmap)
        cbar = plt.colorbar(im, ax=ax_rf_small, label="RF Intensity")
        cbar.set_label("RF Intensity", labelpad=20)
        ax_rf_small.set_xlabel("Azimuth [deg]")
        ax_rf_small.set_ylabel("Elevation [deg]")
        ax_rf_small.set_title("Receptive Field for Small Dot")
        ax_rf_small.set_xticks(ticks=np.linspace(0, 360, 13, endpoint=True),
                               labels=np.linspace(-180, 180, 13, endpoint=True).astype("int64"))
        ax_rf_small.set_yticks(ticks=np.linspace(0, 180, 13, endpoint=True),
                               labels=np.linspace(90, -90, 13, endpoint=True).astype("int64"))
        ax_rf_small.axvline(180, ymin=0, ymax=180, color="white", linestyle="--", linewidth=2)
        ax_rf_small.axhline(90, xmin=0, xmax=360, color="white", linestyle="--", linewidth=2)

    plt.show(block=False)

# %% anatomical regression
# sort cells with their RFs based on their RF center location
# get their anatomical position
# carina stuff
# plotting

# take finished RFs, get good ones
# find out where RFs are located
# azimuth: take midth of window
# identify threshold to determine where cell is tuned to (to which window)
# calculate via z score, or take like 2*std above mean or so
# center of RF: elevation: take max, azimuth: midth of window i guess

# simplest method: 2 histograms one for elevation, one for azimuth
# for more beautiful: overlay of recordings in fiji/oython
# correct for x and y shift
# suite2p gives cell pos, adjust them (registration)
# only do for cells that pass RF selection test (only take those that are good looking in RF matrix)
# each cell has RF location (center detected as above mentioned)
# one map for elevation, one for azimuth
# cbar: from -90 to 0 to 90
# color code cells at their registrated RF location (elevation/azimuth)
# 4 maps in total: differ in dot size, then differ in elevation and azimuth location

# if we manage it: histogram for distribution of anatomical position
# around maps
# histogram of distribution of x and y position (one for each)
