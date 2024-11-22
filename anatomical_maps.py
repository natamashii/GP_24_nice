from analysis_samu import analysis_complete
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib as mpl
#%%
# import data for rf
data_path = "//172.25.250.112/arrenberg_data/shared/GP_24/12112024/GP_12112024_fish1_rec1/"

# define plot name to save each RF plot seperately
# TODO: adjust name to each cell, instead of each recording
#plot_name = data_path.split('/')[-2].split('_')[-2] + '_' + data_path.split('/')[-2].split('_')[-1]

# create rf plots and return variables for anatomical maps
#center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(data_path, plot_name)

plt.rcParams['xtick.labelsize'] = 20  # Size of x-tick labels
plt.rcParams['ytick.labelsize'] = 20  # Size of y-tick labels
plt.rcParams['axes.titlesize'] = 20 * 1.4  # Title font size (axes titles)
plt.rcParams['axes.labelsize'] = 20*1.2  # Label font size for x and y axes
plt.rcParams['font.size'] = 20*1.2  # Default font size for text in plots
plt.rcParams['font.weight'] = 'bold'  # Font weight (use 'bold' instead of 2)
plt.rcParams['axes.titleweight'] = 'bold'  # Bold axes titles
plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
#%% LOOP
# - loop over all files and sort 

gp_Folder = '//172.25.250.112/arrenberg_data/shared/GP_24'  # root folder for loading TIF file

###################################################################################

# make a list of all paths of all recordings (experiments and individual layers).

day_folders = [n for n in os.listdir(gp_Folder) if "112024" in n]  
master_recording_paths = []

for folder in day_folders:

    recording_folders = [n for n in os.listdir(gp_Folder+f'/{folder}') if "rec" in n]
    for recording in recording_folders:
        print(recording)

        # try:
        #     tif_path = [n for n in os.listdir(f'{gp_Folder}/{folder}/{recording}') if ".TIF" in n][0]
        master_recording_paths.append(f'{gp_Folder}/{folder}/{recording}/')

        # except:
        #     print('     no tif')
        #     continue


################# Insert your filtered file lists here ############################################################

# example lists to filter for specific depths
dorsal_60_list = []
dorsal_40_list = []
ventral_20_list = []
ventral_00_list = []

for i in master_recording_paths:
    # Check for files containing "60um_dorsalAC"
    path_60 = [i for n in os.listdir(i) if "60um_dorsalAC" in n]
    if len(path_60) != 0:
        dorsal_60_list.append(path_60[0])
    
    # Check for files containing "40um_dorsalAC"
    path_40 = [i for n in os.listdir(i) if "40um_dorsalAC" in n]
    if len(path_40) != 0:
        dorsal_40_list.append(path_40[0])
    
    # Check for files containing "20um_ventralAC"
    path_20 = [i for n in os.listdir(i) if "20um_ventralAC" in n]
    if len(path_20) != 0:
        ventral_20_list.append(path_20[0])
    
    # Check for files containing "00um_ventralAC"
    path_00 = [i for n in os.listdir(i) if "00um_ventralAC" in n]
    if len(path_00) != 0:
        ventral_00_list.append(path_00[0])




#%% anatomical maps
best_cells_pos_bd_ar_60 = []
best_cells_pos_sd_ar_60 = []
center_rf_bd_ar_60 = []
center_rf_sd_ar_60 = []

# load mean img from suite2p
ref_img_60 = np.load(f'{data_path}/suite2p/plane0/ops.npy', allow_pickle=True).item()['meanImg']
rec_counter = 0
for rec in dorsal_60_list:
    # load cell positions
    #all_cells_pos = pd.read_pickle(f'{rec}/cell_positions_transformed.pkl')
    # Open the CSV file
    all_cells_pos = pd.read_csv(f'{rec}/cell_positions_transformed.csv')
    
    # define plot name dynamically
    plot_name = str(rec.split('/')[-1])+ "_receptive_field"
        
    # create rf plots and return variables for anatomical maps
    center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(rec, plot_name, rec_counter)
    
    # extract positions for relevant cells
    best_cells_pos_bd = all_cells_pos.iloc[cells_index_bd]
    best_cells_pos_sd = all_cells_pos.iloc[cells_index_sd]
    
    #extend lists of all recordings
    best_cells_pos_bd_ar_60.append(best_cells_pos_bd)
    best_cells_pos_sd_ar_60.append(best_cells_pos_sd)
    center_rf_bd_ar_60.append(center_rf_cells_bd)
    center_rf_sd_ar_60.append(center_rf_cells_sd)
    rec_counter += 1

#%%
# Plot Azimuth
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_60, cmap='gray', origin='lower')
ax.set_title("Optic Tectum 60 µm Dorsal of Landmark")

cmap_big = mpl.cm.get_cmap('Wistia')  # andere Farbe nehmen
cmap_small = mpl.cm.get_cmap('cool')#
cmap_big_elevation = mpl.cm.get_cmap('Wistia')  # andere Farbe nehmen
cmap_small_elevation = mpl.cm.get_cmap('cool')

for i in range(len(best_cells_pos_bd_ar_60)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_60[i].drop(columns=best_cells_pos_bd_ar_60[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_60[i].drop(columns=best_cells_pos_sd_ar_60[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        #color = [1, (center_rf_bd_ar_60[i][cell_bd][0]-45)/(360-45), (center_rf_bd_ar_60[i][cell_bd][0]-45)/(360-45)]
        color = cmap_big((center_rf_bd_ar_60[i][cell_bd][0] - (-180)) / (180 - (-180)))
        if center_rf_bd_ar_60[i][cell_bd][0] == 0:
            color = cmap_big(1.0)
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        #color = [(center_rf_sd_ar_60[i][cell_sd][0]-45)/(360-45), (center_rf_sd_ar_60[i][cell_sd][0]-45)/(360-45), 1]
        color = cmap_small((center_rf_sd_ar_60[i][cell_sd][0] - (-180)) / (180 - (-180)))
        if center_rf_sd_ar_60[i][cell_sd][0] == 0:
            #color = [(360-45)/(360-45), (360-45)/(360-45), 1]
            color = cmap_small(1.0)
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        

# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
#cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_big), ax=ax, label='Big Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 9) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-180, 181, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols


# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, normalized, 1]
#colors = [[norm, norm, 1] for norm in normalized]
# Create a colormap from the colors
#cmap_sd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_small), ax=ax, label='Small Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 9) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-180, 181, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols


# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()


#fig.colorbar(sax_bd, ax=ax, label='Azimuth Brightness')
#fig.colorbar(sax_sd, ax=ax, label='Azimuth Brightness')
plt.show()
fig.savefig('anatomical_map_60_azim.svg')
#################
# Plot Elevation
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_60, cmap='gray', origin='lower')
ax.set_title("Optic Tectum 60 µm Dorsal of Landmark")


for i in range(len(best_cells_pos_bd_ar_60)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_60[i].drop(columns=best_cells_pos_bd_ar_60[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_60[i].drop(columns=best_cells_pos_sd_ar_60[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        #color = [1, (center_rf_bd_ar_60[i][cell_bd][1]-31)/(151-31) ,1]
        color = cmap_big_elevation((center_rf_bd_ar_60[i][cell_bd][1]-(-61))/(61-(-61)))
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        #color = [(center_rf_sd_ar_60[i][cell_sd][1]-43)/(139-43), 1 ,(center_rf_sd_ar_60[i][cell_sd][1]-43)/(139-43)]
        color = cmap_small_elevation((center_rf_sd_ar_60[i][cell_sd][1]-(-61))/(61-(-61)))
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
#values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
#greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
#colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
#cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_big_elevation), ax=ax, label='Big Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)


# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols


# Define the range of values
#values = np.linspace(43, 139, 1024)  # Generate values between 43 and 139
#pinks = (values - 43) / (139 - 43)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, 1, normalized]
#colors_sd = [[norm, 1, norm] for norm in pinks]
# Create a colormap from the colors
#cmap_sd = mcolors.ListedColormap(colors_sd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_small_elevation), ax=ax, label='Small Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)


# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_60_elev.svg')


########################################################################################################STOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#%% 40 um dorsal AC
best_cells_pos_bd_ar_40 = []
best_cells_pos_sd_ar_40 = []
center_rf_bd_ar_40 = []
center_rf_sd_ar_40 = []

data_path = "//172.25.250.112/arrenberg_data/shared/GP_24/12112024/GP_12112024_fish1_rec2/"
# load mean img from suite2p
ref_img_40 = np.load(f'{data_path}/suite2p/plane0/ops.npy', allow_pickle=True).item()['meanImg']

for rec in dorsal_40_list:
    # load cell positions
    #all_cells_pos = pd.read_pickle(f'{rec}/cell_positions_transformed.pkl')
    # Open the CSV file
    all_cells_pos = pd.read_csv(f'{rec}/cell_positions_transformed.csv')

    # define plot name dynamically
    plot_name = rec.split('/')[-1]+ "_receptive_field"
        
    # create rf plots and return variables for anatomical maps
    center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(rec, plot_name)
    
    # extract positions for relevant cells
    best_cells_pos_bd = all_cells_pos.iloc[cells_index_bd]
    best_cells_pos_sd = all_cells_pos.iloc[cells_index_sd]
    
    #extend lists of all recordings
    best_cells_pos_bd_ar_40.append(best_cells_pos_bd)
    best_cells_pos_sd_ar_40.append(best_cells_pos_sd)
    center_rf_bd_ar_40.append(center_rf_cells_bd)
    center_rf_sd_ar_40.append(center_rf_cells_sd)
    
#%%
# Plot Azimuth
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_40, cmap='gray', origin='lower')
ax.set_title("Optic Tectum 40 µm Dorsal of Landmark")



for i in range(len(best_cells_pos_bd_ar_40)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_40[i].drop(columns=best_cells_pos_bd_ar_40[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_40[i].drop(columns=best_cells_pos_sd_ar_40[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = cmap_big((center_rf_bd_ar_40[i][cell_bd][0] - (-180)) / (180 - (-180)))
        if center_rf_bd_ar_40[i][cell_bd][0] == 0:
            color = cmap_big(1.0)
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = cmap_small((center_rf_sd_ar_40[i][cell_sd][0] - (-180)) / (180 - (-180)))
        if center_rf_sd_ar_60[i][cell_sd][0] == 0:
            #color = [(360-45)/(360-45), (360-45)/(360-45), 1]
            color = cmap_small(1.0)
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
# Define the range of values
#values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
#normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
#colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
#cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_big), ax=ax, label='Big Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Define the range of values
# values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
# normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, normalized, 1]
# colors = [[norm, norm, 1] for norm in normalized]
# Create a colormap from the colors
# cmap_sd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_small), ax=ax, label='Small Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_40_azim.svg')

#################
# Plot Elevation
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_40, cmap='gray', origin='lower')
ax.set_title("Optic Tectum 40 µm Dorsal of Landmark")

for i in range(len(best_cells_pos_bd_ar_40)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_40[i].drop(columns=best_cells_pos_bd_ar_40[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_40[i].drop(columns=best_cells_pos_sd_ar_40[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = cmap_big_elevation((center_rf_bd_ar_40[i][cell_bd][1]-(-61))/(61-(-61)))
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = cmap_small_elevation((center_rf_sd_ar_40[i][cell_sd][1]-(-61))/(61-(-61)))
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
#values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
#greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
#colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
#cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_big_elevation), ax=ax, label='Big Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)


# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Define the range of values
#values = np.linspace(43, 139, 1024)  # Generate values between 43 and 139
#pinks = (values - 43) / (139 - 43)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, 1, normalized]
#colors_sd = [[norm, 1, norm] for norm in pinks]
# Create a colormap from the colors
#cmap_sd = mcolors.ListedColormap(colors_sd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_small_elevation), ax=ax, label='Small Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels


ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_40_elev.svg')

########################################################################################################
#%% pretectum
best_cells_pos_bd_ar_20 = []
best_cells_pos_sd_ar_20 = []
center_rf_bd_ar_20 = []
center_rf_sd_ar_20 = []

data_path = "//172.25.250.112/arrenberg_data/shared/GP_24/07112024/GP_07112024_fish1_rec3/"
# load mean img from suite2p
ref_img_20 = np.load(f'{data_path}/suite2p/plane0/ops.npy', allow_pickle=True).item()['meanImg']

for rec in ventral_20_list:
    # load cell positions
    #all_cells_pos = pd.read_pickle(f'{rec}/cell_positions_transformed.pkl')
    # Open the CSV file
    all_cells_pos = pd.read_csv(f'{rec}/cell_positions_transformed.csv')

    # define plot name dynamically
    plot_name = str(rec.split('/')[-1])+ "_receptive_field"
        
    # create rf plots and return variables for anatomical maps
    center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(rec, plot_name)
    
    # extract positions for relevant cells
    best_cells_pos_bd = all_cells_pos.iloc[cells_index_bd]
    best_cells_pos_sd = all_cells_pos.iloc[cells_index_sd]
    
    #extend lists of all recordings
    best_cells_pos_bd_ar_20.append(best_cells_pos_bd)
    best_cells_pos_sd_ar_20.append(best_cells_pos_sd)
    center_rf_bd_ar_20.append(center_rf_cells_bd)
    center_rf_sd_ar_20.append(center_rf_cells_sd)
#%%
# Plot Azimuth
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_20, cmap='gray', origin='lower')
ax.set_title("Pretectum 20 µm Ventral of Landmark")


for i in range(len(best_cells_pos_bd_ar_20)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_20[i].drop(columns=best_cells_pos_bd_ar_20[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_20[i].drop(columns=best_cells_pos_sd_ar_20[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = cmap_big((center_rf_bd_ar_20[i][cell_bd][0] - (-180)) / (180 - (-180)))
        if center_rf_bd_ar_20[i][cell_bd][0] == 0:
            color = cmap_big(1.0)
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = cmap_small((center_rf_sd_ar_20[i][cell_sd][0] - (-180)) / (180 - (-180)))
        if center_rf_sd_ar_20[i][cell_sd][0] == 0:
            #color = [(360-45)/(360-45), (360-45)/(360-45), 1]
            color = cmap_small(1.0)
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Big Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)


# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, normalized, 1]
colors = [[norm, norm, 1] for norm in normalized]
# Create a colormap from the colors
cmap_sd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Small Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_20_azim.svg')

#################
# Plot Elevation
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_20, cmap='gray', origin='lower')
ax.set_title("Pretectum 20 µm Ventral of Landmark")

for i in range(len(best_cells_pos_bd_ar_20)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_20[i].drop(columns=best_cells_pos_bd_ar_20[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_20[i].drop(columns=best_cells_pos_sd_ar_20[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = cmap_big_elevation((center_rf_bd_ar_20[i][cell_bd][1]-(-61))/(61-(-61)))
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = cmap_small_elevation((center_rf_sd_ar_20[i][cell_sd][1]-(-61))/(61-(-61)))
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Big Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)


# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}°' for label in tick_labels]) # Add degree symbols

# Define the range of values
values = np.linspace(43, 139, 1024)  # Generate values between 43 and 139
pinks = (values - 43) / (139 - 43)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, 1, normalized]
colors_sd = [[norm, 1, norm] for norm in pinks]
# Create a colormap from the colors
cmap_sd = mcolors.ListedColormap(colors_sd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Small Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13)  
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels


# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_20_elev.svg')

########################################################################################################
#%% AC
best_cells_pos_bd_ar = []
best_cells_pos_sd_ar = []
center_rf_bd_ar = []
center_rf_sd_ar = []

data_path = "//172.25.250.112/arrenberg_data/shared/GP_24/07112024/GP_07112024_fish1_rec4/"
# load mean img from suite2p
ref_img = np.load(f'{data_path}/suite2p/plane0/ops.npy', allow_pickle=True).item()['meanImg']

for rec in ventral_00_list:
    # load cell positions
    #all_cells_pos = pd.read_pickle(f'{rec}/cell_positions_transformed.pkl')
    # Open the CSV file
    all_cells_pos = pd.read_csv(f'{rec}/cell_positions_transformed.csv')

    # define plot name dynamically
    plot_name = str(rec.split('/')[-1])+ "_receptive_field"
        
    # create rf plots and return variables for anatomical maps
    center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(rec, plot_name)
    
    # extract positions for relevant cells
    best_cells_pos_bd = all_cells_pos.iloc[cells_index_bd]
    best_cells_pos_sd = all_cells_pos.iloc[cells_index_sd]
    
    #extend lists of all recordings
    best_cells_pos_bd_ar.append(best_cells_pos_bd)
    best_cells_pos_sd_ar.append(best_cells_pos_sd)
    center_rf_bd_ar.append(center_rf_cells_bd)
    center_rf_sd_ar.append(center_rf_cells_sd)
#%%
# Plot Azimuth
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img, cmap='gray', origin='lower')
ax.set_title("Optic Tectum at Landmark")


for i in range(len(best_cells_pos_bd_ar)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar[i].drop(columns=best_cells_pos_bd_ar[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar[i].drop(columns=best_cells_pos_sd_ar[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = cmap_big((center_rf_bd_ar[i][cell_bd][0] - (-180)) / (180 - (-180)))
        if center_rf_bd_ar[i][cell_bd][0] == 0:
            color = cmap_big(1.0)
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = cmap_small((center_rf_sd_ar[i][cell_sd][0] - (-180)) / (180 - (-180)))
        if center_rf_sd_ar[i][cell_sd][0] == 0:
            #color = [(360-45)/(360-45), (360-45)/(360-45), 1]
            color = cmap_small(1.0)
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Big Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols


# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, normalized, 1]
colors = [[norm, norm, 1] for norm in normalized]
# Create a colormap from the colors
cmap_sd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Small Dot Azimuth [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_00_azim.svg')

#################
# Plot Elevation
fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img, cmap='gray', origin='lower')
ax.set_title("Optic Tectum at Landmark")

for i in range(len(best_cells_pos_bd_ar)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar[i].drop(columns=best_cells_pos_bd_ar[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar[i].drop(columns=best_cells_pos_sd_ar[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = cmap_big_elevation((center_rf_bd_ar[i][cell_bd][1]-(-61))/(61-(-61)))
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = cmap_small_elevation((center_rf_sd_ar[i][cell_sd][1]-(-61))/(61-(-61)))
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Big Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}°' for label in tick_labels]) # Add degree symbols

# Define the range of values
values = np.linspace(43, 139, 1024)  # Generate values between 43 and 139
pinks = (values - 43) / (139 - 43)  # Normalize values between 0 and 1
# Create RGB colors based on [normalized, 1, normalized]
colors_sd = [[norm, 1, norm] for norm in pinks]
# Create a colormap from the colors
cmap_sd = mcolors.ListedColormap(colors_sd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Small Dot Elevation [°]')
#cbar.ax.yaxis.label.set_size(20)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 13) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(-90, 91, 15) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols


# Dynamically determine x-axis range
x_min, x_max = plt.gca().get_xlim()
# Calculate 3 evenly spaced ticks
ticks = np.linspace(x_min, x_max, 3)  # Creates 3 equally spaced positions
labels = ['left', 'medial', 'right']  # Labels for the ticks
# Apply ticks and labels to the x-axis
ax.set_xticks(ticks)  # Set tick positions
ax.set_xticklabels(labels)  # Set tick labels

# Dynamically determine y-axis range
y_min, y_max = ax.get_ylim()  # Get y-axis limits from the axis
# Calculate 2 evenly spaced ticks
yticks = np.linspace(y_min, y_max, 2)  # Creates 2 equally spaced positions
ytick_labels = ['rostral', 'caudal']  # Labels for the ticks
# Apply ticks and labels to the y-axis
ax.set_yticks(yticks)  # Set tick positions
ax.set_yticklabels(ytick_labels)  # Set tick labels

ax.invert_yaxis()
plt.show()
fig.savefig('anatomical_map_00_elev.svg')
