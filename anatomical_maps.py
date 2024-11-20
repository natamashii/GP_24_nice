from analysis_samu import analysis_complete
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import pandas as pd

#%%
# import data for rf
data_path = "//172.25.250.112/arrenberg_data/shared/GP_24/12112024/GP_12112024_fish1_rec1/"

# define plot name to save each RF plot seperately
# TODO: adjust name to each cell, instead of each recording
#plot_name = data_path.split('/')[-2].split('_')[-2] + '_' + data_path.split('/')[-2].split('_')[-1]

# create rf plots and return variables for anatomical maps
#center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(data_path, plot_name)

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
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

for rec in dorsal_60_list:
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
    best_cells_pos_bd_ar_60.append(best_cells_pos_bd)
    best_cells_pos_sd_ar_60.append(best_cells_pos_sd)
    center_rf_bd_ar_60.append(center_rf_cells_bd)
    center_rf_sd_ar_60.append(center_rf_cells_sd)
    

#%%
"""
# define colors
# Blue hues
blue_hues = [
    "#B0E0E6",  # Powder Blue
    "#87CEEB",  # Sky Blue
    "#6495ED",  # Cornflower Blue
    "#1E90FF",  # Dodger Blue
    "#4169E1",  # Royal Blue
    "#0000CD",  # Medium Blue
    "#000080",  # Navy Blue
    "#191970"   # Midnight Blue
]

# Red hues
red_hues = [
    "#FFA07A",  # Light Salmon
    "#FA8072",  # Salmon
    "#F08080",  # Light Coral
    "#CD5C5C",  # Indian Red
    "#DC143C",  # Crimson
    "#B22222",  # Firebrick
    "#8B0000",  # Dark Red
    "#800000"   # Maroon
]
"""
##############
# Plot Azimuth
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_60, cmap='gray', origin='lower')
ax.set_title("Optic tectum 60 um dorsal of anterior Commissure", fontsize = 20)



for i in range(len(best_cells_pos_bd_ar_60)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_60[i].drop(columns=best_cells_pos_bd_ar_60[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_60[i].drop(columns=best_cells_pos_sd_ar_60[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar_60[i][cell_bd][0]-45)/(360-45), (center_rf_bd_ar_60[i][cell_bd][0]-45)/(360-45)]
        if center_rf_bd_ar_60[i][cell_bd][0] == 0:
            color = [1, (360-45)/(360-45) , (360-45)/(360-45)]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar_60[i][cell_sd][0]-45)/(360-45), (center_rf_sd_ar_60[i][cell_sd][0]-45)/(360-45), 1]
        if center_rf_bd_ar_60[i][cell_bd][0] == 0:
            color = [(360-45)/(360-45), (360-45)/(360-45), 1]            
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
    ##########
    """    
    for cell_bd in range(len(current_pos_bd)):
        if center_rf_bd_ar[i][cell_bd][0] == 0 or 360:
            color = red_hues[0]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0,0], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 45:
            color = red_hues[1]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.8,0.8], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 90:
            color = red_hues[2]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.7,0.7], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 135:
            color = red_hues[3]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.6,0.6], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 180:
            color = red_hues[4]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.5,0.5], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 225:
            color = red_hues[5]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.4,0.4], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 270:
            color = red_hues[6]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.2,0.2], s=5)
        if center_rf_bd_ar[i][cell_bd][0] == 315:
            color = red_hues[7]
            ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=[1,0.1,0.1], s=5)
        #print(color)
        # elif center_rf_bd_ar[i][cell_bd][0] == 360:
        #     color = red_hues[7]  # Reuse the darkest red for 360
            
        # Loop through rows and plot
        #for cell_bd in range(np.shape(current_pos_bd.iloc[cell]):
        #ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=1)
        
    for cell_sd in range(len(current_pos_sd)):
        if center_rf_sd_ar[i][cell_sd][0] == 0 or 360:
            color_sd = blue_hues[0]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0,0,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 45:
            color_sd = blue_hues[1]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.8,0.8,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 90:
            color_sd = blue_hues[2]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.7,0.7,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 135:
            color_sd = blue_hues[3]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.6,0.6,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 180:
            color_sd = blue_hues[4]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.5,0.5,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 225:
            color_sd = blue_hues[5]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.4,0.4,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 270:
            color_sd = blue_hues[6]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.2,0.2,1], s=5)
        if center_rf_sd_ar[i][cell_sd][0] == 315:
            color_sd = blue_hues[7]
            ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=[0.1,0.1,1], s=5)
        # elif center_rf_sd_ar[i][cell_sd][0] == 360:
        #     color_sd = blue_hues[7]  # Reuse the darkest red for 360

        #ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color_sd, s=5)
        """



# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols


ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()


#fig.colorbar(sax_bd, ax=ax, label='Azimuth Brightness')
#fig.colorbar(sax_sd, ax=ax, label='Azimuth Brightness')
plt.show()

#################
# Plot Elevation
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_60, cmap='gray', origin='lower')
ax.set_title("Optic tectum 60 um dorsal of anterior Commissure", fontsize = 20)


for i in range(len(best_cells_pos_bd_ar_60)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_60[i].drop(columns=best_cells_pos_bd_ar_60[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_60[i].drop(columns=best_cells_pos_sd_ar_60[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar_60[i][cell_bd][1]-31)/(151-31) ,1]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar_60[i][cell_sd][1]-43)/(139-43), 1 ,(center_rf_sd_ar_60[i][cell_sd][1]-43)/(139-43)]
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()
#plt.savefig(, kwargs)


########################################################################################################
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
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_40, cmap='gray', origin='lower')
ax.set_title("Optic tectum 40 um dorsal of anterior Commissure", fontsize = 20)



for i in range(len(best_cells_pos_bd_ar_40)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_40[i].drop(columns=best_cells_pos_bd_ar_40[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_40[i].drop(columns=best_cells_pos_sd_ar_40[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar_40[i][cell_bd][0]-45)/(360-45), (center_rf_bd_ar_40[i][cell_bd][0]-45)/(360-45)]
        if center_rf_bd_ar_40[i][cell_bd][0] == 0:
            color = [1, (360-45)/(360-45) , (360-45)/(360-45)]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar_40[i][cell_sd][0]-45)/(360-45), (center_rf_sd_ar_40[i][cell_sd][0]-45)/(360-45), 1]
        if center_rf_bd_ar_40[i][cell_bd][0] == 0:
            color = [(360-45)/(360-45), (360-45)/(360-45), 1]            
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()


#################
# Plot Elevation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_40, cmap='gray', origin='lower')
ax.set_title("Optic tectum 40 um dorsal of anterior Commissure", fontsize = 20)

for i in range(len(best_cells_pos_bd_ar_40)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_40[i].drop(columns=best_cells_pos_bd_ar_40[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_40[i].drop(columns=best_cells_pos_sd_ar_40[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar_40[i][cell_bd][1]-31)/(151-31) ,1]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar_40[i][cell_sd][1]-43)/(139-43), 1 ,(center_rf_sd_ar_40[i][cell_sd][1]-43)/(139-43)]
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()


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
    plot_name = rec.split('/')[-1]+ "_receptive_field"
        
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
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_20, cmap='gray', origin='lower')
ax.set_title("Pretectum 20 um ventral of anterior Commisure", fontsize = 20)


for i in range(len(best_cells_pos_bd_ar_20)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_20[i].drop(columns=best_cells_pos_bd_ar_20[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_20[i].drop(columns=best_cells_pos_sd_ar_20[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar_20[i][cell_bd][0]-45)/(360-45), (center_rf_bd_ar_20[i][cell_bd][0]-45)/(360-45)]
        if center_rf_bd_ar_20[i][cell_bd][0] == 0:
            color = [1, (360-45)/(360-45) , (360-45)/(360-45)]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar_20[i][cell_sd][0]-45)/(360-45), (center_rf_sd_ar_20[i][cell_sd][0]-45)/(360-45), 1]
        if center_rf_bd_ar_20[i][cell_bd][0] == 0:
            color = [(360-45)/(360-45), (360-45)/(360-45), 1]            
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)


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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()


#################
# Plot Elevation
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img_20, cmap='gray', origin='lower')
ax.set_title("Pretectum 20 um ventral of anterior Commisure", fontsize = 20)

for i in range(len(best_cells_pos_bd_ar_20)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar_20[i].drop(columns=best_cells_pos_bd_ar_20[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar_20[i].drop(columns=best_cells_pos_sd_ar_20[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar_20[i][cell_bd][1]-31)/(151-31) ,1]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar_20[i][cell_sd][1]-43)/(139-43), 1 ,(center_rf_sd_ar_20[i][cell_sd][1]-43)/(139-43)]
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()


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
    plot_name = rec.split('/')[-1]+ "_receptive_field"
        
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
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img, cmap='gray', origin='lower')
ax.set_title("Optic tectum at anterior Commissure", fontsize = 20)


for i in range(len(best_cells_pos_bd_ar)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar[i].drop(columns=best_cells_pos_bd_ar[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar[i].drop(columns=best_cells_pos_sd_ar[i].columns[0]).reset_index(drop=True)
    ##########

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar[i][cell_bd][0]-45)/(360-45), (center_rf_bd_ar[i][cell_bd][0]-45)/(360-45)]
        if center_rf_bd_ar[i][cell_bd][0] == 0:
            color = [1, (360-45)/(360-45) , (360-45)/(360-45)]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar[i][cell_sd][0]-45)/(360-45), (center_rf_sd_ar[i][cell_sd][0]-45)/(360-45), 1]
        if center_rf_bd_ar[i][cell_bd][0] == 0:
            color = [(360-45)/(360-45), (360-45)/(360-45), 1]            
        sax_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
        
# Define the range of values
values = np.linspace(45, 360, 1024)  # Generate values between 45 and 360
normalized = (values - 45) / (360 - 45)  # Normalize values between 0 and 1
# Create RGB colors based on [1, normalized, normalized]
colors = [[1, norm, norm] for norm in normalized]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Azimuth [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()


#################
# Plot Elevation
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img, cmap='gray', origin='lower')
ax.set_title("Optic tectum at anterior Commissure", fontsize = 20)

for i in range(len(best_cells_pos_bd_ar)):
    # Drop the first column and reset the index for consistent indexing
    current_pos_bd = best_cells_pos_bd_ar[i].drop(columns=best_cells_pos_bd_ar[i].columns[0]).reset_index(drop=True)
    current_pos_sd = best_cells_pos_sd_ar[i].drop(columns=best_cells_pos_sd_ar[i].columns[0]).reset_index(drop=True)
    #####

    for cell_bd in range(len(current_pos_bd)):
        color = [1, (center_rf_bd_ar[i][cell_bd][1]-31)/(151-31) ,1]
        sax_bd = ax.scatter(current_pos_bd.iloc[cell_bd]['y'], current_pos_bd.iloc[cell_bd]['x'], c=color, s=20)

        
    for cell_sd in range(len(current_pos_sd)):
        color = [(center_rf_sd_ar[i][cell_sd][1]-43)/(139-43), 1 ,(center_rf_sd_ar[i][cell_sd][1]-43)/(139-43)]
        sac_sd = ax.scatter(current_pos_sd.iloc[cell_sd]['y'], current_pos_sd.iloc[cell_sd]['x'], c=color, s=20)
        
#create custom colorbars
# Define the range of the green component
values = np.linspace(31, 151, 1024)  # Generate values between 31 and 151
greens = (values - 31) / (151 - 31)  # Normalize green component between 0 and 1
# Create RGB colors based on [1, green, 1]
colors_bd = [[1, green, 1] for green in greens]
# Create a colormap from the colors
cmap_bd = mcolors.ListedColormap(colors_bd)
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_bd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
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
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap_sd), ax=ax, label='Elevation [°]')
cbar.ax.yaxis.label.set_size(16)

# Define tick positions and labels 
tick_positions = np.linspace(0, 1, 8) 
# Normalized ticks from 0 to 1 (9 ticks) 
tick_labels = np.arange(0, 359, 45) 
# Labels from 0 to 360 in steps of 45
# Set ticks and labels 
cbar.set_ticks(tick_positions) 
cbar.set_ticklabels([f'{label}' for label in tick_labels]) # Add degree symbols

ax.set_xlabel("X Position", fontsize = 16)
ax.set_ylabel("Y Position", fontsize = 16)
ax.invert_yaxis()
plt.show()

