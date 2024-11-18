from analysis_samu import analysis_complete
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.pyplot import cm

#%%
# import data for rf
data_path = "//172.25.250.112/arrenberg_data/shared/GP_24/05112024/GP24_fish1_rec1_05112024/"

# define plot name to save each RF plot seperately
# TODO: adjust name to each cell, instead of each recording
plot_name = data_path.split('/')[-2].split('_')[-2] + '_' + data_path.split('/')[-2].split('_')[-1]

# create rf plots and return variables for anatomical maps
center_rf_cells_bd, cells_index_bd, center_rf_cells_sd, cells_index_sd = analysis_complete(data_path, plot_name)



#%% LOOP
# - loop over all files and sort 

gp_Folder = '//172.25.250.112/arrenberg_data/shared/GP_24'  # root folder for loading TIF file

###################################################################################

# make a list of all paths of all recordings (experiments and individual layers).

day_folders = [n for n in os.listdir(gp_Folder) if "112024" in n]  # todo: check if this string pattern is correct.
master_recording_paths = []

for folder in day_folders:

    recording_folders = [n for n in os.listdir(gp_Folder+f'/{folder}') if "rec" in n]  # todo: check if this string pattern is correct.
    for recording in recording_folders:
        print(recording)

        # try:
        #     tif_path = [n for n in os.listdir(f'{gp_Folder}/{folder}/{recording}') if ".TIF" in n][0]
        #     master_recording_paths.append(f'{gp_Folder}/{folder}/{recording}/{tif_path}')

        # except:
        #     print('     no tif')
        #     continue


################# Insert your filtered file lists here ############################################################

# example lists to filter for specific depths
dorsal_60_list = [n for n in master_recording_paths if "60um_dorsalAC" in n] # Alternative for more complicated pattern: do it by hand!
dorsal_40_list = [n for n in master_recording_paths if "40um_dorsalAC" in n]
ventral_20_list = [n for n in master_recording_paths if "20um_ventralAC" in n]
ventral_00_list = [n for n in master_recording_paths if "00um_ventralAC" in n]




#%% anatomical maps
best_cells_pos_bd_ar = []
best_cells_pos_sd_ar = []
for data_path in dorsal_60_list:
    # load mean img from suite2p
    ref_img = np.load(f'{data_path}/suite2p/plane0/ops.npy', allow_pickle=True).item()['meanImg']
    
    # load cell positions
    all_cells_pos = np.load(f'{data_path}/suite2p/plane0/stat.npy', allow_pickle=True)
    
    # extract positions for relevant cels
    best_cells_pos_bd = all_cells_pos[cells_index_bd]
    best_cells_pos_sd = all_cells_pos[cells_index_sd]
    
    #-----
    
    # Normalize function
    def normalize(data, min_alpha=0.2, max_alpha=1.0):
        """Normalize data to a range of alpha values."""
        min_val, max_val = min(data), max(data)
        return [min_alpha + (max_alpha - min_alpha) * (val - min_val) / (max_val - min_val) for val in data]
    
    # Prepare azimuth and elevation data
    azimuth_bd = [entry[0] for entry in center_rf_cells_bd]
    elevation_bd = [entry[1] for entry in center_rf_cells_bd]
    azimuth_sd = [entry[0] for entry in center_rf_cells_sd]
    elevation_sd = [entry[1] for entry in center_rf_cells_sd]
    
    # Normalize azimuth and elevation for alpha levels
    azimuth_alpha_bd = normalize(azimuth_bd)
    elevation_alpha_bd = normalize(elevation_bd)
    azimuth_alpha_sd = normalize(azimuth_sd)
    elevation_alpha_sd = normalize(elevation_sd)
    
    #extend lists of all recordings
    best_cells_pos_bd_ar.append(best_cells_pos_bd)
    best_cells_pos_sd_ar.append(best_cells_pos_sd)
    


# Plot Azimuth
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
cax=ax.imshow(ref_img, cmap='gray', origin='lower')
ax.set_title("Cell Positions with Azimuth Brightness")
for i in range(len(best_cells_pos_bd_ar)):
    for cell_bd, alpha in zip(best_cells_pos_bd_ar[i], azimuth_alpha_bd):
        ax.scatter(cell_bd['xpix'], cell_bd['ypix'], c='red', s=0.2, alpha=alpha)
        
    for cell_sd, alpha in zip(best_cells_pos_sd_ar[i], azimuth_alpha_sd):
        ax.scatter(cell_sd['xpix'], cell_sd['ypix'], c='blue', s=0.2, alpha=alpha)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
fig.colorbar(cax, ax=ax, label='Azimuth Brightness')
plt.show()

# Plot Elevation
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111)
cax = ax.imshow(ref_img, cmap='gray', origin='lower')
ax.set_title("Cell Positions with Elevation Brightness")

for cell_bd, alpha in zip(best_cells_pos_bd, elevation_alpha_bd):
    ax.scatter(cell_bd['xpix'], cell_bd['ypix'], c='red', s=0.2, alpha=alpha)

for cell_sd, alpha in zip(best_cells_pos_sd, elevation_alpha_sd):
    ax.scatter(cell_sd['xpix'], cell_sd['ypix'], c='blue', s=0.2, alpha=alpha)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
fig2.colorbar(cax, ax=ax, label='Elevation Brightness')
plt.show()


#%% plot code old
# ### Plot the reference image
# plt.figure(figsize=(10, 8))
# plt.imshow(ref_img, cmap='gray', origin='lower')
# plt.title("Cell Positions on Reference Image")

# # Overlay cell positions with different colors
# for cell_bd in range(len(best_cells_pos_bd)):
#     plt.scatter(best_cells_pos_bd[cell_bd]['xpix'], best_cells_pos_bd[cell_bd]['ypix'], c='red', s=0.2, alpha=0.8)
    
    
# for cell_sd in range(len(best_cells_pos_sd)):
#     plt.scatter(best_cells_pos_sd[cell_sd]['xpix'], best_cells_pos_sd[cell_sd]['ypix'], c='blue', s=0.2, alpha=0.8)



# # Add legend and labels
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.tight_layout()

# # Show the plot
# plt.show()

