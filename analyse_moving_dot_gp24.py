import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


'''FUNCTIONS FOR RF ANALYSIS'''
###This part is written for the input from one stim phase from one cell
def find_activity_peak(dff_trace):
    """
    Finds the activity peak of each trial from a dff trace for a particular phase.

    Parameters:
    dff_trace (np.array): 1D array representing dff trace for a single phase.

    Returns:
    float: The activity peak value to be used as a weight.
    """
    weight = np.max(dff_trace)
    return weight


# use AUC instead of max
def AUC_phase(mean_dff_one_cell):
    AUC_weight = np.trapz(mean_dff_one_cell)
    return AUC_weight



def calculate_rf(AUC_weights, masks):
    """
    Calculates the receptive field (RF) for a single phase.

    Parameters:
    weight (float): Weight for the RF based on activity peak.
    mask (np.array): 2D binary mask indicating where stimulus was shown.

    Returns:
    np.array: RF matrix with the weight applied.
    """
    rf_matrix_all_cells = []
    for i in range(len(masks)):
        rf_matrix = AUC_weights[i] * masks[i]
        
    rf_matrix_all_cells.append(rf_matrix)
    return rf_matrix_all_cells

###This part is written for the input from one cell
def generate_cell_rf(AUCs_all_cells, stims_list):
    """
    Generates the RF matrix for one cell.
    Loops over all trials of one cell, computes the RF matrix for each trial, 
    and superimposes them to generate the RF matrix for the entire cell.
    
    Parameters:
    dff_traces (list of np.array): List of 1D arrays, each representing a dff trace for a trial.
    stim_traces (list of np.array): List of 2D arrays, each representing stimulus positions for a trial.
    elevation_size (int): Number of elevation bins.
    azimuth_size (int): Number of azimuth bins.

    Returns:
    np.array: RF matrix for the entire cell.
    """
    rf_matrix_total = np.zeros((elevation_size, azimuth_size))
    for cell in range(np.shape(AUCs_all_cells)[0]):
        for dff_trace, stim_trace in zip(AUCs_all_cells, stim_traces):
            # AUC_weight = AUC_phase(mean_dff_one_cell)
            mask = create_stimulus_mask(stim_trace, elevation_size, azimuth_size)
            rf_matrix = calculate_rf(AUC_weight, mask)
            rf_matrix_total += rf_matrix  # Superimpose RFs
    return rf_matrix_total


def plot_rf(rf_matrix):
    """
    Plots the receptive field of one cell using a heatmap.

    Parameters:
    rf_matrix (np.array): 2D array representing the RF matrix for the cell.
    
    Returns:
    None
    """
        
    fig = plt.figure()
    ax = fig.add_subplot(111)  # Add a single subplot to the figure
    
    # Use the Axes object (`ax`) for plotting
    cax = ax.imshow(rf_matrix, origin='upper', cmap='hot', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='RF Intensity')  # Add colorbar to the figure
    
    # Set labels and title using the Axes object
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_title('Receptive Field of Cell')
    
    # Display the plot
    plt.show()


# maybe include in this function that it classifies the rf_center for this cell!! then it can be included in process all cells function
def calculate_rf_center(rf_matrix):
    # classify rf_center: elevation is given by stim where max is, azimuth classification into 8 classes 360/8?
    """
    Calculates RF center for one cell.
    Finds the coordinates of the peak RF value, representing the center of the RF.

    Parameters:
    rf_matrix (np.array): 2D array representing the RF matrix for the cell.

    Returns:
    tuple: Coordinates of the RF center (azimuth, elevation).
    """
    max_idx = np.unravel_index(np.argmax(rf_matrix), rf_matrix.shape)
    elevation_center, azimuth_center = max_idx
    return azimuth_center, elevation_center


def get_anatomical_position(cell_attributes):
    """
    Gets the anatomical position of one cell from datatable attributes.

    Parameters:
    cell_attributes (dict): Dictionary containing the attributes for the cell,
                            including anatomical position info.

    Returns:
    tuple: Coordinates (x, y) of the cell's anatomical position.
    """
    x_position = cell_attributes['x_position']
    y_position = cell_attributes['y_position']
    return x_position, y_position


'''FUNCTIONS FOR ANATOMICAL MAPS'''
### This is written for the input from one recording
def process_all_cells(dff_traces, stim_traces, cell_attributes, good_cells_flags, elevation_size, azimuth_size):
    """
    Processes all good cells in one layer, calculating RFs, RF centers, and anatomical coordinates.

    Parameters:
    dff_traces (list of lists): Nested list where each sublist is a list of dff traces for each cell.
    stim_traces (list of lists): Nested list where each sublist is a list of stim traces for each cell.
    cell_attributes (list of dicts): List of dictionaries, each containing attributes for a cell (including anatomical position).
    good_cells_flags (list of str): List of flags ("small" or "big") for each cell, indicating the stimulus type each cell responds to.
    elevation_size (int): Number of elevation bins for the RF.
    azimuth_size (int): Number of azimuth bins for the RF.

    Returns:
    dict: Dictionary with RF center positions and anatomical coordinates for each cell, separated by stimulus type.
    """
    rf_centers = {'small': [], 'big': []}
    cell_coords = {'small': [], 'big': []}

    for cell_dff_traces, cell_stim_traces, attributes, flag in zip(dff_traces, stim_traces, cell_attributes, good_cells_flags):
        rf_matrix = generate_cell_rf(cell_dff_traces, cell_stim_traces, elevation_size, azimuth_size)
        rf_center = calculate_rf_center(rf_matrix)
        anatomical_position = get_anatomical_position(attributes)

        # Store based on stimulus type
        rf_centers[flag].append(rf_center)
        cell_coords[flag].append(anatomical_position)
    return rf_centers, cell_coords


def classify_rf_centers(rf_centers, azimuth_windows, elevation_windows):
    """
    Classifies RF centers into categories based on their azimuth and elevation positions.

    Parameters:
    rf_centers (dict): Dictionary containing RF center positions for each stimulus type.
    azimuth_windows (list of tuple): List of azimuth ranges defining RF categories.
    elevation_windows (list of tuple): List of elevation ranges defining RF categories.

    Returns:
    dict: Dictionary categorizing cells based on RF center positions for each stimulus type.
    """
    categories = {'small': {}, 'big': {}}
    
    for dot_type, centers in rf_centers.items():
        for az_range, elev_range in zip(azimuth_windows, elevation_windows):
            category_label = f'Az:{az_range[0]}-{az_range[1]}, Elv:{elev_range[0]}-{elev_range[1]}'
            categories[dot_type][category_label] = []

            for az, elev in centers:
                if az_range[0] <= az <= az_range[1] and elev_range[0] <= elev <= elev_range[1]:
                    categories[dot_type][category_label].append((az, elev))
    return categories


def plot_basic_anatomical_map(cell_coords, anatomical_image):
    """
    Plots an anatomical map with cell positions color-coded by stimulus type (small or big dot).

    Parameters:
    cell_coords (dict): Dictionary containing cell coordinates for each type ("small" and "big").
    anatomical_image (np.array): 2D image array representing the anatomical layer.

    Returns:
    None
    """
    plt.imshow(anatomical_image, cmap='gray')
    for dot_type, color in zip(['small', 'big'], ['blue', 'red']):
        x_coords, y_coords = zip(*cell_coords[dot_type])
        plt.scatter(x_coords, y_coords, c=color, label=f'{dot_type.capitalize()} Dot Cells', s=10)
    
    plt.legend()
    plt.title('Anatomical Map of Small vs. Big Dot Cells')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()


def plot_advanced_anatomical_map(cell_coords, rf_categories, anatomical_image):
    """
    Plots an advanced anatomical map with cell positions color-coded by RF center category.

    Parameters:
    cell_coords (dict): Dictionary containing cell coordinates for each stimulus type ("small" and "big").
    rf_categories (dict): Dictionary categorizing RF centers for each stimulus type.
    anatomical_image (np.array): 2D image array representing the anatomical layer.

    Returns:
    None
    """
    plt.imshow(anatomical_image, cmap='gray')

    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, (dot_type, categories) in enumerate(rf_categories.items()):
        for j, (category, centers) in enumerate(categories.items()):
            color = colors[(i * len(categories) + j) % len(colors)]  # Rotate colors
            x_coords, y_coords = zip(*[cell_coords[dot_type][k] for k, center in enumerate(centers)])
            plt.scatter(x_coords, y_coords, color=color, label=f'{dot_type.capitalize()} - {category}', s=10)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Advanced Anatomical Map with RF Center Categories')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()
    
#%% Functions were actually using!!!!!!!!!!
# This function calculates the masks for each stimulus phase and multiplis it with the corresponding weight. It does this for each cell. 
# It returns 2 arrays, with the RFs for each cell that reacts to the big/small stimulus.
def create_stimulus_mask(stims_list, AUCs_cell):
    """
    Creates a binary mask based on the stimulus trace for a particular phase, 
    showing where the stimulus was presented in elevation and azimuth..

    Parameters:
    stim_trace (np.array): 1D array representing the stimulus presence for a phase.
    elevation_size (int): Number of elevation bins in the mask (rows).
    azimuth_size (int): Number of azimuth bins in the mask (columns).

    Returns:
    np.array: 2D binary mask (elevation x azimuth) with 1s where stimulus is shown.
    """
    masks_all_phases_big = []
    masks_all_phases_small = []
    #mask = np.zeros((180, 360))
    #conds_elevs = [[135, 105], [105, 75], [75, 45], [105, 75]]
    for phase in range(len(AUCs_cell)):
        mask = np.zeros((180, 360))

        stim_phase = stims_list[phase]
        
        dot_size = stim_phase[0]
        azim_range = np.arange(stim_phase[1]-np.ceil(dot_size/2).astype(int), stim_phase[1]+180+np.ceil(dot_size/2).astype(int), 1)
        elev_range = np.arange(90+stim_phase[2]+np.ceil(dot_size/2).astype(int), 
                               90+stim_phase[2]-np.floor(dot_size/2).astype(int), -1)

        if dot_size == 30:
            #mask[elev_range, azim_range] = 1
            mask[np.ix_(elev_range, azim_range)] = 1
            masks_all_phases_big.append(mask)
        else:
            #mask[elev_range, azim_range] = 1
            mask[np.ix_(elev_range, azim_range)] = 1
            masks_all_phases_small.append(mask)
    
    
    # now do the fucking weighted mask
    rf_matrix_all_phases_big = []
    rf_matrix_all_phases_small = []
    for i in range(len(masks_all_phases_big)):
        rf_matrix = AUCs_cell[i] * masks_all_phases_big[i]
        rf_matrix_all_phases_big.append(rf_matrix)
        rf_matrix_total_big = np.dstack(rf_matrix_all_phases_big)
        rf_matrix_total_avg_big = np.mean(rf_matrix_total_big, axis=2)
    
    for i in range(len(masks_all_phases_small)):
        rf_matrix = AUCs_cell[i] * masks_all_phases_small[i]
        rf_matrix_all_phases_small.append(rf_matrix)
        rf_matrix_total_small = np.dstack(rf_matrix_all_phases_small)
        rf_matrix_total_avg_small = np.mean(rf_matrix_total_small, axis=2)
    return rf_matrix_total_avg_big, rf_matrix_total_avg_small


# this function returns the AUC values for all cells
def get_AUCs(mean_dff_best_cells, new_inds, num_stim):
    # calculate AUC:
    AUCs_all_cells = np.zeros((np.shape(mean_dff_best_cells)[0], num_stim))
    for cell in range(np.shape(AUCs_all_cells)[0]):
        AUCs_cell = []
        for d in range(len(new_inds)):
            for window in range(len(new_inds[d])):
                for elevation in range(len(new_inds[d, window])):
                    if not np.isnan(new_inds[d, window, elevation]).any():
                        AUC = np.trapz(mean_dff_best_cells[cell, int(new_inds[d, window, elevation, 0]):
                                                           int(new_inds[d, window, elevation, 1])])
                        AUCs_cell.append(AUC)

        AUCs_all_cells[cell, : ] = AUCs_cell
    return AUCs_all_cells
#%% 
# get classification of rfs from weights/activity of cells!
# greyscale for how much cell likes certain stim and then color code for big/small dot!
# histograms (per rec/over all cells??)
# include RF center position calc in rf center value calc??

# Giulia talk:
    # anatomical map: colormap not white in middle
    # one map for elevation, one for azimuth
    # per map: two colors, one for each dot size
    # histogram for anatomical over x and y axes
    # use inkscape (e.g. to insert histograms)

#%% trash code
        # if stims_list[i][1] == -180:
        #     elev = np.arange(conds_elevs[0][0], conds_elevs[0][1], 1) 
        # elif stims_list[i][1] == -90:
        #     elev = np.arange(conds_elevs[1][0], conds_elevs[1][1], 1) 
        # elif stims_list[i][1]== 0:
        #     elev = np.arange(conds_elevs[2][0], conds_elevs[2][1], 1) 
        # elif stims_list[i][1]== 90:
        #     elev = np.arange(conds_elevs[3][0], conds_elevs[3][1], 1) 
        

    # if dot_size == 30:
    #     rf_matrix_all_phases_big = []
    #     for i in range(len(masks_all_phases)):
    #         rf_matrix = AUCs_cell[i] * masks_all_phases[i]
    #         rf_matrix_all_phases_big.append(rf_matrix)
        
    #     # superimpose all phases
    #     rf_matrix_total = np.zeros((180, 360))
    #     # for j in range(len(rf_matrix_all_phases)):
    #     #     rf_matrix_total += rf_matrix_all_phases[j]
        
    #     # for rf_matrix in rf_matrix_all_phases:
    #     #     rf_matrix_total += rf_matrix 
    #     rf_matrix_total = np.dstack(rf_matrix_all_phases_big)
    #     rf_matrix_total_avg_big = np.mean(rf_matrix_total, axis=2)
    # else:
    #     rf_matrix_all_phases_small = []
    #     for i in range(len(masks_all_phases)):
    #         rf_matrix = AUCs_cell[i] * masks_all_phases[i]
    #         rf_matrix_all_phases_small.append(rf_matrix)
        
    #     # superimpose all phases
    #     rf_matrix_total = np.zeros((180, 360))
    #     # for j in range(len(rf_matrix_all_phases)):
    #     #     rf_matrix_total += rf_matrix_all_phases[j]
        
    #     # for rf_matrix in rf_matrix_all_phases:
    #     #     rf_matrix_total += rf_matrix 
    #     rf_matrix_total = np.dstack(rf_matrix_all_phases_small)
    #     rf_matrix_total_avg_small = np.mean(rf_matrix_total, axis=2)
            


