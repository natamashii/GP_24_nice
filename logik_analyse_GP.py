'''
This script is a work in progress. 
It starts after having sorted the raw data into cells that respond to the moving dot stimulus in general (goodCells), 
and to which version they respond (goodCells_small/goodCells_big).

The objective of this script is to identify and plot receptive fields 
for each trial and phase of the moving dot stimulus for each cell. 
This information is stored to find RF centers and plot an anatomical map.
(Cell in this script relates to the biological structure, not code structure.)

Receptive Fields (one cell):
Step 1: function that finds activity peak of each trial
    - takes dff traces (per phase)
    - find activity peak (max)
    - activity peak acts as weight in further analysis
    - returns weight
Step 2: function that creates stimulus mask
    - takes stimulus trace (per phase) 
    - creates binary mask with ones at elevation/azimuth where stimulus is shown
    - returns stimulus mask (y axis: elevation in deg, x axis: azimuth in deg, 0 azimuth/0 elevation in middle of matrix)
Step 3: function to calculate RF
    - takes weight and mask
    - multiply weight with mask to create RF for one trial of one cell
    - returns matrix of RF (y axis: elevation in deg, x axis: azimuth in deg, 0 azimuth/0 elevation in middle of matrix)
Step 4: function that loops over all trials of one cell
    - takes dff and stim traces of all trials of one cell
    - loops over all trials
    - calls function 3 within this loop
    - creates structure of some kind that stores RF matrices of all trials
    - superposition all of these RF matrices to generate RF matrix for whole cell
    - returns RF matrix for one cell
Step 5: function that plots RF of one cell
    - takes RF matrix for one cell 
    - plots the receptive field
    - returns nothing
Step 6: function that calculates center of RF for one cell (this is relevant for a more detailed anatomical map)
    - takes output of function 4: RF matrix of one cell
    - calculates max value of matrix
    - extracts elevation and azimuth of max value to get center position
    - returns RF center position (2 coordinates: azimuth and elevation in deg) 
Step 7: function that gets anatomical position for one cell
    - takes datatable.attributes (check exact path to this info!!)
    - accesses anatomic position information in this structure
    - stores x and y positions of cell (values indicate position of cell in pixel of suite2p image)
    - returns cell coordinates









Anatomical Maps (all cells one layer one fish):
Step 8: function that loops over all goodCells of one layer/recording
    - takes dff and stim traces of all trials of all goodCells of one rec (they should be flagged if they respond to small or big dot)
    - loops over all cells within this structure
    - calls function 4 within this loop -> RF matrix for current cell
    - calls function 6 within this loop -> RF center position for current cell
    - calls function 7 within this loop -> coordinates of current cell
    - if structure; store values according to big/small dot flag
    - stores center position for each cells RF (flagged)
    - stores coordinates for each cell (flagged)
    - returns structure of cell coordinates flagged for small/big dot cells, 
    returns structure of RF center positions flagged for small/big dot cells
Step 9: function that plots basic anatomical map (small vs big dot cells) for one layer of one animal
    - takes structure with cell coordinates (output of step 8), 
    - takes anatomical picture of the layer of rec
    - plots anatomical picture
    - plots position of each cell onto anatomical picture 
    - one color for big dot cells, another color for small dot cells
    - returns nothing
Step 10: function that classifies RF center positions in categories
        -> ask how to define categories? same as how stim windows are defined??
    - takes structure with RF center positions (output of function 8)
    - defines relevant windows for categorizing receptive fields
    - for loop over all cells of this layer
    - read out RF center of this cell
    - check in which window these coordinates lie
    - sorts cells into correct category 
    - returns cells sorted per RF position
Step 11: function that plots advanced anatomical map (color gradient for cells RF center)
    - takes structure with cell coordinates (output of function 8),
    - takes structure with cells sorted per their RF center positions
    - takes anatomical picture of the layer of rec
    - plots anatomical picture
    - plots position of each cell onto anatomical picture
    - one color for each of the categories of RF center
    - returns nothing        



Levels of complexity:
1: one phase    
2: one cell
3: one recording/layer

Todo:
4: function that loops over all files
    - takes filename and path to directory
    - takes necesary variables to call function for one file
    - imports and loads all files and sorts files into layers 
    - parses filename (name has to include '40um_dorsal' for example)
    - loads files that include this in their filename
    - calls function for one file
    - define function that plots same layer of all rec days into one anatomical map (averaging?)
    - do this for all layers that were recorded in 
'''


