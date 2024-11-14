import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# TODO: when at point that everything works, then DO DOCUMENTATION!!!!
data_path = 'Z:/shared/Carina Thomas/_RecordedData/06_09_24/ct_rec5_24_09_06/'

############## load in data ##########################################
# vxpy stuff
camera = h5py.File(data_path+'Camera.hdf5')
display = h5py.File(data_path+'Display.hdf5')
io = h5py.File(data_path+'Io.hdf5')

# suite2p
F = np.load(data_path+'suite2p/plane0/F.npy')
ops = np.load(data_path+'suite2p/plane0/ops.npy', allow_pickle=True).item()
stat = np.load(data_path+'suite2p/plane0/stat.npy', allow_pickle=True)
# stat needs loop through cells. see keys: stat[0].keys()
#stat_cells = []
cell_pos = []
for i in range(len(stat)):
    cell_pos.append(tuple(stat[i]['med']))
    #cell_x_pos[i] = stat[i]['med'][0]
    #cell_y_pos[i] = stat[i]['med'][1]


####################################################################################
# calculate delta F / F (method from Giulias MATLAB scripten)
####################################################################################
dff = np.zeros(np.shape(F))
for i, trace in enumerate(F):
    F0 = np.median(trace)
    dff[i,:] = [(x-F0)/F0 for x in trace]
    # mit dieser Methode ist das ergebnis immer quasi 0...
        # literatur?
        # wie machen das andere leute? Saccadic suppression cells sind genau so.
    # TODO: Teste verschiedene möglichkeiten für F0 - Moving median ev. besser.

# Interpolate frametimes - bec. of dropouts from vxPy.
numberOfFrames = np.shape(F)[1]
frameChanges = np.diff(io['di_frame_sync'][:].flatten()) # Ca-frames from framesynk signal
frameChanges_time = io['di_frame_sync_time'][:-1][frameChanges].flatten()

frametimes = np.linspace(frameChanges_time[0], frameChanges_time[-1], numberOfFrames + 1)
# frametimes um 1 größer number of frames: letztes frame weglassen, da es nicht fertig gemacht wurde und nicht als bild abgespeichert wurde.
if len(frametimes) != numberOfFrames:
    if len(frametimes) > numberOfFrames:
        if len(frametimes)-numberOfFrames > 1: # mehr als 1 frame unterschied
            print(str(len(frametimes)) + ' - frametimes')
            print(str(numberOfFrames) + ' - numberOfFrames')
            raise ValueError('Too many frames detected. CHECK IT!')
        frametimes = frametimes[:-1]
    else:
        raise ValueError('frametimes < numberOfFrames. CHECK IT!')

# TODO: Verwende dise Auskommentierte Section irgendwie - notfall plot bei Problemen.
'''
# visualisation plot:
fig, ax = plt.subplots()
plt.plot(io['di_frame_sync_time'][:],io['di_frame_sync'][:])
plt.xlabel('Time [con]')
plt.ylabel('Framesync signal')
plt.scatter(frameChanges_time, np.ones(np.shape(frameChanges_time)), color='k')
plt.scatter(frameChanges_time_ori[:-1][frameSyncProblems], np.ones(np.shape(frameChanges_time_ori[:-1][frameSyncProblems])).flatten()+0.001, color='r')
plt.scatter(frameSyncProblems_fixed, np.ones(np.shape(frameSyncProblems_fixed))+0.002, color='g')
plt.scatter(frametimes, np.ones(np.shape(frametimes))+0.0015, color='c')
for phase_no in display:
    if 'phase_no' in phase_no:
        startTime = display[phase_no].attrs['__start_time']
        if 'spherical_uniform_background' in display[phase_no].attrs['__visual_module']:   # movement
            col = 'k'
        if 'twoEllipses_test' in display[phase_no].attrs['__visual_module']:               # visual
            col = 'r'
        if 'spherical_grating' in display[phase_no].attrs['__visual_module']:  # grating
            col = 'g'
        ax.axvline(startTime, color=col)
        plt.text(startTime,0.8, phase_no, rotation=90, color=col)
plt.show()

#TODO: cutof of last frame. The frame started but did not finish and therefore is not saved to Tiff file.
'''


##########################################################################################
# collect information of protocol and phases.
##########################################################################################
# protocol information
protocol_phase_colors = []
for prot in range(0,sum(['protocol' in p for p in list(display.keys())])):
    protocol_name = display['protocol' + str(prot)].attrs['__protocol_name']
    this_protocol_colors = [protocol_name.split('_')[-1]] * display['protocol' + str(prot)].attrs['__target_phase_count']
    protocol_phase_colors = protocol_phase_colors + this_protocol_colors

# phase information
occlusion_zero_pos = io['phase0'].attrs['sutter_end_pos_xyz']
all_phases_data = [None]*len(protocol_phase_colors)#np.zeros((len(protocol_phase_colors)))
for phase_no in range(0, sum(['phase' in p for p in list(display.keys())])):
    # general stuff for all phases
    start_time = display['phase'+str(phase_no)].attrs['__start_time']
    end_time = display['phase'+str(phase_no)].attrs['__start_time'] + display['phase'+str(phase_no)].attrs['__target_duration']
    all_phases_data[phase_no] = {'phase_no': phase_no,
                                 'movement_phase': False,
                                 'start_time': start_time,
                                 'end_time': end_time,
                                 'start_frame': min(enumerate(frametimes), key=lambda x: abs(x[1]-start_time))[0],
                                 'end_frame': min(enumerate(frametimes), key=lambda x: abs(x[1]-end_time))[0],
                                 'visual_name': display['phase'+str(phase_no)].attrs['__visual_name'],
                                 'stimulus_color': protocol_phase_colors[phase_no],
                                 'occlusion_pos': [],
                                 'el_mirrored': [],
                                 'el_azimut_pos': []
                                 # neg -> single dot on right?/other site.
                                 }
    if 'UniformBackground' in all_phases_data[phase_no]['visual_name']:  # if there is any movement. No micromanipulator stuff saved during visuals
        all_phases_data[phase_no]['movement_phase'] = True
        #sutter_start_pos = io['phase'+str(phase_no)].attrs['sutter_start_pos']
        #sutter_end_pos = io['phase' + str(phase_no)].attrs['sutter_end_pos']
        #sutter_movementX = sutter_start_pos[0] - sutter_end_pos[0]
        #sutter_movementY = sutter_start_pos[1]-sutter_end_pos[1]
        movement = occlusion_zero_pos - io['phase' + str(phase_no)].attrs['sutter_end_pos_xyz']
        #print(movement)
        if movement[0] == 0 and movement[1] == 0:
            occlusion_pos = 'middle'
        elif movement[0] > 0:
            occlusion_pos = 'left'
            # TODO: check if left and right are actually correct.
        elif movement[0] < 0:
            occlusion_pos = 'right'
        elif movement[1] < 0:
            occlusion_pos = 'down'
        else:
            raise ValueError('Something wrong with occlusion movement calculation.')

        all_phases_data[phase_no]['occlusion_pos'] = occlusion_pos
        #TODO: add cases for actual movement frames of micromanipulator - I am curious.
             #'occlusion_mov_start_time':
             #'occlusion_mov_start_frame':
             #'occlusion_mov_end_time':
             #'occlusion_mov_end_frame':
    elif 'TwoEllipses' in all_phases_data[phase_no]['visual_name']:
        all_phases_data[phase_no]['occlusion_pos'] = all_phases_data[phase_no-1]['occlusion_pos']  # visual phases have same position as previous movement position
        all_phases_data[phase_no]['el_mirrored'] = display['phase' + str(phase_no)].attrs['el_mirror'][0]  # 0 / 1 ???
        all_phases_data[phase_no]['el_azimut_pos'] = display['phase' + str(phase_no)].attrs['pos_azimut_angle'][0],  # neg -> single dot on right?/other site.
    elif 'SphericalBlackWhiteGrating' in all_phases_data[phase_no]['visual_name']:
        all_phases_data[phase_no]['occlusion_pos'] = 'down'
        all_phases_data[phase_no]['angular_period'] = display['phase'+str(phase_no)].attrs['angular_period'][0]
        all_phases_data[phase_no]['angular_velocity'] = display['phase' + str(phase_no)].attrs['angular_velocity'][0]
    else:
        raise ValueError('Someting wrong with visual name. Visual name not found. "'+all_phases_data[phase_no]['visual_name']+'"')


##########################################################################################
# make regressors for all conditions
##########################################################################################
all_conditions_dict = {
    # 1. Filter Conditions
    # blue
    'blue_2dots_OCmiddle':[],
    'blue_2dots_OCleft':[],
    'blue_2dots_OCright':[],
    'blue_2dots_OCdown':[],
    'blue_1dot_left_OCleft':[],
    'blue_1dot_left_OCright':[],
    'blue_1dot_left_OCmiddle':[],
    'blue_1dot_left_OCdown':[],
    'blue_1dot_right_OCleft':[],
    'blue_1dot_right_OCright':[],
    'blue_1dot_right_OCmiddle':[],
    'blue_1dot_right_OCdown':[],
    # red
    'red_2dots_OCmiddle':[],
    'red_2dots_OCleft':[],
    'red_2dots_OCright':[],
    'red_2dots_OCdown':[],
    'red_1dot_left_OCleft':[],
    'red_1dot_left_OCright':[],
    'red_1dot_left_OCmiddle':[],
    'red_1dot_left_OCdown':[],
    'red_1dot_right_OCleft':[],
    'red_1dot_right_OCright':[],
    'red_1dot_right_OCmiddle':[],
    'red_1dot_right_OCdown':[],
    # special cases
    'movement':[],
    'blue_sphericalGrating_static':[],
    'blue_sphericalGrating_moving':[],
    'red_sphericalGrating_static':[],
    'red_sphericalGrating_moving':[]
}
for i, dic in enumerate(all_phases_data):
    if dic['movement_phase']:
        #all_phases_movement.append(i)
        all_conditions_dict['movement'].append(i)
    elif 'SphericalBlackWhiteGrating' in dic['visual_name'] and dic['stimulus_color'] == 'blue':
        if dic['angular_velocity'] == 0:
            all_conditions_dict['blue_sphericalGrating_static'].append(i)
        else:
            all_conditions_dict['blue_sphericalGrating_moving'].append(i)
    elif 'SphericalBlackWhiteGrating' in dic['visual_name'] and dic['stimulus_color'] == 'red':
        if dic['angular_velocity'] == 0:
            all_conditions_dict['red_sphericalGrating_static'].append(i)
        else:
            all_conditions_dict['red_sphericalGrating_moving'].append(i)
    elif 'TwoEllipses' in dic['visual_name']: # fiter: ellipse-phases
        if dic['stimulus_color'] == 'blue':
            if dic['el_mirrored'] == 1:  # 2 dots
                if dic['occlusion_pos'] == 'middle':
                    all_conditions_dict['blue_2dots_OCmiddle'].append(i)
                elif dic['occlusion_pos'] == 'left':
                    all_conditions_dict['blue_2dots_OCleft'].append(i)
                elif dic['occlusion_pos'] == 'right':
                    all_conditions_dict['blue_2dots_OCright'].append(i)
                elif dic['occlusion_pos'] == 'down':
                    all_conditions_dict['blue_2dots_OCdown'].append(i)
            else: #  1 dot
                if dic['el_azimut_pos'][0] > 0: # dot left ???
                    if dic['occlusion_pos'] == 'middle':
                        all_conditions_dict['blue_1dot_left_OCmiddle'].append(i)
                    elif dic['occlusion_pos'] == 'left':
                        all_conditions_dict['blue_1dot_left_OCleft'].append(i)
                    elif dic['occlusion_pos'] == 'right':
                        all_conditions_dict['blue_1dot_left_OCright'].append(i)
                    elif dic['occlusion_pos'] == 'down':
                        all_conditions_dict['blue_1dot_left_OCdown'].append(i)
                else: # dot right
                    if dic['occlusion_pos'] == 'middle':
                        all_conditions_dict['blue_1dot_right_OCmiddle'].append(i)
                    elif dic['occlusion_pos'] == 'left':
                        all_conditions_dict['blue_1dot_right_OCleft'].append(i)
                    elif dic['occlusion_pos'] == 'right':
                        all_conditions_dict['blue_1dot_right_OCright'].append(i)
                    elif dic['occlusion_pos'] == 'down':
                        all_conditions_dict['blue_1dot_right_OCdown'].append(i)
        elif dic['stimulus_color'] == 'red':
            if dic['el_mirrored'] == 1:  # 2 dots
                if dic['occlusion_pos'] == 'middle':
                    all_conditions_dict['red_2dots_OCmiddle'].append(i)
                elif dic['occlusion_pos'] == 'left':
                    all_conditions_dict['red_2dots_OCleft'].append(i)
                elif dic['occlusion_pos'] == 'right':
                    all_conditions_dict['red_2dots_OCright'].append(i)
                elif dic['occlusion_pos'] == 'down':
                    all_conditions_dict['red_2dots_OCdown'].append(i)
            else: #  1 dot
                if dic['el_azimut_pos'][0] > 0: # dot left ???
                    if dic['occlusion_pos'] == 'middle':
                        all_conditions_dict['red_1dot_left_OCmiddle'].append(i)
                    elif dic['occlusion_pos'] == 'left':
                        all_conditions_dict['red_1dot_left_OCleft'].append(i)
                    elif dic['occlusion_pos'] == 'right':
                        all_conditions_dict['red_1dot_left_OCright'].append(i)
                    elif dic['occlusion_pos'] == 'down':
                        all_conditions_dict['red_1dot_left_OCdown'].append(i)
                else: # dot right
                    if dic['occlusion_pos'] == 'middle':
                        all_conditions_dict['red_1dot_right_OCmiddle'].append(i)
                    elif dic['occlusion_pos'] == 'left':
                        all_conditions_dict['red_1dot_right_OCleft'].append(i)
                    elif dic['occlusion_pos'] == 'right':
                        all_conditions_dict['red_1dot_right_OCright'].append(i)
                    elif dic['occlusion_pos'] == 'down':
                        all_conditions_dict['red_1dot_right_OCdown'].append(i)

# Build regressor (convolved)
def CIRF(regressor, n_ca_frames):
    tau = 1.6
    time = np.arange(0, n_ca_frames)
    exp = np.exp(-time / tau)
    reg_conv = np.convolve(regressor, exp)
    reg_conv = reg_conv[:n_ca_frames]
    return reg_conv

regressor_win_buffer = [1,10] # how many frames adding before start (1) and after (2) end of regressor
all_regressors = np.zeros([len(all_conditions_dict),numberOfFrames])
all_regressors_conv = np.zeros(np.shape(all_regressors))
all_ignore_idx = []
all_zscoreBL = []
for i, thisConditions in enumerate(all_conditions_dict.values()): # loop conditions
    if len(thisConditions) == 0: # check if condition exists
        all_ignore_idx.append([])
        all_zscoreBL.append([])
        continue
    regressor_thisCon = np.zeros(len(frametimes))
    ignore_idx_thisCon = np.ones(len(frametimes))
    z_scoreBL_thisCon = []
    for j in thisConditions: # loop phases of this condition
        regressor_thisCon[all_phases_data[j]['start_frame']: all_phases_data[j]['end_frame']] = 1
        ignore_idx_thisCon[all_phases_data[j]['start_frame'] - regressor_win_buffer[0]: all_phases_data[j]['end_frame'] + regressor_win_buffer[1]] = 0 # Hardcoded: this might be not the best way of doing it....
        z_scoreBL_thisCon.append(np.arange(all_phases_data[j]['start_frame'] - 11, all_phases_data[j]['start_frame']-1)) # use last 10 frames of each pause as baseline for zscore. ! exclude start freame (reason start-1 and +9)
    ignore_idx_thisCon = np.where(ignore_idx_thisCon == 1)
    all_regressors[i,:] = regressor_thisCon  # raw version of regressors
    all_regressors_conv[i,:] = CIRF(regressor_thisCon, len(frametimes)) # regressor convolution with CIRF
    all_ignore_idx.append(ignore_idx_thisCon)  # needed later for correlation
    #all_ignore_idx.append(np.where(all_regressors_conv[i,:] < 0.001)) # soft-coded version. problem, includes first frame
    all_zscoreBL.append(z_scoreBL_thisCon)

'''
# test plot for visualizing regressors (convovled) 
plt.figure()
plt.title('Overview of regressors of phases - some excluded. Example ignore index')
for i,t in enumerate(all_regressors_conv):
    # ignore: movement phases, static grating phases
    if (i == list(all_conditions_dict.keys()).index('movement') or
            i == list(all_conditions_dict.keys()).index('blue_sphericalGrating_static') or
            i == list(all_conditions_dict.keys()).index('red_sphericalGrating_static')):
        continue
    plt.plot(t)
plt.scatter(all_ignore_idx[0], np.zeros(len(all_ignore_idx[0][0]))) # for first phase

plt.figure()
plt.title('Overview of regressors of phases - MOVEMENT PHASE')
for i,t in enumerate(all_regressors_conv):
    if i == list(all_conditions_dict.keys()).index('movement'):
        plt.plot(t)
plt.scatter(all_ignore_idx[24], np.zeros(len(all_ignore_idx[24][0]))) # for movement phases
'''


####################################################################################################################
# Correltation dff mit gefilterter bedingung/regressor / zscore -> to baseline
####################################################################################################################
# calculate correlation of all cells with all convolved regressors. Only do correlation where stimulus is shown (all_ignore_idx).
# TODO: find a nice way of storing this data.... pandas? hdf5? dafarame?
all_cells_all_cond_corr = np.zeros([np.shape(dff)[0], np.shape(all_regressors_conv)[0]]) # simple correlation -> 1: cells, 2:conditions
all_cells_all_cond_zScore = list(range(np.shape(dff)[0]))
for c, cell in enumerate(dff): # loop over cells
    all_zscores_thisCell = list(range(len(all_regressors_conv)))
    condition_startFrames = list(range(len(all_regressors_conv)))
    condition_endFrames = list(range(len(all_regressors_conv)))
    for con, condition in enumerate(all_regressors_conv): # loop over conditions / stimuli
        if len(all_ignore_idx[con]) == 0: # skipp conditions that are not shown
            all_cells_all_cond_corr[c,con] = np.nan # mark conditions that are not shown
            all_zscores_thisCell[con] = []
            condition_startFrames[con] = []
            condition_endFrames[con] = []
            continue

        # make a mask for when this stimulus is not shown (ingore_idx)
        mask = np.full(len(cell), True)
        mask[all_ignore_idx[con]] = False
        # do correlation whit mask applied
        all_cells_all_cond_corr[c,con] = np.corrcoef(cell[mask], condition[mask])[0,1]

        # get start and end points of every iteration of stimulus in mask -> for autocorrelation and zscore
        mask_starts = np.where(np.diff(mask))[0][0::2]
        mask_ends = np.where(np.diff(mask))[0][1::2]
        condition_startFrames[con] = np.array(mask_starts)+1  # save condition starts and ends for plotting
        condition_endFrames[con] = np.array(mask_ends)+1
        # TODO: add autocorrelation here: (ignore cells that only sometimes work with stimulus)



        # Z SCORE:
        # get frames of phase: -> from mask, calculate mean + std of baseline for this condition (THIS USES NOT STIMULUS FRAMES!!!)
        bl_mean = np.mean(dff[c,all_zscoreBL[con]])  # this is for mean over all baseline snippets
        bl_std = np.std(dff[c,all_zscoreBL[con]])

        # calculate zscore
        z_scores_thisCon = [] # 3 arrays mit individuellen werten
        z_scores_frames_thisCon = []
        for inst in range(len(mask_starts)):       # loop over stimulus repetitions
            frames_this_stimulus = np.arange(mask_starts[inst]+1,mask_ends[inst]+1)
            z_score = np.zeros(np.shape(frames_this_stimulus))
            #bl_mean = np.mean(dff[c,all_zscoreBL[con][inst]])  # for snippet-individual bl
            #bl_std = np.std(dff[c,all_zscoreBL[con][inst]]) # for snippet-individual bl
            for frame, frame_thisStim in enumerate(frames_this_stimulus):  # loop over all dffs in condition instance
                z_score[frame] = (dff[c,frame_thisStim] - bl_mean) / bl_std
            # alles abspeichern
            z_scores_thisCon.append(z_score)
            z_scores_frames_thisCon.append(frames_this_stimulus)

        all_zscores_thisCell[con] =  z_scores_thisCon

        '''
        # control plot: This condition all z_scores
        conditions_key_list = list(all_conditions_dict.keys())
        all_zscores = np.array([])
        for x in range(len(z_scores_thisCon)):
            all_zscores = np.append(all_zscores, z_scores_thisCon[x])
        if np.mean(all_zscores) > 1:
            plt.figure()
            plt.title('Cell: ' + str(c) + ' - ZSCORE: ' + str(np.round(np.mean(all_zscores), 3)) + ' - Condition: ' + conditions_key_list[con])
            plt.axhline(y=np.mean(all_zscores))
            plt.axhline(y=0, linestyle='--', color=[0.5,0.5,0.5])
            plt.ylabel('zScore')
            plt.xlabel('frames')
            plt.ylim([-2,7])
            for i in range(len(z_scores_thisCon)):
                this_phaseFrames = z_scores_frames_thisCon[i]
                this_phase_zScore = z_scores_thisCon[i]
                x_values = np.arange(0, len(this_phase_zScore))  # z_scores
                plt.plot(x_values, all_regressors_conv[con, this_phaseFrames])  # regressor of this stimulus presentation
                plt.plot(x_values, this_phase_zScore)  # dff of this stimulus presentation
        
        ####################################################################################################################
           
        # control plot: This condition all dffs
        conditions_key_list = list(all_conditions_dict.keys())
        plt.figure()
        plt.title('Cell: ' + str(c) + ' - corr: ' +str(np.round(all_cells_all_cond_corr[c,con],2)) + ' - Condition: ' + conditions_key_list[con])
        plt.ylabel('dff')
        plt.xlabel('frames')
        for i in all_conditions_dict[conditions_key_list[con]]:
            this_phaseFrames = np.arange(all_phases_data[i]['start_frame'] - regressor_win_buffer[0], all_phases_data[i]['end_frame'] + regressor_win_buffer[1])
            this_phase_dff = cell[this_phaseFrames] # dff values
            x_values = np.arange(0, len(this_phase_dff))
            plt.plot(x_values, all_regressors_conv[con,this_phaseFrames]) # regressor of this stimulus presentation
            plt.plot(x_values, this_phase_dff) # dff of this stimulus presentation
        '''

    all_cells_all_cond_zScore[c] = all_zscores_thisCell
    # need some saving with all the frames ????????

#################################################################################################################
# make some overview plots.
#################################################################################################################
# 7. plot cells that correlate significantly
#(8. make some anatomical marker where they should go)

# ----> cells that reakt to moving dot

# Make some example plots (good cells)

# FINAL RESULT: RASTERPLOT!!!!

'''
# pixelplot correlation
fig, ax = plt.subplots()
plt.title('Correlations of Cells with regressors of all stimuli')
plt.xlabel('Conditions')
plt.ylabel('Cells (Sorted location of hightest correltaion)')

# cut out conditions that are not shown and make custom labels of conditions
all_cells_all_cond_corr_cleaned = np.delete(all_cells_all_cond_corr, np.where(np.isnan(all_cells_all_cond_corr[0])), axis=1)
all_cells_all_cond_corr_cleaned = all_cells_all_cond_corr_cleaned[np.argsort(np.argmax(all_cells_all_cond_corr_cleaned, 1)),:]

conditions_key_list = np.array(list(all_conditions_dict.keys()))
conditions_key_list_valid = np.delete(conditions_key_list,np.where(np.isnan(all_cells_all_cond_corr[0])), axis=0)
plt.xticks(range(np.shape(all_cells_all_cond_corr_cleaned)[1]), conditions_key_list_valid, rotation=90)

plt.imshow(all_cells_all_cond_corr_cleaned, cmap='seismic')
plt.colorbar()
plt.clim(-1,1)
plt.xlim([-2,len(conditions_key_list_valid)+1])
'''


# pixelplot z_score (very hacky way)
all_cell_zScore = np.array([])
cell_highestActivity_condIDX = []
for c,cell in enumerate(all_cells_all_cond_zScore):
    all_conditions = np.array([])
    xtick_list = []
    av_zScore_perCondition = []
    for con, condition in enumerate(cell):
        if len(condition) > 0:  # only do valid conditions
            all_rep = np.array([])
            for rep in condition:
                if len(rep) < 33:  # this is super random
                    rep = np.append(rep,0)
                all_rep = np.concatenate([all_rep, rep])
            # condition start indices for plotting. Marks the end of the conditions
            if con == 0:
                xtick_list.append(len(all_rep))
            else:
                xtick_list.append(xtick_list[-1]+len(all_rep))
            all_conditions = np.concatenate([all_conditions,all_rep])
            av_zScore_perCondition.append(np.mean(all_rep))
        # stack everything together
    if c == 0:
        all_cell_zScore = np.concatenate([all_cell_zScore, all_conditions])
    else:
        all_cell_zScore = np.vstack([all_cell_zScore, all_conditions])
    cell_highestActivity_condIDX.append(np.where(av_zScore_perCondition==np.max(av_zScore_perCondition))[0][0])

# pixelplot dff
hightest_dff = []
all_cell_dff = np.array([])
for c, cell in enumerate(dff):
    dff_allConditions = np.array([])
    av_dff_perCondition = []
    for con in range(len(condition_startFrames)):
        condition_stats = condition_startFrames[con]
        condition_end = condition_endFrames[con]
        dff_allRep = np.array([])
        for rep in range(len(condition_stats)):
            dff_allRep = np.append(dff_allRep, cell[condition_stats[rep]:condition_end[rep]+1])
            if len(cell[condition_stats[rep]:condition_end[rep]+1]) < 33:
                dff_allRep = np.append(dff_allRep,0)
        dff_allConditions = np.append(dff_allConditions, dff_allRep)
        av_dff_perCondition.append(np.mean(dff_allRep))
    if c == 0:
        all_cell_dff = np.concatenate([all_cell_dff, dff_allConditions])
    else:
        all_cell_dff = np.vstack([all_cell_dff, dff_allConditions])
    hightest_dff.append(np.where(av_dff_perCondition == np.nanmax(av_dff_perCondition))[0][0])


####### actual plots

# general stuff:
conditions_key_list = np.array(list(all_conditions_dict.keys()))
conditions_key_list_valid = np.delete(conditions_key_list,np.where(np.isnan(all_cells_all_cond_corr[0])), axis=0)
norm = colors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)

# unsorted cells
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('z-scores of all cells for all conditions - UNSORTED')
plt.xlabel('Conditions')
plt.ylabel('Cells - unsorted')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
plt.imshow(all_cell_zScore, cmap='seismic',norm=norm, aspect='auto')
plt.colorbar()

'''
# cells sorted by position max z_score (in any frame)
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('z-scores of all cells for all conditions - MAX ZSCORE POSITION')
plt.xlabel('Conditions')
plt.ylabel('Cells sorted - frame with max zScore')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
###
all_cell_zScore = all_cell_zScore[np.argsort(np.argmax(all_cell_zScore, 1)),:]
plt.imshow(all_cell_zScore, cmap='seismic',norm=norm, aspect='auto')
plt.colorbar()
'''

# cells sorted by max average z_score in condition
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('z-scores of all cells for all conditions - MAX ZSCORE CONDITION')
plt.xlabel('Conditions')
plt.ylabel('Cells sorted - max av. zScore per Condition')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
###
all_cell_zScore = all_cell_zScore[np.argsort(cell_highestActivity_condIDX),:]
plt.imshow(all_cell_zScore, cmap='seismic',norm=norm, aspect='auto')
plt.colorbar()

# Reine dff
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('dff of all cells for all conditions')
plt.xlabel('Conditions')
plt.ylabel('Cells unstorted')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
###
all_cell_dff_show = all_cell_dff
plt.imshow(all_cell_dff_show, aspect='auto')
plt.colorbar()

# DFF sorted
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('dff of all cells for all conditions')
plt.xlabel('Conditions')
plt.ylabel('Cells sorted - max av. dff per Condition')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
###
all_cell_dff_show = all_cell_dff[np.argsort(hightest_dff),:]
plt.imshow(all_cell_dff_show, vmax=1.5, aspect='auto')
plt.colorbar()

# display only the cells (dff and zScore) that have their peak av. aktivity somewhere in moving dot phases
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('Cells with av. peak dff in moving dot conditions')
plt.xlabel('Conditions')
plt.ylabel('Cells sorted - max av. dff per Condition')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
#ax.xaxis.labelpad = 20
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
###
hightest_dff = np.array(hightest_dff)
all_cell_dff_show = all_cell_dff[hightest_dff<19 ,:]
new_highest_dff = hightest_dff[np.where(hightest_dff<19)]
all_cell_dff_show = all_cell_dff_show[np.argsort(new_highest_dff),:]
plt.imshow(all_cell_dff_show, vmax=1.5, aspect='auto')
plt.colorbar()

# ONLY CELLS DOT CONDITIONS: cells sorted by max average z_score in condition
fig, ax = plt.subplots()
plt.gcf().subplots_adjust(bottom=0.2)
plt.title('Celly av. peak z-scores in dot conditoin')
plt.xlabel('Conditions')
plt.ylabel('Cells sorted - max av. zScore per Condition')
plt.xticks(np.array(xtick_list)-np.diff([0]+xtick_list)/2, conditions_key_list_valid, rotation=90)
plt.vlines(x=xtick_list, ymin=0, ymax=np.shape(all_cell_zScore)[1], colors='k')
###
cell_highestActivity_condIDX = np.array(cell_highestActivity_condIDX)
all_cell_zscore_show = all_cell_zScore[cell_highestActivity_condIDX<2 ,:]
new_highest_zscore = cell_highestActivity_condIDX[np.where(cell_highestActivity_condIDX<2)]
all_cell_zscore_show = all_cell_zscore_show[np.argsort(new_highest_zscore),:]
plt.imshow(all_cell_zscore_show, cmap='seismic',norm= colors.TwoSlopeNorm(vcenter=0, vmax=5), aspect='auto')
plt.colorbar()



plt.show()
