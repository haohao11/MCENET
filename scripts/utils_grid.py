# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:14:50 2019

@author: lray
"""

import numpy as np
import os
import cv2

from occupancy import circle_group_model_input
from heatmap import heatmap_main
from localscene import ind_heatmap, rgb_image, ind_image


def save_function(data, frame_num, info_col, name):
    # define the repository to store the processed data 
    data_dir = '../processed_data'
    # Define the path in which the process data would be stored
    name = os.path.join(data_dir, name)
    relay_data = []
    relay_data.append(data)
    relay_data = np.reshape(relay_data, [-1, frame_num * info_col])
    np.savetxt(name, relay_data, delimiter = ',')  


def load_function(frame_num, info_col, name):
    # define the repository to store the processed data
    data_dir = '../processed_data'
    # Define the path in which the process data would be stored
    name = os.path.join(data_dir, name)
    relay_data = np.genfromtxt(name, delimiter=',')
    data = np.reshape(relay_data, [-1,frame_num, info_col])
    return data


def normalize(matrix, diagonal):
    '''
    This is matrix transformation and xy coordinate normalization
    This can be optimaized by numpy transpose and division
    '''
    matrix = np.divide(matrix, [1, 1, diagonal, diagonal, 1])
    return matrix


def preprocess(data_dir, filename):
    '''
    This is the function to read the trajectory data from csv
    return: trajectory data [userId, frameId, x, y, userType]
    '''
    file_path = os.path.join(data_dir, filename)
    data = np.genfromtxt(file_path, delimiter=',')
    return data

    

def get_traj_like(data, observed_frame_num, predicting_frame_num, normalized_to_meter):
    '''
    This is the function to get trajectory sorted by frameId, in order to get scenario trajectory
    '''  
    seq_length = observed_frame_num + predicting_frame_num   
    frameList = np.unique(data[:, 1])
    min_frameId, max_frameId = min(frameList), max(frameList)    
    # Set the increment_size
    # If the increment_size >= seq_length, there will be no overlap 
    # Otherwise, overlap is allowed
    increment_size = int(seq_length/1) #
    obs = np.empty((0, 5))
    pred = np.empty((0, 5))   
    start = min_frameId
    end = start + seq_length
    for i in range(int(len(frameList)/increment_size)):
        if end <= max_frameId:
            scenario_data = data[data[:,1]>=start]
            scenario_data = scenario_data[scenario_data[:,1]<end, :]
            obs_, pred_ = get_scenario_trajectory(scenario_data, observed_frame_num, seq_length)
            obs = np.vstack((obs, obs_))
            pred = np.vstack((pred, pred_))                        
            start = start + increment_size
            end = start + seq_length                        
    obs = np.reshape(obs, (-1, observed_frame_num, 5))
    pred = np.reshape(pred, (-1, predicting_frame_num, 5))    
    # Normalize the raw_data to meters
    raw_data = normalize(data, normalized_to_meter)
    print("There are %.0f unique road user"%len(np.unique(raw_data[:, 0])))
    return obs, pred, raw_data 
        
def get_scenario_trajectory(scenario_data, observed_frame_num, seq_length):
    obs = np.empty((0, 5))
    pred = np.empty((0, 5))
    userIds = np.unique(scenario_data[:, 0])
    for i in userIds:
        user_traj = scenario_data[scenario_data[:, 0]==i, :]
        if len(user_traj) == seq_length:
            obs = np.vstack((obs, user_traj[:observed_frame_num, :]))
            pred = np.vstack((pred, user_traj[observed_frame_num:seq_length, :]))
    return obs, pred
    

def type_classification(data, frame_num):
    '''
    This is the function to sort trajectory according to userType
    1: pedestrian
    2: cyclist
    3: vehicle
    '''
    type_1 = []
    type_2 = []
    type_3 = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][4] == 1:
                type_1.append([data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3]])
            elif data[i][j][4] == 2:
                type_2.append([data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3]])
            elif data[i][j][4] == 3:
                type_3.append([data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3]]) 
    data_type_1 = np.reshape(type_1, [-1, frame_num, 4]) 
    data_type_2 = np.reshape(type_2, [-1, frame_num, 4])
    data_type_3 = np.reshape(type_3, [-1, frame_num, 4])    
    return data_type_1, data_type_2, data_type_3


def dataLoader(dirs, filename, args, dataname, background = 'heatmap', trict_untouched=True):
    '''
    This is the function to load data
    return:
        training: train_obs, train_pred, train_raw, train_occuGrid, train_hmaps
        test: test_obs, test_pred, test_raw, test_occuGrid, test_hmaps
    '''
    
    # Store the directories for the data
    train_dir = dirs[1]
    test_dir = dirs[2]
    
    # Store the scale for normalization
    scale = args.scale
    # Store the scale for normalized_to_meter
    # The raw_data needs to be scaled back to meters for normalization
    normalized_to_meter = args.real_scale / args.scale
        
    # Read the trajectory into numpy array [userId, frameId, y, x, userType]
    ## Training data
    train_data = normalize(preprocess(train_dir, filename), scale)
    train_obs, train_pred, train_raw = get_traj_like(train_data, args.obs_seq, args.pred_seq, normalized_to_meter)
    
    
    ## Validation data
    test_data = normalize(preprocess(test_dir, filename), scale)
    test_obs, test_pred, test_raw = get_traj_like(test_data, args.obs_seq, args.pred_seq, normalized_to_meter)
    
        
    # Compute the occupancy grid
    ## Compute the occupancy grid for training data
    train_obs_og = circle_group_model_input(train_obs[:, :, 0:4],
                                              args.neighSize,
                                              args.gridRadius,
                                              args.gridAngle,
                                              train_raw,
                                              args)
    train_pred_og = circle_group_model_input(train_pred[:, :, 0:4],
                                              args.neighSize,
                                              args.gridRadius,
                                              args.gridAngle,
                                              train_raw,
                                              args)
    
    # Compute the occupancy grid for test data
    test_obs_og = circle_group_model_input(test_obs[:, :, 0:4],
                                            args.neighSize,
                                            args.gridRadius,
                                            args.gridAngle,
                                            test_raw,
                                            args)
    test_pred_og = circle_group_model_input(test_pred[:, :, 0:4],
                                            args.neighSize,
                                            args.gridRadius,
                                            args.gridAngle,
                                            test_raw,
                                            args)

    
    
    if background == 'heatmap':
        print("The scene context is heatmap")
        # each layer a different user type
        # TODO, this need to be changed to only use training data to get the heatmap for a strict unseen for validation
        if trict_untouched == False:
            all_data = np.concatenate((np.reshape(train_data, [-1, 5]), np.reshape(test_data, [-1, 5])), axis=0)
        else:
            all_data = np.reshape(train_data, [-1, 5])
        print('shape for all_data', all_data.shape)
        hmap_ped = heatmap_main(all_data.T, args.dimensions, scale, dataname, user_type=1, sigma=args.sigma)
        hmap_cyc = heatmap_main(all_data.T, args.dimensions, scale, dataname, user_type=2, sigma=args.sigma)
        hmap_veh = heatmap_main(all_data.T, args.dimensions, scale, dataname, user_type=3, sigma=args.sigma)
        heatmaps = [hmap_ped, hmap_cyc, hmap_veh]
        ## Compute the heatmap grid for training data
        train_obs_hmap = ind_heatmap(train_obs, heatmaps, scale, hgrid_size=args.hgrid_size)
        train_obs_hmaps = repeat(train_obs_hmap, 3)
        train_pred_hmap = ind_heatmap(train_pred, heatmaps, scale, hgrid_size=args.hgrid_size)
        train_pred_hmaps = repeat(train_pred_hmap, 3)    
        ## Compute the heatmap grid for validation data
        test_obs_hmap = ind_heatmap(test_obs, heatmaps, scale, hgrid_size=args.hgrid_size)
        test_obs_hmaps = repeat(test_obs_hmap, 3)
        test_pred_hmap = ind_heatmap(test_pred, heatmaps, scale, hgrid_size=args.hgrid_size)
        test_pred_hmaps = repeat(test_pred_hmap, 3)

    elif background == 'segmented_map':
        print("The scene context is segmented map")
        images = get_images(dataname)
        ## Compute the heatmap grid for training data
        train_obs_hmaps = ind_image(train_obs, images, scale, hgrid_size=args.hgrid_size)
        train_pred_hmaps = ind_image(train_pred, images, scale, hgrid_size=args.hgrid_size)
        ## Compute the heatmap grid for validation data
        test_obs_hmaps = ind_image(test_obs, images, scale, hgrid_size=args.hgrid_size)
        test_pred_hmaps = ind_image(test_pred, images, scale, hgrid_size=args.hgrid_size)

    elif background == 'aerial_photograph':
        print("The scene context is aerial photograph")
        image = get_rgb_image(dataname)
        ## Compute the heatmap grid for training data
        train_obs_hmaps = rgb_image(train_obs, image, scale, hgrid_size=args.hgrid_size)
        train_pred_hmaps = rgb_image(train_pred, image, scale, hgrid_size=args.hgrid_size)
        ## Compute the heatmap grid for validation data
        test_obs_hmaps = rgb_image(test_obs, image, scale, hgrid_size=args.hgrid_size)
        test_pred_hmaps = rgb_image(test_pred, image, scale, hgrid_size=args.hgrid_size)
    
    # Return the data
    training = [train_obs, train_pred, train_raw, train_obs_og, train_pred_og, train_obs_hmaps, train_pred_hmaps]
    test = [test_obs, test_pred, test_raw, test_obs_og, test_pred_og, test_obs_hmaps, test_pred_hmaps]	
    training_data_name = '../processed/%s_training_data_grid.npz'%dataname
    test_data_name = '../processed/%s_test_data_grid.npz'%dataname
    save_data(training_data_name, training)
    save_data(test_data_name, test) 
    return training, test


def repeat(data, times):
    shape = np.append(data.shape, [times])
    new_data = np.zeros(shape)
    for i in range(times):
        new_data[..., i] = data
    return new_data


def filter_type(data, length, userType=1):
    '''
    This is the function only filter the target user based on the user type,
    The coexisting will be retained in the raw data
    '''
    print('data shape before filter', data.shape)
    data = np.reshape(data, [-1, 5])
    data_filter = data[data[:, 4]==userType, :]
    print('data shape after filter', data.shape)
    new_data = np.reshape(data_filter, [-1, length, 5])
    return new_data
	

def save_data(name, data):
	# Save all the processed data
    # Even the the variable name is "training", this only holds for training and testing on the same dataset.
    # But if we use the leave-one-out cross validation, 
    # the training and validation will be later be merged to form leaveoneout_validation as the whole dataset for testing
    np.savez(name, 
             obs=data[0], 
             pred=data[1], 
             raw=data[2], 
             obs_og=data[3], 
             pred_og=data[4], 
             obs_hmaps=data[5], 
             pred_hmaps=data[6])


def get_images(dataname):
    '''
    This is the function to read the background image into numpy array
    '''
    ped_dir, cyc_dir, veh_dir, img_path = map_dirs(dataname)
    ped_img = cv2.imread(ped_dir) / 255
    cyc_img = cv2.imread(cyc_dir) / 255
    veh_img = cv2.imread(veh_dir) / 255
    
    return [ped_img, cyc_img, veh_img]
    
        
def get_rgb_image(dataname):
    '''
    This is the function to read the background image into numpy array
    '''
    ped_dir, cyc_dir, veh_dir, img_path = map_dirs(dataname)
    img = cv2.imread(img_path)
    img = img / 255.0    
    return img


def map_dirs(dataname):
    '''
    This is the function to specify all the layout map according to user type
    '''
    # This is the repository for HBS
    if dataname == 'HBS':
        bg_image = '../data/HBS/HH_Bergedorf.jpg'
        ped_dir = '../cvat/out/HH_Bergedorf_ped.png'
        cyc_dir = '../cvat/out/HH_Bergedorf_cyc.png'
        veh_dir = '../cvat/out/HH_Bergedorf_veh.png'        
        
    # This is the repository for HC
    if dataname == 'HC':
        bg_image = '../data/HC/background.png'
        ped_dir = '../cvat/out/HC_ped.png'
        cyc_dir = '../cvat/out/HC_cyc.png'
        veh_dir = '../cvat/out/HC_veh.png'        
        
    # This is the repository for SDD/gates/3
    if dataname == 'SDDgates3':
        bg_image = '../data/SDD/gates/video3/reference.jpg'
        ped_dir = '../cvat/out/gates3_ped.png'
        cyc_dir = '../cvat/out/gates3_cyc.png'
        veh_dir = '../cvat/out/gates3_veh.png'
       
    return ped_dir, cyc_dir, veh_dir, bg_image
