# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:46:11 2019
This is the module to get the occupancy data for each user
@author: cheng
"""

import numpy as np
import math

from group_detection import get_prediction


def circle_group_model_input(obs_data, neighborhood_radius, grid_radius, grid_angle, data, args):
    '''
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        obs_data: these are the selected trajectory, numpy with [numUsers, obs_length, 5]
                  5 refers to UserId, frameId, x, y, userType
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: these are all the trajectories, including all the discarded trajectories
    '''
    group_model_input = []
    raw_data = data
        
    # Get the friends data
    # Sort the data by colomn: frameId, UserId, x, y, userType 
    order = [1, 0, 2, 3, 4]
    i = np.argsort(order)
    raw_data_ordered = raw_data[:, i]
    friends = get_prediction(raw_data_ordered, opt_pixel_thre=args.dist_thre, ratio=args.overlap, max_friends=40)
    #print('friends /n', friends)
        
    for ego_data in obs_data:
        o_map = get_circle_occupancy_map(ego_data, raw_data, neighborhood_radius, grid_radius, grid_angle, friends)
        group_model_input.append(o_map)
    group_model_input = np.reshape(group_model_input, [len(group_model_input), len(ego_data), -1])
    return group_model_input    


def get_circle_occupancy_map(ego_data, raw_data, neighborhood_radius, grid_radius, grid_angle, friends, islog=False):
    '''
    This is the function to get the occupancy for each ego user
    '''
    o_map = np.zeros((len(ego_data), int(neighborhood_radius / grid_radius), int(360 / grid_angle)))
    egoId = ego_data[0, 0]
    egoFrameList = ego_data[:, 1]
    
    # Get the ego user's friends
    ego_friends = friends[friends[:, 0]==egoId, :]
    
    for i, f in enumerate(egoFrameList):
        frame_data = raw_data[raw_data[:, 1]==f, :]
        otherIds = frame_data[:, 0]
        current_x, current_y = ego_data[i, 2], ego_data[i, 3]
        
        for otherId in otherIds:
            if egoId != otherId:
                ### incorporate friend detection
                if otherId in ego_friends:
                    print('%s and %s are frineds'%(str(int(egoId)), str(int(otherId))))
                    # Set the occupancy as 0
                    continue
                
                [other_x, other_y] = frame_data[frame_data[:, 0]==otherId, 2:4][0]
                distance = math.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
                ### TODO, log grid
                d = 0.05
                if distance < d:
                    if islog == False:                    
                        angle= np.rad2deg(np.math.atan2(other_y-current_y,other_x-current_x))
                        cell_radius = int(distance*8 / d)
                        cell_angle = int(angle / grid_angle)
                        o_map[i, cell_radius, cell_angle] += 1
                    else:
                        angle= np.rad2deg(np.math.atan2(other_y-current_y,other_x-current_x))
                        r = 1/(-np.log2(distance)+1e-8)
                        cell_radius = int((8*r) / (1/(-np.log2(d)+1e-8)))
                        cell_angle = int(angle / grid_angle)
                        o_map[i, cell_radius, cell_angle] += 1
    return np.reshape(o_map, [len(egoFrameList), -1])            
 

