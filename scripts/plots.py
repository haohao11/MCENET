# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:04:23 2019

@author: cheng
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.spatial import ConvexHull
    
    
def plot_loss(history, scale, real_scale, timestr, dataname):
    loss = history.history['loss']   
    val_loss = history.history['val_loss']
    loss = [math.sqrt(x)*scale/real_scale for x in loss]
    val_loss = [math.sqrt(x)*scale/real_scale for x in val_loss]
    epochs = [i+1 for i in range(len(loss))]
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, loss, label='train_loss')
    ax1.plot(epochs, val_loss, label='val_loss')
    ax1.set_xlim(left=0)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    plt.tight_layout()
    np.savetxt('../loss/%s_loss_%s.txt'%(dataname, timestr), np.concatenate((np.reshape(epochs, [-1, 1]), 
                                                   np.reshape(loss, [-1, 1]), 
                                                   np.reshape(val_loss, [-1, 1])), axis=1).T, delimiter=',')
    plt.savefig("../loss/%s_loss_%s.png"%(dataname, timestr), dpi=100)
    plt.gcf().clear()
    plt.close()

        
def plot_scenarios(raw, observation, ground_truth, predictions, bg_image, scale, real_scale, plot_dir, convex=True):
    test_obs = observation
    test_pred = ground_truth
    indexed_predictions = predictions  
    test_raw = raw
    obs_seq =  observation.shape[-2]
    pred_seq = ground_truth.shape[-2]
    

    # Get the scenario index
    start_end_list = get_scenario_index(test_obs, test_pred)    
    for i in range(len(start_end_list)):
        test_obs_sce = np.empty((0, 5))
        test_pred_sce = np.empty((0, 5))
        num_paths = indexed_predictions.shape[1]
        indexed_predictions_sce = np.empty((0, pred_seq, 5))
        start, end = start_end_list[i]
        for j in range(len(test_obs)):
            # Sort the observed trajectory based on the scenario ID
            if test_obs[j, 0, 1] == start and test_obs[j, -1, 1] == start+obs_seq-1:
                test_obs_sce = np.vstack((test_obs_sce, test_obs[j]))
            # Sort the ground truth trajectory based on scenario ID
            if test_pred[j, 0, 1] == start+obs_seq and test_pred[j, -1, 1] == start+obs_seq+pred_seq-1:
                test_pred_sce = np.vstack((test_pred_sce, test_pred[j]))
            # Sort the (multiple) predicted trajectory based on scenario ID
            if indexed_predictions[j, 0, 0, 1] == start+obs_seq and indexed_predictions[j, 0, -1, 1] == start+obs_seq+pred_seq-1:
                indexed_predictions_sce = np.concatenate((indexed_predictions_sce, indexed_predictions[j]), axis=0)
                
        test_obs_sce = np.reshape(test_obs_sce, [-1, obs_seq, 5]) 
        test_pred_sce = np.reshape(test_pred_sce, [-1, pred_seq, 5])
        indexed_predictions_sce = np.reshape(indexed_predictions_sce, [-1, num_paths, pred_seq, 5])
        userIds_sce = np.unique(test_obs_sce.reshape([-1, 5])[:, 0])
        print("%.0f/%.0f\n"%(i+1, len(start_end_list)))
        test_raw_sce = get_raw_sce(test_raw, start, end, real_scale)
        plot_each_scenario(userIds_sce, test_raw_sce, test_obs_sce, test_pred_sce, indexed_predictions_sce, bg_image, scale, real_scale, i, plot_dir)
                
    
def get_raw_sce(test_raw, start, end, real_scale):
    '''
    '''
    test_raw_sce = test_raw[test_raw[:, 1]>=start, :]
    test_raw_sce = test_raw_sce[test_raw_sce[:, 1]<=end, :]
    test_raw_sce = np.multiply(test_raw_sce, [1, 1, real_scale, real_scale, 1])
    return test_raw_sce   
    
    
def plot_each_scenario(userIds_sce, raw, observation, ground_truth, predictions, bg_image, scale, real_scale, index, plot_dir, convex=True):
    '''
    This is the function to plot all the users in the same scenarios
    All the users:
        users whose trajectory has the same length as the predefined length. Their trajectories are also predicted
        users whose trajectory has shorter length as the predefined length. Their trajectories are not predicted. They are treated as co-existing road users 
    '''
    # Load the background image
    im = plt.imread(bg_image)       
    fig, ax = plt.subplots()
    ax.imshow(im)
    cmap = plt.cm.get_cmap("prism", 37)            
    for idx, user_ob in enumerate(observation):
        obs_seq = len(user_ob) 
        user_gt = ground_truth[idx]        
        pred_seq = len(user_gt)
        steps = obs_seq + pred_seq
        user_gt = np.vstack((user_ob[-1, :], user_gt))
        user_pds = predictions[idx]
        # Plot observation
        ax.plot(user_ob[:, 2]*scale, user_ob[:, 3]*scale, color='k', linestyle='-', linewidth=0.5, fillstyle='none', marker='x', markersize=0.75, alpha=0.7)
        # Plot Ground truth
        ax.plot(user_gt[:, 2]*scale, user_gt[:, 3]*scale, color ='navy', linestyle='-', linewidth=0.5, fillstyle='none', marker='*', markersize=0.75, alpha=0.7)               
        points = np.empty([0, 2])
        # Plot Prediction(s)
        for j, user_pd in enumerate(user_pds):
            user_pd = np.vstack((user_ob[-1, :], user_pd))
            ax.plot(user_pd[:, 2]*scale, user_pd[:, 3]*scale, linewidth=0.5, c=cmap(idx%29), marker='.', markersize=0.5, fillstyle='none')
            points = np.vstack((points, user_pd[:, 2:4]))
        if convex:
            points = points*scale
            hull = ConvexHull(points)
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'navy', alpha=0.3)
                
    # Plot the co existing road users whoes trajectory is short than the pre-defined length    
    plot_cousers = False   
    if plot_cousers == True:                   
        co_userIds = np.unique(raw[:, 0])
        for co_user in co_userIds:
            if co_user not in userIds_sce: 
                co_traj = raw[raw[:, 0]==co_user, 2:4]
                # filter the short co existing users
                if len(co_traj)>=(steps/1.5):
                    ax.plot(co_traj[:, 0], co_traj[:, 1],'k-', linewidth=0.5, alpha=0.4)
                    end_x, end_y = None, None
                    for i in range(2, pred_seq):
                        if co_traj[-1, 0]-co_traj[-i, 0] == 0 or co_traj[-1, 1]-co_traj[-i, 1] == 0:
                            i += 1
                            continue
                        end_x, end_y = co_traj[-1, 0]-co_traj[-i, 0], co_traj[-1, 1]-co_traj[-i, 1]
                        break
                    if end_x is not None and end_y is not None:                       
                        start_x, start_y = co_traj[-2, 0], co_traj[-2, 1]
                        ax.arrow(start_x, start_y, end_x, end_y, shape='full', lw=0.5, length_includes_head=True, head_width=3)
                        
    # Plot the legend
    if plot_cousers == True:
        ax.plot([], [], linewidth=1, color='k', linestyle='-', label='neighbors', markersize=0.75, alpha=0.4)    
    ax.plot([], [], linewidth=1, color='k', linestyle='-', label='obs.', fillstyle='none', marker='x', markersize=2, alpha=1)
    ax.plot([], [], linewidth=1, color='navy', linestyle='-', label='gt.', fillstyle='none', marker='*', markersize=2, alpha=1)
    ax.plot([], [], linewidth=1, color='k', linestyle='-', label='pred.', marker='.', markersize=2, alpha=1)
    
    
    # HBS
#    plt.legend(ncol=4, loc='upper center') 
    # SDD/gates/video3
#    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.17))
    # HC
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    
    
    name = os.path.join(plot_dir, "prediction_scenario_%s.png"%(index))
    plt.savefig(name, dpi=200, bbox_inches="tight")
#    plt.show()
    plt.gcf().clear()
    plt.close()
        


def get_scenario_index(obs, pred):
    '''
    This is the function to get the start and end index for scenarios
    '''
    start_end_list = []
    for i in range(len(obs)):
        start, end = min(obs[i, :, 1]), max(pred[i, :, 1])
        if [start, end] not in start_end_list:
            start_end_list.append([start, end])
    return start_end_list     
            
        




    
        
    
    
    

