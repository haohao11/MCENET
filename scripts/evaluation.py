# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:52:48 2019
This is the module for evaluation metrics 
@author: Cheng
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:31:32 2019
@author: cheng
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def get_classified_errors(test_pred, indexed_predictions, scale):
    '''
    Measure the errors in relation to user types
    userType:
        1: pedestrian
        2: cyclist
        3: vehicle
    '''
    _, num_pred, pred_seq, _ = indexed_predictions.shape
    all_preds_prime = np.reshape(indexed_predictions, [-1, 5])
    all_test_pred = np.reshape(test_pred, [-1, 5])

    # Calculate the corresponding errors
    mixed_errors = get_evaluation(test_pred, indexed_predictions, num_pred, scale)
    print('\nmixed_errors by ADE and FDE\n', np.array_str(mixed_errors[0:2, 2], precision=2, suppress_small=True))
    
    # Differentiate errors by user types    
    ped_test_preds = np.reshape(all_test_pred[all_test_pred[:, 4]==1, :], [-1, pred_seq, 5])
    cyc_test_preds = np.reshape(all_test_pred[all_test_pred[:, 4]==2, :], [-1, pred_seq, 5])
    veh_test_preds = np.reshape(all_test_pred[all_test_pred[:, 4]==3, :], [-1, pred_seq, 5])    
    ped_preds_prime = np.reshape(all_preds_prime[all_preds_prime[:, 4]==1, :], [-1, num_pred, pred_seq, 5])
    cyc_preds_prime = np.reshape(all_preds_prime[all_preds_prime[:, 4]==2, :], [-1, num_pred, pred_seq, 5])
    veh_preds_prime = np.reshape(all_preds_prime[all_preds_prime[:, 4]==3, :], [-1, num_pred, pred_seq, 5])
    ped_errors = get_evaluation(ped_test_preds, ped_preds_prime, num_pred, scale)
#    print('\nped_errors \n', np.array_str(ped_errors, precision=2, suppress_small=True))
    cyc_errors = get_evaluation(cyc_test_preds, cyc_preds_prime, num_pred, scale)
#    print('\ncyc_errors \n', np.array_str(cyc_errors, precision=2, suppress_small=True))
    veh_errors = get_evaluation(veh_test_preds, veh_preds_prime, num_pred, scale)
#    print('\nveh_errors \n', np.array_str(veh_errors, precision=2, suppress_small=True))    
    errors = np.vstack((mixed_errors, ped_errors, cyc_errors, veh_errors))    
    return errors
    

def get_evaluation(test_pred, predictions, num_pred, scale):    
    # Evaluation
    evaluations = np.zeros([len(predictions), num_pred, 5])
    for i, user_gt in enumerate(test_pred):
        user_preds = predictions[i]
        for j, user_pred in enumerate(user_preds):
            evaluations[i, j, :] = get_eva_values(user_gt[:, 2:4]*scale, user_pred[:, 2:4]*scale)
    # Compute the average errors across all users and all predictions
    mean_evaluations = np.reshape(evaluations, [-1, 5])
    mean_errors =  np.mean(mean_evaluations, axis=0)
    mean_std = np.std(mean_evaluations, axis=0)
    # Comput the minimum errors across all users for the best prediction
    min_evaluations = np.amin(evaluations, axis=1)
    min_errors = np.mean(min_evaluations, axis=0)
    min_std = np.std(min_evaluations, axis=0)
    # Save the evaluation results
    errors = np.concatenate((np.reshape(mean_errors, [-1, 1]), 
                             np.reshape(mean_std, [-1, 1]),
                             np.reshape(min_errors, [-1, 1]),
                             np.reshape(min_std, [-1, 1])), axis=1)
    return errors


def get_eva_values(y_t, y_p):
    '''
    y_t: 2d numpy array for true trajectory. Shape: steps*2
    y_p: 2d numpy array for predicted trajectory. Shape: steps*2
    '''    
    Euclidean = get_euclidean(y_t, y_p)    
    last_disp = get_last_disp(y_t, y_p)   
    Hausdorff = get_hausdorff(y_t, y_p)    
    speed_dev = get_speeddev(y_t, y_p)    
    heading_error = get_headerror(y_t, y_p)    
    # Store Euclidean, last_disp, Hausdorff, speed_dev, heading_error as a list
    eva_values = [Euclidean, last_disp, Hausdorff, speed_dev, heading_error]    
    return eva_values
   
def get_euclidean(y_true, y_prediction):
    Euclidean = np.linalg.norm((y_true - y_prediction), axis=1)
    Euclidean = np.mean(Euclidean)
    #Euclidean = np.around(Euclidean, decimals=4)
    return Euclidean

def get_last_disp(y_true, y_prediction):
    last_disp = np.linalg.norm((y_true[-1, :] - y_prediction[-1, :]))
    #last_disp = np.around(last_disp, decimals=4)
    return last_disp
        
def get_hausdorff(y_true, y_prediction):
    '''
    Here is the directed Hausdorff distance, but it computes both directions and output the larger value
    '''
    Hausdorff = max(directed_hausdorff(y_true, y_prediction)[0], directed_hausdorff(y_prediction, y_true)[0])
    #Hausdorff = np.around(Hausdorff, decimals=4)
    return Hausdorff
    
def get_speeddev(y_true, y_prediction):
    if len(y_true) == 1:
        return 0
    else:       
        speed_dev = 0.0
        for t in range(len(y_true)-1):
            speed_t = np.linalg.norm(y_true[t+1] - y_true[t])       
            speed_p = np.linalg.norm(y_prediction[t+1] - y_prediction[t])
            speed_dev += abs(speed_t - speed_p)
        speed_dev /=  (len(y_true)-1)
        #speed_dev = np.around(speed_dev, decimals=4)    
        return speed_dev     

def get_headerror(y_true, y_prediction):
    if len(y_prediction) == 1:
        return 0
    else:
        heading_error = 0.0
        for t in range(len(y_true)-1):
            xcoor_t = y_true[t+1, 0] - y_true[t, 0]
            ycoor_t = y_true[t+1, 1] - y_true[t, 1]
            angle_t = np.arctan2(ycoor_t, xcoor_t)
            xcoor_p = y_prediction[t+1, 0] - y_prediction[t, 0]
            ycoor_p = y_prediction[t+1, 1] - y_prediction[t, 1]
            angle_p = np.arctan2(ycoor_p, xcoor_p)
            angle = np.rad2deg((abs(angle_t - angle_p)) % (np.pi))
            heading_error += angle
        heading_error /= len(y_true)-1
        #heading_error = np.around(heading_error, decimals=4) 
        return heading_error
     
        