# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:37:50 2020

@author: cheng
"""

from contextlib import redirect_stdout

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.mobilenet_v2 import MobileNetV2

import numpy as np
import os
from scipy.stats import multivariate_normal
import time


from dparser import get_data_repo, generate_data
from evaluation import get_classified_errors
from hyperparameters import parse_args
from plots import plot_loss, plot_scenarios
from model import MCENET


def main():
    '''
    This is the MCENET model for trajectory prediction. 
    It leverages scene, occupancy grid, and trajectry sequence for trajectory prediction
    Training:
        X-Encoder for encoding the information from observation
        Y-Encoder for encoding the information from ground truth
        Both encoded information are concatenated for parameterizing the latent variable z, 
            which is pushed towards a Gaussian distribution
        z and the encoded information from X-Encoder is used as the condition for reconstructing the future trajectory        
    Inference:
        X-Encoder for encoding the information from observation
        Y-Encoder is removed
        z is sampled from the Gaussian distribution
        z and the encoded information from X-Encoder is used as the condition for predicting the future trajectory
    '''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dataname = 'HC'
    print(dataname)
    # Make all the necessary folders
    mak_dir()
    bg_image, data_dir, dirs, filename = get_data_repo(dataname)
    
    args = parse_args(dataname)
    
    # # Generate data
    if args.data_process == True:
        generate_data(dataname, leaveoneout=args.leaveoneout)
        
    # specify which GPU(s) to be used, gpu device starts from 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # # Use the default CPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    
    # CHECK POINT AND SAVE THE BEST MODEL
    filepath="../models/mcenet_mixed_%s_%0.f_%s.hdf5"%(dataname, args.epochs, timestr)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]
    
    # INSTANTIATE THE MODEL
    mcenet = MCENET(args)
    # Construct the training model
    train = mcenet.training()
    train.summary()
    
    
    # Load the test data
    # Note the one step difference between position and offset
    obs_seq = args.obs_seq -1
    
    
    #################### START TRAINING THE MCENet MODEL ####################
    if args.train_mode:
        print("\nload the training data")
        T_ = np.load('../processed/%s_training_data_grid.npz'%dataname)
        obs, pred, obs_og, pred_og, obs_hmaps, pred_hmaps = T_['obs'], T_['pred'], T_['obs_og'], T_['pred_og'], T_['obs_hmaps'], T_['pred_hmaps']
        print('You are using the scene context from %s'%args.sceneType)
        print('Data loaded!')        
        # Shift one time step for occupancy grid and scene attention
        obs_og = obs_og[:, 1:, :]
        obs_hmaps = obs_hmaps[:, 1:, ...]    
        # Get the residual of the trajectories
        traj = np.concatenate((obs[:, :, 2:4], pred[:, :, 2:4]), axis=1)
        traj_r = traj[:, 1:, :] - traj[:, :-1, :]
        traj_r = traj_r*args.resi_scale
        obs_r, pred_r = traj_r[:, :obs_seq, :], traj_r[:, obs_seq:, :]                       
        ## Get User type one-hot encoding
        obs_type = get_type(obs, 3, isObservation=True)    
        obs_r = np.concatenate((obs_r, obs_type), axis=-1) 
        obs_hpinput = mobilenet(obs_hmaps, obs_seq)
        pred_hpinput = mobilenet(pred_hmaps, args.pred_seq)
                            
        # Here is the training related data
        print('Check the input data...')    
        print('the shape of obs', obs_r.shape)
        print('the shape of pred', pred_r.shape)  
        #print('the number of road users in train_raw', len(np.unique(train_raw[:, 0])))
        print('the shape of obs_og', obs_og.shape)
        print('the shape of pred_og', pred_og.shape)
        print('the shape of obs_maps', obs_hmaps.shape)
        print('the shape of pred_maps', pred_hmaps.shape)                    
               
        # Get the data fro training and validation
        np.random.seed(10)
        train_val_split = np.random.rand(len(traj)) < args.split
        train_obs_hpinput = obs_hpinput[train_val_split]
        train_pred_hpinput = pred_hpinput[train_val_split]
        train_obs_og = obs_og[train_val_split]
        train_pred_og = pred_og[train_val_split]
        train_obs_r = obs_r[train_val_split]
        train_pred_r = pred_r[train_val_split]

        val_obs_hpinput = obs_hpinput[~train_val_split]
        val_pred_hpinput = pred_hpinput[~train_val_split]
        val_obs_og = obs_og[~train_val_split]
        val_pred_og = pred_og[~train_val_split]
        val_obs_r = obs_r[~train_val_split]
        val_pred_r = pred_r[~train_val_split]
        
        print('Start training the MCENet model...')
        history = train.fit(x=[train_obs_hpinput, train_pred_hpinput, train_obs_og, train_pred_og, train_obs_r, train_pred_r],
                           y=train_pred_r,
                           shuffle=True,
                           epochs=args.epochs,
                           batch_size=args.batch_size,
                           verbose=1,
                           callbacks=callbacks_list,
                           validation_data=([val_obs_hpinput, val_pred_hpinput, val_obs_og, val_pred_og, val_obs_r, val_pred_r], 
                                            val_pred_r))
        print('Training the MCENet model done!')
        
        print('Plotting loss...')
        plot_loss(history, args.scale, args.real_scale, timestr, dataname)
        print('Plotting done!')
                      
        ### Here load the best trained model
        train.load_weights(filepath)
        
    else:
        print("\nLoad the trained model")
        if args.sceneType == 'heatmap':
            print("The scene context is heatmap")
            trained_model = "../trained_model/....hdf5"
        elif args.sceneType == "aerial_photograph":
            print("The scene context is aerial photograph")
            trained_model = "../trained_model/....hdf5"            
        elif args.sceneType == 'segmented_map':            
            print("The scene context is segmented map")
            trained_model = "../models/mcenet_mixed_HC_2000_20200618-213259.hdf5"            
        train.load_weights(trained_model)    
    
    
    #################### START TESTING ####################
    # NOTE THAT IN TESTING PHASE, ONLY x (observed trajectory) is available
    # construct the encoder to get the x_encoded_dense, including scene, occupancy, and trajectory sequence information
    print('Start testing...')   
    print('Load the test data ...')    
    test_ = np.load('../processed/%s_test_data_grid.npz'%dataname)
    test_obs, test_obs_og, test_obs_hmaps = test_['obs'], test_['obs_og'], test_['obs_hmaps']
    # Shift one time step for occupancy grid and scene attention                  
    test_obs_og = test_obs_og[:, 1:, :]
    test_obs_hmaps = test_obs_hmaps[:, 1:, ...]                    
    # Get the residual of the trajectories     
    test_obs_r = test_obs[:, 1:, 2:4] - test_obs[:, :-1, 2:4]
    test_obs_r = test_obs_r*args.resi_scale
               
    ## Get User type one-hot encoding      
    test_obs_type = get_type(test_obs, 3, isObservation=True)
    test_obs_r = np.concatenate((test_obs_r, test_obs_type), axis=-1)
    # Construct the heatmap/scene inputs for the model
    test_obs_hpinput = mobilenet(test_obs_hmaps, obs_seq)
                
    # Double check the test data shape
    print('the shape of test_obs', test_obs_r.shape)   
    print('the shape of test_obs_og', test_obs_og.shape)
    print('the shape of test_obs_hmaps', test_obs_hmaps.shape)
    
    
    # Retrieve the x_encoder and the decoder
    x_encoder=mcenet.X_encoder()
    generator = mcenet.Decoder() 
    
    # get the x_encoded_dense as latent feature for prediction
    x_latent = x_encoder.predict([test_obs_hpinput, test_obs_og, test_obs_r], 
                                 batch_size=args.batch_size)
    
     # Save the summary of the model
    with open('../models/mcenet_mixed_%s_model_summary_%s.txt'%(dataname, timestr), 'w') as f:
        with redirect_stdout(f):
            train.summary()
            x_encoder.summary()
            generator.summary()
            
            
    # START PREDICTING USING THE x_encoded_dense AND SAMPLED LATENT VARIABLE z
    print('Start predicting...')
        
    ### Change the residual back to positions
    last_obs_test = test_obs[:, -1, 2:4]
        
    predictions = []
    for i, x_ in enumerate(x_latent):
        last_pos = last_obs_test[i]
        x_ = np.reshape(x_, [1, -1])
        for i in range(args.num_pred):
            # sampling z from a normal distribution
            z_sample = np.random.rand(1, args.z_dim)
            y_p = generator.predict(np.column_stack([z_sample, x_]))
            y_p = y_p / args.resi_scale
            y_p_ = np.concatenate(([last_pos], np.squeeze(y_p)), axis=0)
            y_p_sum = np.cumsum(y_p_, axis=0)
            predictions.append(y_p_sum[1:, :])
    predictions = np.reshape(predictions, [-1, args.num_pred, args.pred_seq, 2])
    
    test_pred, test_raw = test_['pred'], test_['raw']    
    indexed_predictions = get_index_prediction(test_obs, test_pred, predictions, args.num_pred, args.pred_seq)
    print('Predicting done!')
    
    ## this is the error for the sampling (aveage and minimum)    
    classfied_errors = get_classified_errors(test_pred, indexed_predictions, args.scale/args.real_scale)
    
    ## Select the most likely prediction based on Gaussian rank
    selected_preds = np.zeros((0, 5))
    for pred in indexed_predictions:
        select_index = gauss_rank(pred)
        for m, traj in enumerate(pred):
            if m == select_index:
                selected_preds = np.vstack((selected_preds, traj))
    selected_preds = selected_preds.reshape(indexed_predictions.shape[0], 1, indexed_predictions.shape[2], indexed_predictions.shape[3])
    selected_errors = get_classified_errors(test_pred, selected_preds, args.scale/args.real_scale)           
    
    np.savetxt('../results/mcenet_mixed_%s_statistic_results_%.2f_%s.txt'%(dataname, classfied_errors[0, 2], timestr), 
               classfied_errors, delimiter=',')
    np.savetxt('../selected_results/mcenet_mixed_%s_statistic_results_%.2f_%s.txt'%(dataname, selected_errors[0, 2], timestr), 
               selected_errors, delimiter=',')    
    
    # Save the predictions
    print('Start saving the predictions...')
    np.save('../predictions/mcenet_mixed_%s_test_obs_grid_%.2f_%s.npy'%(dataname, classfied_errors[0, 2], timestr), test_obs)
    np.save('../predictions/mcenet_mixed_%s_test_gt_grid_%.2f_%s.npy'%(dataname, classfied_errors[0, 2], timestr), test_pred)
    np.save('../predictions/mcenet_mixed_%s_test_predictions_grid_%.2f_%s.npy'%(dataname, classfied_errors[0, 2], timestr), 
            indexed_predictions)
    print('Results saved...')
    
    # Plotting the predicted trajectories
    print('Start plotting the predictions...')
    plot_dir = '../plots/mcenet_mixed_%s_plot_%.2f_%s'%(dataname, classfied_errors[0, 2], timestr)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print("Directory mcenet_mixed_%s_plot_%.2f_%s"%(dataname, classfied_errors[0, 2], timestr))
    else:
        print("Directory mcenet_mixed_%s_plot_%.2f_%s"%(dataname, classfied_errors[0, 2], timestr))       
    plot_scenarios(test_raw, test_obs, test_pred, indexed_predictions, bg_image, args.scale, args.real_scale, plot_dir)
    print('Plotting Done!')           
        
    # Save the model configuration
    with open('../hyperparameters/mcenet_mixed_%s_args_%.2f_%s.txt'%(dataname, classfied_errors[0, 2], timestr), 'w') as f:
        for arg in vars(args):
            params = ('%s = %s'%(str(arg), str(getattr(args, arg))))
            f.writelines(params)
            f.writelines('\n')     
   
    
def mobilenet(scenes, seq):
    '''
    This is function to use MobileNetV2
    '''
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    num_seq = scenes.shape[0]
    scenes_r = scenes.reshape(-1, scenes.shape[-3], scenes.shape[-2], scenes.shape[-1])
    output = model.predict(scenes_r)
    output = np.reshape(output, [num_seq, seq, -1])        
    return output 


def get_index_prediction(test_obs, test_pred, predictions, num_pred, pred_seq):
    indexed_predictions = np.zeros([len(predictions), num_pred, pred_seq, 5])
    for u, user_obs in enumerate(test_obs):
        # Get the userId, corresponding frameId, and userType
        # Store them into col0, col1, and col4 respectively
        userId = user_obs[0, 0]
        userType = user_obs[0, 4]
        last_step = user_obs[-1, 1]
        col0 = np.full((pred_seq, 1), userId)
        col1 = [x for x in range(int(last_step+1), int(last_step+pred_seq+1), 1)]
        col1 = np.reshape(col1, [pred_seq, 1])
        col4 = np.full((pred_seq, 1), userType)
        # Concatenate userId, corresponding frameId, and userType for the relative user at each prediction
        user_preds = predictions[u]
        for i, user_pred in enumerate(user_preds):
            user_pred_prime = np.concatenate((col0, col1, user_pred, col4), axis=1)
            indexed_predictions[u, i, :, :] = user_pred_prime        
    return indexed_predictions        
    
    
def get_type(traj, nb_classes, isObservation=True):
    '''
    This is the function to get the one hot encoding type
    '''
    if isObservation:
        # the length should be one step short for observation in residual motion
        num_seqs, length, _ = traj.shape
        length = length - 1
        # user type starts from 0 for one-hot encoding
        targets = traj[:, 1:, -1].astype(int) - np.array([1])
        type_encoding = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        type_encoding = type_encoding.reshape(num_seqs, length, nb_classes)
    else:
        # the length is the same for prediction
        num_seqs, length, _ = traj.shape
        # user type starts from 0 for one-hot encoding
        targets = traj[:, :, -1].astype(int) - np.array([1])         
        type_encoding = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        type_encoding = type_encoding.reshape(num_seqs, length, nb_classes)
    return type_encoding    
    
    
def mak_dir():
    """
    Make all the folders to save the intermediate results
    """
    model_dir = "../models"
    loss_dir = "../loss"
    results_dir = "../results"
    selected_results_dir = "../selected_results"
    plots_dir = "../plots"
    predictions_dir = "../predictions"
    hyperparameters_dir = "../hyperparameters"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print('%s created'%model_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        print('%s created'%results_dir)
    if not os.path.exists(selected_results_dir):
        os.mkdir(selected_results_dir)
        print('%s created'%selected_results_dir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
        print('%s created'%plots_dir)
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
        print('%s created'%predictions_dir)
    if not os.path.exists(hyperparameters_dir):
        os.mkdir(hyperparameters_dir)
        print('%s created'%hyperparameters_dir)
    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        print('%s created'%loss_dir) 
        
        
def gauss_rank(pred_trajs, addnoise=False):
    '''
    pred_trajs: numberofpredictions*length*[userId, FrameId, x, y, type]
    '''
    # Only extract x and y positions
    pred_trajs = pred_trajs[:, :, 2:4]
    # Swap time axis to the first
    pred_trajs_t = np.swapaxes(pred_trajs, 1, 0)
    rank = np.zeros((0, pred_trajs.shape[0]))
    for pred_poss in pred_trajs_t:
 
        # pred_poss is the sampled positions at each time step
        # pred_poss will be used to fit a bivariable gaussian distribution 
        if addnoise == True:
            pred_poss = pred_poss + np.random.normal(0, 1, pred_poss.shape)
        mu = np.mean(pred_poss, axis=0)
        covariance = np.cov(pred_poss.T)
        pos_pdf = multivariate_normal.pdf(pred_poss, mean=mu, cov=covariance)
        rank = np.vstack((rank, pos_pdf))
    rank = np.mean(np.log(rank), axis=0)
    return np.argmax(rank)
    
    
    
    
if __name__ == '__main__':
    main() 