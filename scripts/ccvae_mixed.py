# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 08:47:05 2019
This is the model for trajectory prediction using cvae,
with the consideration of scene context and non-group and group neighhood users
Scene context is modeled using heatmap/aerial_photograph/segmented_map 
Neighborhood interaction is modeled by occupancy map
The motion feature is the residual (x and y-speed) of two consecutive time steps, and one-hot encoding type
The actual time steps has one step less than the observed time steps due to the residual
@author: cheng
"""

from contextlib import redirect_stdout
import numpy as np
import os

from keras.layers import Input, Dense, Lambda, concatenate, LSTM, Activation 
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.core import RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import mse

#from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2


import time
from plots import plot_loss, plot_scenarios
from evaluation import get_classified_errors
from dparser import get_data_repo, generate_data
from hyperparameters import parse_args

from scipy.stats import multivariate_normal
    
    
def main():
    '''
    This is the CVAE model for trajectory prediction. It has:
        three induvidual encoders for heatmap/scene, occupancy grid, and trajectry sequence
        one CVAE encoder for latent z
        one CVAE encoder for generating predictions
    
    '''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dataname = 'HC'
    print(dataname)
    # Make all the necessary folders
    mak_dir()
    bg_image, data_dir, dirs, filename = get_data_repo(dataname)
   
    # Get the hyperparameters
    args = parse_args(dataname)   
    num_pred = args.num_pred
    obs_seq = args.obs_seq - 1 ### minus one is for residual
    pred_seq = args.pred_seq
    neighSize = args.neighSize
    gridRadius = args.gridRadius
    gridAngle = args.gridAngle
    train_mode = args.train_mode
#    retrain_mode = args.retrain_mode
    sceneType = args.sceneType
    n_hidden = args.n_hidden
    z_dim = args.z_dim
    encoder_dim = args.encoder_dim
    z_decoder_dim = args.z_decoder_dim
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    h_drop = args.h_drop
    o_drop = args.o_drop
    s_drop = args.s_drop
    dropout = args.dropout
    lr = args.lr
    epochs = args.epochs
    scale = args.scale
    real_scale = args.real_scale
    beta = args.beta
    resi_scale = args.resi_scale
    
        
    # Generate data
    if args.data_process == True:
        generate_data(dataname, leaveoneout=args.leaveoneout)
    
         
    #################### MODEL CONSTRUCTION STARTS FROM HERE ####################
    # Define the ResNet for scene    
    # Get the parsed last shape of the heatmap input
    parse_dim = 1280
    
    # CONSTRUCT THREE LSTM ENCODERS TO ENCODE HEATMAP, OCCUPANCY GRID, AND TRAJECTORY INFORMATION IN PARALLEL       
    # Construct the heatmap scene model
    # heatmap sence model for the observed data
    h_obs_in = Input(shape=(obs_seq, parse_dim), name='h_obs_in')
    h_obs_out = LSTM(hidden_size,
                     return_sequences=False,
                     stateful=False,
                     dropout=h_drop,
                     name='h_obs_out')(h_obs_in)
    h_obs_Model = Model(h_obs_in, h_obs_out)
    h_obs_Model.summary()        
    # heatmap scene model for the conditioned data
    h_pred_in = Input(shape=(pred_seq, parse_dim), name='h_pred_in')
    h_pred_out = LSTM(hidden_size,
                      return_sequences=False,
                      stateful=False,
                      dropout=h_drop,
                      name='h_pred_out')(h_pred_in)
        
    # Construct the occupancy grid model
    # occupancy grid model for the observed data
    o_obs_in = Input(shape=(obs_seq, int(neighSize/gridRadius*360/gridAngle)), name='o_obs_in')
    o_obs_out = LSTM(hidden_size,
                     return_sequences=False,
                     stateful=False,
                     dropout=o_drop,
                     name='o_obs_out')(o_obs_in)
    o_obs_Model = Model(o_obs_in, o_obs_out)
    o_obs_Model.summary()
    # occupancy grid model for the conditioned data
    o_pred_in = Input(shape=(pred_seq, int(neighSize/gridRadius*360/gridAngle)), name='o_pred_in')
    o_pred_out = LSTM(hidden_size,
                      return_sequences=False,
                      stateful=False,
                      dropout=o_drop,
                      name='o_pred_out')(o_pred_in)
    o_pred_Model = Model(o_pred_in, o_pred_out)
    o_pred_Model.summary()
        
    # Construct the sequence model
    # sequence model for the observed data
    # x_state
    x = Input(shape=(obs_seq, 5), name='x') # including the 3-dimensions for one-hot encoding
    x_conv1d = Conv1D(n_hidden//16, kernel_size=3, strides=1, padding='same', name='x_conv1d')(x)
    # Do I need to have a activation function?
    x_dense = Dense(n_hidden//8, activation='relu', name='x_dense')(x_conv1d)
    x_state = LSTM(n_hidden//8,
                   return_sequences=False,
                   stateful=False,
                   dropout=s_drop,
                   name='x_state')(x_dense) # (1, 64)
    # encoded x
    x_endoced = concatenate([x_state, h_obs_out, o_obs_out], name='x_endoced')
    x_encoded_dense = Dense(encoder_dim, activation='relu', name='x_encoded_dense')(x_endoced)
    
    # sequence model for the conditioned model    
    # y_state
    y = Input(shape=(pred_seq, 2), name='y') 
    y_conv1d = Conv1D(n_hidden//16, kernel_size=3, strides=1, padding='same', name='y_conv1d')(y)
    y_dense = Dense(n_hidden//8, activation='relu', name='y_dense')(y_conv1d)
    y_state = LSTM(n_hidden//8,
                   return_sequences=False,
                   stateful=False,
                   dropout=s_drop,
                   name='y_state')(y_dense) # (1, 64)
    # encoded y
    y_encoded = concatenate([y_state, h_pred_out, o_pred_out], name='y_encoded')
    y_encoded_dense = Dense(encoder_dim, activation='relu', name='y_encoded_dense')(y_encoded)
        
    # CONSTRUCT THE CVAE ENCODER BY FEEDING THE CONCATENATED ENCODED HEATMAP, OCCUPANCY GRID, AND TRAJECTORY INFORMATION
    # the concatenated input
    inputs = concatenate([x_encoded_dense, y_encoded_dense], name='inputs') # (1, 256)
    xy_encoded_d1 = Dense(n_hidden, activation='relu', name='xy_encoded_d1')(inputs) # (1, 512)
    xy_encoded_d2 = Dense(n_hidden//2, activation='relu', name='xy_encoded_d2')(xy_encoded_d1) # (1, 256)
    mu = Dense(z_dim, activation='linear', name='mu')(xy_encoded_d2) # 2
    log_var = Dense(z_dim, activation='linear', name='log_var')(xy_encoded_d2) # 2
        
    # THE REPARAMETERIZATION TRICK FOR THE LATENT VARIABLE z
    # sampling function
    def sampling(params):
        mu, log_var = params
        eps = K.random_normal(shape=(K.shape(mu)[0], z_dim), mean=0., stddev=1.0)
        return mu + K.exp(log_var/2.) * eps
    # sampling z
    z = Lambda(sampling, output_shape=(z_dim,), name='z')([mu, log_var])
    # concatenate the z and x_encoded_dense
    z_cond = concatenate([z, x_encoded_dense], name='z_cond')
        
    # CONSTRUCT THE CVAE DECODER
    z_decoder1 = Dense(n_hidden//2, activation='relu', name='z_decoder1')
    z_decoder2 = RepeatVector(pred_seq, name='z_decoder2')
    z_decoder3 = LSTM(z_decoder_dim,
                      return_sequences=True,
                      stateful=False,
                      dropout=dropout,
                      name='z_decoder3')
    z_decoder4 = Activation('tanh', name='z_decoder4')
    z_decoder5 = Dropout(dropout, name='z_decoder5')
    y_decoder = TimeDistributed(Dense(2), name='y_decoder') # (12, 2)
    
    # Instantiate the decoder by feeding the concatenated z and x_encoded_dense
    z_d1 = z_decoder1(z_cond)
    z_d2 = z_decoder2(z_d1)
    z_d3 = z_decoder3(z_d2)
    z_d4 = z_decoder4(z_d3)
    z_d5 = z_decoder5(z_d4)
    y_prime = y_decoder(z_d5)
                
    # CONSTRUCT THE LOSS FUNCTION FOR THE CVAE MODEL        
    ### Update the utility of the loss function
    def vae_loss(y, y_prime):
        '''
        This is the customized loss function
        '''
        reconstruction_loss = K.mean(mse(y, y_prime)*pred_seq)
        kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis=-1)
        cvae_loss = K.mean(reconstruction_loss*beta + kl_loss*(1-beta))
        return cvae_loss

                
    # BUILD THE CVAE MODEL
    cvae = Model([h_obs_in, h_pred_in, o_obs_in, o_pred_in, x, y], [y_prime])
    opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
    cvae.compile(optimizer=opt, loss=vae_loss)
    cvae.summary()
                
    # CHECK POINT AND SAVE THE BEST MODEL
    filepath="../models/cvae_mixed_%s_%0.f_%s.hdf5"%(dataname, epochs, timestr)
    ## ToDo, Eraly stop
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [earlystop, checkpoint]
    
            
    # Load the data
    # No mattter train_mode or retrain mode, the validation and test data are the same and they are loaded once the model is built.
    # For the 7030 tarining and test seting on the same set, the name of training (30%) and validation (70%) are swapped
    print('Load the validation data ...')    
    V_ = np.load('../processed/%s_validation_data_grid.npz'%dataname)
    val_obs, val_pred, val_raw, val_obs_og, val_pred_og, val_obs_hmaps, val_pred_hmaps = V_['obs'], V_['pred'], V_['raw'], V_['obs_og'], V_['pred_og'], V_['obs_hmaps'], V_['pred_hmaps']
    # Shift one time step for occupancy grid and scene attention                  
    val_obs_og = val_obs_og[:, 1:, :]
    val_obs_hmaps = val_obs_hmaps[:, 1:, ...]                    
    # Get the residual of the trajectories     
    val_traj = np.concatenate((val_obs[:, :, 2:4], val_pred[:, :, 2:4]), axis=1)
    val_traj_r = val_traj[:, 1:, :] - val_traj[:, :-1, :]
    val_traj_r = val_traj_r*resi_scale
    val_obs_r, val_pred_r = val_traj_r[:, :obs_seq, :], val_traj_r[:, obs_seq:, :]                
    ## Get User type one-hot encoding      
    val_obs_type = get_type(val_obs, 3, isObservation=True)
    val_obs_r = np.concatenate((val_obs_r, val_obs_type), axis=-1)            
    # Double check the data shape
    # Here is the validation realted data
    print('the shape of val_obs', val_obs_r.shape)
    print('the shape of val_pred', val_pred_r.shape)
    print('the number of road users in val_raw', len(np.unique(val_raw[:, 0])))
    print('the shape of val_obs_og', val_obs_og.shape)
    print('the shape of val_pred_og', val_pred_og.shape)   
    print('the shape of val_obs_hmaps', val_obs_hmaps.shape)
    print('the shape of val_pred_hmaps', val_pred_hmaps.shape)        
    # Construct the heatmap/scene inputs for the model
    # Please note that during training and validation, in order to calculate latent z, y (ground truth prediction) is visiable
    # y (ground truth prediction) is not available in testing
    val_obs_hpinput = mobilenet(val_obs_hmaps, obs_seq)
    val_pred_hpinput = mobilenet(val_pred_hmaps, pred_seq) 
    
    print('\nLoad the test data ...')  
# =============================================================================
#     test = np.load('../processed/%s_validation_data_grid.npz'%dataname)
#     test_obs, test_pred, test_raw, test_obs_og, test_obs_hmaps = test['obs'], test['pred'], test['raw'], test['obs_og'], test['obs_hmaps']
#     # Shift one time step for occupancy grid and scene attention 
#     test_obs_og = test_obs_og[:, 1:, :]
#     test_obs_hmaps = test_obs_hmaps[:, 1:, ...]       
#     # Get the residual of the trajectories        
#     test_traj = np.concatenate((test_obs[:, :, 2:4], test_pred[:, :, 2:4]), axis=1)
#     test_traj_r = test_traj[:, 1:, :] - test_traj[:, :-1, :]
#     test_traj_r = test_traj_r*resi_scale
#     test_obs_r = test_traj_r[:, :obs_seq, :]            
#     ## Get User type one-hot encoding            
#     test_obs_type = get_type(test_obs, 3, isObservation=True)
#     test_obs_r = np.concatenate((test_obs_r, test_obs_type), axis=-1) 
# =============================================================================
    test_raw = val_raw
    test_obs = val_obs
    test_pred = val_pred
    test_obs_r = val_obs_r
#    test_pred_r = val_pred_r
    test_obs_og = val_obs_og
#    test_pred_og = val_pred_og
    test_obs_hmaps = val_obs_hmaps
#    test_pred_hmaps = val_pred_hmaps
    test_obs_hpinput = val_obs_hpinput
#    test_pred_hpinput = val_pred_hpinput
                  
    # Double check the data shape
    # Here is the testing related data
    print('the shape of test_obs', test_obs_r.shape)   
    print('the number of road users in test_raw', len(np.unique(test_raw[:, 0])))
    print('the shape of test_obs_og', test_obs_og.shape)
    print('the shape of test_obs_hmaps', test_obs_hmaps.shape)            
    # Construct the heatmap/scene inputs for the model
    # Please note that during training and validation, in order to calculate latent z, y (ground truth prediction) is visiable
    # y (ground truth prediction) is not available in testing
    test_obs_hpinput = mobilenet(test_obs_hmaps, obs_seq)
    
    
    

    #################### START TRAINING THE CVAE MODEL ####################
    if train_mode:
        print("\nload the training data")
        T_ = np.load('../processed/%s_training_data_grid.npz'%dataname)
        train_obs, train_pred, train_obs_og, train_pred_og, train_obs_hmaps, train_pred_hmaps = T_['obs'], T_['pred'], T_['obs_og'], T_['pred_og'], T_['obs_hmaps'], T_['pred_hmaps']
        print('You are using the scene context from %s'%sceneType)
        print('Data loaded!')        
        # Shift one time step for occupancy grid and scene attention
        train_obs_og = train_obs_og[:, 1:, :]
        train_obs_hmaps = train_obs_hmaps[:, 1:, ...]    
        # Get the residual of the trajectories
        train_traj = np.concatenate((train_obs[:, :, 2:4], train_pred[:, :, 2:4]), axis=1)
        train_traj_r = train_traj[:, 1:, :] - train_traj[:, :-1, :]
        train_traj_r = train_traj_r*resi_scale
        train_obs_r, train_pred_r = train_traj_r[:, :obs_seq, :], train_traj_r[:, obs_seq:, :]                       
        ## Get User type one-hot encoding
        train_obs_type = get_type(train_obs, 3, isObservation=True)    
        train_obs_r = np.concatenate((train_obs_r, train_obs_type), axis=-1)                             
        # Here is the training related data
        print('Check the input data...')    
        print('the shape of train_obs', train_obs_r.shape)
        print('the shape of train_pred', train_pred_r.shape)  
        #print('the number of road users in train_raw', len(np.unique(train_raw[:, 0])))
        print('the shape of train_obs_og', train_obs_og.shape)
        print('the shape of train_pred_og', train_pred_og.shape)
        print('the shape of train_obs_hmaps', train_obs_hmaps.shape)
        print('the shape of train_pred_hmaps', train_pred_hmaps.shape)            
        # Construct the heatmap/scene inputs for the model
        # Please note that during training and validation, in order to calculate latent z, y (ground truth prediction) is visiable
        # y (ground truth prediction) is not available in testing
        train_obs_hpinput = mobilenet(train_obs_hmaps, obs_seq)
        train_pred_hpinput = mobilenet(train_pred_hmaps, pred_seq)
        
        
        print('Start training the CVAE model...')
        history = cvae.fit(x=[train_obs_hpinput, train_pred_hpinput, train_obs_og, train_pred_og, train_obs_r, train_pred_r],
                           y=train_pred_r,
                           shuffle=True,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=1,
                           callbacks=callbacks_list,
                           validation_data=([val_obs_hpinput, val_pred_hpinput, val_obs_og, val_pred_og, val_obs_r, val_pred_r], val_pred_r))
        print('Training the CVAE model done!')
        
        print('Plotting loss...')
        plot_loss(history, scale, real_scale, timestr, dataname)
        print('Plotting done!')
                      
        ### Here load the best trained model
        cvae.load_weights(filepath)
    else:
        print("\nLoad the trained model")
        if args.sceneType == 'heatmap':
            print("The scene context is heatmap")
            trained_model = "../trained_model/hm_gp_8_8_models/cvae_mixed_HC_10000_20191223-214916.hdf5"
        elif args.sceneType == "aerial_photograph":
            print("The scene context is aerial photograph")
            trained_model = "../trained_model/ap_gp_8_8_models/cvae_mixed_HC_10000_20191224-102920.hdf5"            
        elif args.sceneType == 'segmented_map':            
            print("The scene context is segmented map")
            trained_model = "../trained_model/sm_gp_8_8_models/cvae_mixed_HC_10000_20191224-112444.hdf5"            
        cvae.load_weights(trained_model)
        
    
        
    #################### START TESTING ####################
    # NOTE THAT IN TESTING PHASE, ONLY x (observed trajectory) is available
    # construct the encoder to get the x_encoded_dense, including heatmap, occupancy, and trajectory sequence information
    print('Start testing...')
    print('Construct the CVAE encoder')
        
    x_encoder = Model([h_obs_in, o_obs_in, x], x_encoded_dense)
    x_encoder.summary()
    # get the x_encoded_dense as latent feature for prediction
    x_latent = x_encoder.predict([test_obs_hpinput, test_obs_og, test_obs_r], batch_size=batch_size)
        
    # CONSTRUCT THE DECODER
    print('Construct the CVAE decoder')
    decoder_input = Input(shape=(z_dim+encoder_dim, ), name='decoder_input')
    _z_d1 = z_decoder1(decoder_input)
    _z_d2 = z_decoder2(_z_d1)
    _z_d3 = z_decoder3(_z_d2)
    _z_d4 = z_decoder4(_z_d3)
    _z_d5 = z_decoder5(_z_d4)
    _y_prime = y_decoder(_z_d5)
    generator = Model(decoder_input, _y_prime)
    generator.summary() 
        
    # Save the summary of the model
    with open('../models/cvae_mixed_%s_model_summary_%s.txt'%(dataname, timestr), 'w') as f:
        with redirect_stdout(f):
            h_obs_Model.summary()
            o_obs_Model.summary()
            o_pred_Model.summary()
            cvae.summary()
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
        for i in range(num_pred):
            # sampling z from a normal distribution
            z_sample = np.random.rand(1, z_dim)
            y_p = generator.predict(np.column_stack([z_sample, x_]))
            y_p = y_p / resi_scale
            y_p_ = np.concatenate(([last_pos], np.squeeze(y_p)), axis=0)
            y_p_sum = np.cumsum(y_p_, axis=0)
            predictions.append(y_p_sum[1:, :])
    predictions = np.reshape(predictions, [-1, num_pred, pred_seq, 2])
        
    indexed_predictions = get_index_prediction(test_obs, test_pred, predictions, num_pred, pred_seq)
    print('Predicting done!')
    
    ## this is the error for the sampling (aveage and minimum)    
    classfied_errors = get_classified_errors(test_pred, indexed_predictions, scale/real_scale)
    
    ## Select the most likely prediction based on Gaussian rank
    selected_preds = np.zeros((0, 5))
    for pred in indexed_predictions:
        select_index = gauss_rank(pred)
        for m, traj in enumerate(pred):
            if m == select_index:
                selected_preds = np.vstack((selected_preds, traj))
    selected_preds = selected_preds.reshape(indexed_predictions.shape[0], 1, indexed_predictions.shape[2], indexed_predictions.shape[3])
    selected_errors = get_classified_errors(test_pred, selected_preds, scale/real_scale)           
    
    np.savetxt('../results/cvae_mixed_%s_statistic_results_%.2f_%s.txt'%(dataname, classfied_errors[0, 2], timestr), classfied_errors, delimiter=',')
    np.savetxt('../selected_results/cvae_mixed_%s_statistic_results_%.2f_%s.txt'%(dataname, selected_errors[0, 2], timestr), selected_errors, delimiter=',')
    
    
    # Save the predictions
    print('Start saving the predictions...')
    np.save('../predictions/cvae_mixed_%s_test_obs_grid_%.2f_%s.npy'%(dataname, classfied_errors[0, 2], timestr), test_obs)
    np.save('../predictions/cvae_mixed_%s_test_gt_grid_%.2f_%s.npy'%(dataname, classfied_errors[0, 2], timestr), test_pred)
    np.save('../predictions/cvae_mixed_%s_test_predictions_grid_%.2f_%s.npy'%(dataname, classfied_errors[0, 2], timestr), indexed_predictions)
    print('Results saved...')
    
    # Plotting the predicted trajectories
    print('Start plotting the predictions...')
    plot_dir = '../plots/cvae_mixed_%s_plot_%.2f_%s'%(dataname, classfied_errors[0, 2], timestr)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print("Directory cvae_mixed_%s_plot_%.2f_%s"%(dataname, classfied_errors[0, 2], timestr))
    else:
        print("Directory cvae_mixed_%s_plot_%.2f_%s"%(dataname, classfied_errors[0, 2], timestr))       
    plot_scenarios(test_raw, test_obs, test_pred, indexed_predictions, bg_image, scale, real_scale, plot_dir)
    print('Plotting Done!')           
        
    # Save the model configuration
    with open('../hyperparameters/cvae_mixed_%s_args_%.2f_%s.txt'%(dataname, classfied_errors[0, 2], timestr), 'w') as f:
        for arg in vars(args):
            params = ('%s = %s'%(str(arg), str(getattr(args, arg))))
            f.writelines(params)
            f.writelines('\n')    


def mobilenet(hmaps, seq):
    '''
    This is function to use ResNet50
    '''
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

    output = []
    for u, user in enumerate(hmaps):
        for s, step in enumerate(user):
            hmap_input = np.expand_dims(step, axis=0)
            hmap_parsed = model.predict(hmap_input)
            output.append(hmap_parsed)
    output = np.reshape(output, [hmaps.shape[0], seq, -1])
    return output 
    


def parse_hmap(hmaps, hmap_model, seq):
    '''
    This is the fucntion to parse the observed trajectory for each user using CNN
    '''
    output = []
    for u, user in enumerate(hmaps):
        for s, step in enumerate(user):
            hmap_input = np.expand_dims(step, axis=0)
            hmap_parsed = hmap_model.predict(hmap_input)
            output.append(hmap_parsed)
    output = np.reshape(output, [hmaps.shape[0], seq, -1])
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


def mak_dir():
    # Make all the folders to save the intermediate results
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
    

        

    
