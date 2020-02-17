# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:21:19 2019
This is the module to set the hyperparameters for each dataset 

@author: cheng

Please find the data for ['HBS', 'HC', 'SDDgates3'] in the following paper

HBS
@article{rinke2017multi,
  title={A multi-layer social force approach to model interactions in shared spaces using collision prediction},
  author={Rinke, N and Schiermeyer, C and Pascucci, F and Berkhahn, V and Friedrich, B},
  journal={Transportation Research Procedia},
  volume={25},
  pages={1249--1267},
  year={2017},
}

HC
@inproceedings{cheng2019pedestrian,
  title={Pedestrian Group Detection in Shared Space},
  author={Cheng, Hao and Li, Yao and Sester, Monika},
  booktitle={Intelligent Vehicles Symposium},
  pages={1707--1714},
  year={2019},
  organization={IEEE}
}

SDDgates3
@inproceedings{robicquet2016learning,
  title={Learning social etiquette: Human trajectory understanding in crowded scenes},
  author={Robicquet, Alexandre and Sadeghian, Amir and Alahi, Alexandre and Savarese, Silvio},
  booktitle={ECCV},
  pages={549--565},
  year={2016},
}

"""

import argparse
import os
import numpy as np
import glob

from utils_grid import dataLoader

def generate_data():
    '''
    This is the function to generate the leave-one-out cross validation data
    '''
    print("Start generating data...")  
#    datanames = ['HBS', 'HC', 'SDDgates3']
    datanames = ['HC']
    
    # Make the folder to store all the processed data
    processed_dir = "../processed"
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
        print('%s created'%processed_dir)
        
    # Make the folder to store the precissed heatmap
    heatmasp_dir = "../heatmaps"
    if not os.path.exists(heatmasp_dir):
        os.mkdir(heatmasp_dir)
        print('%s created'%heatmasp_dir)
    
    
    for dataname in datanames:
        args = parse_args(dataname)
        bg_image, data_dir, dirs, filename = get_data_repo(dataname)
        
        # process data
        if args.data_process:
            print("Start processing the data...")
            # Split the same data into test and validation      
            partition_data(data_dir, dirs, args.ratios, filename)
            training, validation = dataLoader(dirs=dirs, filename=filename, args=args, dataname=dataname, background=args.sceneType)
            [train_obs, train_pred, train_raw, train_obs_og, train_pred_og, train_obs_hmaps, train_pred_hmaps] = training
            [val_obs, val_pred, val_raw, val_obs_og, val_pred_og, val_obs_hmaps, val_pred_hmaps] = validation
            print("Processing the data done!")
        # If the data is already processed, no need to processed again 
        else:
        # load the data
            print('Load the data...')
            T_ = np.load('../processed/%s_training_data_grid.npz'%dataname)
            train_obs, train_pred, train_raw, train_obs_og, train_pred_og, train_obs_hmaps, train_pred_hmaps = T_['obs'], T_['pred'], T_['raw'], T_['obs_og'], T_['pred_og'], T_['obs_hmaps'], T_['pred_hmaps']
            V_ = np.load('../processed/%s_validation_data_grid.npz'%dataname)
            val_obs, val_pred, val_raw, val_obs_og, val_pred_og, val_obs_hmaps, val_pred_hmaps = V_['obs'], V_['pred'], V_['raw'], V_['obs_og'], V_['pred_og'], V_['obs_hmaps'], V_['pred_hmaps']
            print('Data loaded!')
            
        # Double check the data shape
        # Here is the training related data
        print('Check the input data...')    
        print('the shape of train_obs', train_obs.shape)
        print('the shape of train_pred', train_pred.shape)
        print('the number of road users in train_raw', len(np.unique(train_raw[:, 0])))
        print('the shape of train_obs_og', train_obs_og.shape)
        print('the shape of train_pred_og', train_pred_og.shape)
        print('the shape of train_obs_hmaps', train_obs_hmaps.shape)
        print('the shape of train_pred_hmaps', train_pred_hmaps.shape)
        # Here is the validation realted data
        print('the shape of val_obs', val_obs.shape)
        print('the shape of val_pred', val_pred.shape)
#        print('the shape of val_raw', len(val_raw))
        print('the number of road users in val_raw', len(np.unique(val_raw[:, 0])))
        print('the shape of val_obs_og', val_obs_og.shape)
        print('the shape of val_pred_og', val_pred_og.shape)   
        print('the shape of val_obs_hmaps', val_obs_hmaps.shape)
        print('the shape of val_pred_hmaps', val_pred_hmaps.shape)
        
    
    # Using the one leaveout for training data
    merge = False
    if merge == True:
        print("generate the leave-one-out cross validation data") 
#        datanames_ = ['HBS', 'HC', 'SDDgates3', 'SDDhyang4', 'SDDdeathCircle0', 'SDDdeathCircle1']
        datanames_ = ['HBS', 'HC', 'SDDgates3']
        for dataname in datanames_:
            args = parse_args(dataname)
            og_size = int((args.neighSize / args.gridRadius) * (360 / args.gridAngle))
            [hm_size_h, hm_size_w] = args.hmap_dim
            merge_data(dataname, args.obs_seq, args.pred_seq, og_size, hm_size_h, hm_size_w)
            merge_testval(dataname, args.obs_seq, args.pred_seq, og_size, hm_size_h, hm_size_w)


def parse_args(data_name):
    desc = "Tensorflow implementation of RNN GRU for trajectory prediction"
    parser = argparse.ArgumentParser(description=desc)
    
    # Define the distance threshold for DBSCAN
    meter = 1.5
    
    ### The size for each dataset
    # This is the size for HBS   
    if data_name == 'HBS':
        print('\nYou are working on HBS dataset')
        parser.add_argument('--scale', type=float, default=1288.0, help='The normalization scale to pixel')
        parser.add_argument('--dimentions', type=int, default=[709, 1076], help='The height and width of the background image')
        parser.add_argument('--dist_thre', type=float, default=meter, help='The distance threshold for friends detection')
        parser.add_argument('--real_scale', type=float, default=12.0, help='The normalization scale to meters')
        
    if data_name == 'HC':
        print('\nYou are working on HC dataset')
        parser.add_argument('--scale', type=float, default=1991.0, help='The normalization scale to pixel')
        parser.add_argument('--dimentions', type=int, default=[1761, 767], help='The height and width of the background image')
        parser.add_argument('--dist_thre', type=float, default=meter, help='The distance threshold for friends detection')
        parser.add_argument('--real_scale', type=float, default=21.185660421977854, help='The normalization scale to meters')
   
    # This is the size for SDD/gates/3
    if data_name == 'SDDgates3':
        print('\nYou are working on SDD_GATES_VIDEO3')	
        parser.add_argument('--scale', type=float, default=2972.0, help='The normalization scale to pixel')
        parser.add_argument('--dimentions', type=int, default=[2002, 1432], help='The height and width of the background image')
        parser.add_argument('--dist_thre', type=float, default=meter, help='The distance threshold for friends detection')
        parser.add_argument('--real_scale', type=float, default=29.027883858956827, help='The normalization scale to meters')
	
	
    # Hyperparameters for the dataset
    datanames = ['HBS', 'HC', 'SDDgates3']
    parser.add_argument('--data_name', type=str, choices=datanames, default='HC', help='specify the input data name')
    parser.add_argument('--ratios', type=float, default=[0.7, 1.0], help='This is the ratios for data partition')
    parser.add_argument('--num_pred', type=int, default=10, help='This is the number of predictions for each agent')
    parser.add_argument('--data_process', type=bool, default=True, help='This is the flag for data partitioning')
    parser.add_argument('--num_units', type=int, default=128, help='Number of GRU units for x or y input encoder')
    parser.add_argument('--obs_seq', type=int, default=8, help='Number of time steps observed')
    parser.add_argument('--pred_seq', type=int, default=8, help='Number of time steps to be predicted')
    parser.add_argument('--neighSize', type=int, default=32, help='The size of neighborhood')
    parser.add_argument('--overlap', type=float, default=0.99, help='The overlap ratio for coexisting time between friends')
    parser.add_argument('--gridRadius', type=int, default=4, help='The radius of neighborhood grid')
    parser.add_argument('--gridAngle', type=int, default=45, help='The angle size of neighborhood gird')
    parser.add_argument('--sceneType', type=str, choices=['heatmap', 'segmented_map', 'aerial_photograph'], default='segmented_map', help='The background scene type')    
    parser.add_argument('--hmap_dim', type=int, default=[128, 128], help='The height and width of the attended heatmap grid')
    parser.add_argument('--sigma', type=int, default=16, help='The kernel size for heatmap histogram sigma')
    parser.add_argument('--hgrid_size', type=int, default=64, help='The size of the headmap grid')    
    parser.add_argument('--train_mode', type=bool, default=True, help='This is the training mode')
    parser.add_argument('--o_dim', type=int, default=512, help='The dimension of the CNN output')
    parser.add_argument('--n_hidden', type=int, default=512, help='This is the hidden size of the cvae') 
    parser.add_argument('--z_dim', type=int, default=16, help='This is the size of the latent variable')
    parser.add_argument('--encoder_dim', type=int, default=16, help='This is the size of the encoder output dimension')
    parser.add_argument('--z_decoder_dim', type=int, default=128, help='This is the size of the decoder LSTM dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='The size of GRU hidden state')
    parser.add_argument('--conv1d_size', type=int, default=8, help='The filters for the Conv1D')
    parser.add_argument('--batch_size', type=int, default=700, help='Batch size')
    parser.add_argument('--h_drop', type=float, default=0.7, help='The dropout rate for heatmap grid')
    parser.add_argument('--o_drop', type=float, default=0.3, help='The dropout rate for occupancy grid')
    parser.add_argument('--s_drop', type=float, default=0.0, help='The dropout rate for trajectory sequence')
    parser.add_argument('--c_drop', type=float, default=0.2, help='The dropout rate for concatenated input')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for prediction')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0, help='Decay rate')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of batches')
    parser.add_argument('--beta', type=float, default=0.99, help='Loss weight')
    parser.add_argument('--resi_scale', type=float, default=50.0, help='The displacement scale')
    args = parser.parse_args()
    return args


def get_data_repo(data_name):
    '''
    This is the function to get the corresponding repository for each dataset
    '''
    # This is the repository for HBS
    if data_name == 'HBS':
        bg_image = '../data/HBS/HH_Bergedorf.jpg'
        data_dir = '../data/HBS/trajectories.csv'
        dirs = ['../data/HBS', 
                '../data/HBS/training', 
                '../data/HBS/validation', 
                '../data/HBS/testing']
        make_repo(dirs)
        filename = 'HBS.csv'
        
    # This is the repository for HC
    if data_name == 'HC':
        bg_image = '../data/HC/background.png'
        data_dir = '../data/HC/merged/trajectories.csv'
        dirs = ['../data/HC/merged', 
            '../data/HC/merged/training', 
            '../data/HC/merged/validation', 
            '../data/HC/merged/testing']
        make_repo(dirs)
        filename = 'HC.csv'
        
        
    # This is the repository for SDD/gates/3
    if data_name == 'SDDgates3':
        bg_image = '../data/SDD/gates/video3/reference.jpg'
        data_dir = '../data/SDD/gates/video3/trajectories.csv'
        dirs = ['../data/SDD/gates/video3', 
            '../data/SDD/gates/video3/training', 
            '../data/SDD/gates/video3/validation', 
            '../data/SDD/gates/video3/testing']
        make_repo(dirs)
        filename = 'gates3.csv'
        
    return bg_image, data_dir, dirs, filename
    


def make_repo(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
            print('%s created'%d)
        else:
            print("%s exists"%d)
            
            
def partition_data(data_dir, dirs, ratios, filename):
    '''
    This is the function to partition the data into training_data, validation_data
    In the leave-one-out cross validation, we do not need to train, valid, and test on the same dataset,,
    so here we only split the data for training and validation, we can still train and test on the same dataset.
    But in this case, the validation and test dataset will be the same
    In the leave-one-out cross validation, the splitted data (within the same dataset) will be merged as the tes_val dataset;
    The other datasets will be merged as the training dataset
    '''
    data = np.genfromtxt(data_dir, delimiter=',')
    # All the time steps    
    num_frames = len(np.unique(data[:, 0]))
    train_r = ratios[0]
    train_ratio = int(num_frames*train_r)
    # Sort the data by colomn: userId, frameId, x, y, userType 
    order = [1, 0, 2, 3, 4]
    i = np.argsort(order)
    ordered_data = data[:, i]
    # Save the ordered data
    # np.savetxt(os.path.join(dirs[0], 'ordered.csv'), ordered_data, delimiter=',') 
    # Save the training data    
    train = ordered_data[ordered_data[:, 1]<=train_ratio, :]
    np.savetxt(os.path.join(dirs[1], filename), train, delimiter=',')    
    # Save the validation data
    validation = ordered_data[ordered_data[:, 1]>train_ratio]
    np.savetxt(os.path.join(dirs[2], filename), validation, delimiter=',')


def merge_data(dataname, obs_seq, pred_seq, og_size, hm_size_h, hm_size_w):
    
    train_obs = np.empty([0, obs_seq, 5])
    train_pred = np.empty([0, pred_seq, 5])
    train_raw = np.empty([0, 5])
    train_obs_og = np.empty([0, obs_seq, og_size])
    train_pred_og = np.empty([0, pred_seq, 64])
    train_obs_hmaps = np.empty([0, obs_seq, hm_size_h, hm_size_w, 3])
    train_pred_hmaps = np.empty([0, pred_seq, hm_size_h, hm_size_w, 3])
    
    filenames = glob.glob('../processed/*.npz')
    for filename in filenames:
        if (dataname in filename) or ("leveloneout_training" in filename) or ("leveloneout_testing" in filename):
            continue
        obs, pred, raw, obs_og, pred_og, obs_hmaps, pred_hmaps = get_data(filename)
        train_obs = np.concatenate((train_obs, obs))
        train_pred = np.concatenate((train_pred, pred))
        train_raw = np.concatenate((train_raw, raw))
        train_obs_og = np.concatenate((train_obs_og, obs_og))
        train_pred_og = np.concatenate((train_pred_og, pred_og))
        train_obs_hmaps = np.concatenate((train_obs_hmaps, obs_hmaps))
        train_pred_hmaps = np.concatenate((train_pred_hmaps, pred_hmaps))
        
    print('Check the input data...')    
    print('the shape of train_obs', train_obs.shape)
    print('the shape of train_pred', train_pred.shape)
    print('the shape of train_obs_og', train_obs_og.shape)
    print('the shape of train_pred_og', train_pred_og.shape)
    print('the shape of train_obs_hmaps', train_obs_hmaps.shape)
    print('the shape of train_pred_hmaps', train_pred_hmaps.shape)
    
    np.savez('../processed/%s_leveloneout_training_data_grid.npz'%dataname, 
             obs=train_obs, pred=train_pred, raw=train_raw, obs_og=train_obs_og, pred_og=train_pred_og, obs_hmaps=train_obs_hmaps, pred_hmaps=train_pred_hmaps)
    print('%s_leveloneout_training_data is saved'%dataname) 


def merge_testval(dataname, obs_seq, pred_seq, og_size, hm_size_h, hm_size_w):
    train_obs = np.empty([0, obs_seq, 5])
    train_pred = np.empty([0, pred_seq, 5])
    train_raw = np.empty([0, 5])
    train_obs_og = np.empty([0, obs_seq, og_size])
    train_pred_og = np.empty([0, pred_seq, 64])
    train_obs_hmaps = np.empty([0, obs_seq, hm_size_h, hm_size_w, 3])
    train_pred_hmaps = np.empty([0, pred_seq, hm_size_h, hm_size_w, 3])
    filenames = glob.glob('../processed/*.npz')
    for filename in filenames:
        if (dataname in filename) and ("leveloneout_training" not in filename) and ("leveloneout_testing" not in filename):
            obs, pred, raw, obs_og, pred_og, obs_hmaps, pred_hmaps = get_data(filename)
            train_obs = np.concatenate((train_obs, obs))
            train_pred = np.concatenate((train_pred, pred))
            train_raw = np.concatenate((train_raw, raw))
            train_obs_og = np.concatenate((train_obs_og, obs_og))
            train_pred_og = np.concatenate((train_pred_og, pred_og))
            train_obs_hmaps = np.concatenate((train_obs_hmaps, obs_hmaps))
            train_pred_hmaps = np.concatenate((train_pred_hmaps, pred_hmaps))
    print('Check the input data...')    
    print('the shape of train_obs', train_obs.shape)
    print('the shape of train_pred', train_pred.shape)
    print('the shape of train_obs_og', train_obs_og.shape)
    print('the shape of train_pred_og', train_pred_og.shape)
    print('the shape of train_obs_hmaps', train_obs_hmaps.shape)
    print('the shape of train_pred_hmaps', train_pred_hmaps.shape)
    
    np.savez('../processed/%s_leveloneout_testing_data_grid.npz'%dataname, 
             obs=train_obs, pred=train_pred, raw=train_raw, obs_og=train_obs_og, pred_og=train_pred_og, obs_hmaps=train_obs_hmaps, pred_hmaps=train_pred_hmaps)
    print('%s_leveloneout_testing_data is saved'%dataname)
    
    
def get_data(data_dir):
    D = np.load(data_dir)
    print('get data from %s'%(data_dir))
    obs, pred, raw, obs_og, pred_og, obs_hmaps, pred_hmaps = D['obs'], D['pred'], D['raw'], D['obs_og'], D['pred_og'], D['obs_hmaps'], D['pred_hmaps']  
    return obs, pred, raw, obs_og, pred_og, obs_hmaps, pred_hmaps




if __name__ == "__main__":
    generate_data()
