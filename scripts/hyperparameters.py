# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:41:29 2020
This is the function to store all the hyperparameters

@author: cheng
"""
import sys
import argparse


def parse_args(data_name):
    desc = "Tensorflow/keras implementation of RNN LSTM for trajectory prediction"
    parser = argparse.ArgumentParser(description=desc)
    
    # Define the distance threshold for DBSCAN
    meter = 1.5
    
    ### The size for each dataset
    # This is the size for HBS   
    if data_name == 'HBS':
        print('\nYou are working on HBS dataset')
        parser.add_argument('--scale', type=float, default=1288.0, help='The normalization scale to pixel')
        parser.add_argument('--dimensions', type=int, default=[709, 1076], help='The height and width of the background image')
        parser.add_argument('--dist_thre', type=float, default=meter, help='The distance threshold for friends detection')
        parser.add_argument('--real_scale', type=float, default=12.0, help='The normalization scale to meters')
        
    if data_name == 'HC':
        print('\nYou are working on HC dataset')
        parser.add_argument('--scale', type=float, default=1991.0, help='The normalization scale to pixel')
        parser.add_argument('--dimensions', type=int, default=[1761, 767], help='The height and width of the background image')
        parser.add_argument('--dist_thre', type=float, default=meter, help='The distance threshold for friends detection')
        parser.add_argument('--real_scale', type=float, default=21.185660421977854, help='The normalization scale to meters')
   
    # This is the size for SDD/gates/3
    if data_name == 'SDDgates3':
        print('\nYou are working on SDD_GATES_VIDEO3')	
        parser.add_argument('--scale', type=float, default=2972.0, help='The normalization scale to pixel')
        parser.add_argument('--dimensions', type=int, default=[2002, 1432], help='The height and width of the background image')
        parser.add_argument('--dist_thre', type=float, default=meter, help='The distance threshold for friends detection')
        parser.add_argument('--real_scale', type=float, default=29.027883858956827, help='The normalization scale to meters')
	
	
    # Hyperparameters for the dataset
#    datanames = ['HBS', 'HC', 'SDDgates3']
#    parser.add_argument('--data_name', type=str, choices=datanames, default='HBS', help='specify the input data name')
    parser.add_argument('--ratios', type=float, default=[0.3, 1.0], help='This is the ratios for data partition')
    parser.add_argument('--num_pred', type=int, default=10, help='This is the number of predictions for each agent')
    parser.add_argument('--data_process', type=bool, default=True, help='This is the flag for data partitioning')
    parser.add_argument('--split', type=float, default=0.8, help='This is the split for training and validation')
    parser.add_argument('--leaveoneout', type=bool, default=False, help='Whether process data using leave-one-out policy')
    parser.add_argument('--num_units', type=int, default=128, help='Number of GRU units for x or y input encoder')
    parser.add_argument('--obs_seq', type=int, default=8, help='Number of time steps observed')
    parser.add_argument('--pred_seq', type=int, default=8, help='Number of time steps to be predicted')
    parser.add_argument('--neighSize', type=int, default=32, help='The size of neighborhood')
    parser.add_argument('--overlap', type=float, default=0.99, help='The overlap ratio for coexisting time between friends')
    parser.add_argument('--gridRadius', type=int, default=4, help='The radius of neighborhood grid')
    parser.add_argument('--gridAngle', type=int, default=45, help='The angle size of neighborhood gird')
    parser.add_argument('--sceneType', type=str, choices=['heatmap', 'segmented_map', 'aerial_photograph'], default='segmented_map', help='The background scene type')    
    parser.add_argument('--hmap_dim', type=int, default=[224, 224], help='The height and width of the attended heatmap grid')
    parser.add_argument('--sigma', type=int, default=16, help='The kernel size for heatmap histogram sigma')
    parser.add_argument('--hgrid_size', type=int, default=112, help='The size of the headmap grid')    
    parser.add_argument('--train_mode', type=bool, default=True, help='This is the training mode')
    parser.add_argument('--n_hidden', type=int, default=512, help='This is the hidden size of the cvae') 
    parser.add_argument('--z_dim', type=int, default=16, help='This is the size of the latent variable')
    parser.add_argument('--encoder_dim', type=int, default=16, help='This is the size of the encoder output dimension')
    parser.add_argument('--z_decoder_dim', type=int, default=128, help='This is the size of the decoder LSTM dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='The size of GRU hidden state')
    parser.add_argument('--conv1d_size', type=int, default=8, help='The filters for the Conv1D')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # parser.add_argument('--h_drop', type=float, default=0.7, help='The dropout rate for heatmap grid')
    parser.add_argument('--o_drop', type=float, default=0.3, help='The dropout rate for occupancy grid')
    parser.add_argument('--s_drop', type=float, default=0.1, help='The dropout rate for trajectory sequence')
    parser.add_argument('--c_drop', type=float, default=0.7, help='The dropout rate for scene context')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for prediction')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0, help='Decay rate')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of batches')
    parser.add_argument('--beta', type=float, default=0.99, help='Loss weight')
    parser.add_argument('--resi_scale', type=float, default=50.0, help='The displacement scale')
    
                        
    args = parser.parse_args(sys.argv[1:])
    return args

