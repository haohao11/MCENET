# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:23:17 2020

@author: cheng
"""

from keras.layers import Input, Dense, Lambda, concatenate, LSTM, Activation
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.core import RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.losses import mse



class MCENET():
    
    def __init__(self, args):
        # Store the hyperparameters
        self.args = args
        self.num_pred = args.num_pred
        self.obs_seq = args.obs_seq - 1 ### minus one is for residual
        self.pred_seq = args.pred_seq
        self.neighSize = args.neighSize
        self.gridRadius = args.gridRadius
        self.gridAngle = args.gridAngle
        self.train_mode = args.train_mode
        self.sceneType = args.sceneType
        self.n_hidden = args.n_hidden
        self.z_dim = args.z_dim
        self.encoder_dim = args.encoder_dim
        self.z_decoder_dim = args.z_decoder_dim
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.c_drop = args.c_drop
        self.o_drop = args.o_drop
        self.s_drop = args.s_drop
        self.dropout = args.dropout
        self.lr = args.lr
        self.epochs = args.epochs
        self.scale = args.scale
        self.real_scale = args.real_scale
        self.beta = args.beta
        self.resi_scale = args.resi_scale
        self.parse_dim = 1280 # The output dimension of MobileNet
        
        # CONSTRUCT THREE LSTM ENCODERS TO ENCODE SCENE, OCCUPANCY GRID, AND TRAJECTORY INFORMATION IN PARALLEL
        ## Construct the context scene model
        ## Sence model for the observed data
        self.s_obs_in = Input(shape=(self.obs_seq, self.parse_dim), name='s_obs_in')
        self.s_obs_out = LSTM(self.hidden_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=self.c_drop,
                         name='s_obs_out')(self.s_obs_in)
        self.s_obs_Model = Model(self.s_obs_in, self.s_obs_out)
        ## Scene model for the ground truth data in prediction time
        self.s_pred_in = Input(shape=(self.pred_seq, self.parse_dim), name='s_pred_in')
        self.s_pred_out = LSTM(self.hidden_size,
                          return_sequences=False,
                          stateful=False,
                          dropout=self.c_drop,
                          name='s_pred_out')(self.s_pred_in)
        
        ## Construct the occupancy grid model
        ## Occupancy grid model for the observed data
        self.o_obs_in = Input(shape=(self.obs_seq, int(self.neighSize/self.gridRadius*360/self.gridAngle)), name='o_obs_in')
        self.o_obs_out = LSTM(self.hidden_size,
                         return_sequences=False,
                         stateful=False,
                         dropout=self.o_drop,
                         name='o_obs_out')(self.o_obs_in)
        self.o_obs_Model = Model(self.o_obs_in, self.o_obs_out)
        ## Occupancy grid model for the ground truth data in prediction time
        self.o_pred_in = Input(shape=(self.pred_seq, int(self.neighSize/self.gridRadius*360/self.gridAngle)), name='o_pred_in')
        self.o_pred_out = LSTM(self.hidden_size,
                          return_sequences=False,
                          stateful=False,
                          dropout=self.o_drop,
                          name='o_pred_out')(self.o_pred_in)
        self.o_pred_Model = Model(self.o_pred_in, self.o_pred_out)
        
        ## Construct the sequence model
        ## Sequence model for the observed data
        ## x_state
        self.x = Input(shape=(self.obs_seq, 5), name='x') # including the 3-dimensions for one-hot encoding
        self.x_conv1d = Conv1D(self.n_hidden//16, kernel_size=3, strides=1, padding='same', name='x_conv1d')(self.x)
        self.x_dense = Dense(self.n_hidden//8, activation='relu', name='x_dense')(self.x_conv1d)
        self.x_state = LSTM(self.n_hidden//8,
                       return_sequences=False,
                       stateful=False,
                       dropout=self.s_drop,
                       name='x_state')(self.x_dense) # (1, 64)
        ## Encoded x
        self.x_endoced = concatenate([self.x_state, self.s_obs_out, self.o_obs_out], name='x_endoced')
        self.x_encoded_dense = Dense(self.encoder_dim, activation='relu', name='x_encoded_dense')(self.x_endoced)
        
        ## Sequence model for the ground truth data in prediction time    
        ## y_state
        self.y = Input(shape=(self.pred_seq, 2), name='y') 
        self.y_conv1d = Conv1D(self.n_hidden//16, kernel_size=3, strides=1, padding='same', name='y_conv1d')(self.y)
        self.y_dense = Dense(self.n_hidden//8, activation='relu', name='y_dense')(self.y_conv1d)
        self.y_state = LSTM(self.n_hidden//8,
                       return_sequences=False,
                       stateful=False,
                       dropout=self.s_drop,
                       name='y_state')(self.y_dense)
        ## Encoded y
        self.y_encoded = concatenate([self.y_state, self.s_pred_out, self.o_pred_out], name='y_encoded')
        self.y_encoded_dense = Dense(self.encoder_dim, activation='relu', name='y_encoded_dense')(self.y_encoded)
        
            
        # CONSTRUCT THE MCENet ENCODER BY FEEDING THE CONCATENATED ENCODED SCENE, OCCUPANCY GRID, AND TRAJECTORY INFORMATION
        ## Get the concatenated input
        self.inputs = concatenate([self.x_encoded_dense, self.y_encoded_dense], name='inputs')
        self.xy_encoded_d1 = Dense(self.n_hidden, activation='relu', name='xy_encoded_d1')(self.inputs)
        self.xy_encoded_d2 = Dense(self.n_hidden//2, activation='relu', name='xy_encoded_d2')(self.xy_encoded_d1)
        ## Get the mean and log variance for the z variable using two side-by-side fc layers 
        self.mu = Dense(self.z_dim, activation='linear', name='mu')(self.xy_encoded_d2)
        self.log_var = Dense(self.z_dim, activation='linear', name='log_var')(self.xy_encoded_d2)
        
        
        # THE REPARAMETERIZATION TRICK FOR THE LATENT VARIABLE z
        # Sampling function
        z_dim = self.z_dim
        def sampling(params):
            mu, log_var = params
            eps = K.random_normal(shape=(K.shape(mu)[0], z_dim), mean=0., stddev=1.0)
            return mu + K.exp(log_var/2.) * eps        
        ## Sampling z
        self.z = Lambda(sampling, output_shape=(self.z_dim,), name='z')([self.mu, self.log_var])
        ## Concatenate the z and x_encoded_dense
        self.z_cond = concatenate([self.z, self.x_encoded_dense], name='z_cond')
        
        
        # CONSTRUCT THE MCENet DECODER
        self.z_decoder1 = Dense(self.n_hidden//2, activation='relu', name='z_decoder1')
        self.z_decoder2 = RepeatVector(self.pred_seq, name='z_decoder2')
        self.z_decoder3 = LSTM(self.z_decoder_dim,
                          return_sequences=True,
                          stateful=False,
                          dropout=self.dropout,
                          name='z_decoder3')
        self.z_decoder4 = Activation('relu', name='z_decoder4')
        self.z_decoder5 = Dropout(self.dropout, name='z_decoder5')
        self.y_decoder = TimeDistributed(Dense(2), name='y_decoder')
        
        # Instantiate the decoder by feeding the concatenated z and x_encoded_dense
        self.z_d1 = self.z_decoder1(self.z_cond)
        self.z_d2 = self.z_decoder2(self.z_d1)
        self.z_d3 = self.z_decoder3(self.z_d2)
        self.z_d4 = self.z_decoder4(self.z_d3)
        self.z_d5 = self.z_decoder5(self.z_d4)
        self.y_prime = self.y_decoder(self.z_d5)
        
        
    def training(self):
        """
        Construct the MCENET model in the training time
        Both the observation data and ground truth data are available
        Note, y is the ground truth trajectory
        """
        print("Construct the MCENET model for training")
        
        def vae_loss(y, y_prime):
            """
            This is the customized loss function
            It consists of L2 and KL loss
            """
            reconstruction_loss = K.mean(mse(y, self.y_prime)*self.pred_seq)
            kl_loss = 0.5 * K.sum(K.square(self.mu) + K.exp(self.log_var) - self.log_var - 1, axis=-1)
            cvae_loss = K.mean(reconstruction_loss*self.beta + kl_loss*(1-self.beta))
            return cvae_loss
        
        # BUILD THE MCENET TRAINING MODEL
        mcenet = Model([self.s_obs_in, self.s_pred_in, self.o_obs_in, self.o_pred_in, self.x, self.y], 
                       [self.y_prime])
        opt = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        mcenet.compile(optimizer=opt, loss=vae_loss)        
        return mcenet
    
    
    def X_encoder(self):
        """
        Constructure the X_encoder to get the encoded information from observation 
        in the inference time
        """
        print('Construct the X-Encoder for inference') 
        x_encoder = Model([self.s_obs_in, self.o_obs_in, self.x], self.x_encoded_dense)
        return x_encoder
    
    
    def Decoder(self):
        """
        Construct the Decoder
        """
        print('Constructure the Decoder for trajectory prediction')
        decoder_input = Input(shape=(self.z_dim+self.encoder_dim, ), name='decoder_input')
        _z_d1 = self.z_decoder1(decoder_input)
        _z_d2 = self.z_decoder2(_z_d1)
        _z_d3 = self.z_decoder3(_z_d2)
        _z_d4 = self.z_decoder4(_z_d3)
        _z_d5 = self.z_decoder5(_z_d4)
        _y_prime = self.y_decoder(_z_d5)
        generator = Model(decoder_input, _y_prime)
        return generator
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        