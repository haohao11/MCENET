# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:41:13 2019

@author: cheng
"""

import numpy as np


def ind_heatmap(obs, heatmaps, scale, hgrid_size):
    '''
    This is the function to get the heatmap according to user type
    params:
        obs: are the observed trajectories with user types
    '''
    print('the dimensions for the heatmap: ', heatmaps[0].shape)
    [heatmap_ped, heatmap_cyc, heatmap_veh] = heatmaps
    heatmap_ped = np.pad(heatmap_ped,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size)),'constant',constant_values = (0,0))
    heatmap_cyc = np.pad(heatmap_cyc,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size)),'constant',constant_values = (0,0))
    heatmap_veh = np.pad(heatmap_veh,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size)),'constant',constant_values = (0,0))
    
    # Convert the x, y coordinates back to pixel values
    obs = np.multiply(obs, [1, 1, scale, scale, 1])
    # define the attention map
    att_map = np.zeros((obs.shape[0], obs.shape[1], hgrid_size*2, hgrid_size*2))
    # count the wrong position
    count=0
    for i, user in enumerate(obs):
        for j, frame in enumerate(user):
            current_y = int(np.ceil(frame[2])) + hgrid_size
            current_x = int(np.ceil(frame[3])) + hgrid_size
            userType = frame[4]
            x, y = heatmap_ped[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size].shape
            if x!=hgrid_size*2 or y!=hgrid_size*2:
# =============================================================================
#                 print('The dimension is not correct at position current_x:%s, current_y:%s'%(str(current_x), str(current_y)))
#                 print('the wrong frame', frame)
# =============================================================================
                count+=1
            else:
                if userType==1:
                    att_map[i, j, :, :] = heatmap_ped[current_x-hgrid_size:current_x+hgrid_size, current_y-hgrid_size:current_y+hgrid_size]
                elif userType==2:
                    att_map[i, j, :, :] = heatmap_cyc[current_x-hgrid_size:current_x+hgrid_size, current_y-hgrid_size:current_y+hgrid_size]
                elif userType==3:
                    att_map[i, j, :, :] = heatmap_veh[current_x-hgrid_size:current_x+hgrid_size, current_y-hgrid_size:current_y+hgrid_size]
#    print('the number of wrong postions in the heatmap %s'%str(count))
    return att_map



def ind_image(obs, images, scale, hgrid_size):
    '''
    This is the function to get the image attention according to user type
    '''
    print('the dimensions for the background image:', images[0].shape)
    img_ped, img_cyc, img_veh = images
    # This is the zero paddings for the background image
    img_ped = np.pad(img_ped,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size),(0,0)),'constant',constant_values = (0,0))
    img_cyc = np.pad(img_cyc,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size),(0,0)),'constant',constant_values = (0,0))
    img_veh = np.pad(img_veh,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size),(0,0)),'constant',constant_values = (0,0))
    
    # Convert the x, y coordinates back to pixel values
    obs = np.multiply(obs, [1, 1, scale, scale, 1])
    
    # define the attention map
    att_map = np.zeros((obs.shape[0], obs.shape[1], hgrid_size*2, hgrid_size*2, img_ped.shape[-1]))
    
    # count the wrong position
    count=0
    for i, user in enumerate(obs):
        for j, frame in enumerate(user):
            current_y = int(np.ceil(frame[2])) + hgrid_size
            current_x = int(np.ceil(frame[3])) + hgrid_size
            userType = frame[4]
            x, y = img_ped[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size, 0].shape
            if x!=hgrid_size*2 or y!=hgrid_size*2:
                count+=1
            else:
                if userType==1:
                    att_map[i, j, :, :, :] = img_ped[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size, :]
                elif userType==2:
                    att_map[i, j, :, :, :] = img_cyc[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size, :]
                elif userType==3:
                    att_map[i, j, :, :, :] = img_veh[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size, :]
#    print('the number of wrong postions in the background image %s'%str(count))
    return att_map


def rgb_image(obs, image, scale, hgrid_size):
    '''
    This is the function to get the image attention regardless of user type
    '''
    print('the dimensions for the background image:', image.shape)
    
    image = np.pad(image,((hgrid_size,hgrid_size),(hgrid_size,hgrid_size),(0,0)),'constant',constant_values = (0,0))

    # Convert the x, y coordinates back to pixel values
    obs = np.multiply(obs, [1, 1, scale, scale, 1])
    
    # define the attention map
    att_map = np.zeros((obs.shape[0], obs.shape[1], hgrid_size*2, hgrid_size*2, image.shape[-1]))
    
    # count the wrong position
    count=0
    for i, user in enumerate(obs):
        for j, frame in enumerate(user):
            current_y = int(np.ceil(frame[2])) + hgrid_size
            current_x = int(np.ceil(frame[3])) + hgrid_size

            x, y = image[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size, 0].shape
            if x!=hgrid_size*2 or y!=hgrid_size*2:
                count+=1
            else:
                att_map[i, j, :, :, :] = image[current_x - hgrid_size:current_x + hgrid_size, current_y - hgrid_size:current_y + hgrid_size, :]
#    print('the number of wrong postions in the background image %s'%str(count))
    return att_map
                    
    
