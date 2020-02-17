# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:01:50 2019

@author: cheng
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:46:58 2019

@author: cheng

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter


def heatmap(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    nor_heatmap = heatmap / (heatmap.max() - heatmap.min())
    return heatmap.T, extent, nor_heatmap

def heatmap_main(data , dimensions, diagonal, dataname, user_type=1, sigma=16):
    '''
    This is the main function to get the heatmap for scene context
    '''
    type_dic = {0:'all', 1:'ped', 2:'cyc', 3:'veh'}
    # Gaussian Sigma
    s = sigma
    data = np.multiply(data.T, [1, 1, diagonal, diagonal, 1]).T
    x, y = filter_type(data, user_type)
    
    # Here need to check if the filtered data is empty
    if x.size == 0 or y.size == 0:
        img_extent = [0, dimensions[1], dimensions[0], 0]
        bins_real = [int(dimensions[1]), int(dimensions[0])]
        bin_x_min = 0
        bin_y_min = 0
    else:
        #### solve the offset problem
        img_extent = [0, dimensions[1], dimensions[0], 0]
        bin_x_max, bin_x_min = int(np.ceil(min(np.max(x), dimensions[1]))), int(np.floor(max(np.min(x), 0)))
        bin_y_max, bin_y_min = int(np.ceil(min(np.max(y), dimensions[0]))), int(np.floor(max(np.min(y), 0)))
        bin_width = bin_x_max - bin_x_min
        bin_heiht = bin_y_max - bin_y_min
        # Get the actual heatmap where data can be seen
        bins_real = [bin_width, bin_heiht]
    fig, ax = plt.subplots()
    img, extent, nor_heatmap =  heatmap(x, y, s, bins_real)
    # Align the data to the image size
    aligned_img = np.zeros(dimensions)
    for r, row in enumerate(img):
        for c, column in enumerate(row):
            aligned_img[r+bin_y_min, c+bin_x_min] = column
    # Plot the distribution of trajectories
    #ax.plot(x, y, 'k.', markersize=1, alpha=0.3)
    img = ax.imshow(aligned_img, extent=img_extent, origin='upper', cmap=cm.jet)
    ax.set_title("%s, smoothing with $\sigma$ = %d" %(str(type_dic[user_type]), s))
    #img = ax.imshow(image_file)
    plt.colorbar(img, ax=ax)
    plt.savefig("../heatmaps/%s_heatmap_%s_sigma_%d"%(dataname, str(type_dic[user_type]), s), dpi=200)
    plt.show()
    plt.gcf().clear()
    plt.close()

    return(aligned_img)
        
     
def filter_type(data, user_type):
    # Filter by user type
    if user_type == 0:
        x = data[2, :]
        y = data[3, :]
    else:
        x = data[2, data[4, :]==user_type]
        y = data[3, data[4, :]==user_type]
    return x, y
    
def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Rows are scores for each class. 
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)

