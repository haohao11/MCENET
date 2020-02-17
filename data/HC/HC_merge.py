# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:56:17 2019

@author: cheng
"""

import numpy as np

dir1 = '../data/HC/D1/trajectories.csv'
dir2 = '../data/HC/D2/trajectories.csv'
dir3 = '../data/HC/D3/trajectories.csv'

data1 = np.genfromtxt(dir1, delimiter=',')
data2 = np.genfromtxt(dir2, delimiter=',')
data3 = np.genfromtxt(dir3, delimiter=',')

#print(np.unique(data1[:, 0]))
print(len(np.unique(data1[:, 0])))
#print(np.unique(data1[:, 1]))
#print(len(np.unique(data1[:, 1])))

#print(np.unique(data2[:, 0]))
print(len(np.unique(data2[:, 0])))
#print(np.unique(data2[:, 1]))
#print(len(np.unique(data2[:, 1])))

#print(np.unique(data3[:, 0]))
print(len(np.unique(data3[:, 0])))
#print(np.unique(data3[:, 1]))
#print(len(np.unique(data3[:, 1])))

d1_fmin, d1_umin = np.min(data1[:, 0]), np.min(data1[:, 1])
d1_fmax, d1_umax = np.max(data1[:, 0]), np.max(data1[:, 1])

d2_fmin, d2_umin = np.min(data2[:, 0]), np.min(data2[:, 1])
d2_fmax, d2_umax = np.max(data2[:, 0]), np.max(data2[:, 1])

d3_fmax, d3_umax = np.max(data3[:, 0]), np.max(data3[:, 1])
d3_fmin, d3_umin = np.min(data3[:, 0]), np.min(data3[:, 1])

print(d1_fmin, d1_fmax, d1_umin, d1_umax)
print(d2_fmin, d2_fmax, d2_umin, d2_umax)
print(d3_fmin, d3_fmax, d3_umin, d3_umax)


new_data2 = np.add(data2, [d1_fmax+1, d1_umax+1, 0, 0, 0])
new_data3 = np.add(data3, [d1_fmax+d2_fmax+2, d1_umax+d2_umax+2, 0, 0, 0])

d_merge = np.concatenate((data1, new_data2, new_data3), axis=0)

print(np.unique(d_merge[:, 0]))
print(len(np.unique(d_merge[:, 0])))
print(np.unique(d_merge[:, 1]))
print(len(np.unique(d_merge[:, 1])))

np.savetxt('../data/HC/merged/trajectories.csv', d_merge, delimiter=',')