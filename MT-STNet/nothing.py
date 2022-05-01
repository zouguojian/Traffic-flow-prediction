# -- coding: utf-8 --

import  matplotlib.pyplot as plt
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:09:06 2018

@author: butany
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

a=np.divide(np.array([0]),np.array([0]))

rmse = np.nan_to_num(a)
print(rmse)

gman_mae_1=[3.428062 ,3.362646 ,3.390813 ,3.494071 ,3.531799 ,3.571153 ]
mtstnet_mae_1 =[3.337196 ,3.262034 ,3.309998 ,3.393569 ,3.452100 ,3.490424]
stnet_mae_1 =[3.397154 ,3.315932 ,3.352641 ,3.436237 ,3.487889 ,3.522225]
stnet_1_mae_1 =[3.611186 ,3.527211 ,3.554060 ,3.636007 ,3.688783 ,3.713170]
stnet_2_mae_1 =[3.389414 ,3.331613 ,3.374541 ,3.468928 ,3.520837 ,3.547431]
stnet_3_mae_1 =[3.397154 ,3.394028 ,3.414805 ,3.505591 ,3.555709 ,3.588656]
stnet_4_mae_1 =[3.594652 ,3.557736 ,3.597997 ,3.693960 ,3.759612 ,3.834248]

gman_mae_2=[3.210767 ,3.202774 ,3.227582 ,3.326689 ,3.390800 ,3.401304]
mtstnet_mae_2 =[3.134595 ,3.120462 ,3.155602 ,3.261914 ,3.335388 ,3.352315]
stnet_mae_2 =[3.238022 ,3.216114 ,3.213459 ,3.311007 ,3.368181 ,3.386188]
stnet_1_mae_2 =[3.395396 ,3.388695 ,3.402937 ,3.496401 ,3.555645 ,3.556630]
stnet_2_mae_2 =[3.236276 ,3.230162 ,3.244592 ,3.354224 ,3.403931 ,3.412322]
stnet_3_mae_2 =[3.238022 ,3.257916 ,3.266579 ,3.372933 ,3.434382 ,3.455262]
stnet_4_mae_2 =[3.362322 ,3.392057 ,3.418516 ,3.529217 ,3.607550 ,3.637214]

gman_mae_3=[4.958639 ,4.911810 ,4.936396 ,5.005800 ,5.099811 ,5.113480]
mtstnet_mae_3 =[4.898097 ,4.796436 ,4.847729 ,4.917308 ,5.015742 ,5.026833]
stnet_mae_3 =[5.009089 ,4.912931 ,4.939495 ,4.991030 ,5.073441 ,5.071135]
stnet_1_mae_3 =[5.103732 ,5.009642 ,5.047273 ,5.089467 ,5.176653 ,5.171041]
stnet_2_mae_3 =[4.970406 ,4.902922 ,4.935338 ,4.995029 ,5.080034 ,5.081162]
stnet_3_mae_3 =[5.009089 ,4.916643 ,4.946118 ,4.996597 ,5.079686 ,5.075325]
stnet_4_mae_3 =[5.121350 ,5.076481 ,5.121463 ,5.221449 ,5.328916 ,5.357304]

gman_rsme_1=[5.254955 ,5.204915 ,5.229750 ,5.483340 ,5.530793 ,5.608531 ]
gman_rsme_2=[4.931747 ,5.076912 ,5.040418 ,5.350675 ,5.415388 ,5.388327]
gman_rsme_3=[7.099093 ,7.202662 ,7.175812 ,7.334400 ,7.482317 ,7.495110]

gman_mape_1=[0.373112 ,0.375637 ,0.373050 ,0.379447 ,0.375342 ,0.375640 ]
gman_mape_2=[0.356836 ,0.358082 ,0.353682 ,0.354211 ,0.352952 ,0.354543]
gman_mape_3=[0.271792 ,0.284824 ,0.281732 ,0.283210 ,0.282705 ,0.283630]

plt.figure()
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
plt.ylabel('Loss(ug/m3)',font2)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

a=np.array([1.2, 1.9])
print(np.round(a))

import matplotlib.pyplot as plt
import csv
# test_y_=[21.0, 21.0, 23.0, 29.0, 25.0, 27.0, 17.0, 22.0, 25.0, 32.0, 30.0, 54.0, 101.0, 92.0, 128.0, 144.0, 148.0, 162.0, 74.0, 90.0, 37.0, 24.0, 23.0, 44.0, 57.0, 60.0, 94.0, 132.0, 104.0, 122.0, 114.0, 113.0, 122.0, 148.0, 117.0, 152.0, 160.0, 173.0, 148.0, 54.0, 38.0, 29.0, 20.0, 15.0, 11.0, 17.0, 17.0, 18.0, 16.0, 13.0, 15.0, 13.0, 13.0, 12.0, 10.0, 15.0, 28.0, 27.0, 44.0, 44.0, 54.0, 59.0, 26.0, 30.0, 21.0, 22.0, 16.0, 19.0, 20.0, 22.0, 23.0, 21.0, 36.0, 52.0, 70.0, 66.0, 69.0, 85.0, 118.0, 149.0, 147.0, 126.0, 138.0, 164.0, 171.0, 172.0, 171.0, 39.0, 61.0, 28.0, 20.0, 27.0, 39.0, 26.0, 53.0, 81.0, 73.0, 74.0, 74.0, 53.0, 64.0, 58.0, 41.0, 67.0, 42.0, 36.0, 31.0, 32.0, 32.0, 28.0, 22.0, 43.0, 30.0, 34.0, 47.0, 91.0, 155.0, 174.0, 168.0, 182.0, 116.0, 81.0, 86.0, 37.0, 36.0, 22.0, 17.0, 15.0, 15.0, 14.0, 14.0, 17.0, 19.0, 19.0, 17.0, 14.0, 14.0, 23.0, 30.0, 11.0, 8.0, 26.0, 42.0, 54.0, 20.0, 14.0, 15.0, 27.0, 45.0, 46.0, 32.0, 28.0, 29.0, 33.0, 49.0, 83.0, 87.0, 75.0, 74.0, 77.0]
# lstm=[30.751698, 17.727066, 23.690172, 30.909302, 23.26216, 26.953888, 17.20324, 24.358753, 29.745918, 32.466156, 29.921616, 53.750828, 93.41886, 92.46564, 109.80635, 126.95114, 126.95243, 126.95243, 66.735855, 76.09878, 72.08246, 36.86182, 27.748272, 34.51977, 53.87351, 56.372463, 74.42778, 116.182724, 93.047386, 99.93741, 111.55981, 114.0785, 114.24001, 116.13046, 93.02787, 123.31676, 126.95242, 126.95243, 127.030136, 47.99202, 45.650005, 34.22887, 16.447601, 11.042307, 9.728447, 15.358141, 15.720802, 16.47887, 15.720724, 15.7568245, 15.788059, 9.986174, 16.077154, 14.7404995, 10.512214, 13.468962, 24.673944, 25.472223, 43.365948, 43.748947, 52.234165, 72.83601, 25.313221, 31.525368, 15.135357, 26.501083, 11.897305, 11.482635, 25.288748, 17.68246, 16.863317, 18.73354, 33.569775, 51.17573, 73.63948, 68.6946, 55.888683, 77.46185, 108.93254, 126.95242, 126.95243, 114.3782, 117.09589, 126.95243, 126.95243, 126.952515, 126.964676, 38.27724, 57.322567, 29.432613, 22.16314, 22.26508, 36.195705, 17.573545, 36.410263, 74.44796, 74.400665, 69.52792, 74.40893, 54.16889, 62.82105, 55.646034, 37.826847, 61.01657, 35.98039, 35.85402, 32.230686, 33.995007, 28.467007, 30.254416, 22.072472, 51.89529, 22.515303, 32.336033, 50.085255, 90.75559, 126.952415, 126.96614, 127.060036, 127.05977, 124.30838, 86.58644, 93.83348, 36.999546, 34.23879, 26.464943, 13.351826, 15.297886, 15.727814, 15.225983, 14.532322, 16.38696, 21.034473, 16.960356, 22.053703, 21.910316, 21.005028, 21.740044, 33.379883, 12.998972, 9.491955, 17.560755, 38.772404, 53.42855, 17.824362, 10.226382, 15.770744, 24.299265, 45.999943, 50.609917, 26.628263, 27.673094, 29.812965, 35.24548, 53.64097, 79.604324, 93.486015, 69.06604, 64.71947, 64.67985]
# cnn_lstm=[29.280191, 21.464802, 23.31498, 24.090784, 23.575891, 20.91396, 19.411085, 18.969503, 16.814491, 28.11272, 24.989388, 41.72264, 91.99086, 84.88394, 129.68945, 139.67342, 139.99835, 143.29314, 56.636684, 82.196884, 81.10626, 38.92935, 29.072144, 43.30697, 58.422928, 58.08319, 70.73058, 112.735306, 95.45541, 91.29062, 92.47461, 97.42151, 112.66903, 114.5868, 83.46471, 110.31955, 137.70882, 138.19757, 130.88179, 26.015903, 37.867947, 30.822578, 24.180637, 16.64291, 16.458103, 15.891394, 19.641478, 20.517971, 20.084663, 20.13374, 18.950468, 18.78441, 18.251196, 16.905954, 17.440273, 19.404238, 23.588448, 27.99667, 34.97045, 36.554176, 35.833496, 70.08014, 24.868168, 17.860939, 13.518156, 21.17403, 17.50316, 19.94941, 26.168621, 21.68789, 23.103544, 22.651596, 27.805592, 47.27333, 68.53558, 68.03697, 50.425377, 74.28932, 89.0047, 137.36311, 141.42683, 113.83853, 117.83724, 137.66557, 141.40947, 142.94267, 142.2929, 48.088867, 38.909145, 30.73684, 20.264843, 22.552353, 23.079079, 21.421741, 21.761333, 74.61645, 79.191666, 63.096104, 69.41557, 28.656862, 43.860756, 44.587196, 38.588333, 60.85347, 34.535423, 28.872223, 25.718412, 24.48484, 29.400314, 19.39906, 22.612364, 32.80706, 19.340965, 38.023537, 54.748077, 100.79452, 141.57661, 139.11017, 141.91887, 144.1241, 128.18367, 79.24384, 90.24802, 52.194244, 30.034626, 27.566807, 21.47528, 17.852291, 18.757278, 16.848974, 15.996511, 17.812054, 22.231325, 16.80072, 15.237344, 26.922668, 22.210766, 18.646873, 32.445354, 14.068464, 11.926083, 24.341608, 38.40429, 39.47667, 24.25716, 21.524368, 19.98589, 24.175312, 34.48566, 49.767532, 28.816822, 26.790257, 28.741234, 28.702333, 41.419876, 60.331314, 96.99054, 70.149124, 73.407875, 67.41491]
# gc_lstm=[25.09303, 16.481585, 20.011435, 24.29231, 23.584532, 26.21135, 22.480673, 16.412998, 29.238346, 24.804895, 28.969276, 40.961422, 93.12666, 87.947624, 128.59344, 139.05672, 142.6569, 145.23459, 59.667847, 83.134514, 77.703575, 33.430325, 25.512812, 36.093246, 51.195194, 74.95928, 96.996666, 127.6917, 100.76174, 107.13145, 102.60085, 106.92017, 111.64754, 121.397385, 100.68754, 139.20827, 144.57945, 138.67775, 138.00945, 43.885265, 38.350586, 29.908285, 24.804852, 17.7572, 14.286681, 16.433764, 20.759758, 20.139093, 19.753426, 16.166079, 18.731085, 16.497852, 16.503292, 16.315786, 13.365427, 15.160015, 28.366148, 27.451689, 45.53986, 45.30903, 46.04928, 62.416744, 29.334541, 27.559504, 14.755451, 20.316008, 12.213644, 17.058397, 16.34856, 17.890783, 17.36175, 22.862623, 30.62871, 49.496372, 66.57667, 83.71722, 63.328094, 95.4994, 123.83979, 144.74986, 145.37665, 134.0324, 133.02065, 144.35121, 145.62784, 145.07278, 145.5608, 49.86901, 51.173645, 35.567547, 11.5421, 12.751381, 32.33025, 22.71004, 37.293163, 83.24233, 92.78784, 81.94455, 92.97757, 50.65094, 46.449192, 53.34135, 54.241367, 55.785793, 38.725822, 22.351057, 24.677362, 27.404022, 34.300243, 28.995419, 26.337145, 34.714012, 22.675962, 33.20493, 46.423397, 93.09615, 142.86452, 145.04294, 145.69392, 145.7076, 139.06339, 84.563446, 107.117226, 52.92613, 30.993124, 26.705227, 21.310394, 13.224564, 18.07543, 18.692078, 14.523851, 15.11443, 20.213245, 15.877609, 12.560291, 20.688719, 18.853952, 16.984255, 32.694958, 20.754026, 11.25792, 23.571012, 46.21414, 48.861675, 23.6029, 18.278715, 16.640825, 27.116863, 38.481808, 44.388245, 32.439575, 33.65973, 24.47406, 32.28618, 51.852345, 76.88882, 110.07586, 74.37321, 74.919136, 80.018814]
#
# rcl=[22.738247, 20.059422, 20.739676, 26.003866, 22.904007, 26.492905, 21.198606, 24.601261, 26.308996, 33.00296, 33.05211, 55.953964, 129.3604, 99.06331, 122.82876, 138.44603, 147.50497, 161.7219, 68.89431, 108.54088, 49.869415, 34.920204, 26.954254, 41.662464, 53.711914, 57.180916, 79.0829, 127.44161, 106.83887, 110.64573, 114.42798, 114.18193, 119.662994, 134.46356, 114.52107, 145.77657, 150.24255, 162.27486, 137.35684, 50.602352, 42.97056, 37.35704, 21.147436, 16.88111, 15.306157, 14.74591, 15.516311, 19.450096, 16.0842, 19.424988, 20.224096, 12.7399435, 18.063295, 13.321813, 13.033454, 15.837091, 28.653067, 29.90061, 48.30726, 46.52002, 44.20468, 50.056892, 23.192804, 30.240057, 18.507639, 23.855276, 14.194512, 16.485521, 24.542557, 22.377474, 23.35272, 25.266813, 32.703888, 50.655544, 63.681824, 66.475975, 67.32435, 87.81343, 121.42914, 150.86229, 152.47495, 125.356926, 140.34846, 159.33221, 167.12444, 168.88388, 174.03514, 47.040936, 60.576088, 25.192766, 19.405787, 25.429375, 36.93495, 26.134895, 45.570923, 87.123245, 70.8487, 71.17146, 69.07947, 57.235123, 67.71796, 63.051575, 39.742012, 66.591156, 40.94357, 34.70962, 28.997202, 30.624475, 30.278334, 29.443909, 30.504803, 45.48939, 19.915537, 42.38515, 63.48817, 100.51588, 151.01022, 169.50246, 176.1217, 184.52724, 136.04134, 91.100784, 101.55942, 36.226555, 36.81511, 28.757702, 16.177351, 13.355011, 16.087408, 15.207308, 15.76856, 18.34579, 18.881636, 20.002508, 21.318142, 19.349924, 14.664257, 19.06787, 32.06086, 13.554066, 9.246055, 24.607779, 50.035053, 57.757233, 18.211645, 13.950619, 16.349228, 23.835575, 43.601944, 45.333668, 27.068626, 27.655838, 28.377012, 32.35911, 53.777805, 78.21798, 93.25842, 73.43584, 75.38863, 75.140564]

gman=pd.read_csv('results/gman.csv',encoding='utf-8').values
gman_pre_1=[]
gman_obs_1=[]
gman_pre_2=[]
gman_obs_2=[]
gman_pre_3=[]
gman_obs_3=[]
K = 700
for i in range(66*(K-100),66*K,66):
    print((i)//66-600,gman[i,6:19])
    gman_obs_1.append(gman[i:i+13,19:25])
    gman_pre_1.append(gman[i:i + 13, 25:])

    gman_obs_2.append(gman[i+13:i + 26, 19:25])
    gman_pre_2.append(gman[i+13:i + 26, 25:])

    gman_obs_3.append(gman[i+26:i + 66, 19:25])
    gman_pre_3.append(gman[i+26:i + 66, 25:])

mtstnet=pd.read_csv('results/mtstnet.csv',encoding='utf-8').values
mtstnet_pre_1=[]
mtstnet_obs_1=[]
mtstnet_pre_2=[]
mtstnet_obs_2=[]
mtstnet_pre_3=[]
mtstnet_obs_3=[]
for i in range(66*(K-100),66*K,66):
    mtstnet_obs_1.append(mtstnet[i:i+13,19:25])
    mtstnet_pre_1.append(mtstnet[i:i + 13, 25:])

    mtstnet_obs_2.append(mtstnet[i+13:i + 26, 19:25])
    mtstnet_pre_2.append(mtstnet[i+13:i + 26, 25:])

    mtstnet_obs_3.append(mtstnet[i+26:i + 66, 19:25])
    mtstnet_pre_3.append(mtstnet[i+26:i + 66, 25:])


# plt.subplot(2,3,1)
# i,j=10,2
# plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_1[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_1[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_1[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# plt.ylabel("Taffic flow", font2)
# plt.title("Entrance toll dataset (sample 1)", font2)
#
# plt.subplot(2,3,2)
# i,j=10,3
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_2[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_2[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_2[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Exit toll dataset (sample 1)", font2)
#
# plt.subplot(2,3,3)
# i,j=10,39
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_3[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_3[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_3[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# # plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Gantry dataset (sample 1)", font2)
#
# plt.subplot(2,3,4)
# i,j=10,4
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1,7), ['2021.8.26 7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_1[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_1[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_1[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# plt.ylabel("Taffic flow", font2)
# plt.title("Entrance toll dataset (sample 2)", font2)
#
# plt.subplot(2,3,5)
# i,j=10,6
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_2[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_2[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_2[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Exit toll dataset (sample 2)", font2)
#
# plt.subplot(2,3,6)
# i,j=10,16
# plt.xticks(range(1,7), ['7:50','7:55','8:00','8:05','8:10','8:15'])
# plt.plot(range(1,7),mtstnet_obs_3[i][j],marker='d',color= 'black', label=u'Observed value', linewidth=1)
# plt.plot(range(1,7),mtstnet_pre_3[i][j],marker='o', color= 'orange', label=u'MT-STNet', linewidth=1)
# plt.plot(range(1,7),gman_pre_3[i][j],marker='s', color= '#0cdc73', label=u'GMAN', linewidth=1)
# plt.legend(loc='upper left',prop=font2)
# plt.xlabel("Target time steps", font2)
# # plt.ylabel("Taffic flow", font2)
# plt.title("Gantry dataset (sample 2)", font2)
#
# plt.show()





plt.figure()
plt.subplot(2,3,1)
plt.scatter(mtstnet_obs_1,mtstnet_pre_1,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MT-STNet',linewidths=1)
a=[i for i in range(220)]
b=[i for i in range(220)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
plt.ylabel("Predicted traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,2)
plt.scatter(mtstnet_obs_2,mtstnet_pre_2,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MT-STNet',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
plt.title("Exit tall dataset", font2)
# plt.xlabel("Observed PM2.5 (μg/m3)", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,3)
plt.scatter(mtstnet_obs_3,mtstnet_pre_3,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MT-STNet',linewidths=1)
c=[i for i in range(330)]
d=[i for i in range(330)]
plt.plot(c,d,'black',linewidth=2)
plt.title("Gantry dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (μg/m3)", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,4)
plt.scatter(gman_obs_1,gman_pre_1,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic flow", font2)
plt.ylabel("Predicted traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,5)
plt.scatter(gman_obs_2,gman_pre_2,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic flow", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,6)
plt.scatter(gman_obs_3,gman_pre_3,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
plt.plot(c,d,'black',linewidth=2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic flow", font2)
# plt.ylabel("Predicted PM2.5 (μg/m3)", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()






# x = np.arange(1, 8, 1)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
# plt.subplot(1,2,1)
#
# rmse_1=[5.6085,5.5612,5.7057,5.8235,5.6626,6.0323,5.5336]
# rmse_2=[5.3883,5.4198,5.5582,5.6798,5.5089,5.7972,5.3924]
# rmse_3=[7.4951,7.4619,7.5817,7.5810,7.4643,7.8891,7.4103]
# mape_1=[0.3756,0.3498,0.4027,0.3600,0.3527,0.3900,0.3516]
# mape_2=[0.3545,0.3255,0.3665,0.3464,0.3340,0.3653,0.3282]
# mape_3=[0.2836,0.2789,0.2810,0.2735,0.2685,0.2871,0.2765]
# plt.ylim(4,8)
# plt.xticks(range(1,9),['GMAN','MT-STNet','STNet','STNet-1','STNet-2','STNet-3','STNet-3'])
# plt.bar(x, rmse_1, width=width,label='Entrance toll dataset',color = 'red')
# plt.bar(x + width, rmse_2, width=width,label='Exit toll dataset',color = 'black')
# plt.bar(x + 2 * width, rmse_3, width=width,label='Gantry dataset',color='salmon')
# plt.ylabel('RMSE',font2)
# # plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
# plt.legend()
#
# plt.subplot(1,2,2)
# plt.ylim(0.2, 0.45)
# plt.xticks(range(1,9),['GMAN','MT-STNet','STNet','STNet-1','STNet-2','STNet-3','STNet-3'])
# plt.bar(x, mape_1, width=width,label='Entrance toll dataset',color = 'red')
# plt.bar(x + width, mape_2, width=width,label='Exit toll dataset',color = 'black')
# plt.bar(x + 2 * width, mape_3, width=width,label='Gantry dataset',color='salmon')
# plt.ylabel('MAPE',font2)
# # plt.title('Target time steps $Q$ = 6 ([0-30 min])',font2)
# plt.legend()
# plt.show()





# plt.subplot(1,3,1)
#
# plt.plot(range(1,7,1),gman_mae_1,marker='o',color='orange',linestyle='-', linewidth=1,label='GMAN')
# plt.plot(range(1,7,1), mtstnet_mae_1,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='MT-STNet')
# plt.plot(range(1,7,1), stnet_mae_1,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='STNet')
# plt.plot(range(1,7,1),stnet_1_mae_1,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='STNet-1')
# plt.plot(range(1,7,1),stnet_2_mae_1 ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='STNet-2')
# plt.plot(range(1,7,1), stnet_3_mae_1,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='STNet-3')
# plt.plot(range(1,7,1), stnet_4_mae_1,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STNet-4')
# plt.legend(loc='upper left',prop=font1)
# plt.grid(axis='y')
# plt.ylabel('MAE',font2)
# plt.xlabel('Target time steps',font2)
# plt.title('Entrance tall dataset',font2)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#
# plt.subplot(1,3,2)
# # plt.xticks(range(1,8), range(0,31,5))
# plt.plot(range(1,7,1),gman_mae_2,marker='o',color='orange',linestyle='-', linewidth=1,label='GMAN')
# plt.plot(range(1,7,1), mtstnet_mae_2,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='MT-STNet')
# plt.plot(range(1,7,1), stnet_mae_2,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='STNet')
# plt.plot(range(1,7,1),stnet_1_mae_2,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='STNet-1')
# plt.plot(range(1,7,1),stnet_2_mae_2 ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='STNet-2')
# plt.plot(range(1,7,1), stnet_3_mae_2,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='STNet-3')
# plt.plot(range(1,7,1), stnet_4_mae_2,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STNet-4')
# plt.legend(loc='upper left',prop=font1)
# plt.grid(axis='y')
# # plt.ylabel('RMSE value',font2)
# plt.xlabel('Target time steps',font2)
# plt.title('Exit tall dataset',font2)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#
# plt.subplot(1,3,3)
# plt.plot(range(1,7,1),gman_mae_3,marker='o',color='orange',linestyle='-', linewidth=1,label='GMAN')
# plt.plot(range(1,7,1), mtstnet_mae_3,marker='s', color='#0cdc73',linestyle='-',linewidth=1,label='MT-STNet')
# plt.plot(range(1,7,1), stnet_mae_3,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='STNet')
# plt.plot(range(1,7,1),stnet_1_mae_3,marker='^',color='#d0c101',linestyle='-', linewidth=1,label='STNet-1')
# plt.plot(range(1,7,1),stnet_2_mae_3 ,marker='d', color='#ff5b00',linestyle='-',linewidth=1,label='STNet-2')
# plt.plot(range(1,7,1), stnet_3_mae_3,marker='*', color='#82cafc',linestyle='-',linewidth=1,label='STNet-3')
# plt.plot(range(1,7,1), stnet_4_mae_3,marker='X', color='#a55af4',linestyle='-',linewidth=1,label='STNet-4')
# # plt.ylabel('R$^2$ value',font2)
# plt.xlabel('Target time steps',font2)
# plt.title('Gantry dataset',font2)
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.legend(loc='upper left',prop=font1)
# plt.grid(axis='y')
# plt.show()