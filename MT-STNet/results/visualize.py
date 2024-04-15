# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from model.inits import *
from matplotlib.ticker import MaxNLocator
import seaborn as sns

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 17.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
mean=  29.996377569846402
std=  31.95731962018177

T_GCN_YC = np.load('T-GCN-YINCHUAN.npz')['prediction']
LABEL_YC = np.load('T-GCN-YINCHUAN.npz', allow_pickle=True)['truth']
STGNN_YC = np.load('STGNN-YINCHUAN.npz')['prediction'].transpose([0, 2, 1])[20:]
DCRNN_YC = np.load('DCRNN-YINCHUAN.npz')['prediction'].transpose([1,2,0])[20:] * std + mean
AGCRN_YC = np.load('AGCRN-YINCHUAN.npz')['prediction'].squeeze(axis=-1).transpose([0,2,1])[20:]
ASTGCN_YC = np.load('ASTGCN-YINCHUAN.npz')['prediction'][20:]
MSTGCN_YC = np.load('MSTGCN-YINCHUAN.npz')['prediction'][20:]
Graph_WaveNet_YC = np.load('GraphWaveNet-YINCHUAN.npz')['prediction'][20:]
GMAN_YC = np.load('GMAN-YINCHUAN.npz', allow_pickle=True)['prediction']
ST_GRAT_YC = np.load('ST-GRAT-YINCHUAN.npz', allow_pickle=True)['prediction'].transpose([0,2,1])
MTGNN_YC = np.load('MTGNN-YINCHUAN.npz', allow_pickle=True)['prediction'][20:]
RGSL_YC = np.load('RGSL-YINCHUAN.npz', allow_pickle=True)['prediction'][20:].squeeze(axis=-1).squeeze(axis=-1).transpose([0,2,1])
MT_STNet_YC = np.load('MT-STNet-YINCHUAN.npz', allow_pickle=True)['prediction']
label = np.load('MT-STNet-YINCHUAN.npz', allow_pickle=True)['truth']

print(T_GCN_YC.shape, 'T-GCN')
print(LABEL_YC.shape, 'LABEL')
print(STGNN_YC.shape, 'STGNN_YC')
print(DCRNN_YC.shape, 'DCRNN')
print(AGCRN_YC.shape, 'AGCRN')
print(ASTGCN_YC.shape,'ASTGCN')
print(MSTGCN_YC.shape, 'MSTGCN')
print(Graph_WaveNet_YC.shape, 'Graph-WaveNet')
print(GMAN_YC.shape, 'GMAN')
print(ST_GRAT_YC.shape, 'ST-GRAT')
print(MTGNN_YC.shape, 'MTGNN')
print(RGSL_YC.shape, 'RGS')
print(MT_STNet_YC.shape, 'MT-STNet')

print(T_GCN_YC[0,0])
print(LABEL_YC[0,0])
print(STGNN_YC[0,0])
print(DCRNN_YC[0,0])
print(AGCRN_YC[0,0])
print(ASTGCN_YC[0,0])
print(MSTGCN_YC[0,0])
print(Graph_WaveNet_YC[0,0])
print(GMAN_YC[0,0])
print(ST_GRAT_YC[0,0])
print(MTGNN_YC[0,0])
print(RGSL_YC[0,0])
print(MT_STNet_YC[0,0])
print(label[0,0])

# '''
# 用于展示YINCHUAN中600个测试样本的拟合程度
plt.figure()
mean = 80.13087428380071
std = 30.2782227196378

plt.subplot(3, 1, 1)
road_index = 3
total=1200
plt.plot(np.concatenate([list(LABEL_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(MTGNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(RGSL_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='RGSL')
plt.plot(np.concatenate([list(MT_STNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='MT-STNet')
plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
road_index = 13
plt.plot(np.concatenate([list(LABEL_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(MTGNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(RGSL_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='RGSL')
plt.plot(np.concatenate([list(MT_STNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='MT-STNet')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
road_index = 43
plt.plot(np.concatenate([list(LABEL_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='black', linestyle='-',
         linewidth=0.7, label='Observed')
plt.plot(np.concatenate([list(DCRNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)], axis=-1), color='red', linestyle='-',
         linewidth=0.5, label='DCRNN')
plt.plot(np.concatenate([list(AGCRN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1),  color='blue', linestyle='-', linewidth=0.5,
         label='AGCRN')
plt.plot(np.concatenate([list(ASTGCN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#d0c101', linestyle='-',
         linewidth=0.5, label='ASTGCN')
plt.plot(np.concatenate([list(Graph_WaveNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#0cdc73', linestyle='-',
         linewidth=0.5, label='Graph-WaveNet')
plt.plot(np.concatenate([list(GMAN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#f504c9', linestyle='-', linewidth=0.5,
         label='GMAN')
plt.plot(np.concatenate([list(MTGNN_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#ff5b00', linestyle='-', linewidth=0.5,
         label='MTGNN')
plt.plot(np.concatenate([list(RGSL_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='orange', linestyle='-', linewidth=0.5,
         label='RGSL')
plt.plot(np.concatenate([list(MT_STNet_YC[sample_index, road_index]) for sample_index in range(12, total, 12)],axis=-1), color='#a55af4', linestyle='-', linewidth=0.5,
         label='MT-STNet')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Target time steps', font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# '''




# YC实际的预测过程展示，样例
# '''
mean = 80.13087428380071
std = 30.2782227196378

plt.subplot(3, 1, 1)
road_index = 3
sample_index =768
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_YC[sample_index - 12, road_index], LABEL_YC[sample_index, road_index, 0:1]],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_YC[sample_index, road_index], marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_YC[sample_index, road_index], marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_YC[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_YC[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_YC[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_YC[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), MTGNN_YC[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), RGSL_YC[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='RGSL')
plt.plot(range(13, 25, 1), MT_STNet_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='MT-STNet')
plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
road_index = 13
sample_index =768
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_YC[sample_index - 12, road_index], LABEL_YC[sample_index, road_index, 0:1]],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_YC[sample_index, road_index], marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_YC[sample_index, road_index], marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_YC[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_YC[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_YC[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_YC[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), MTGNN_YC[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), RGSL_YC[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='RGSL')
plt.plot(range(13, 25, 1), MT_STNet_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='MT-STNet')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
road_index = 43
sample_index =768
plt.plot(range(1, 14, 1), np.concatenate(
    [LABEL_YC[sample_index - 12, road_index], LABEL_YC[sample_index, road_index, 0:1]],
    axis=-1), marker='P', color='black', linestyle='--', linewidth=1)
plt.plot(range(13, 25, 1), LABEL_YC[sample_index, road_index], marker='P', color='black', linestyle='--',
         linewidth=1, label='Observed')
plt.plot(range(13, 25, 1), DCRNN_YC[sample_index, road_index], marker='P', color='red', linestyle='-',
         linewidth=1, label='DCRNN')
plt.plot(range(13, 25, 1), AGCRN_YC[sample_index, road_index], marker='h', color='blue', linestyle='-', linewidth=1,
         label='AGCRN')
plt.plot(range(13, 25, 1), ASTGCN_YC[sample_index, road_index], marker='o', color='orange', linestyle='-', linewidth=1,
         label='ASTGCN')
plt.plot(range(13, 25, 1), Graph_WaveNet_YC[sample_index, road_index], marker='s', color='#0cdc73', linestyle='-',
         linewidth=1, label='Graph-WaveNet')
plt.plot(range(13, 25, 1), GMAN_YC[sample_index, road_index], marker='p', color='#f504c9', linestyle='-', linewidth=1,
         label='GMAN')
plt.plot(range(13, 25, 1), MTGNN_YC[sample_index, road_index], marker='d', color='#ff5b00', linestyle='-', linewidth=1,
         label='MTGNN')
plt.plot(range(13, 25, 1), RGSL_YC[sample_index, road_index], marker='^', color='#d0c101', linestyle='-',
         linewidth=1, label='RGSL')
plt.plot(range(13, 25, 1), MT_STNet_YC[sample_index, road_index], marker='X', color='#a55af4', linestyle='-', linewidth=1,
         label='MT-STNet')
# plt.legend(loc='upper left', prop=font1)
# plt.grid(axis='y')
plt.ylabel('Traffic flow', font2)
# plt.xlabel('Number of samples',font2)
# plt.title('Exit tall dataset',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()
# '''


# y=x的拟合可视化图
# '''
begin = 1000
total=1500

LABEL_obs=np.concatenate([list(LABEL_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
DCRNN_pre=np.concatenate([list(DCRNN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
AGCRN_pre=np.concatenate([list(AGCRN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
Graph_WaveNet_pre=np.concatenate([list(Graph_WaveNet_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
GMAN_pre=np.concatenate([list(GMAN_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
RGSL_pre=np.concatenate([list(RGSL_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
MT_STNet_pre=np.concatenate([list(MT_STNet_YC[sample_index]) for sample_index in range(begin, total, 12)],axis=-1)
print(LABEL_obs.shape, MT_STNet_YC.shape)
# plt.figure()
plt.subplot(2,3,1)
plt.scatter(LABEL_obs,DCRNN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'DCRNN',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.ylabel("Predicted traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,2)
plt.scatter(LABEL_obs,AGCRN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'AGCRN',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,3)
plt.scatter(LABEL_obs,Graph_WaveNet_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'Graph-WaveNet',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
# plt.xlabel("Observed PM2.5 (ug/m3)", font2)
# plt.ylabel("Predicted traffic spedd", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,4)
plt.scatter(LABEL_obs,GMAN_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'GMAN',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Entrance tall dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic flow", font2)
plt.ylabel("Predicted traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,5)
plt.scatter(LABEL_obs,RGSL_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'RGSL',linewidths=1)
plt.plot(a,b,'black',linewidth=2)
#plt.scatter(a, b)
#plt.plot(test_y_out,'r*:',label=u'predicted value')
# plt.title("Exit tall dataset", font2)
plt.xlabel("Observed traffic flow", font2)
plt.legend(loc='upper left',prop=font2)

plt.subplot(2,3,6)
plt.scatter(LABEL_obs,MT_STNet_pre,alpha=0.7,color='dimgray',edgecolor = "black",marker='o',label=u'MT-STNet',linewidths=1)
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'black',linewidth=2)
# plt.title("Gantry dataset", font2)
#设置横纵坐标的名称以及对应字体格式
plt.xlabel("Observed traffic flow", font2)
plt.legend(loc='upper left',prop=font2)
plt.show()
# '''



# '''
import matplotlib.gridspec as gridspec
sns.set_theme(style='ticks', font_scale=2.,font='Times New Roman')
data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(DCRNN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'DCRNN')
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'dimgray',linewidth=1.5)
plt.legend(loc='lower right',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(AGCRN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'AGCRN')
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'dimgray',linewidth=1.5)
plt.legend(loc='lower right',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(Graph_WaveNet_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'Graph-WaveNet')
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'dimgray',linewidth=1.5)
plt.legend(loc='lower right',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(GMAN_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'GMAN')
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'dimgray',linewidth=1.5)
plt.legend(loc='lower right',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(RGSL_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'RGSL')
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'dimgray',linewidth=1.5)
plt.legend(loc='lower right',prop=font2)
plt.show()

data_df = pd.DataFrame(np.concatenate([np.reshape(LABEL_obs,[-1,1]),np.reshape(MT_STNet_pre, [-1,1])],axis=-1), columns=['truth','prediction'])
g = sns.jointplot(x="truth", y="prediction", data=data_df, label=u'MT-STNet')
# g = sns.jointplot(data=MT_STAN, x="segment_label_"+str(i), y="segment_pre_"+str(i), hue="vehicle type", kind="kde")
g.set_axis_labels(xlabel='Observed traffic flow', ylabel='Predicted traffic flow')
a=[i for i in range(300)]
b=[i for i in range(300)]
plt.plot(a,b,'dimgray',linewidth=1.5)
plt.legend(loc='lower right',prop=font2)
plt.show()
# '''