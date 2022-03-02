# Traffic-flow-prediction

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements的包进行安装，才能正常的运行程序！！！</font>
  
>首先，使用conda创建一个虚拟环境，如‘conda create traffic_flow’；  
激活环境，conda activate traffic_flow；  
安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0
---

## 模型实验结果
### LSTM
>MAE is : 6.432934  
RMSE is : 9.824185  
R is: 0.969684  
R<sup>2</sup> is: 0.940265  

### ST-GAT  
>MAE is : 6.343116  
RMSE is : 9.503859  
R is: 0.972036  
R<sup>2</sup> is: 0.944097  