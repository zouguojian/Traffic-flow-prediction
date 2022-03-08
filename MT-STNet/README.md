# Traffic-flow-prediction

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements的包进行安装，才能正常的运行程序！！！</font>
  
>首先，使用conda创建一个虚拟环境，如‘conda create traffic_flow’；  
激活环境，conda activate traffic_flow；  
安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0
---

## 模型实验结果
### LSTM
> MAE is : 6.101465  
RMSE is : 9.301329  
R is: 0.972954  
R<sup>2</sup> is: 0.946454  

### ST-GAT  
#### 1-blocks and 1 heads for spatial, 4-blocks and 1 heads for temporal  
>MAE is : 6.005778  
RMSE is : 9.141615  
R is: 0.974661  
R<sup>2</sup> is: 0.948277  

#### 1-blocks and 4 heads for spatial, 4-blocks and 4 heads for temporal 
> embedding size is 256  
MAE is : 5.930677  
RMSE is : 9.035396  
R is: 0.974623  
R<sup>2</sup> is: 0.949472  

#### 4-blocks and 8 heads  
> embedding size is 512  
>MAE is : 5.977749  
RMSE is : 9.063117  
R is: 0.974713  
R<sup>2</sup> is: 0.949162  

### GMAN

> embedding size is 256  

 