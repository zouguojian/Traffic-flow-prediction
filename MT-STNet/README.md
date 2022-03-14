# Traffic-flow-prediction

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements.txt文件中指示的包进行安装，才能正常的运行程序！！！</font>
  
>* 首先，使用conda创建一个虚拟环境，如‘conda create traffic_flow’；  
> * 激活环境，conda activate traffic_flow；  
> * 安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0；  
> * 如果安装的是最新的tensorflow环境，也没问题，tensorflow的包按照以下方式进行导入即可：import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()；  
> * 点击 run_train.py文件即可运行代码。
---

## 模型实验结果
### LSTM (1-step)

|  评价指标   | 6-1 step  |6-3 step  |6-6 step  |
|  ----  | ----  |  ----  |  ----  |
| MAE  | 6.101465 |  |  6.985751|
| RMSE  | 9.301329 |  | 11.021970|
| R  | 0.972954 |  |0.961918|
| R<sup>2</sup>  | 0.946454 |  | 0.924858|

> embedding size is 64  
especially input length: 6-1  
MAE is : 6.101465  
RMSE is : 9.301329  
R is: 0.972954  
R<sup>2</sup> is: 0.946454  

### ST-GAT (1-step)  
#### 1-blocks and 1 heads for spatial, 4-blocks and 1 heads for temporal  
> embedding size is 64  
especially input length: 6-1   
MAE is : 6.005778  
RMSE is : 9.141615  
R is: 0.974661  
R<sup>2</sup> is: 0.948277  

#### 1-blocks and 1 heads for spatial, 4-blocks and 1 heads for temporal
> embedding size is 64  
especially input length: 4-1  
MAE is : 6.097268  
RMSE is : 9.299411  
R is: 0.973530  
R<sup>2</sup> is: 0.946475 

#### 1-blocks and 4 heads for spatial, 4-blocks and 4 heads for temporal 
> embedding size is 256  
especially input length: 6-1  
MAE is : 5.930677  
RMSE is : 9.035396  
R is: 0.974623  
R<sup>2</sup> is: 0.949472  

#### 4-blocks and 8 heads  
> embedding size is 512  
especially input length: 6-1  
>MAE is : 5.977749  
RMSE is : 9.063117  
R is: 0.974713  
R<sup>2</sup> is: 0.949162  
 

### GMAN (1step)  

> Embedding size is 256  


### ST-GAT (6-steps)  
#### 1-blocks and 1 heads for spatial, 4-blocks and 1 heads for temporal  
> embedding size is 64  
especially input length: 4-6   
MAE is : 14.322435  
RMSE is : 20.525616  
R is: 0.927124  
R<sup>2</sup> is: 0.739462 

> especially input length: 5-6  
MAE is : 6.550879  
RMSE is : 10.132798  
R is: 0.968368  
R^$2$ is: 0.936499  

> especially input length: 6-6  
epoch 100   
MAE is : 6.372804  
RMSE is : 9.840441  
R is: 0.970164  
R<sup>2</sup> is: 0.940104   

> especially input length: 7-6   
epoch 100  
MAE is : 6.439330  
RMSE is : 9.963966  
R is: 0.969012  
R<sup>2</sup> is: 0.938585 

> especially input length: 8-6   
epoch 100  
MAE is : 6.479654  
RMSE is : 10.042691  
R is: 0.968415  
R<sup>2</sup> is: 0.937646  

> especially input length: 9-6   
epoch 100  
MAE is : 6.501210  
RMSE is : 10.095425  
R is: 0.968201  
R<sup>2</sup> is: 0.936981  

> especially input length: 10-6  
MAE is : 8.980754  
RMSE is : 13.065266  
R is: 0.950789  
R<sup>2</sup> is: 0.894437  

|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |
