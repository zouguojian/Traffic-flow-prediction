# Traffic-flow-prediction  

We have added all the baseline models‘ results and codes to our GitHub page, including HA, LSTM_BILSTM, GMAN, AST-GAT, T-GCN, Etc! Our task is using the 6 historical time steps to prediction the future 6 target time steps traffic flow of highway, and our time granularity is 5 minutes. If you have any questions, don't hesitate to connect us, thanks！

## WHAT SHOULD WE PAY ATTENTION TO FOCUS ON THE RUNNING ENVIRONMENT?

<font face="微软雅黑" >Note that we need to install the right packages to guarantee the model runs according to the file requirements.txt！！！</font>
  
>* first, please use the conda tool to create a virtual environment, such as ‘conda create traffic speed’;  
> * second, active the environment, and conda activate traffic speed;   
> * third, build environment, the required environments have been added in the file named requirements.txt; you can use conda as an auxiliary tool to install or pip, e.g., conda install tensorflow==1.13.1;    
> * if you have installed the last TensorFlow version, it’s okay; import tensorflow.compat.v1 as tf and tf.disable_v2_behavior();    
> * finally, please click the run_train.py file; the codes are then running;  
> * Note that our TensorFlow version is 1.14.1 and can also be operated on the 2.0. version.  
---
## DATA DESCRIPTION  
> This study uses three real-world datasets from highways, including the gantry, entrance toll, and exit toll datasets from the ETC intelligent monitoring sensors at the gantries and the toll stations of the highway in Ningxia Province, China, as shown in Fig. 7. The 66 ETC intelligent monitoring sensors record the traffic flow in real time, including 13 highway toll stations (each toll station contains an entrance and exit) and 20 highway gantries (each gantry has two directions, upstream and downstream). The highway traffic flow data includes four factors: traffic flow, time, road position, and distance between monitoring sensors, and the time range is from June 1, 2021, to August 31, 2021. For this paper, the data from each monitoring sensor were recorded every 5 minutes to obtain the time series form of traffic flow 1. ([dataset link](https://github.com/zouguojian/Traffic-flow-prediction/tree/main/MT-STNet/data)).
---
## Experimental Results