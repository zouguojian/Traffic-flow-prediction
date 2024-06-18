# Traffic-flow-prediction  

We have added all the baseline models‘ results and codes to our GitHub page, including SARIMA, LSTM_BILSTM, GMAN, ASTGCN, T-GCN, Etc. Our task is to use the twelve historical time steps to predict the future twelve target time steps traffic flow of the highway, and our time granularity is 5 minutes. If you have any questions, don't hesitate to contact us, thanks！

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
> This study uses real-world monitoring datasets from the highway network, including the gantry, entrance toll, and exit toll sources from the ETC intelligent monitoring sensors at the gantries and the toll stations of the highway in Yinchuan, Ningxia Province, China, as shown in Fig. 7. The 66 ETC intelligent monitoring sensors record the traffic flow in realtime, including 13 highway toll stations (each toll station contains an entrance and exit) and 20 highway gantries (each gantry has two directions, upstream and downstream). In particular, directed connectivity generally exists between the adjacent monitoring sensors, such as flow passing the entrance toll to the gantry. In addition, the degree is then defined according to the directed connectivity: the in-degree is the sum of the first-order connected upstream, and the out-degree is the sum of the first-order connected downstream. Moreover, the distance between two adjacent sensors is computed via the sensors’ longitude and latitude values. ([dataset link](https://github.com/zouguojian/Traffic-flow-prediction/tree/main/MT-STNet/data)).  
> Furthermore, the highway traffic flow data includes six factors: traffic flow, timestamp, monitoring sensor position (i.e., longitude and latitude), connectivity matrices, degree of the sensor, and distance between monitoring sensors, and the time range is from June 1, 2021, to August 31, 2021. For this paper, the data from each monitoring sensor were recorded every 5 minutes to obtain the time series form of traffic flow. All these essential elements are added to our GitHub 2 page, such as sensor position, connectivity matrices, source samples, etc.

## Experimental Results


* Latex inference:
  
      @ARTICLE{10559778,  
        author={Zou, Guojian and Lai, Ziliang and Wang, Ting and Liu, Zongshi and Li, Ye},  
        journal={IEEE Transactions on Intelligent Transportation Systems},  
        title={MT-STNet: A Novel Multi-Task Spatiotemporal Network for Highway Traffic Flow Prediction},   
        year={2024},  
        volume={},  
        number={},  
        pages={1-16},  
        doi={10.1109/TITS.2024.3411638}  
      }    

* paper link [click](https://github.com/zouguojian/Personal-Accepted-Research/blob/main/MT-STNet%20A%20Novel%20Multi-Task%20Spatiotemporal%20Network%20for%20Highway%20Traffic%20Flow%20Prediction/manuscript.pdf)
