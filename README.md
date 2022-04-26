# Traffic-flow-prediction
This work considers combine multi-tricks with highway network to achieve traffic flow prediction accurately.  

## 注意事项

<font face="微软雅黑" >需要注意的是，需要根据requirements.txt文件中指示的包进行安装，才能正常的运行程序！！！</font>
  
>* 首先，使用conda创建一个虚拟环境，如‘conda create traffic_flow’；  
> * 激活环境，conda activate traffic_flow；  
> * 安装环境，需要安装的环境已经添加在requirements.txt中，可以用conda安装，也可以使用pip安装，如：conda install tensorflow==1.12.0；  
> * 如果安装的是最新的tensorflow环境，也没问题，tensorflow的包按照以下方式进行导入即可：import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()；  
> * 点击 run_train.py文件即可运行代码。
> * 需要注意的是，我们在tensorflow的1.12和1.14版本环境中都可以运行
---

## Hyperparameters for MT-STNet

self.parser.add_argument('--save_path', type=str, default='weights/STNet/', help='save path')  
self.parser.add_argument('--model_name', type=str, default='STNet', help='training or testing model name')  

self.parser.add_argument('--divide_ratio', type=float, default=0.8, help='data_divide')  
self.parser.add_argument('--is_training', type=bool, default=True, help='is training')  
self.parser.add_argument('--epoch', type=int, default=100, help='epoch')  
self.parser.add_argument('--step', type=int, default=1, help='step')  
self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')  
self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')  
self.parser.add_argument('--dropout', type=float, default=0.3, help='drop out')  
self.parser.add_argument('--site_num', type=int, default=66, help='total number of road')  
self.parser.add_argument('--edge_num', type=int, default=108, help='total number of edge')  
self.parser.add_argument('--num_heads', type=int, default=4, help='total number of head attentions')  
self.parser.add_argument('--num_blocks', type=int, default=1, help='total number of attention layers')  

#每个点表示一个监测站点，目前66个监测站点  
self.parser.add_argument('--emb_size', type=int, default=128, help='embedding size')  
self.parser.add_argument('--features', type=int, default=1, help='numbers of the feature')  
self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')  
self.parser.add_argument('--input_length', type=int, default=6, help='input length')  
self.parser.add_argument('--output_length', type=int, default=6, help='output length')  
self.parser.add_argument('--predict_length', type=int, default=6, help='predict length')  

self.parser.add_argument('--hidden1', type=int, default=32, help='number of units in hidden layer 1')  
self.parser.add_argument('--gcn_output_size', type=int, default=64, help='model string')  
self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')  
self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')  

self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')  
self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')  

self.parser.add_argument('--training_set_rate', type=float, default=0.7, help='training set rate')   
self.parser.add_argument('--validate_set_rate', type=float, default=0.15, help='validate set rate')  
self.parser.add_argument('--test_set_rate', type=float, default=0.15, help='test set rate')  

self.parser.add_argument('--file_train', type=str, default='data/train.csv',help='training set address')  
self.parser.add_argument('--file_val', type=str, default='data/val.csv', help='validation set address')  
self.parser.add_argument('--file_test', type=str, default='data/test.csv', help='test set address')  
self.parser.add_argument('--file_sp', type=str, default='data/sp.csv', help='sp set address')  
self.parser.add_argument('--file_dis', type=str, default='data/dis.csv', help='dis set address')  
self.parser.add_argument('--file_in_deg', type=str, default='data/in_deg.csv', help='in_deg set address')  
self.parser.add_argument('--file_out_deg', type=str, default='data/out_deg.csv', help='out_deg set address')  