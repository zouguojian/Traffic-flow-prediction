# -- coding: utf-8 --

import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser
        self.parser.add_argument('--save_path', type=str, default='weights/MT_STNet-7/', help='save path')
        self.parser.add_argument('--granularity', type=int, default=5, help='minute granularity')

        self.parser.add_argument('--model_name', type=str, default='MT_STNet', help='model name')
        self.parser.add_argument('--file_train_f', type=str, default='data/YINCHUAN/train.csv',help='training set')
        self.parser.add_argument('--file_sp', type=str, default='data/YINCHUAN/sp.csv', help='shortest path')
        self.parser.add_argument('--file_dis', type=str, default='data/YINCHUAN/dis.csv', help='distance')
        self.parser.add_argument('--file_in_deg', type=str, default='data/YINCHUAN/in_deg.csv', help='in degree')
        self.parser.add_argument('--file_out_deg', type=str, default='data/YINCHUAN/out_deg.csv', help='out degree')
        self.parser.add_argument('--file_adj', type=str, default='data/YINCHUAN/adjacent_fully.csv',help='adjacent')

        self.parser.add_argument('--train_ratio', type=float, default=0.7, help='train data divide')
        self.parser.add_argument('--validate_ratio', type=float, default=0.1, help='validate divide')
        self.parser.add_argument('--test_ratio', type=float, default=0.2, help='test divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--is_physical', type=bool, default=True, help='using physical information')
        self.parser.add_argument('--epochs', type=int, default=100, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.4, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=66, help='total number of road')
        self.parser.add_argument('--edge_num', type=int, default=108, help='total number of edge')
        self.parser.add_argument('--num_heads', type=int, default=8, help='total number of head attentions')
        self.parser.add_argument('--num_blocks', type=int, default=1, help='total number of attention layers')
        self.parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')

        #每个点表示一个监测站点，目前66个监测站点
        self.parser.add_argument('--emb_size', type=int, default=64, help='embedding size') # 对比模型中可能为256
        self.parser.add_argument('--features', type=int, default=1, help='numbers of the feature')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=12, help='input length')
        self.parser.add_argument('--output_length', type=int, default=12, help='output length')
        self.parser.add_argument('--pre_len', type=int, default=6, help='previous length')

        self.parser.add_argument('--hidden1', type=int, default=32, help='number of units in hidden layer 1')
        self.parser.add_argument('--gcn_output_size', type=int, default=64, help='model string')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)