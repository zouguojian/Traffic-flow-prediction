import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
import os
import datetime

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:3',help='')
parser.add_argument('--data',type=str,default='data/YINCHUAN',help='data path')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adjacent.npz',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=66,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='experiments/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

parser.add_argument('--runs',type=int,default=10,help='number of runs')



args = parser.parse_args()
torch.set_num_threads(3)


def main():
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # predefined_A = load_adj(args.adj_data)
    predefined_A = load_adjacent(args.adj_data) # 专对yinchuan
    predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    predefined_A = predefined_A.to(device)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)


    engine.model.load_state_dict(torch.load(args.save + args.data.split('/')[-1]))

    #test data
    outputs = []
    realy = torch.from_numpy(dataloader['y_test'].astype(np.float)).float().to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    start_time = datetime.datetime.now()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x = x.astype(np.float)
        testx = torch.from_numpy(x).float().to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(axis=1))
    
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total running times is : %f" % total_time.total_seconds())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    np.savez_compressed('data/MTGNN-' + 'YINCHUAN', **{'prediction': scaler.inverse_transform(yhat).cpu().numpy(), 'truth': realy.cpu().numpy()})

    mae = []
    mape = []
    rmse = []
    print('                MAE\t\tRMSE\t\tMAPE')
    for (l,r) in [(0,13),(13,26),(26,66)]:
        for i in range(args.seq_out_len):
            pred = scaler.inverse_transform(yhat[:, l:r, i])
            real = realy[:, l:r, i]
            metrics = metric(pred, real)
            print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, metrics[0], metrics[2], metrics[1] * 100))
            # log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            # print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        metrics = metric(scaler.inverse_transform(yhat[:,l:r]),realy[:,l:r])
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(metrics[0], metrics[2], metrics[1] * 100))
        print('\n')

if __name__ == "__main__":

    main()




