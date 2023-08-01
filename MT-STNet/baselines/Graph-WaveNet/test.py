import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '5'


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:5',help='')
parser.add_argument('--data',type=str,default='data/YINCHUAN',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adjacent.npz',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=66,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,default='./garage/yinchuan', help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')


args = parser.parse_args()

def main():
    device = torch.device(args.device)

    if  args.adjdata == 'data/sensor_graph/adjacent.npz':
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_graph_adj(args.adjdata,args.adjtype)
    else: sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()


    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.from_numpy(dataloader['y_test'].astype(np.float)).float().to(device)
    print('first realy',realy.shape)
    realy = realy.transpose(1,3)[:,0,:,:]
    print('second realy',realy.shape)

    start_time = datetime.datetime.now()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x = x.astype(np.float)
        testx = torch.from_numpy(x).float().to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze(axis=1))
    
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total running times is : %f" % total_time.total_seconds())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    amae = []
    amape = []
    armse = []
    print('                MAE\t\tRMSE\t\tMAPE')
    for (l,r) in [(0,66)]:
        for i in range(12):
            pred = scaler.inverse_transform(yhat[:,l:r,i])
            real = realy[:,l:r,i]
            metrics = util.metric(pred,real)
            # log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, metrics[0], metrics[2], metrics[1] * 100))
            # print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
        metrics = util.metric(scaler.inverse_transform(yhat[:,l:r]),realy[:,l:r])
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(metrics[0], metrics[2], metrics[1] * 100))
        print('\n')

    np.savez_compressed('data/GraphWaveNet-' + 'YINCHUAN', **{'prediction': scaler.inverse_transform(yhat).cpu().numpy(), 'truth': realy.cpu().numpy()})

    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(np.mean(amae), np.mean(armse), np.mean(amape) * 100))
    # print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))


    # if args.plotheatmap == "True":
    #     adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
    #     device = torch.device('cpu')
    #     adp.to(device)
    #     adp = adp.cpu().detach().numpy()
    #     adp = adp*(1/np.max(adp))
    #     df = pd.DataFrame(adp)
    #     sns.heatmap(df, cmap="RdYlBu")
    #     plt.savefig("./emb"+ '.pdf')

    # y12 = realy[:,99,11].cpu().detach().numpy()
    # yhat12 = scaler.inverse_transform(yhat[:,99,11]).cpu().detach().numpy()

    # y3 = realy[:,99,2].cpu().detach().numpy()
    # yhat3 = scaler.inverse_transform(yhat[:,99,2]).cpu().detach().numpy()

    # df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    # df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    main()
