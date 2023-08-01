import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.evaluate import MAE_torch
from RGSL_Config import args
from RGSL_Utils import *
from RGSL_Trainer import Trainer
from model.rgsl import RGSL as Network


def load_data(args):
    dataloader = get_load_dataset(args.dataset, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    adj_mx = load_adjacent(args.graph_path) # 专对yinchuan
    print("The shape of adjacency matrix : ", adj_mx.shape)
    return adj_mx, dataloader, scaler


def generate_model_components(args, cheb_polynomials, L_tilde):
    init_seed(args.seed)
    # 1. model
    model = Network(
        num_nodes=args.num_node,
        input_dim=args.input_dim,
        rnn_units=args.hidden_dim,
        embed_dim=args.embed_dim,
        output_dim=args.output_dim,
        horizon=args.horizon,
        cheb_k=args.cheb_k,
        num_layers=args.num_layers,
        default_graph=args.default_graph,
        cheb_polynomials=cheb_polynomials,
        L_tilde=L_tilde,
        dev=device
    )
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    def masked_mae_loss(scaler, mask_value):
        def loss(preds, labels):
            if scaler:
                preds = scaler.inverse_transform(preds)
                labels = scaler.inverse_transform(labels)
            mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
    # 4. learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    return model, loss, optimizer, lr_scheduler


def get_log_dir(model, dataset, debug):
    # current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficFlow
    log_dir = os.path.join(current_dir, model, dataset.split('/')[0])
    if os.path.isdir(log_dir) == False and not debug:
        os.makedirs(log_dir, exist_ok=True)  # run.log
    return log_dir


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
        device = torch.device(args.device)
    else:
        args.device = 'cpu'
        device = torch.device(args.device)
    adj_mx, dataloader, scaler = load_data(args)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in cheb_polynomial(L_tilde, args.cheb_k)]
    adj_mx = torch.from_numpy(adj_mx).type(torch.FloatTensor).to(args.device)
    L_tilde = torch.from_numpy(L_tilde).type(torch.FloatTensor).to(args.device)

    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)
    model, loss, optimizer, lr_scheduler = generate_model_components(args, cheb_polynomials, L_tilde)
    trainer = Trainer(
        args=args,
        dataloader=dataloader,
        scaler=scaler, 
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        device=device
    )

    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "data/YINCHUAN_RGSL_best_model.pth"
        trainer.test(model, args, dataloader['test_loader'], scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError