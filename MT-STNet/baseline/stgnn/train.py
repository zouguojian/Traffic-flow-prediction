from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"


from baseline.stgnn.utils import log_string, loadPEMSData
from baseline.stgnn.model import STGNN

parser = argparse.ArgumentParser()
# parser.add_argument('--time_slot', type = int, default = 5,
#                     help = 'a time step is 5 mins')
parser.add_argument('--P', type=int, default=6,
                    help='history steps')
parser.add_argument('--Q', type=int, default=6,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--dropout', type=float, default=0.3, help='drop out')
parser.add_argument('--K', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=64,
                    help='dims of each head attention outputs')

parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='training set [default : 0.8]')
# parser.add_argument('--val_ratio', type=float, default=0.2,
#                     help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
# parser.add_argument('--patience', type = int, default = 10,
#                     help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='initial learning rate')
parser.add_argument('--traffic_file', default='data/train.npz',
                    help='traffic file')
parser.add_argument('--SE_file', default='**.npy',
                    help='spatial emebdding file')
parser.add_argument('--model_file', default='model_layer2_head8_attdim16',
                    help='save the model to disk')
parser.add_argument('--log_file', default='log',
                    help='log file')

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

trainX, trainTE, trainY, testX, testTE, testY, mean, std = loadPEMSData(args)
# SE = torch.from_numpy(SE).to(device)


log_string(log, "loading end....")


def res(model, testX, testTE, testY, mean, std):
    model.eval()  # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_val = testX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(testX[start_idx: end_idx]).float().to(device)
                y = testY[start_idx: end_idx]
                # te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat = model(X)

                pred.append(y_hat.cpu().numpy() * std + mean)
                label.append(y)

    pred = np.concatenate(pred, axis=0)
    label = np.concatenate(label, axis=0)

    print(pred.shape, label.shape)
    maes = []
    rmses = []
    mapes = []
    wapes = []
    cors = []
    r2s = []

    # k=6
    # print('1')
    # metric(pred[:, :k, :13], label[:, :k, :13])
    # print('2')
    # metric(pred[:, :k, 13:26], label[:, :k, 13:26])
    # print('3')
    # metric(pred[:, :k, 26:], label[:, :k, 26:])
    # print('4')
    # metric(pred[:, :k, :], label[:, :k, :])

    for i in range(6):
        mae, rmse, mape, cor, r2 = metric(pred[:, :i+1, :], label[:, :i+1, :])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        cors.append(cor)
        r2s.append(r2)
        # if i == 11:
        log_string(log, 'step %d, mae: %.6f, rmse: %.6f, mape: %.6f,cor:%.6f,r2:%.6f' % (
            i + 1, mae, rmse, mape, cor, r2))
        # print('step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))

    mae, rmse, mape, cor, r2 = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f,cor:%.4f,r2:%.4f' % (
        mae, rmse, mape, cor, r2))

    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(cors, 0), np.stack(r2s, 0)


def train(model, trainX, trainTE, trainY,testX, testTE, testY, mean, std):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15],
    #                                                         gamma=0.2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                              verbose=False, threshold=0.001, threshold_mode='rel',
                                                              cooldown=0, min_lr=2e-6, eps=1e-08)

    for epoch in tqdm(range(1, args.max_epoch + 1)):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        # trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(trainX[start_idx: end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx: end_idx]).float().to(device)
                # te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)

                optimizer.zero_grad()

                y_hat = model(X)

                y_d = y
                y_hat_d = y_hat

                loss = _compute_loss(y, y_hat * std + mean)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                train_l_sum += loss.cpu().item()
                # print(f"\nbatch loss: {l.cpu().item()}")
                n += y.shape[0]
                batch_count += 1
                pbar.update(1)
        # lr = lr_scheduler.get_lr()
        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                   % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        # print('epoch %d, lr %.6f, loss %.4f, time %.1f sec'
        #       % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        mae, rmse, mape, cor, r2 = res(model, testX, testTE, testY, mean, std)
        # lr_scheduler.step()
        lr_scheduler.step(mae[-1])
        if mae[-1] < min_loss:
            min_loss = mae[-1]
            torch.save(model, args.model_file)


def test(model,testX, testTE, testY, mean, std):
    model = torch.load(args.model_file,map_location='cpu')
    mae, rmse, mape, cor, r2 = res(model, testX, testTE, testY, mean, std)
    return mae, rmse, mape, cor, r2


def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        # mae = np.nan_to_num(mae * mask)
        # wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        # rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
        print('mae is : %.6f'%mae)
        print('rmse is : %.6f'%rmse)
        print('mape is : %.6f'%mape)
        print('r is : %.6f'%cor)
        print('r$^2$ is : %.6f'%r2)
    return mae, rmse, mape, cor, r2


if __name__ == '__main__':
    maes, rmses, mapes, cors, r2s = [], [], [], [], []
    # for i in range(1):
    log_string(log, "model constructed begin....")
    model = STGNN(1, args.K * args.d, args.L, args.d).to(device)
    log_string(log, "model constructed end....")
    # log_string(log, "train begin....")
    # train(model, trainX, trainTE, trainY, testX, testTE, testY, mean, std)
    # log_string(log, "train end....")
    log_string(log, 'test start....')
    mae, rmse, mape, cor, r2 = test(model, testX, testTE, testY, mean, std)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    cors.append(cor)
    r2s.append(r2)
    log_string(log, "\n\nresults:")
    maes = np.stack(maes, 1)
    rmses = np.stack(rmses, 1)
    mapes = np.stack(mapes, 1)
    cors = np.stack(cors, 1)
    r2s = np.stack(r2s, 1)

    for i in range(6):
        print('end')
        # log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f,cor%.4f,r2s %.4f' % (
        #     i + 1, maes[i].mean(), rmses[i].mean(), mapes[i].mean(), cors[i].mean(), r2s[i].mean()))
        # log_string(log,
        #            'step %d, mae %.4f, rmse %.4f, mape %.4f,cor%.4f,r2s %.4f' % (
        #                i + 1, maes[i].std(), rmses[i].std(), mapes[i].std(), cors[i].std(), r2s[i].std()))
        #  平均step
        # log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f,cor%.4f,r2s %.4f' % (
        #     i + 1, maes[:i+1].mean(), rmses[:i+1].mean(), mapes[:i+1].mean(), cors[:i+1].mean(), r2s[:i+1].mean()))
        # log_string(log,
        #            'step %d, mae %.4f, rmse %.4f, mape %.4f,cor%.4f,r2s %.4f' % (
        #                i + 1, maes[:i].std(), rmses[:i].std(), mapes[:i].std(), cors[:i].std(), r2s[:i].std()))

    # log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f,cor%.4f,r2s %.4f' % (
    #     maes[-1].mean(), rmses[-1].mean(), mapes[-1].mean(), cors[-1].mean(), r2s[-1].mean()))
    # log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f,cor%.4f,r2s %.4f' % (
    #     maes[-1].std(), rmses[-1].std(), mapes[-1].std(), cors[-1].std(), r2s[-1].std()))

