import sys
sys.path.append('../')

import os
import copy
import torch
import numpy as np
import time
# from timm.utils import ModelEmaV2
from tqdm import tqdm
from lib.utils import get_logger
from lib.evaluate import All_Metrics
import datetime

class Trainer(object):
    def __init__(self, args, dataloader, scaler, model, loss, optimizer, lr_scheduler=None, device=''):
        super(Trainer, self).__init__()
        self.args = args
        # self.ema_model = ModelEmaV2(model, decay=args.model_ema_decay)
        self.dataloader = dataloader
        self.scaler = scaler
        # 模型、损失函数、优化器
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device=device
        # 日志与模型的保存路径
        self.best_path = os.path.join(args.log_dir, '{}_{}_best_model.pth'.format(args.dataset.split('/')[-1], args.model))
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)  # run.log
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info("Experiment log path in: {}".format(args.log_dir))


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        self.dataloader['train_loader'].shuffle()
        start_time = datetime.datetime.now()
        iteration=0
        for _, (data, target) in enumerate(self.dataloader['train_loader'].get_iterator()):
            iteration+=1
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            data = data.astype(np.float)
            data = torch.from_numpy(data).float().to(self.device)
            label = label.astype(np.float)
            label = torch.from_numpy(label).float().to(self.device)
            # data and target shape: B, T, N, D; output shape: B, T, N, D
            self.optimizer.zero_grad()
            output = self.model(data)  # directly predict the true value
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)  # 若模型预测的真实值
            loss = self.loss(output.to(self.device), label)
            loss.backward()

            if iteration==100:
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                print("Total running times is : %f" % total_time.total_seconds())

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            # self.ema_model.update(self.model)
            total_loss += loss.item()
        
        train_epoch_loss = total_loss / self.dataloader['train_loader'].num_batch
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss


    def val_epoch(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.dataloader['val_loader'].get_iterator()):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                data = data.astype(np.float)
                data = torch.from_numpy(data).float().to(self.device)
                label = label.astype(np.float)
                label = torch.from_numpy(label).float().to(self.device)
                output = self.model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)  # 若模型预测的真实值
                loss = self.loss(output.to(self.device), label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / self.dataloader['val_loader'].num_batch
        return val_loss


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            t1 = time.time()
            train_epoch_loss = self.train_epoch()
            t2 = time.time()
            # 验证, 如果是Encoder-Decoder结构，则需要将epoch作为参数传入
            val_epoch_loss = self.val_epoch()
            t3 = time.time()
            self.logger.info('Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f} secs, Inference Time: {:.4f} secs.'.format(epoch, train_epoch_loss, val_epoch_loss, (t2 - t1), (t3 - t2)))
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if val_epoch_loss < best_loss:
                print('Val loss decrease from {:.3f} to {:.3f}, saving to {}'.format(best_loss, val_epoch_loss, self.best_path))
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # is or not early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                # self.logger.info("Current best model saved!")
                # best_model = copy.deepcopy(self.ema_model.state_dict())
                torch.save(self.model.state_dict(), self.best_path)
            
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        self.logger.info("Saving current best model to " + self.best_path)
        # load model and test
        # self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.dataloader['test_loader'], self.scaler, self.logger)


    @staticmethod
    def test(model, args, data_loader, scaler, logger, save_path=None):
        if save_path != None:
            model.load_state_dict(torch.load(save_path))
            model.to(args.device)
            print("load saved model...")
        model.eval()
        y_pred = []
        y_true = []
        start_time = datetime.datetime.now()
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader.get_iterator()):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                data = data.astype(np.float)
                data = torch.from_numpy(data).float().to(args.device)
                label = label.astype(np.float)
                label = torch.from_numpy(label).float().to(args.device)
                output = model(data)
                y_true.append(label)
                y_pred.append(output)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())
        # y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        if args.real_value:   # 预测值是真实值还是归一化的结果
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = scaler.inverse_transform(y_pred)
        y_true = torch.cat(y_true, dim=0)
        y_true = torch.unsqueeze(y_true,dim=-1)
        y_pred = torch.unsqueeze(y_pred,dim=-1)
        print(y_true.shape)
        np.savez_compressed('data/RGSL-' + 'YINCHUAN', **{'prediction': y_pred.cpu().numpy(), 'truth': y_true.cpu().numpy()})

        maes = []
        rmses = []
        mapes = []
        print('                MAE\t\tRMSE\t\tMAPE')
        for (l,r) in [(0,13),(13,26),(26,66)]:
            for t in range(12):
                mae, rmse , mape= All_Metrics(y_pred[:, t, l:r], y_true[:, t, l:r], args.mae_thresh, args.mape_thresh)
                maes.append(mae)
                rmses.append(rmse)
                mapes.append(mape)
                # if i == 11:
                print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (t + 1, mae, rmse, mape * 100))
            mae, rmse, mape = All_Metrics(y_pred[:,:,l:r], y_true[:,:,l:r], args.mae_thresh, args.mape_thresh)
            print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
            print('\n')

        # for t in range(y_true.shape[1]):
        #     mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
        #     logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        # mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        # logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))