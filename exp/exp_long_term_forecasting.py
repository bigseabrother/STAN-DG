import pickle

from matplotlib import pyplot as plt
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # model = self.model_dict[self.args.model].Model(self.args,self.device).float()
        if 'TimeXer_' in self.args.model:
            model = self.model_dict[self.args.model].Model(self.args,self.device).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()
        
        total_params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        print(
            f"Trainable parameters: "
            f"{total_params:,} "
            f"({total_params / 1e6:.3f} M)"
        )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.HuberLoss()
        # nn.MSELoss()
        return criterion
    def plot_ls(self,train_ls,pred_ls):
        # 创建一个新的图表
        plt.figure(figsize=(20, 16))

        # 绘制损失曲线
        plt.plot(train_ls, label='Training Loss', color='blue')
        plt.plot(pred_ls, label='Pred Loss', color='orange')


        # 添加标签和标题
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Convergence')

        # 添加网格
        plt.grid(True)

        # 添加图例
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "train_"+str(self._max_epochs)+ '_ls.png'))
        # 显示图表
        plt.show()
    def plot_ls_epoch(self,train_ls,pred_ls):
        # 创建一个新的图表
        plt.figure(figsize=(20, 16))

        # 绘制损失曲线
        plt.plot(train_ls, label='Training Loss', color='blue')
        plt.plot(pred_ls, label='Pred Loss', color='orange')


        # 添加标签和标题
        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.title('Training Convergence')

        # 添加网格
        plt.grid(True)

        # 添加图例
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "train_"+str(self._max_epochs)+ '_ls_epoch.png'))
        # 显示图表
        plt.show()
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # batch_x_mark = None
                # batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :,:]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len,:, :], dec_inp], dim=1).float().to(self.device)

                if len(batch_x.shape)==4:
                    b,  t , n,c= batch_x.shape
                #     batch_x = batch_x.permute(0,2,1, 3 ).reshape(b*n,t,  c)
                #     _,_,_,times= batch_x_mark.shape
                    _,labellen,_,_ = dec_inp.shape
                #     batch_x_mark = batch_x_mark.permute(0,2,1, 3 ).reshape(b*n,t,  times)
                #     _,labellen,_,_= batch_y_mark.shape #torch.Size([4, 40, 471, 4])
                #     dec_inp = dec_inp.permute(0,2,1, 3 ).reshape(b*n,labellen,  c)
                #     batch_y_mark = batch_y_mark.permute(0,2,1, 3 ).reshape(b*n,labellen,  times)
                    

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs,attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # if len(batch_x.shape)==4:
                outputs = outputs.reshape(b,n,labellen,-1)  .permute(0, 2, 1, 3)
                
                f_dim = 3#4 if self.args.features == 'MS' else 4
                outputs = outputs[:, -self.args.pred_len:,:, f_dim: f_dim+1]
                batch_y = batch_y[:, -self.args.pred_len:,:, f_dim: f_dim+1].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # batch_x_mark = None
                # batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:,:, :]).float() #torch.Size([32, 1, 52, 4])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :,:], dec_inp], dim=1).float().to(self.device)


                if len(batch_x.shape)==4:
                    b,  t , n,c= batch_x.shape
                #     batch_x = batch_x.permute(0,2,1, 3 ).reshape(b*n,t,  c)
                #     _,_,_,times= batch_x_mark.shape
                #     _,labellen,_,_ = dec_inp.shape
                #     batch_x_mark = batch_x_mark.permute(0,2,1, 3 ).reshape(b*n,t,  times)
                    _,labellen,_,_= batch_y_mark.shape #torch.Size([4, 40, 471, 4])
                #     dec_inp = dec_inp.permute(0,2,1, 3 ).reshape(b*n,labellen,  c)
                #     batch_y_mark = batch_y_mark.permute(0,2,1, 3 ).reshape(b*n,labellen,  times)
     
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # if len(batch_x.shape)==4:
                        outputs = outputs.reshape(b,n,labellen, -1)  .permute(0, 2, 1, 3)
                    
                        f_dim = 3#4 if self.args.features == 'MS' else 4
                        outputs = outputs[:, -self.args.pred_len:,:, f_dim: f_dim+1]
                        batch_y = batch_y[:, -self.args.pred_len:,:, f_dim: f_dim+1].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs,attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #torch.Size([32, 1, 52, 4])
                    # if len(batch_x.shape)==4:
                    outputs = outputs.reshape(b,n,labellen, -1)  .permute(0, 2, 1, 3)
                        # print(outputs.shape)
              
                    f_dim =3# 4 if self.args.features == 'MS' else 4 
                    outputs = outputs[:, -self.args.pred_len:,:, f_dim: f_dim+1]
                    batch_y = batch_y[:, -self.args.pred_len:, :,f_dim: f_dim+1].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        attnlist=[]
        preds_scaler = []
        trues_scaler = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/data/scaler_financialgeo/scaler.pkl', 'rb') as f:
            scaler=pickle.load(f)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_ori, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x_ori = batch_x_ori.float().to(self.device)
                batch_y = batch_y.float().to(self.device) #torch.Size([1, 1, 52, 4])

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # batch_x_mark = None
                # batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:,:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len,:, :], dec_inp], dim=1).float().to(self.device)


                if len(batch_x_ori.shape)==4:
                    b,  t , n,c= batch_x_ori.shape
                    batch_x = batch_x_ori#.permute(0,2,1, 3 ).reshape(b*n,t,  c)
                #     _,_,_,times= batch_x_mark.shape
                    _,labellen,_,_ = dec_inp.shape
                #     batch_x_mark = batch_x_mark.permute(0,2,1, 3 ).reshape(b*n,t,  times)
                #     _,labellen,_,_= batch_y_mark.shape #torch.Size([4, 40, 471, 4])
                #     dec_inp = dec_inp.permute(0,2,1, 3 ).reshape(b*n,labellen,  c)
                #     batch_y_mark = batch_y_mark.permute(0,2,1, 3 ).reshape(b*n,labellen,  times)
     
                attn=None
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs,attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs,attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #torch.Size([1, 1, 52, 4])
                # if len(batch_x.shape)==4:
                outputs = outputs.reshape(b,n,labellen, -1)  .permute(0, 2, 1, 3)
              

                f_dim = 3#4 if self.args.features == 'MS' else 4
                outputs = outputs[:, -self.args.pred_len:,:, :]
                batch_y = batch_y[:, -self.args.pred_len:,:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # print(type(attn))
                if attn is not None:
                    attnlist.append(attn.detach().cpu().numpy())
                # if test_data.scale and self.args.inverse:
                #     shape = batch_y.shape
                #     if outputs.shape[-1] != batch_y.shape[-1]:
                #         outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                #     outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :,:, f_dim: f_dim+1]
                batch_y = batch_y[:, :,:, f_dim: f_dim+1]
                shape = batch_y.shape



                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                outputs = ((outputs.reshape(shape[0] * shape[1]*shape[2], -1) * scaler.scale_[f_dim]) + scaler.mean_[f_dim]).reshape(shape)
                batch_y = ((batch_y.reshape(shape[0] * shape[1]*shape[2], -1)* scaler.scale_[f_dim]) + scaler.mean_[f_dim]).reshape(shape)
                pred_scaler = outputs
                true_scaler = batch_y
                preds_scaler.append(pred_scaler)
                trues_scaler.append(true_scaler)


                if i % 20 == 0:
                    input = batch_x_ori.detach().cpu().numpy()
                    shape = input.shape

                    input = ((input.reshape(shape[0] * shape[1]*shape[2], -1) * scaler.scale_[f_dim]) + scaler.mean_[f_dim]).reshape(shape)

                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :,0, 0], true_scaler[0, :,0, 0]), axis=0)
                    pd = np.concatenate((input[0, :,0, 0], pred_scaler[0, :,0, 0]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if len(attnlist)>0:
            attnlist = np.concatenate(attnlist, axis=0)
        else:
            attnlist=None
        preds_scaler = np.concatenate(preds_scaler, axis=0)
        trues_scaler = np.concatenate(trues_scaler, axis=0)  
        # preds = np.array(preds)
        # trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        preds_scaler = preds_scaler.reshape(-1, preds_scaler.shape[-2], preds_scaler.shape[-1])
        trues_scaler = trues_scaler.reshape(-1, trues_scaler.shape[-2], trues_scaler.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe,r2 = metric(preds, trues)
        print('mse:{}, mae:{},rmse:{}, mape:{}, mspe:{}, r2:{}'.format(mse, mae,rmse, mape, mspe,r2))

        mae_scaler, mse_scaler, rmse_scaler, mape_scaler, mspe_scaler,r2_scaler = metric(preds_scaler, trues_scaler)
        print('mse_scaler:{}, mae_scaler:{},rmse_scaler:{}, mape_scaler:{}, mspe_scaler:{}, r2_scaler:{}'.format(
            mse_scaler, mae_scaler,rmse_scaler, mape_scaler, mspe_scaler, r2_scaler))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{},rmse:{}, mape:{}, mspe:{}, r2:{}'.format(mse, mae,rmse, mape, mspe,r2))
        f.write('mse_scaler:{}, mae_scaler:{},rmse_scaler:{}, mape_scaler:{}, mspe_scaler:{}, r2_scaler:{}'.format(
            mse_scaler, mae_scaler,rmse_scaler, mape_scaler, mspe_scaler,r2_scaler))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics_scaler.npy', np.array([mae_scaler, mse_scaler, rmse_scaler, mape_scaler, mspe_scaler,r2_scaler]))
        np.save(folder_path + 'pred_scaler.npy', preds_scaler)
        np.save(folder_path + 'true_scaler.npy', trues_scaler)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if not os.path.exists('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/attn/' + setting):
            os.makedirs('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/attn/' + setting)
        if self.args.graph_flag:
            if attnlist is not None:
                np.save('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/attn/' + setting+'/adjlist.npy', attnlist)

        else:
            if attnlist is not None:
                np.save('/disk4/cl/yc/yc/transformer/Time-Series-Library-main-transformer/Time-Series-Library-main-transformer/attn/' + setting+'/attnlist.npy', attnlist)

        return
