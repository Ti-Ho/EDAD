from exp.exp_basic import Exp_Basic
from models import Preformer, Preformer_Recons
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from data_provider.data_util import *
from data_provider.data_loader import Dataset_Custom_with_EAD

import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np
from detection import Detector
import random
import statsmodels.api as sm

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

def replace_anomalies(train_data, fill_data, indices, targets):
    for index, target in zip(indices, targets):
        train_data.iloc[index, target] = fill_data.iloc[index, target]

    return train_data

def process_fill_data(df_fill, fill_type):
    col_name = df_fill.columns[0]
    # process fill data
    if fill_type == "season_mean_4":
        lag_list = [7, 14, 21, 28]
        for lag in lag_list:
            df_fill[f'lag{lag}'] = df_fill[col_name].shift(lag)
        df_fill[fill_type] = df_fill[[f'lag{lag}' for lag in lag_list]].mean(axis=1)
        # fill nan
        null_index = df_fill[df_fill[fill_type].isnull()].index
        fill_values = df_fill.iloc[null_index][col_name]
        df_fill.loc[null_index.values.tolist(), fill_type] = fill_values
    elif fill_type == "mean":
        lag_list = range(-5, 5)
        for lag in lag_list:
            df_fill[f"lag{lag}"] = df_fill[col_name].shift(lag)
        df_fill[fill_type] = df_fill[[f'lag{lag}' for lag in lag_list]].mean(axis=1)
    else:  # lowess
        lowess = sm.nonparametric.lowess
        lowess_result = lowess(df_fill[col_name], df_fill.index, frac=0.1, it=3, delta=0.0, return_sorted=False)
        df_fill[fill_type] = lowess_result
        # plt.plot(df_fill.index, df_fill[fill_type], label='lowess')
        # plt.plot(df_fill.index, df_fill[col_name], label='real_data')
        # plt.legend()
        # plt.grid()
        # plt.show()

    return df_fill[fill_type]


def get_fill_data(train_data, fill_type):
    print("-- getting fill data --")
    fill_df = pd.DataFrame(columns=train_data.columns)
    for col_i in tqdm(train_data.columns, desc="preprocessing fill data"):
        fill_df[col_i] = process_fill_data(train_data[[col_i]], fill_type)
    return fill_df

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Preformer': Preformer,
            'Preformer_Recons': Preformer_Recons
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, recons, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, recons, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # f_dim = -1 if self.args.features == 'MS' else 0
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                pred_loss = criterion(pred, true)

                target_x = batch_x[:, :, f_dim].detach().cpu()
                recons = recons.squeeze(2)
                recons = recons.detach().cpu()
                recon_loss = torch.sqrt(criterion(target_x, recons))

                loss = pred_loss + self.args.recons_loss_ratio * recon_loss

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, args):
        # _, train_loader = self._get_data(flag='train')
        # _, vali_loader = self._get_data(flag='val')
        # _, test_loader = self._get_data(flag='test')

        train_data, val_data, test_data = load_data(dataset=args.dataset,
                                                    val_split=args.val_split,
                                                    del_epidemic=args.del_epidemic)

        timeenc = 0 if args.embed != 'timeF' else 1
        train_dataset = Dataset_Custom_with_EAD(data=train_data, size=[args.seq_len, args.label_len, args.pred_len],
                                       features=args.features, scale=False, timeenc=timeenc, freq=args.freq)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
        val_dataset = Dataset_Custom_with_EAD(data=val_data, size=[args.seq_len, args.label_len, args.pred_len],
                                     features=args.features, scale=False, timeenc=timeenc, freq=args.freq)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=True)
        test_dataset = Dataset_Custom_with_EAD(data=test_data, size=[args.seq_len, args.label_len, args.pred_len],
                                      features=args.features, scale=False, timeenc=timeenc, freq=args.freq)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=True)

        fill_data = get_fill_data(train_data, args.fill_data_type)

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

        n_decay = 1
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, recons, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs, recons, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)
                    pred_loss = criterion(outputs, batch_y)

                    target_x = batch_x[:, :, f_dim]
                    recons = recons.squeeze(2)
                    recon_loss = torch.sqrt(criterion(target_x, recons))

                    loss = pred_loss + self.args.recons_loss_ratio * recon_loss

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
            vali_loss = self.vali(val_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # EAD process
            if epoch % args.detect_per_epoch == 0:
                # # recons_loss_decay
                # self.recons_loss_ratio = self.recons_loss_ratio * self.recons_decay

                # detector init
                n_features = len(train_data.columns)
                time_len = len(train_data)

                detector = Detector(self.model, train_loader, self.device, n_features, time_len, args)
                # detect anomalies
                anomalies_indices, anomalies_targets = detector.detect_anomalies_and_fill(epoch=epoch, n_decay=n_decay)
                n_decay += 1
                # replace anomalies with fill data
                train_data = replace_anomalies(train_data, fill_data, anomalies_indices, anomalies_targets)
                print(f"########## Data Std: {train_data.std().mean()} ##########")

                seed_everything()
                # reconstruct dataset and dataloader
                train_dataset = Dataset_Custom_with_EAD(data=train_data,
                                                        size=[args.seq_len, args.label_len, args.pred_len],
                                                        features=args.features, scale=False, timeenc=timeenc,
                                                        freq=args.freq)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, drop_last=True)
                print(f"Detect and Replace {len(anomalies_indices)} anomalies, accounts for "
                      f"{len(anomalies_indices) * 100 / len(train_dataset):.2f}%, Dataloader has been updated")


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, args, test=0):

        _, _, test_data = load_data(dataset=args.dataset,
                                    val_split=args.val_split,
                                    del_epidemic=args.del_epidemic)
        timeenc = 0 if args.embed != 'timeF' else 1
        test_dataset = Dataset_Custom_with_EAD(data=test_data, size=[args.seq_len, args.label_len, args.pred_len],
                                      features=args.features, scale=False, timeenc=timeenc, freq=args.freq)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=True)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # all_mse_min = 100
        # all_index_min = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, recons, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                else:
                    if self.args.output_attention:
                        outputs, recons, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mae:{}'.format(rmse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mae:{}'.format(rmse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return