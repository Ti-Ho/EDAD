import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')


# load data
def load_data(dataset=None, val_split=0.0, del_epidemic=True):
    """
    get data
    :param val_split:
    :param dataset: choose dataset from ["NON10", "NON12", "SP500"]
    :param del_epidemic: whether del epidemic period data
    :return: data shape (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size] or None))
    """
    print(f"-- loading {dataset} --")

    train_df = pd.read_csv(f"./data/{dataset}/train.csv", index_col=0)
    test_df = pd.read_csv(f"./data/{dataset}/test.csv", index_col=0)
    # # ================= for debug ================= #
    # train_df = train_df.iloc[:100]
    # test_df = test_df.iloc[:100]
    # # ================= for debug ================= #
    train_df.index = pd.to_datetime(train_df.index)
    test_df.index = pd.to_datetime(test_df.index)

    if del_epidemic and "NON" in dataset:
        epidemic_time_period = pd.date_range(start="2020-01-05", end="2020-06-01")
        train_df = train_df[~(train_df.index.isin(epidemic_time_period))]

    # rename the columns name
    train_df.columns = np.arange(len(train_df.columns))
    test_df.columns = np.arange(len(test_df.columns))

    # train_df.reset_index(inplace=True, drop=True)
    # test_df.reset_index(inplace=True, drop=True)
    train_df.fillna(0.0, inplace=True)
    test_df.fillna(0.0, inplace=True)

    val_df = None
    if val_split != 0.0:
        split = int(np.floor(len(train_df.index) * (1.0 - val_split)))
        val_df = train_df[split:]
        # val_df.reset_index(inplace=True, drop=True)
        train_df = train_df[:split]
        # train_df.reset_index(inplace=True, drop=True)

    return train_df, val_df, test_df


# Dataset
# class SlidingWindowDataset(Dataset):
#     def __init__(self, data, window):
#         self.data = torch.tensor(data=data.values.T, dtype=torch.float32)
#         self.window = window
#         self.node_num, self.time_len = self.data.shape
#         self.node_list = list(range(len(data.columns)))
#         self.st, self.target_node = self.process()  # start_point, target_node
#
#     def process(self):
#         st_arr = np.array(list(range(0, self.time_len - self.window)) * self.node_num)  # start point
#         node_arr = np.concatenate(
#             ([[node] * (self.time_len - self.window) for node in self.node_list]))  # correspond target node
#         return st_arr, node_arr
#
#     def __len__(self):
#         return len(self.st)
#
#     def __getitem__(self, item):
#         start_point = self.st[item]
#         target_node = self.target_node[item]
#
#         target_data = self.data[target_node, start_point:start_point+self.window].reshape(1, -1)
#         ref_data = self.data[np.arange(self.node_num) != target_node, start_point:start_point+self.window]
#         X = torch.cat((target_data, ref_data), dim=0)
#         X = X.permute(1, 0)  # (batch_size, window_len, feature)
#         y = self.data[target_node, start_point + self.window]
#
#         return X, y, target_node, start_point
