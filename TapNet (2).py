
from __future__ import division
from __future__ import print_function
tappy_data_path='/home1/vulli/physionet.org/files/tappy/1.0.0/data/Tappy Data'
users_data_path= '/home1/vulli/physionet.org/files/tappy/1.0.0/users/Archived users'

index = 9
enbedding_path = "/home1/vulli/embeddings.txt"
chunk_size=100
sliding_step=50

import re
import glob
import os
import random
import sys
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder




import torch
import math
import sys
import time
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch

import torch
import torchvision
if torch.cuda.is_available():
    devi = 'cuda:0'
elif torch.backends.mps.is_available():
    devi = 'mps'
else:
    devi = 'cpu'
device = torch.device(devi)
print(f"Using device: {device}")


nb_data_per_person = np.array([0])
def get_user_data():
    import os
    directory = users_data_path
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename)) as f:
                file_contents = f.read()
            file_data = {}
            for line in file_contents.split('\n'):
                # print(line)
                if line.strip() != '':
                    key, value = line.split(': ')
                    file_data[key.strip()] = value.strip()

            file_data['ID'] = filename.split("_")[1].split(".")[0]
            # Append the file data to the list
            data.append(file_data)
    df = pd.DataFrame(data)
    df = df[["ID", "Parkinsons"]]
    df["Parkinsons"] = df["Parkinsons"].replace({"False": 0, "True": 1})
    parkinsons_dict = df.set_index('ID')['Parkinsons'].to_dict()
    return parkinsons_dict


positive_files = []
negative_files = []
fils = sorted(glob.glob(os.path.join(tappy_data_path, '*txt')))
random.shuffle(fils)
parkinsons_dict = get_user_data()
files = []
for file in fils:
    f = file.split("/")[index].split("_")[0]
    if f in parkinsons_dict:
        files.append(file)
        if parkinsons_dict[f] == 1:
          positive_files.append(file)
        else:
          negative_files.append(file)
print(len(files), len(positive_files), len(negative_files))

from sklearn.model_selection import train_test_split
from datetime import datetime
train, test = train_test_split(files,  train_size=0.8,  shuffle=True)

def gen_data2(temp, had_parkinson_user, data_list, input, output):
    count = 0
    # print("gen", temp.shape)
    
    nb_datas = int(temp.shape[0] - chunk_size)
    num_cols = temp.shape[1]-1

    for start in range(0, nb_datas, sliding_step):
        end = start + chunk_size
        data = temp[start:end, :]
        # print("differenece",temp[end-1, num_cols] - temp[start, num_cols])
        time_list.append(temp[end-1, num_cols] - temp[start, num_cols])
        if  temp[end-1, num_cols] - temp[start, num_cols] <1000:
          input.append(data[:,:-1])
          # cv+=1
          output.append(had_parkinson_user)
          count = count + 1
        
        
def one_hot(df, value, cat):
    one_hot_df = pd.get_dummies(df[value], prefix=value, columns=cat)
    df = pd.concat([df, one_hot_df], axis=1)
    df.drop(value, axis=1, inplace=True)
    return df

time_list = []
cv=0
train_inputs = []
train_labels = []
test_inputs = []
test_labels = []
def get_system_time(row):
    # Combine the date and time columns into a single datetime object
    dt = datetime.combine(row['date'], row['time'])
    # Calculate the system time in seconds
    return (dt - datetime(1970, 1, 1)).total_seconds()
def load_data(files, input, output, data_type = 1):
  
  for i in range(0, len(files)):
      # print(files[i])
      # print("i:", i)
      
      try:
        ref_date = pd.Timestamp('2000-01-01')
        data = pd.read_csv(files[i], sep = "\t").iloc[:, :-1]
        data.columns = ["user_id","date","time","hand", "hold_time", "dir", "latency_time", "flight_time"]

        
        data = data[np.in1d(data['hand'], ["L", "R", "S"])
                    & data['date'].apply(lambda x: re.search(r'^\d{6}$', str(x)) is not None)
                    & data['time'].apply(lambda x: re.search(r'^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.\d{3}$', str(x)) is not None)
                    & data['hold_time'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)
                    & np.in1d(data['dir'], ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'SS'])
                    & data['latency_time'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)
                    & data['flight_time'].apply(lambda x: re.search(r"[^\d.]", str(x)) is None)]

        # print(data)
      
        data['date'] = (pd.to_datetime(data['date'], format='%y%m%d') )
        data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S.%f').dt.time
        # data['time'] = data['time'].apply(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)).dt.total_seconds() / 86400
        data['datetime'] = data.apply(get_system_time, axis=1)
        # data['datetime'] = data['date'] + " "+ data['time']
        # print(data['date'][0], data['time'][0], data['datetime'][0])
        # data['datetime'] = data['date'] + data['time']/(24*60*60)
        had_parkinson_user = parkinsons_dict[data.iloc[:, 0].unique()[0]]
        data.drop(data.columns[[0, 1, 2]], axis=1, inplace=True) 
        categories_dir = ['LL', 'LR', 'RL', 'RR', 'LS', 'SL', 'RS', 'SR', 'SS']
        categories_hand = ['L','R','S']
        data = one_hot(data, "hand", categories_hand)
        data = one_hot(data, "dir", categories_dir)
        data = data.reindex(columns=['hold_time', 'latency_time', 'flight_time', 'hand_L', 'hand_R',
          'hand_S', 'dir_LL', 'dir_LR', 'dir_LS', 'dir_RL', 'dir_RR', 'dir_RS',
          'dir_SL', 'dir_SR', 'dir_SS', 'datetime'], fill_value = 0)
      except Exception as e:
        print(e)
        continue
      # if data_type == 1 and had_parkinson_user == 0 and data.shape[0] < 14000:
      #   print("Before", data.shape)
      #   data = apply_smote(data)
      #   print("After", data.shape)
      # print(data.columns)
      data = data.to_numpy().astype('float64')
      if data.shape[0] < chunk_size:
        input.append(data[:,:-1])
        output.append(had_parkinson_user)
        continue
      gen_data2(data, had_parkinson_user, nb_data_per_person, input, output)
      # print(input, output)

load_data(train, train_inputs, train_labels)

load_data(test, test_inputs, test_labels, data_type = 0)

train_inputs = np.array(tf.keras.preprocessing.sequence.pad_sequences(
    train_inputs, padding="post"
))
train_labels = np.array(train_labels)

test_inputs = np.array(tf.keras.preprocessing.sequence.pad_sequences(
    test_inputs, padding="post"
))

test_labels = np.array(test_labels)

print(train_inputs.shape)
print(train_labels.shape)
print(test_inputs.shape)
print(test_labels.shape)

print(train_labels)



import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a

# print(train_inputs.shape)
# print(train_labels.shape)
# print(test_inputs.shape)
# print(test_labels.shape)
def load_raw_ts(tensor_format=True):
    x_train = train_inputs
    y_train = train_labels
    x_test = test_inputs
    y_test = test_labels
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

#     ts = np.concatenate((x_train, x_test), axis=0)
#     # ts = np.transpose(ts, axes=(0, 2, 1))
#     labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(y_train)) + 1


#     train_size = y_train.shape[0]

#     total_size = labels.shape[0]
#     idx_train = range(train_size)
#     idx_val = range(train_size, total_size)
#     idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        x_train = torch.FloatTensor(x_train)
        y_train = torch.LongTensor(y_train)
        x_test = torch.FloatTensor(x_test)
        y_test = torch.LongTensor(y_test)
        
#         ts = torch.FloatTensor(np.array(ts))
#         labels = torch.LongTensor(labels)

#         idx_train = torch.LongTensor(idx_train)
#         idx_val = torch.LongTensor(idx_val)
#         idx_test = torch.LongTensor(idx_test)

    return x_train, y_train, x_test, y_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score



def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file=enbedding_path):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TapNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_metric=False, use_lstm=False, use_cnn=True, lstm_dim=128):
        super(TapNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params

        if True:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = lstm_dim
            self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

            paddings = [0, 0, 0]
            if self.use_rp:
                self.conv_1_models = nn.ModuleList()
                self.idx = []
                for i in range(self.rp_group):
                    self.conv_1_models.append(nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0]))
                    self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
            else:
                self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0])

            self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

            self.conv_bn_2 = nn.BatchNorm1d(filters[1])

            self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

            self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # compute the size of input for fully connected layers
            fc_input = 0
            if self.use_cnn:
                conv_size = len_ts
                for i in range(len(filters)):
                    conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
                fc_input += conv_size 
                #* filters[-1]
            if self.use_lstm:
                fc_input += conv_size * self.lstm_dim
            
            if self.use_rp:
                fc_input = self.rp_group * filters[2] + self.lstm_dim


        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):

                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)

        
    def forward(self, input, labels):
        x = input  # x is N * L, where L is the time-series feature dimension
#         print("input shape in forward pass", x.shape)
        idx_train = torch.LongTensor(range(x.shape[0])).cuda()

        if True:
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                x_lstm = x_lstm.mean(1)
                x_lstm = x_lstm.view(N, -1)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        #x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1)

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv
            #

        # linear mapping to low-dimensional space
        x = self.mapping(x)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1).to(device)
            if self.use_att:
                A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k

                class_repr = torch.mm(A, x[idx_train][idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        proto_dists = torch.exp(-0.5*proto_dists)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)

        dump_embedding(x_proto, x, labels)
        return torch.exp(-0.5*dists), proto_dist





os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

data_path = "/content/drive/MyDrive/"
dataset = "NATOPS"

no_cuda = False
seed = 42

epochs = 30
lr = 1e-5
wd = 1e-3
stop_thres = 1e-9

use_cnn = True
use_lstm = True
use_rp = True
rp_params = '-1, 3'
use_metric = False
metric_param = 0.01
filters = "256,256,128"
kernels = "8,5,3"
dilation = 1
layers = "500,300"
dropout = 0
lstm_dim = 128

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

batch_size = 64

np.random.seed(seed)
torch.manual_seed(seed)
sparse = True
print(layers)
layers = [int(l) for l in layers.split(",")]
kernels = [int(l) for l in kernels.split(",")]
filters = [int(l) for l in filters.split(",")]
rp_params = [float(l) for l in rp_params.split(",")]

if not use_lstm and not use_cnn:
    print("Must specify one encoding method: --use_lstm or --use_cnn")
    print("Program Exiting.")
    exit(-1)



print("Loading dataset", dataset, "...")
# Model and optimizer
model_type = "TapNet" 

if model_type == "TapNet":
    x_train, y_train, x_test, y_test, nclass = load_raw_ts()
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    x_test = x_test.cuda()
    y_test = y_test.cuda()
    


    # update random permutation parameter
    if rp_params[0] < 0:
        dim = x_train.shape[1]
        rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = x_train.shape[1]
        rp_params[1] = math.floor(dim / rp_params[1])
    
    rp_params = [int(l) for l in rp_params]
    print("rp_params:", rp_params)

    # update dilation parameter
    if dilation == -1:
        dilation = math.floor(x_train.shape[2] / 64)

    print("Data shape:", x_train.size())
    model = TapNet(nfeat=x_train.shape[1],
                   len_ts=x_train.shape[2],
                   layers=layers,
                   nclass=nclass,
                   dropout=dropout,
                   use_lstm=use_lstm,
                   use_cnn=use_cnn,
                   filters=filters,
                   dilation=dilation,
                   kernels=kernels,
                   use_metric=use_metric,
                   use_rp=use_rp,
                   rp_params=rp_params,
                   lstm_dim=lstm_dim
                   )
    
   


    model = model.cuda()
    input = x_train.cuda()

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=wd)


dataset = TimeSeriesDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_data = TimeSeriesDataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print("len:", len(dataloader))
print(dataloader)
# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    
    for epoch in range(epochs):
        model.train()
        print("epoch:", epoch)
        for i, data in enumerate(dataloader):
            (input, labels) = data
            t = time.time()
            optimizer.zero_grad()
            output, proto_dist = model(input, labels)
            loss_train = F.cross_entropy(output, torch.squeeze(labels))
            if use_metric:
                loss_train = loss_train + metric_param * proto_dist


            loss_list.append(loss_train.item())

            acc_train = accuracy(output, labels)
            loss_train.backward()
            optimizer.step()

            if i % 100 == 0 :
              print('Batch: {:04d}'.format(i),
                    'loss_train: {:.8f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    'time: {:.4f}s'.format(time.time() - t))

        model.eval()
        all_predictions = []
        true_preds = []
        with torch.no_grad():
            for batch_data in test_dataloader:
                batch_predictions, _ = model(batch_data[0], batch_data[1])  # Predict the output for the batch
                all_predictions.append(batch_predictions)
                true_preds.append(batch_data[1])

        # Concatenate the predictions for all batches into a single tensor
        all_predictions = torch.cat(all_predictions)
        true_preds = torch.cat(true_preds)
        loss_val = F.cross_entropy(all_predictions, torch.squeeze(true_preds))
        acc_val = accuracy(all_predictions, true_preds)
        print('loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()))
        # print("test_acc: " + str(test_acc))
        # print("best possible: " + str(test_best_possible))

# test function
def test():
    output, proto_dist = model(input)
    loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
    if use_metric:
        loss_test = loss_test - metric_param * proto_dist

    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()

