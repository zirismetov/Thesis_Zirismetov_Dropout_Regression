import logging
import random
import importlib
import matplotlib.pylab as plt
import numpy as np, pandas as pd
import sys
import os
import sklearn.metrics
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse
import time

from datetime import datetime
from tqdm import tqdm

sys.path.append('taskgen_files')
import csv_utils_2
import file_utils
import args_utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
parser = argparse.ArgumentParser(description="Weather hypermarkets")

parser.add_argument('-is_debug',
                    default=True,
                    type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()
args = args_utils.ArgsUtils.add_other_args(args, args_other)
args.sequence_name_orig = args.sequence_name
args.sequence_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))

removebrac = ['[',']',"'", '"']
for key in args.__dict__:
    if isinstance(args.__dict__[key], str):
        value = (args.__dict__[key])
        for char in removebrac:
            value = value.replace(char, "")
        args.__dict__.update({key: value})
    elif isinstance(args.__dict__[key], list):
        value = str(args.__dict__[key])
        for char in removebrac:
            value = value.replace(char, '')
        args.__dict__.update({key: value})

args.dataset = args.dataset.lower()

args.sequence_name = ''.join(args.sequence_name).replace(',','').replace(' ', '')
args.sequence_name_orig = ''.join(args.sequence_name_orig).replace(',','').replace(' ', '')

args.layers_size = ''.join(args.layers_size)
args.layers_size = args.layers_size.split(',')
if args.dropoutModule != 'advancedDropout':
    args.drop_p = ''.join(args.drop_p)
    args.drop_p = args.drop_p.split(',')

path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
file_utils.FileUtils.createDir(path_run)
file_utils.FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)
csv_utils_2.CsvUtils2.create_global(path_sequence)
csv_utils_2.CsvUtils2.createOverall('./results/')
csv_utils_2.CsvUtils2.create_local(path_sequence, args.run_name)


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        if args.dataset == 'calcofi':
            data_raw = pd.read_csv('dataset/CalCOFI.csv', low_memory=False)
            # Predict temperature of water 1 features: salinity
            data_raw = data_raw[['Salnty', 'T_degC']]
            data_raw['Salnty'].replace(0, np.nan, inplace=True)
            data_raw['T_degC'].replace(0, np.nan, inplace=True)
            data_raw.fillna(method='pad', inplace=True)

            self.X = data_raw['Salnty'].to_numpy()
            np_x = np.copy(self.X)
            self.X[:] = ((np_x[:] - np.min(np_x[:])) / (np.max(np_x[:]) - np.min(np_x[:]))) + 1
            self.X = np.expand_dims(self.X, axis=1)

            self.y = data_raw['T_degC'].to_numpy()
            np_y = np.copy(self.y)
            self.y[:] = ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:]))) + 1
            self.y = np.expand_dims(self.y, axis=1)
        else:
            data_raw = pd.read_csv('dataset/weatherHistory.csv')
            data_raw.drop("Loud Cover", axis=1, inplace=True)
            data_raw['Pressure (millibars)'].replace(0, np.nan, inplace=True)
            data_raw.fillna(method='pad', inplace=True)

            # Predict Weather with 2 features: Humidity and Pressure
            data_X = data_raw.drop(['Formatted Date', 'Summary', 'Precip Type', 'Daily Summary',
                                    'Apparent Temperature (C)', 'Temperature (C)', 'Visibility (km)',
                                    'Wind Bearing (degrees)', 'Wind Speed (km/h)'], axis=1)
            self.X = data_X.to_numpy()
            np_x = np.copy(self.X)
            for i in range(np_x.shape[-1]):
                self.X[:, i] = ((np_x[:, i] - np.min(np_x[:, i])) / (np.max(np_x[:, i]) - np.min(np_x[:, i])))

            self.y = data_raw['Temperature (C)'].to_numpy()
            np_y = np.copy(self.y)
            self.y[:] = ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])))
            self.y = np.expand_dims(self.y, axis=1)

    def __len__(self):
        if args.is_debug:
            return 100
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


dataset = LoadDataset()
tr_idx = np.arange(len(dataset))
subset_train_data, subset_test_data = train_test_split(
    tr_idx,
    test_size=float(args.test_size),
    random_state=0)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=int(args.batch_size),
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=int(args.batch_size),
                                              shuffle=False)

# python taskgen.py -dropoutModule advancedDropout gaussianDropout simpleDropout dropConnect -lr 0.1 0.01 0.001 -test_size 0.20 0.33 -batch_size 64 128 256 -layers_size 1,32,32,32,1 1,32,64,32,1 1,64,64,64,1 1,64,128,64,1 -drop_p 0,0.5,0.5 0.2,0.5,0.5 -epoch 5 10 15
CustomDropout = getattr(__import__('DropoutModules.' + args.dropoutModule, fromlist=['Dropout']), 'Dropout')
class Model(torch.nn.Module):
    def __init__(self, layers_size=[1, 32, 32, 32, 1], drop_p = [0,0.5,0.5]):
        super().__init__()

        self.layers = torch.nn.Sequential()
        for l in range(len(layers_size) - 2):

            if args.dropoutModule == 'gaussianDropout' or args.dropoutModule == 'simpleDropout' :
                self.layers.add_module(f'{args.dropoutModule}_{l + 1}',
                                       CustomDropout(float(drop_p[l])))
                self.layers.add_module(f'linear_layer_{l + 1}',
                                       torch.nn.Linear(int(layers_size[l]), int(layers_size[l + 1])))

            elif args.dropoutModule == 'dropConnect' :
                self.layers.add_module(f'DropConnect_layer_{l + 1}',
                                       CustomDropout(in_features=int(layers_size[l]),out_features=int(layers_size[l + 1]),
                                                    weight_dropout=float(drop_p[l])))
            else:
                self.layers.add_module(f'linear_layer_{l + 1}',
                                       torch.nn.Linear(int(layers_size[l]), int(layers_size[l + 1])))

            self.layers.add_module(f'LeakyReLU_layer_{l + 1}',
                                   torch.nn.LeakyReLU())

            if args.dropoutModule == 'advancedDropout' and l < 2:
                self.layers.add_module(f'advanceDropout_layer_{l + 1}',
                                       CustomDropout(int(layers_size[l + 1])))

        self.layers.add_module("last_linear_layer",
                               torch.nn.Linear(int(layers_size[-2]), int(layers_size[-1])))

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model(layers_size=args.layers_size,
              drop_p=args.drop_p)
model.to(device)
if args.dropoutModule == 'advancedDropout':
    dp_params = []
    res_params = []
    for m in model.layers:
        if isinstance(m, CustomDropout):
            dp_params.append(m.weight_h)
            dp_params.append(m.bias_h)
            dp_params.append(m.weight_mu)
            dp_params.append(m.bias_mu)
            dp_params.append(m.weight_sigma)
            dp_params.append(m.bias_sigma)
        elif isinstance(m, torch.nn.Linear):
            res_params.append(m.weight)
            if hasattr(m, "bias"):
                res_params.append(m.bias)

    opt = torch.optim.SGD([{'params': res_params, 'lr': float(args.lr)},
                           {'params': dp_params, 'lr': 1e-4}], momentum=0.9, weight_decay=5e-4)
else:
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr)
    )

best_loss_test = []
best_R2_test = []
losses_train = []
losses_test = []
R2_train = []
R2_test = []
metrics_mean_dict = {'loss_train': None,
                     'R^2_train': None,
                     'loss_test': 0,
                     'R^2_test': 0,
                     'best_loss_test': 0,
                     'best_R^2_test': 0
                     }
for epoch in range(int(args.epoch)):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        R2_s = []
        if dataloader is dataloader_test:
            model.eval()
            mode = 'test'
        else:
            model.train()
            mode = 'train'

        metrics_mean_dict[f'loss_{mode}'] = []
        metrics_mean_dict[f'R^2_{mode}'] = []

        for x, y in tqdm(dataloader, desc=mode):

            x = torch.FloatTensor(x.float()).to(device)
            y = torch.FloatTensor(y.float()).to(device)
            y_prim = (model.forward(x))

            loss = torch.mean(torch.abs(y - y_prim))
            R2 = sklearn.metrics.r2_score(y.detach().cpu(), y_prim.detach().cpu())

            metrics_mean_dict[f'loss_{mode}'].append(loss)
            metrics_mean_dict[f'R^2_{mode}'].append(R2)

            losses.append(loss.item())
            R2_s.append(R2.item())

            if dataloader is dataloader_train:
                loss.backward()
                opt.step()
                opt.zero_grad()

        metrics_mean_dict[f'loss_{mode}'] = round((torch.mean(torch.FloatTensor(metrics_mean_dict[f'loss_{mode}'])))
                                                  .numpy().item(), 4)
        metrics_mean_dict[f'R^2_{mode}'] = round((torch.mean(torch.FloatTensor(metrics_mean_dict[f'R^2_{mode}'])))
                                                 .numpy().item(), 4)

        if dataloader is dataloader_test:
            best_loss_test.append(metrics_mean_dict['loss_test'])
            best_R2_test.append(metrics_mean_dict['R^2_test'])
            metrics_mean_dict['best_loss_test'] = min(best_loss_test)
            metrics_mean_dict['best_R^2_test'] = max(best_R2_test)

        csv_utils_2.CsvUtils2.add_hparams(
            path_sequence,
            args.run_name,
            args.__dict__,
            metrics_mean_dict,
            epoch
        )
        if dataloader is dataloader_train:
            losses_train.append(np.mean(losses))
            R2_train.append(np.mean(R2_s))
        else:
            losses_test.append(np.mean(losses))
            R2_test.append(np.mean(R2_s))

name = ""
for string in args.run_name:
    string = string.replace("-", "")
    name += string
last = name[-6:]
script_dir = path_run
results_dir = os.path.join(script_dir, 'Results_img/')
sample_file_name = f"sample"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

plt.subplot(2, 1, 1)
plt.title('loss')
plt.plot(losses_train, label="loss_trian")
plt.plot(losses_test, label="loss_test")
plt.legend(loc='upper right', shadow=False, fontsize='medium')

plt.subplot(2, 1, 2)
plt.title('R2')
plt.plot(R2_train, label="R2_trian")
plt.plot(R2_test, label="R2_test")
plt.legend(loc='lower right', shadow=False, fontsize='medium')
plt.savefig(results_dir + last + sample_file_name)
