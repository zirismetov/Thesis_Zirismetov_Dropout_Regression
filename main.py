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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from datetime import datetime
from tqdm import tqdm

sys.path.append('taskgen_files')
import csv_utils_2
import file_utils
import args_utils

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="hypermarkets")

parser.add_argument('-is_debug',
                    default=False,
                    type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()
args = args_utils.ArgsUtils.add_other_args(args, args_other)
args.sequence_name_orig = str(args.sequence_name[0])
args.sequence_name += ('-'+ f'{args.dropoutModule}-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))

removebrac = ['[', ']', "'", '"']
for key in args.__dict__:
    if isinstance(args.__dict__[key], str):
        value = (args.__dict__[key])
        for char in removebrac:
            value = value.replace(char, "")
        args.__dict__.update({key: value})
    elif isinstance(args.__dict__[key], list):
        value = str(args.__dict__[key])
        for char in removebrac:
            value = value.replace(char, "")
        args.__dict__.update({key: value})

args.dataset = args.dataset.lower()
args.sequence_name = ''.join(args.sequence_name).replace(',', '').replace(' ', '')
args.sequence_name_orig = ''.join(args.sequence_name_orig).replace(',', '').replace(' ', '')
args.layers_size = ''.join(args.layers_size)
if args.dropoutModule != 'advancedDropout':
    args.drop_p = ''.join(args.drop_p)

path_sequence = f'./results/{args.sequence_name_orig}/{args.sequence_name}'
args.run_name += ('-' + f'{args.dropoutModule}-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
# path_run = f'./{path_sequence}/{args.run_name}'
path_overall_results = f'./results/{args.sequence_name_orig}'
# file_utils.FileUtils.createDir(path_run)
file_utils.FileUtils.createDir(path_sequence)
file_utils.FileUtils.writeJSON(f'{path_sequence}/args.json', args.__dict__)
csv_utils_2.CsvUtils2.create_global(path_sequence)
csv_utils_2.CsvUtils2.createOverall(path_overall_results)
csv_utils_2.CsvUtils2.create_local(path_sequence, args.run_name)

total_avg_y = 0
class LoadDataset(torch.utils.data.Dataset):
    def __init__(self):
        global total_avg_y
        if args.dataset == 'calcofi':
            data_raw = pd.read_csv('datasets/CalCOFI.csv', low_memory=False)
            # Predict temperature of water 1 features: salinity
            data_raw = data_raw[['Salnty', 'Depthm', 'T_degC']]

            self.X = data_raw[['Salnty', 'Depthm']].copy().to_numpy()
            self.y = data_raw['T_degC'].to_numpy()

            np_x = np.copy(self.X)
            np_y = np.copy(self.y)
            np_y = np.expand_dims(np_y, axis=1)

            nans = np.argwhere(np.isnan(np_x))
            np_x = np.delete(np_x, nans, axis=0)
            np_y = np.delete(np_y, nans, axis=0)

            nans = np.argwhere(np.isnan(np_y))
            np_x = np.delete(np_x, nans, axis=0)
            np_y = np.delete(np_y, nans, axis=0)

            # self.X = (np_x - np.mean(np_x, axis=0))/np.std(np_x, axis=0)
            # self.y = (np_y - np.mean(np_y, axis=0))/np.std(np_y, axis=0)
            self.X = np_x
            self.y = np_y
            total_avg_y = np.average(self.y, axis=0)
        else:
            data_raw = pd.read_csv('datasets/weatherHistory.csv')
            data_raw.drop("Loud Cover", axis=1, inplace=True)
            data_raw['Pressure (millibars)'].replace(0, np.nan, inplace=True)
            data_raw.fillna(method='pad', inplace=True)

            self.X = data_raw[['Wind Speed (km/h)', 'Humidity', 'Wind Bearing (degrees)']].to_numpy()
            np_x = np.copy(self.X)
            self.X = (np_x - np.mean(np_x, axis=0))/np.std(np_x, axis=0)

            self.y = data_raw['Temperature (C)'].to_numpy()
            np_y = np.copy(self.y)
            self.y = (np_y - np.mean(np_y, axis=0))/np.std(np_y, axis=0)
            self.y = np.expand_dims(self.y, axis=1)
            total_avg_y = np.average(self.y, axis=0)

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

if args.dropoutModule != 'noDrop':
    CustomDropout = getattr(__import__('DropoutModules.' + args.dropoutModule, fromlist=['Dropout']), 'Dropout')


class Model(torch.nn.Module):
    def __init__(self, layers_size, drop_p):
        super().__init__()

        self.drop_p = drop_p.split(',')
        self.layers_size = layers_size.split(',')
        self.layers = torch.nn.Sequential()

        for l in range(len(self.layers_size) - 2):

            if args.dropoutModule == 'dropConnect':
                self.layers.add_module(f'dropConnect_{l + 1}',
                                       CustomDropout(float(self.drop_p[l]), self.layers_size, l))
            else:
                self.layers.add_module(f'linear_layer_{l + 1}',
                                       torch.nn.Linear(int(self.layers_size[l]), int(self.layers_size[l + 1])))

            self.layers.add_module(f'LeakyReLU_layer_{l + 1}',
                                   torch.nn.LeakyReLU())

            if args.dropoutModule != "noDrop" and args.dropoutModule != "dropConnect":
                self.layers.add_module(f'{args.dropoutModule}_{l + 1}',
                                       CustomDropout(float(self.drop_p[l]), self.layers_size, l))

        self.layers.add_module("last_linear_layer",
                               torch.nn.Linear(int(self.layers_size[-2]), int(self.layers_size[-1])))

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
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr)
    )


def R2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - total_avg_y) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])

    return float(output_scores)


best_loss_test = []
best_R2_test = []
losses_train = []
losses_test = []
R2s_train = []
R2s_test = []
metrics_mean_dict = {'loss_train': None,
                     'loss_test': 0,
                     'R^2_train': None,
                     'R^2_test': 0,
                     'best_loss_test': 0,
                     'best_R^2_test': 0,

                     }

for epoch in range(int(args.epoch)):
    logging.warning(f"epochs left: {int(args.epoch) - epoch} ")

    for dataloader in [dataloader_train, dataloader_test]:

        if dataloader is dataloader_test:
            model.eval()
            mode = 'test'
        else:
            model.train()
            mode = 'train'

        metrics_mean_dict[f'loss_{mode}'] = []
        metrics_mean_dict[f'R^2_{mode}'] = []

        np_y_prim = []
        np_y = []
        losses = []

        for x, y in tqdm(dataloader, desc=mode):

            x = torch.FloatTensor(x.float()).to(device)
            y = torch.FloatTensor(y.float()).to(device)
            y_prim = (model.forward(x))
            loss = torch.mean(torch.abs(y - y_prim))

            if model.training:
                loss.backward()
                opt.step()
                opt.zero_grad()

            np_y += y.detach().cpu().numpy().tolist()
            np_y_prim += y_prim.detach().cpu().numpy().tolist()
            metrics_mean_dict[f'loss_{mode}'].append(loss.item())
            losses.append(loss.item())

        # for i in range(len(pred_y)):
        np_y = np.vstack(np_y)
        np_y_prim = np.vstack(np_y_prim)

        # np_y  = np.array(sum(np_y, []))
        # np_y_prim  = np.array(sum(np_y_prim, []))
        # r2 = R2_score(np_y, np_y_prim)
        r2 = R2_score(np_y, np_y_prim)

        metrics_mean_dict[f'loss_{mode}'] = round((torch.mean(torch.FloatTensor(metrics_mean_dict[f'loss_{mode}'])))
                                                  .numpy().item(), 4)
        metrics_mean_dict[f'R^2_{mode}'] = round(r2, 4)

        if dataloader is dataloader_test:
            best_loss_test.append(metrics_mean_dict['loss_test'])
            best_R2_test.append(metrics_mean_dict['R^2_test'])
            metrics_mean_dict['best_loss_test'] = min(best_loss_test)
            metrics_mean_dict['best_R^2_test'] = max(best_R2_test)

        csv_utils_2.CsvUtils2.add_hparams(
            path_sequence,
            path_overall_results,
            args.run_name,
            args.__dict__,
            metrics_mean_dict,
            epoch
        )
        if dataloader is dataloader_train:
            losses_train.append(np.mean(losses))
            R2s_train.append(r2)
        else:
            losses_test.append(np.mean(losses))
            R2s_test.append(r2)

name = ""
for string in args.run_name:
    string = string.replace("-", "")
    name += string
last = name[-6:]
script_dir = path_sequence
results_dir = os.path.join(script_dir, 'Results_img/')
sample_file_name = f"sample"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

moving_averages_train = pd.DataFrame(losses_train).rolling(3, min_periods=1).mean()
moving_averages_test = pd.DataFrame(losses_test).rolling(3, min_periods=1).mean()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), dpi=100)

axes[0].set_title('Simple Moving Average of Loss', fontsize=12)
axes[0].plot(moving_averages_train, label="loss_trian")
axes[0].plot(moving_averages_test, label="loss_test")
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].legend(loc='upper right', shadow=False, fontsize='medium')

axes[1].set_title('R2_log', fontsize=12)
axes[1].plot(R2s_train, label="R2s_trian")
axes[1].plot(R2s_test, label="R2s_test")
axes[1].set_yscale('log')
axes[1].set_xlabel('Epochs', fontsize=12)
axes[1].legend(loc='lower right', shadow=False, fontsize='medium')

moving_averages_train_r2s = pd.DataFrame(R2s_train).rolling(3, min_periods=1).mean()
moving_averages_test_r2s = pd.DataFrame(R2s_test).rolling(3, min_periods=1).mean()
axes[2].set_title('R2_MEAN_of_3', fontsize=12)
axes[2].plot(moving_averages_train_r2s, label="R2s_trian")
axes[2].plot(moving_averages_test_r2s, label="R2s_test")
axes[2].set_xlabel('Epochs', fontsize=12)
axes[2].legend(loc='lower right', shadow=False, fontsize='medium')
plt.savefig(results_dir + last + sample_file_name)