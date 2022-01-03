import logging
import random
import importlib
import matplotlib.pylab as plt
from matplotlib import pyplot
import numpy as np, pandas as pd
import sys
import os
import sklearn.metrics
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.append('taskgen_files')
import csv_utils_2
import file_utils
import args_utils

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

parser = argparse.ArgumentParser(description="hypermarkets")

# parser.add_argument('-is_debug',
#                     default=False,
#                     type=lambda x: (str(x).lower() == 'true'))
#
# parser.add_argument('-sequence_name',
#                     default='mining-test-8_lowerererlr',
#                     type=str)
#
# parser.add_argument('-dropoutModule',
#                     default='noDrop',
#                     type=str)
#
# parser.add_argument('-dataset',
#                     default='mining',
#                     type=str)
#
# parser.add_argument('-layers_size',
#                     default='22,128,128,128,1',
#                     type=str)
#
# parser.add_argument('-drop_p',
#                     default='0,0,0',
#                     type=str)
#
# parser.add_argument('-test_size',
#                     default=0.8,
#                     type=float)
#
# parser.add_argument('-lr',
#                     default=0.000001,
#                     type=float)
#
# parser.add_argument('-batch_size',
#                     default=8,
#                     type=int)
#
# parser.add_argument('-epoch',
#                     default=2500,
#                     type=int)
#
#
parser.add_argument('-is_debug',
                    default=False,
                    type=lambda x: (str(x).lower() == 'true'))

# parser.add_argument('-sequence_name',
#                     default='calcofi_testing',
#                     type=str)
#
# parser.add_argument('-dropoutModule',
#                     default='noDrop',
#                     type=str)
#
# parser.add_argument('-dataset',
#                     default='artificial',
#                     type=str)
#
# parser.add_argument('-layers_size',
#                     default='1,64,64,1',
#                     type=str)
#
# parser.add_argument('-drop_p',
#                     default='0,0,0',
#                     type=str)
#
# parser.add_argument('-test_size',
#                     default=0.8,
#                     type=float)
#
# parser.add_argument('-lr',
#                     default=0.01,
#                     type=float)
#
# parser.add_argument('-batch_size',
#                     default=16,
#                     type=int)
#
# parser.add_argument('-epoch',
#                     default=2000,
#                     type=int)
#

args, args_other = parser.parse_known_args()
args = args_utils.ArgsUtils.add_other_args(args, args_other)
args.sequence_name_orig = str(args.sequence_name)
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
# args.run_name += ('-' + f'{args.dropoutModule}-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
args.run_name = args.sequence_name + ('-' + f'{args.dropoutModule}-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))

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
        if args.dataset == 'weather':
            df = pd.read_csv('datasets/weatherHistory.csv')
            df = df[['Wind Speed (km/h)', 'Humidity', 'Wind Bearing (degrees)', 'Temperature (C)']]
            df.dropna(inplace=True)

            np_x = df[['Wind Speed (km/h)', 'Humidity', 'Wind Bearing (degrees)']].to_numpy().astype(np.float32)
            np_y = df['Temperature (C)'].to_numpy().astype(np.float32)
            np_y = np.expand_dims(np_y, axis=1)

            self.X = ((np_x - np.mean(np_x, axis=0))/np.std(np_x, axis=0)).astype(np.float32)
            self.y = ((np_y - np.mean(np_y, axis=0))/np.std(np_y, axis=0)).astype(np.float32)

            total_avg_y = np.mean(self.y)

        elif args.dataset == 'cal_houses':
            np_x,np_y  = fetch_california_housing(return_X_y = True)
            np_y = np.expand_dims(np_y, axis=1)

            self.X = ((np_x - np.mean(np_x, axis=0)) / np.std(np_x, axis=0)).astype(np.float32)
            self.y = ((np_y - np.mean(np_y, axis=0)) / np.std(np_y, axis=0)).astype(np.float32)

            total_avg_y = np.mean(self.y)

        elif args.dataset == 'bng':
            pd.set_option('display.float_format', lambda x: '%.5f' % x)
            df = pd.read_csv('datasets/BNG.csv')
            df.dropna(inplace=True)

            np_x = df.drop('class', axis = 1).to_numpy()
            np_y = (df['class']).to_numpy()
            np_y = np.expand_dims(np_y, axis=1)

            self.X = ((np_x - np.mean(np_x, axis=0)) / np.std(np_x, axis=0)).astype(np.float32)
            self.y = ((np_y - np.mean(np_y, axis=0)) / np.std(np_y, axis=0)).astype(np.float32)

            total_avg_y = np.mean(self.y)

        elif args.dataset == 'boston_houses':
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
            np_x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            np_y = raw_df.values[1::2, 2]
            np_y = np.expand_dims(np_y, axis=1)

            self.X = ((np_x - np.mean(np_x, axis=0)) / np.std(np_x, axis=0)).astype(np.float32)
            self.y = ((np_y - np.mean(np_y, axis=0)) / np.std(np_y, axis=0)).astype(np.float32)

            total_avg_y = np.mean(self.y)


    def __len__(self):
        if args.is_debug:
            return 500
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = LoadDataset()
train_idx = (int(len(dataset) * (1-float(args.test_size))))
test_idx = len(dataset) - train_idx


dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_idx, test_idx])

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=int(args.batch_size),
                                               shuffle=True,
                                               drop_last=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=int(args.batch_size),
                                              shuffle=False,
                                              drop_last=True)

if args.dropoutModule != 'noDrop':
    CustomDropout = getattr(__import__('DropoutModules.' + args.dropoutModule, fromlist=['Dropout']), 'Dropout')
# 10,128,128,128,1
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

            if args.dropoutModule != "noDrop" and args.dropoutModule != "dropConnect" and l != 0:
                self.layers.add_module(f'{args.dropoutModule}_{l + 1}',
                                       CustomDropout(float(self.drop_p[l]), self.layers_size, l))

        self.layers.add_module("last_linear_layer",
                               torch.nn.Linear(int(self.layers_size[-2]), int(self.layers_size[-1])))

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


model = Model(layers_size=args.layers_size,
              drop_p=args.drop_p)
model.to(DEVICE)

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
                           {'params': dp_params, 'lr': 1e-4}], momentum=0.9, weight_decay=5e-3)
else:
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr)
    )

def get_param_count(model):
 params = list(model.parameters())
 result = 0
 for param in params:
     count_param = np.prod(param.size())
     result += count_param
 return result

size = get_param_count(model)


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

def R2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum() + 1e-8
    denominator = ((y_true - total_avg_y) ** 2).sum() + 1e-8
    result = 1 - numerator / denominator
    return result

best_loss_test = []
best_R2_test = []
losses_train = []
losses_test = []
R2s_train = []
R2s_test = []
metrics_mean_dict = {'loss_train': 0,
                     'loss_test': 0,
                     'R^2_train': 0,
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

        np_y_prim = []
        np_y = []
        losses = []

        for x, y in tqdm(dataloader, desc=mode):

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_prim = model.forward(x)
            loss = torch.mean(torch.abs(y - y_prim))

            if model.training:
                loss.backward()
                opt.step()
                opt.zero_grad()

            np_y += y.detach().cpu().numpy().squeeze().tolist()
            np_y_prim += y_prim.detach().cpu().numpy().squeeze().tolist()

            losses.append(loss.item())

        np_y = np.array(np_y)
        np_y_prim = np.array(np_y_prim)
        r2 = R2_score(np_y, np_y_prim)

        metrics_mean_dict[f'loss_{mode}'] = round(np.mean(losses), 4)
        metrics_mean_dict[f'R^2_{mode}'] = round(r2, 4)

        if not model.training:
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
        if model.training:
            losses_train.append(metrics_mean_dict[f'loss_train'])
            R2s_train.append(metrics_mean_dict[f'R^2_train'])
        else:
            losses_test.append(metrics_mean_dict[f'loss_test'])
            R2s_test.append(metrics_mean_dict[f'R^2_test'])

    moving_averages_train = pd.DataFrame(losses_train).rolling(3, min_periods=1).mean()
    moving_averages_test = pd.DataFrame(losses_test).rolling(3, min_periods=1).mean()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), dpi=100)

    axes[0].set_title('Simple Moving Average of Loss', fontsize=12)
    axes[0].plot(moving_averages_train, label="loss_trian")
    axes[0].plot(moving_averages_test, label="loss_test")
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].legend(loc='upper right', shadow=False, fontsize='medium')

    moving_averages_train_r2s = pd.DataFrame(R2s_train).rolling(3, min_periods=1).mean()
    moving_averages_test_r2s = pd.DataFrame(R2s_test).rolling(3, min_periods=1).mean()
    axes[1].set_title('R2', fontsize=12)
    axes[1].plot(moving_averages_train_r2s, label="R2s_trian")
    axes[1].plot(moving_averages_test_r2s, label="R2s_test")
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].legend(loc='lower right', shadow=False, fontsize='medium')

    plt.savefig(results_dir + last + sample_file_name)
