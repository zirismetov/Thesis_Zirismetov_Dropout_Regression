import logging
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import sys
import sklearn.metrics
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse
import time
from GaussianDroput_module import GaussianDropout
from datetime import datetime
from tqdm import tqdm

sys.path.append('/Users/zafarzhonirismetov/PycharmProjects/Thesis_Dropout_Regression/taskgen_files')
import csv_utils_2
import file_utils
import args_utils

parser = argparse.ArgumentParser(description="Weather hypermarkets")
parser.add_argument('-sequence_name',
                    type=str,
                    default='sequence')
parser.add_argument('-run_name',
                    type=str,
                    default=str(time.time()))
parser.add_argument('-lr',
                    type=float,
                    default=0.01)
parser.add_argument('-batch_size',
                    type=int,
                    default=64)
parser.add_argument('-test_size',
                    type=float,
                    default=0.20)
parser.add_argument('-epoch',
                    type=int,
                    default=10)

parser.add_argument('-drop_p',
                    default='0, 0.5, 0.5',
                    type=str)

parser.add_argument('-layers_size',
                    type=str,
                    default='1,32,64,32,1')

parser.add_argument('-is_debug',
                    default=False,
                    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-data_file_path',
                    type=str,
                    default='/Users/zafarzhonirismetov/Desktop/Work/CalCOFI/bottle.csv')

args, args_other = parser.parse_known_args()
args = args_utils.ArgsUtils.add_other_args(args, args_other)
args.sequence_name_orig = args.sequence_name
args.sequence_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))

data_raw = pd.read_csv(args.data_file_path, low_memory=False)

# Predict temperature of water 1 features: salinity
data_raw = data_raw[['Salnty', 'T_degC']]
data_raw['Salnty'].replace(0, np.nan, inplace=True)
data_raw['T_degC'].replace(0, np.nan, inplace=True)
data_raw.fillna(method='pad', inplace=True)

args.layers_size = ''.join(args.layers_size)
args.drop_p = ''.join(args.drop_p)

removebrac = "[]''"
for key in args.__dict__:
    if isinstance(args.__dict__[key], str) :
        value = (args.__dict__[key])
        for char in removebrac:
            value = value.replace(char, "")
        args.__dict__.update({key: value})
    elif isinstance(args.__dict__[key], list):
        value = str(args.__dict__[key])
        for char in removebrac:
            value = value.replace(char, "")
        args.__dict__.update({key: value})

args.layers_size = args.layers_size.split(',')
args.drop_p = args.drop_p.split(',')


class DatasetLoadColCOFI(torch.utils.data.Dataset):

    def __init__(self):
        self.X = data_raw['Salnty'].to_numpy()
        np_x = np.copy(self.X)
        self.X[:] = ((np_x[:] - np.min(np_x[:])) / (np.max(np_x[:]) - np.min(np_x[:])))
        self.X = np.expand_dims(self.X, axis=1)

        self.y = data_raw['T_degC'].to_numpy()
        np_y = np.copy(self.y)
        self.y[:] = ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])))
        self.y = np.expand_dims(self.y, axis=1)

    def __len__(self):
        if args.is_debug:
            return 100
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


dataset = DatasetLoadColCOFI()

tr_idx = np.arange(len(dataset))

subset_train_data, subset_test_data = train_test_split(
    tr_idx,
    test_size=float(args.test_size),
    random_state=0)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=int(args.batch_size),
                                               shuffle=False)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=int(args.batch_size),
                                              shuffle=False)


class Model(torch.nn.Module):
    def __init__(self, d_prob=[0, 0.5, 0.5], layer_size=[1, 32, 32, 32, 1]):
        super().__init__()

        self.layers = torch.nn.Sequential()
        for l in range(len(layer_size) - 2):
            self.layers.add_module(f'SimpleDropout_layer_{l + 1}',
                                   GaussianDropout(float(d_prob[l])))
            self.layers.add_module(f'linear_layer_{l + 1}',
                                   torch.nn.Linear(int(layer_size[l]), int(layer_size[l + 1])))
            self.layers.add_module(f'LeakyReLU_layer_{l + 1}',
                                   torch.nn.LeakyReLU())

        self.layers.add_module("last_linear_layer",
                               torch.nn.Linear(int(layer_size[-2]), int(layer_size[-1])))

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
file_utils.FileUtils.createDir(path_run)
file_utils.FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)
csv_utils_2.CsvUtils2.create_global(path_sequence)
csv_utils_2.CsvUtils2.create_local(path_sequence, args.run_name)

model = Model(d_prob=args.drop_p,
              layer_size=args.layers_size)

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

            y_prim = model.forward(torch.FloatTensor(x.float()))
            y = torch.FloatTensor(y.float())

            loss = torch.mean(torch.abs(y - y_prim))
            R2 = sklearn.metrics.r2_score(y.detach(), y_prim.detach())

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
plt.show()