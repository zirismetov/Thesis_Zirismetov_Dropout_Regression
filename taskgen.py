import argparse
import copy
import logging
import os
import shlex
import time
from datetime import datetime
import sys
sys.path.append('taskgen_files')
import args_utils, file_utils
import torch
from sklearn.model_selection import ParameterGrid
import subprocess
import json
import numpy as np

parser = argparse.ArgumentParser(description="batch_thesis")
parser.add_argument('-sequence_name', type=str, default='sequence')
parser.add_argument('-run_name', type=str, default=str(time.time()))
parser.add_argument(
                    '-script',
                    default='main.py',
                    type=str)

parser.add_argument(
                    '-num_repeat',
                    help='how many times each set of parameters should be repeated for testing stability',
                    default=1,
                    type=int)

parser.add_argument(
                    '-template',
                    default='template_loc.sh',
                    type=str)

parser.add_argument(
                    '-num_tasks_in_parallel',
                    default=6,
                    type=int)

parser.add_argument(
                     '-num_cuda_devices_per_task',
                     default=1,
                     type=int)

parser.add_argument('-is_single_task', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-is_force_start', default=True, type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()
args = args_utils.ArgsUtils.add_other_args(args, args_other)
args.sequence_name_orig = args.sequence_name
args.sequence_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))

file_utils.FileUtils.createDir('./results')
path_sequence = f'./results/{args.sequence_name}'
path_sequence_scripts = f'{path_sequence}/scripts'
file_utils.FileUtils.createDir(path_sequence)
file_utils.FileUtils.createDir(path_sequence_scripts)
file_utils.FileUtils.createDir('./logs')
file_utils.FileUtils.createDir('./artifacts')

rootLogger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
rootLogger.level = logging.DEBUG  # level

fileHandler = logging.FileHandler(f'{path_sequence}/log.txt')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

if hasattr(args, 'class_group_0'):
    if isinstance(args.class_group_0, list):
        args.class_group_0 = ' '.join(args.class_group_0)
if hasattr(args, 'class_group_1'):
    if isinstance(args.class_group_1, list):
        args.class_group_1 = ' '.join(args.class_group_1)
if hasattr(args, 'class_group_2'):
    if isinstance(args.class_group_2, list):
        args.class_group_2 = ' '.join(args.class_group_2)
if hasattr(args, 'classes_include'):
    if isinstance(args.classes_include, list):
        args.classes_include = ' '.join(args.classes_include)
if hasattr(args, 'datasource_jsons'):
    if isinstance(args.datasource_jsons, list):
        args.datasource_jsons = ' '.join(args.datasource_jsons)
if hasattr(args, 'truncate_datasource'):
    if isinstance(args.truncate_datasource, list):
        args.truncate_datasource = ' '.join(args.truncate_datasource)

# Skip these
if hasattr(args, 'tf_path_test'):
    if isinstance(args.tf_path_test, list):
        args.tf_path_test = ' '.join(args.tf_path_test)
    if isinstance(args.tf_path_train, list):
        args.tf_path_train = ' '.join(args.tf_path_train)

args_with_multiple_values = {}
for key, value in args.__dict__.items():
    if isinstance(value, list):
        if len(value) > 1:
            args_with_multiple_values[key] = value

grid_runs = list(ParameterGrid(args_with_multiple_values))

runs = []
for grid_each in grid_runs:
    for _ in range(args.num_repeat):
        run = copy.deepcopy(args.__dict__)
        run.update(grid_each)
        runs.append(run)

if len(runs) == 0:
    logging.error('no grid search combinations found')
    exit()

logging.info(f'planned runs: {len(runs)}')
logging.info(f'grid_runs:\n{json.dumps(grid_runs, indent=4)}')

if not args.is_force_start:
    print('are tests ok? proceed?')
    if input('[y/n]: ') != 'y':
        exit()



max_cuda_devices = 0
cuda_devices_available = 0
if not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info('CUDA NOT AVAILABLE')
else:
    max_cuda_devices = torch.cuda.device_count()
    cuda_devices_available = np.arange(max_cuda_devices).astype(np.int).tolist()

cuda_devices_in_use = []
parallel_processes = []

idx_cuda_device_seq = 0
cuda_devices_list = np.arange(0, max_cuda_devices, dtype=np.int).tolist()

for idx_run, run in enumerate(runs):
    cmd_params = ['-' + key + ' ' + str(value) for key, value in run.items()]
    str_cmd_params = ' '.join(cmd_params)

    str_cuda = ''
    cuda_devices_for_run = []
    if max_cuda_devices > 0:
        if args.num_tasks_in_parallel <= args.num_cuda_devices_per_task:
            for device_id in cuda_devices_available:
                if device_id not in cuda_devices_in_use:
                    cuda_devices_for_run.append(device_id)
                    if len(cuda_devices_for_run) >= args.num_cuda_devices_per_task:
                        break

            if len(cuda_devices_for_run) < args.num_cuda_devices_per_task:
                # reuse existing devices #TODO check to reuse by least recent device
                for device_id in cuda_devices_in_use:
                    cuda_devices_for_run.append(device_id)
                    if len(cuda_devices_for_run) >= args.num_cuda_devices_per_task:
                        break

            for device_id in cuda_devices_for_run:
                if device_id not in cuda_devices_in_use:
                    cuda_devices_in_use.append(device_id)
        else:
            while len(cuda_devices_for_run) < args.num_cuda_devices_per_task:
                cuda_devices_for_run.append(cuda_devices_list[idx_cuda_device_seq])
                idx_cuda_device_seq += 1
                if idx_cuda_device_seq >= len(cuda_devices_list):
                    idx_cuda_device_seq = 0

        if len(cuda_devices_for_run):
            str_cuda = f'CUDA_VISIBLE_DEVICES={",".join([str(it) for it in cuda_devices_for_run])} '

    # detect HPC
    if '/mnt/home/' in os.getcwd():
        str_cuda = ''

    run_name = args.sequence_name_orig + ('-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))
    path_run_sh = f'{path_sequence_scripts}/{run_name}.sh'
    cmd = f'{str_cuda}python3 {args.script} -id {idx_run + 1} -run_name {args.sequence_name_orig}-{idx_run + 1}-run {str_cmd_params}'
    print(path_run_sh)
    print(cmd)
    file_utils.FileUtils.write_text_file(
        path_run_sh,
        file_utils.FileUtils.readAllAsString(args.template) +
        f'\n{cmd} > ./logs/{args.sequence_name_orig}-{idx_run + 1}-run.log'
    )

    cmd = f'chmod +x {path_run_sh}'
    stdout = subprocess.call(shlex.split(cmd))

    logging.info(f'{idx_run}/{len(runs)}: {path_run_sh}\n{cmd}')
    process = subprocess.Popen(
        path_run_sh,
        start_new_session=True,
        shell=False)
    process.cuda_devices_for_run = cuda_devices_for_run
    parallel_processes.append(process)

    time.sleep(1.5)  # delay for timestamp based naming

    while len(parallel_processes) >= args.num_tasks_in_parallel:
        time.sleep(1)
        parallel_processes_filtred = []
        for process in parallel_processes:
            if process.poll() is not None:
                logging.info(process.stdout)
                logging.error(process.stderr)
                # finished
                for device_id in process.cuda_devices_for_run:
                    if device_id in cuda_devices_in_use:
                        cuda_devices_in_use.remove(device_id)
            else:
                parallel_processes_filtred.append(process)
        parallel_processes = parallel_processes_filtred

    if args.is_single_task:
        logging.info('Single task test debug mode completed')
        exit()

while len(parallel_processes) > 0:
    time.sleep(1)
    parallel_processes_filtred = []
    for process in parallel_processes:
        if process.poll() is not None:
            # finished
            for device_id in process.cuda_devices_for_run:
                if device_id in cuda_devices_in_use:
                    cuda_devices_in_use.remove(device_id)
        else:
            parallel_processes_filtred.append(process)
    parallel_processes = parallel_processes_filtred

logging.info('TaskGen finished')
