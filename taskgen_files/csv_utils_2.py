import copy
import os
import time
import logging
import sys
import traceback
import traceback, sys
import numpy as np
from taskgen_files import file_utils as fu


class CsvUtils2():

    @staticmethod
    def create_global(path_sequence):
        if not os.path.exists(f'{path_sequence}/sequence-{os.path.basename(path_sequence)}.csv'):
            with open(f'{path_sequence}/sequence-{os.path.basename(path_sequence)}.csv', mode='w') as csv_file:
                csv_file.write('')

    # results for each test instance/task
    @staticmethod
    def create_local(path_sequence, run_name):
        if not os.path.exists(f'{path_sequence}/{run_name}.csv'):
            with open(f'{path_sequence}/{run_name}.csv', mode='w') as csv_file:
                csv_file.write('')

    @staticmethod
    def add_hparams(path_sequence, run_name, args_dict, metrics_dict, global_step):
        try:
            path_local_csv = f'{path_sequence}/{run_name}.csv'
            path_global_csv = f'{path_sequence}/sequence-{os.path.basename(path_sequence)}.csv'

            args_dict = copy.copy(args_dict)
            metrics_dict = copy.copy(metrics_dict)
            for each_dict in [args_dict, metrics_dict]:
                for key in list(each_dict.keys()):
                    if not isinstance(each_dict[key], float) and \
                        not isinstance(each_dict[key], int) and \
                        not isinstance(each_dict[key], str) and \
                        not isinstance(each_dict[key], np.float) and \
                        not isinstance(each_dict[key], np.int) and \
                        not isinstance(each_dict[key], np.float32):
                        del each_dict[key]

            for path_csv in [path_local_csv, path_global_csv]:

                if os.path.exists(path_csv):
                    with open(path_csv, 'r+') as outfile:
                        fu.FileUtils.lock_file(outfile)
                        lines_all = outfile.readlines()
                        lines_all = [it.replace('\n', '').split(',') for it in lines_all if ',' in it]
                        if len(lines_all) == 0 or len(lines_all[0]) < 2:
                            headers = ['step'] + list(args_dict.keys()) + list(metrics_dict.keys())
                            headers = [str(it).replace(',', '_') for it in headers]
                            lines_all.append(headers)
                        else:
                            headers = lines_all[0]

                        values = [global_step]
                        for key in headers[1:]:
                            if key in args_dict:
                                values.append(args_dict[key])
                            elif key in metrics_dict:
                                values.append(metrics_dict[key])
                            else:
                                values.append('')

                        values = [str(it).replace(',', '_') for it in values]
                        if path_csv == path_local_csv:
                            lines_all.append(values)
                        else:
                            # global
                            existing_line_idx = -1
                            args_values = []
                            for key in headers[1:]:
                                if key in args_dict:
                                    args_values.append(args_dict[key])
                                else:
                                    if key in metrics_dict:
                                        break
                                    args_values.append('')

                            args_values = [str(it).replace(',', '_') for it in args_values]
                            for idx_line, line in enumerate(lines_all):
                                if len(line) > 1:
                                    is_match = True
                                    for idx_arg in range(len(args_values)):
                                        if line[idx_arg + 1] != args_values[idx_arg]:
                                            is_match = False
                                            break
                                    if is_match:
                                        existing_line_idx = idx_line
                                        break
                            if existing_line_idx >= 0:
                                lines_all[existing_line_idx] = values
                            else:
                                lines_all.append(values)

                        outfile.truncate(0)
                        outfile.seek(0)
                        outfile.flush()
                        rows = [','.join(it) for it in lines_all]
                        rows = [it for it in rows if len(it.replace('\n', '').strip()) > 0]
                        outfile.write('\n'.join(rows).strip())
                        outfile.flush()
                        os.fsync(outfile)
                        fu.FileUtils.unlock_file(outfile)

        except Exception as e:
            logging.exception(e)
