import argparse
import os
from alphaminer.backtesting.backtest_pipeline import pipeline
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
import time
import importlib.util
from copy import copy
from easydict import EasyDict as dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_known_args()[0]
    # load configs
    config_path = args.config
    config_name = str(config_path.split('/')[-1].split('.')[0])
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)  # creates a new module based on spec
    spec.loader.exec_module(config)  # executes the module in its own namespace when a module is imported or reloaded.
    configs = [dict() for _ in range(len(config.alphas))]
    for i, alpha in enumerate(config.alphas):
        configs[i].task = config.task
        configs[i].task['dataset']["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]['config']['feature'] = alpha['feature']
        configs[i].name = alpha['name']
        configs[i].port_analysis_config = config.port_analysis_config
    print('Backtesting {} alphas...'.format(len(configs)))

    start_time = time.time()
    result = Parallel(n_jobs=args.njobs)(delayed(pipeline)
             (config, args.save_dir, True) for config in configs)
    tot_time = time.time() - start_time
    print('Running time:', tot_time)
