import argparse
import os
from alphaminer.backtesting import pipeline
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_known_args()[0]
    # load configs
    config_path = args.config_path
    configs = [f for f in listdir(config_path) if isfile(join(config_path, f)) and f.endswith('.py')]

    start_time = time.time()
    result = Parallel(n_jobs=args.njobs)(delayed(pipeline)
             (os.path.join(config_path, config), args.save_dir, True) for config in configs)
    tot_time = time.time() - start_time
    print('Running time:', tot_time)
