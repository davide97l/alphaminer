from qlib.utils import init_instance_by_config
import qlib
import importlib.util
from pandas_profiling import ProfileReport
import argparse
import pandas as pd

# usage example:
# python backtesting/data_visualization.py --config ../configs/config_lightgbm_cohlv_csi500.py --stock SH600000  --save_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--segment', type=str, default='train', choices=('train', 'valid', 'test'))
    parser.add_argument('--stock', type=str, default=None)
    parser.add_argument('--save_dataset', action='store_true')
    args = parser.parse_known_args()[0]

    # choose a stock to analyze
    stock = args.stock
    # choose dataset portion
    segment = args.segment
    # whether to save the dataset as csv
    save_dataset = args.save_dataset

    # load config
    config_path = str(args.config)
    config_name = str(config_path.split('/')[-1].split('.')[0])
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    config = importlib.util.module_from_spec(spec)  # creates a new module based on spec
    spec.loader.exec_module(config)  # executes the module in its own namespace when a module is imported or reloaded.

    qlib.init()  # must init before running all the other commands

    # init dataset
    dataset = init_instance_by_config(config.task["dataset"])
    df = dataset.prepare(segment)
    print(df)
    if save_dataset:
        df_name = '{}.csv'.format(segment)
        df.to_csv(df_name, sep='\t', encoding='utf-8')
        print("Data saved to: {}".format(df_name))

    # generate data analysis report
    # if too many rows pandas profiling may crash, you can change this value according to your computing power
    if stock is not None:
        assert len(df.columns) <= 20
        assert stock in list(df.index.get_level_values('instrument')), \
            "Error: {} not in dataset".format(stock)
        df_stock = df[df.index.get_level_values('instrument') == stock]
        profile = ProfileReport(df_stock, title="Pandas Profiling Report")
        report_path = "{}_set_visualization_{}.html".format(segment, stock)
        profile.to_file(report_path)
        print("Report saved to: {}".format(report_path))
