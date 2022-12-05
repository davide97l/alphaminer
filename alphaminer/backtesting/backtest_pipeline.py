import numpy as np
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import qlib
import pandas as pd
from qlib.contrib.report import analysis_model, analysis_position
import importlib.util
import argparse
import os
from alphaminer.backtesting.utils import benchmark_metrics, detailed_report
import time
import random


def pipeline(config_path, save_dir: str = '', cache_dataset: bool = True):
    random.seed(0)
    np.random.seed(0)
    if cache_dataset:
        cache_dataset = "SimpleDatasetCache"
        print('Using cached dataset if available')
    else:
        cache_dataset = None
    start = time.time()
    # load config by file
    if isinstance(config_path, str):
        config_name = str(config_path.split('/')[-1].split('.')[0])
        spec = importlib.util.spec_from_file_location(config_name, config_path)
        config = importlib.util.module_from_spec(spec)  # creates a new module based on spec
        spec.loader.exec_module(config)  # executes the module in its own namespace when a module is imported or reloaded.
    elif isinstance(config_path, dict):
        config = config_path
        config_name = config["name"]
    else:
        raise Exception("Type not supported")

    try:
        config.provider_uri
        print('reading data from', config.provider_uri)
        qlib.init(dataset_cache=cache_dataset, provider_uri=config.provider_uri)
    except:
        qlib.init(dataset_cache=cache_dataset)

    # model and dataset initiaiton
    model = init_instance_by_config(config.task["model"])
    dataset = init_instance_by_config(config.task["dataset"])

    # train and test the model
    with R.start(experiment_name="workflow"):
        model.fit(dataset)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        rid = rec.id  # save the record id

        # Inference and saving signal
        sr = SignalRecord(model, dataset, rec)
        sr.generate()

    # backtest and analysis
    with R.start(experiment_name='exp', recorder_id=rid, resume=True):
        # signal-based analysis
        rec = R.get_recorder()
        sar = SigAnaRecord(rec)  # get IC, ICIR, Rank IC, Rank ICIR
        sar.generate()

        # portfolio-based analysis: backtest
        # get mean, std, annualized_return, information_ratio, max_drawdown
        par = PortAnaRecord(rec, config.port_analysis_config, "day")
        par.generate()

    # load recorder
    recorder = R.get_recorder(recorder_id=rid, experiment_name='exp')
    # load previous results
    pred_df = recorder.load_object("pred.pkl")
    ic_df = recorder.load_object("sig_analysis/ic.pkl")
    rankic_df = recorder.load_object("sig_analysis/ric.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    indicators_normal = recorder.load_object("portfolio_analysis/indicators_normal_1day.pkl")

    plots = []

    # https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
    result = analysis_position.report_graph(report_normal_df, show_notebook=False)
    plots += result

    # https://qlib.readthedocs.io/en/latest/component/report.html#graphical-result
    result = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    plots += result

    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ['label']
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    result = analysis_position.score_ic_graph(pred_label, show_notebook=False)
    plots += result

    result = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    plots += result

    # save report to html
    report_name = 'report.html'
    os.makedirs(os.path.join(save_dir, config_name), exist_ok=True)
    report_path = os.path.join(save_dir, config_name, report_name)
    with open(report_path, 'w') as f:
        for r in plots:
            f.write(r.to_html(full_html=False, include_plotlyjs='cdn'))
    print('Report saved to:', report_path)

    # save detailed report
    df_stats, df_trades = detailed_report(report_normal_df, positions, pred_df)
    report_path = os.path.join(save_dir, config_name, 'trading_stats.csv')
    df_stats.to_csv(report_path, encoding='utf-8', index=False)
    print('Trading stats report saved to:', report_path)
    report_path = os.path.join(save_dir, config_name, 'trading_positions.csv')
    df_trades.to_csv(report_path, encoding='utf-8', index=False)
    print('Trading positions report saved to:', report_path)

    # csv benchmark report
    os.makedirs(save_dir, exist_ok=True)
    benchmark_dir = os.path.join(save_dir, 'benchmark.csv')
    benchmark_results = benchmark_metrics(config_name, analysis_df, ic_df, rankic_df, pred_label, report_normal_df)
    df = pd.DataFrame([benchmark_results])
    if os.path.isfile(benchmark_dir):
        benchmark = pd.read_csv(benchmark_dir, index_col=False)
        benchmark = pd.concat([benchmark, df])
        # remove duplicated configs
        benchmark = benchmark.drop_duplicates(subset='config', keep='last')
    else:
        benchmark = df
    benchmark.to_csv(benchmark_dir, encoding='utf-8', index=False)
    print('Added {} to benchmark report: {}'.format(config_name, benchmark_dir))
    # save metrics report also as single file
    report_path = os.path.join(save_dir, config_name, 'metrics.csv')
    df.to_csv(report_path, encoding='utf-8', index=False)
    print('Metrics report saved to:', report_path)

    print('Process finished in {}s'.format(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--cache_dataset', action='store_true')
    args = parser.parse_known_args()[0]
    pipeline(args.config, args.save_dir, args.cache_dataset)
