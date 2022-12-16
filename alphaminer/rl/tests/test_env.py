from alphaminer.rl.env import DataSource, TradingEnv, TradingPolicy, Portfolio, TradingRecorder, RandomSampleEnv, TopkOptimizer
from os import path as osp
from qlib.data.dataset import DataHandler
from qlib.data import D
import os
import pandas as pd
import numpy as np
import qlib
import tempfile
import shutil
from unittest.mock import patch


def get_data_path() -> str:
    dirname = osp.dirname(osp.realpath(__file__))
    return osp.realpath(osp.join(dirname, "data"))


qlib.init(provider_uri=get_data_path(), region="cn")


class SimpleDataHandler(DataHandler):
    """
    Fit qlib data to RL env.
    """

    def __init__(self, instruments, start_time, end_time, init_data=True, fetch_orig=True):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config()
                }
            },
        }
        super().__init__(instruments, start_time, end_time, data_loader, init_data, fetch_orig)

    def _get_feature_config(self):
        fields = ["$close", "$open", "$factor"]
        names = ["CLOSE", "OPEN", "FACTOR"]
        return fields, names


def test_data_source():
    ds = DataSource(
        start_date="2010-01-01",
        end_date="2020-01-01",
        market="csi500",
        data_handler=SimpleDataHandler(D.instruments("csi500"), start_time="2010-01-01", end_time="2020-01-01")
    )

    # Check query
    data = ds.query_trading_data("2012-01-04")
    assert data.shape[0] == 2

    data = ds.query_trading_data("2019-01-04")
    assert data.shape[0] == 3


def test_portfolio():
    ds = DataSource(
        start_date="2010-01-01",
        end_date="2020-01-01",
        market="csi500",
        data_handler=SimpleDataHandler(D.instruments("csi500"), start_time="2010-01-01", end_time="2020-01-01")
    )

    cash = 0
    pf = Portfolio(cash=cash)
    assert len(pf.positions) == 0

    date = "2012-06-15"
    code = "SH600006"

    pf.positions = pd.Series({code: 1000}, dtype=np.float64)

    price = ds.query_trading_data(date, [code])["close"]
    assert pf.nav(price) > 0

    # Test suspended stocks with null value
    pf = Portfolio(cash=0)
    codes = ["SH600006", "SH600021"]
    pf.positions = pd.Series([1000, 1000], index=codes, dtype=np.float64)
    price = ds.query_trading_data(date="2011-11-01", instruments=codes)["close"]
    old_nav = pf.nav(price)
    price = ds.query_trading_data(
        date="2011-11-02",  # The day with null value
        instruments=codes
    )["close"]
    new_nav = pf.nav(price)
    assert new_nav / old_nav > 0.9

    # Test delisted stock that still exists in portfolio
    pf = Portfolio(cash=0)
    codes = ["SH600006", "SH600021", "SH600607"]
    pf.positions = pd.Series([1000, 1000, 1000], index=codes, dtype=np.float64)
    price = ds.query_trading_data(date="2011-11-01", instruments=codes)["close"]
    assert pf.nav(price) < 3300  # Price of delisted stock will not include in nav


def test_trading_policy():
    ds = DataSource(
        start_date="2010-01-01",
        end_date="2020-01-01",
        market="csi500",
        data_handler=SimpleDataHandler(D.instruments("csi500"), start_time="2010-01-01", end_time="2020-01-01")
    )
    pf = Portfolio(cash=800000)

    action = pd.Series({
        "SH600006": 1.1,  # Buy
        "SH600008": 0,
    })
    tp = TradingPolicy(data_source=ds)
    date = "2012-06-15"
    pf, log_change = tp.take_step(date, action, portfolio=pf)
    assert log_change < 0 and log_change > -0.01
    price = ds.query_trading_data(date, pf.positions.index.tolist())["close"]
    old_nav = pf.nav(price)

    date = "2012-10-08"
    new_action = pd.Series({
        "SH600006": 0.6,  # Buy
        "SH600008": 0.5,  # Buy
    })
    pf, log_change = tp.take_step(date, new_action, portfolio=pf)
    assert log_change > np.log(0.9) and log_change < np.log(1.1)
    price = ds.query_trading_data(date, pf.positions.index.tolist())["close"]
    new_nav = pf.nav(price)

    assert new_nav / old_nav < 0.81


def test_trading_policy_with_benchmark_index():
    ds = DataSource(
        start_date="2010-01-01",
        end_date="2020-01-01",
        market="csi500",
        data_handler=SimpleDataHandler(D.instruments("csi500"), start_time="2010-01-01", end_time="2020-01-01")
    )
    pf = Portfolio(cash=800000)

    action = pd.Series({
        "SH600006": 1.1,  # Buy
        "SH600008": 0,
    })
    tp = TradingPolicy(data_source=ds, use_benchmark=True, benchmark_index="SH000905")
    date = "2012-06-15"
    pf, log_change = tp.take_step(date, action, portfolio=pf)
    assert log_change > 0 and log_change < 0.02


def test_trading_env():
    ds = DataSource(
        start_date="2011-11-01",
        end_date="2011-11-08",
        market="csi500",
        data_handler=SimpleDataHandler(D.instruments("csi500"), start_time="2010-01-01", end_time="2020-01-01")
    )
    tp = TradingPolicy(data_source=ds)
    env = TradingEnv(data_source=ds, trading_policy=tp, max_episode_steps=5)
    obs = env.reset()
    assert obs.shape[1] > 1
    done = False
    rewards = []
    for _ in range(5):
        action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        assert isinstance(reward, float)
    assert np.unique(rewards).shape[0] == 5
    assert done

    # Test recorder
    tempdir = osp.join(tempfile.gettempdir(), "records")
    os.mkdir(tempdir)
    try:
        recorder = TradingRecorder(data_source=ds, dirname=tempdir)
        env = TradingEnv(data_source=ds, trading_policy=tp, max_episode_steps=5, recorder=recorder)
        obs = env.reset()
        for _ in range(5):
            action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
            env.step(action)
        env.reset()
        record_files = os.listdir(tempdir)
        assert len(record_files) == 1
        df = pd.read_csv(osp.join(tempdir, record_files[0]), index_col=0)
        print(df)
        assert df.shape == (5, 6)
        assert "2011-11" in str(df.index[0])
    finally:
        if osp.exists(tempdir):
            shutil.rmtree(tempdir)


def test_stable_stock_index():
    """
    Env should keep the original position of stocks when new stocks
    been added to the index.
    """
    start_date = pd.to_datetime("2011-11-01")
    end_date = pd.to_datetime("2011-11-08")

    def mock_list_instruments(instruments, start_time, end_time, as_list):
        if start_time == start_date and end_time == end_date:
            return ['sh600006', 'sh600021', "sh600008"]
        elif start_time == start_date:
            return ['sh600006', 'sh600008']
        else:
            return ['sh600008', 'sh600021']

    with patch.object(D, 'list_instruments', mock_list_instruments):
        ds = DataSource(
            start_date=start_date,
            end_date=end_date,
            market="csi500",
            data_handler=SimpleDataHandler(
                ['sh600006', 'sh600021', "sh600008"], start_time="2010-01-01", end_time="2020-01-01"
            )
        )
    tp = TradingPolicy(data_source=ds)
    env = TradingEnv(data_source=ds, trading_policy=tp, max_episode_steps=5)
    # Reset
    obs = env.reset()
    assert obs.index.tolist() == ['sh600006', 'sh600008']
    # Step1
    action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
    obs, *_ = env.step(action)
    assert obs.index.tolist() == ['sh600021', 'sh600008']
    # Step2
    action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
    obs, *_ = env.step(action)
    assert obs.index.tolist() == ['sh600021', 'sh600008']


def test_random_sample_env():
    ds = DataSource(
        start_date="2011-11-01",
        end_date="2011-11-08",
        market="csi500",
        data_handler=SimpleDataHandler(D.instruments("csi500"), start_time="2010-01-01", end_time="2020-01-01")
    )
    tp = TradingPolicy(data_source=ds)
    env = RandomSampleEnv(n_sample=1, data_source=ds, trading_policy=tp, max_episode_steps=5)
    obs = env.reset()
    assert obs.shape[0] == 1
    code = obs.index[0]
    done = False
    for _ in range(5):
        action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
        obs, _, done, _ = env.step(action)
        assert obs.shape[0] == 1
        assert obs.index[0] == code
    assert done


def test_portfolio_optimizer():
    action = pd.Series(np.random.rand(50))

    topk = TopkOptimizer(10, equal_weight=True)
    weight = topk.get_weight(action=action)
    assert all(weight == 0.1)

    topk = TopkOptimizer(10, equal_weight=False)
    weight = topk.get_weight(action=action)
    assert round(weight.sum(), 6) == 1


def test_paused_stock():
    start_date = pd.to_datetime("2021-08-30")
    end_date = pd.to_datetime("2021-09-03")

    def mock_list_instruments(instruments, start_time, end_time, as_list):
        return ['sh600006', 'sh600068']

    with patch.object(D, 'list_instruments', mock_list_instruments):
        ds = DataSource(
            start_date=start_date,
            end_date=end_date,
            market="csi500",
            data_handler=SimpleDataHandler(['sh600006', 'sh600068'], start_time=start_date, end_time=end_date)
        )
    tp = TradingPolicy(data_source=ds)
    recorder = TradingRecorder(data_source=ds)
    env = TradingEnv(data_source=ds, trading_policy=tp, max_episode_steps=-1, recorder=recorder)
    # Reset
    obs = env.reset()
    done = False
    while not done:
        action = pd.Series(1, index=['sh600006', 'sh600068'])
        obs, _, done, _ = env.step(action)
    replay = recorder.records
    # Paused stocks should still in position
    # NAV need to include the value of the last day of delisted stocks
    assert replay["nav"].iloc[-1] > 1e6
    assert replay["position"]["sh600068"].iloc[-1] > 0
