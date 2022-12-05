import pandas as pd
import numpy as np
import qlib
from alphaminer.rl.gtja_env import GTJADataSource
from alphaminer.rl.env import TradingPolicy, RandomSampleEnv
from os import path as osp
from unittest.mock import patch
from qlib.data import D


def get_data_path() -> str:
    dirname = osp.dirname(osp.realpath(__file__))
    return osp.realpath(osp.join(dirname, "gtja_data"))


qlib.init(provider_uri=get_data_path(), region="cn")


def test_gtja_env():
    def mock_qlib_features(codes, *args, **kwargs):
        if codes[0] == "SH000300":
            df = pd.read_csv(osp.join(get_data_path(), "csi300.csv"))
        else:
            df = pd.read_csv(osp.join(get_data_path(), "qlib_data.csv"))
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= "2019-01-01")
                & (df["datetime"] <= "2019-01-10")]
        df.set_index(["instrument", "datetime"], inplace=True)
        return df

    def mock_qlib_list_instruments(*args, **kwargs):
        return [""]

    def mock_qlib_calendar(*args, **kwargs):
        return pd.to_datetime(
            pd.Series([
                '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-07',
                '2019-01-08', '2019-01-09', '2019-01-10'
            ]))

    with patch.object(D, "features", mock_qlib_features), patch.object(
            D, "list_instruments",
            mock_qlib_list_instruments), patch.object(D, "calendar",
                                                      mock_qlib_calendar):
        ds = GTJADataSource(start_date="2019-01-01",
                            end_date="2019-01-10",
                            data_dir=get_data_path())
        tp = TradingPolicy(data_source=ds)
        env = RandomSampleEnv(n_sample=500,
                              data_source=ds,
                              trading_policy=tp,
                              max_episode_steps=5)
        obs = env.reset()
        init_codes = obs.index.tolist()
        assert len(init_codes) > 1
        assert obs.shape[0] == 500  # 500 samples
        assert obs.shape[1] == 10  # 10 factors
        done = False
        for _ in range(5):
            action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
            obs, _, done, _ = env.step(action)
            assert (obs.index == init_codes).sum(
            ) > 490  # Stocks delisted during this period will be replaced by other stocks
        assert done
