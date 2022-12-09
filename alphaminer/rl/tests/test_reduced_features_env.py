from alphaminer.rl.env import DataSource, TradingEnv, TradingPolicy, Portfolio, TradingRecorder
from alphaminer.rl.reduced_features_env import ReducedFeaturesEnv
from alphaminer.rl.tests.test_env import get_data_path, SimpleDataHandler
from alphaminer.rl.encoder.vae import VAE, Encoder, Decoder
import pandas as pd
import numpy as np
import qlib
from qlib.contrib.data.handler import Alpha158
from easydict import EasyDict


qlib.init(provider_uri=get_data_path(), region="cn")

start_time = '2010-01-01'
end_time = '2020-01-08'

data_handler = dict(
        type='alpha158',
        market='csi500',
        start_time=start_time,
        end_time=end_time,
        fit_start_time=start_time,
        fit_end_time=end_time,
    )


def test_reduced_obs_env():
    config = EasyDict(data_handler)
    dh = Alpha158(**config)
    ds = DataSource(start_date="2011-11-01",
                    end_date="2011-11-08",
                    market="csi500",
                    data_handler=dh)
    tp = TradingPolicy(data_source=ds)
    e = Encoder(158, 64, 30)  # 2 stocks in index
    d = Decoder(30, 64, 158)
    model = VAE(e, d)
    env = ReducedFeaturesEnv(model=model, data_source=ds, trading_policy=tp, max_episode_steps=5)
    obs = env.reset()
    assert obs.values.shape[1] == 30
    assert obs.values.flatten().shape[-1] == 30 * 2
    done = False
    rewards = []
    for _ in range(5):
        action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        assert isinstance(reward, float)
    assert np.unique(rewards).shape[0] == 5
    assert done

    e = Encoder(158 * 2, 128, 30)  # 2 stocks in index
    d = Decoder(30, 128, 158 * 2)
    model = VAE(e, d)
    env = ReducedFeaturesEnv(model=model, flatten_obs=True, data_source=ds, trading_policy=tp, max_episode_steps=5)
    obs = env.reset()
    assert obs.values.flatten().shape[-1] == 30
    done = False
    rewards = []
    for _ in range(5):
        action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        assert isinstance(reward, float)
    assert np.unique(rewards).shape[0] == 5
    assert done


def test_reduced_obs_dataset():
    config = EasyDict(data_handler)
    dh = Alpha158(**config)
    ds = DataSource(start_date="2011-11-01",
                    end_date="2011-11-08",
                    market="csi500",
                    data_handler=dh)
    tp = TradingPolicy(data_source=ds)

    e = Encoder(158, 64, 30)
    d = Decoder(30, 64, 158)
    model = VAE(e, d)
    env = ReducedFeaturesEnv(model=model, data_source=ds, trading_policy=tp, max_episode_steps=5)
    env.compress_all_obs()

    ds = DataSource(start_date="2011-11-01",
                    end_date="2011-11-08",
                    market="csi500",
                    data_handler=dh)
    tp = TradingPolicy(data_source=ds)

    e = Encoder(158 * 2, 64, 30)  # 2 stocks in index
    d = Decoder(30, 64, 158 * 2)
    model = VAE(e, d)
    env = ReducedFeaturesEnv(model=model, flatten_obs=True, len_index=2, data_source=ds, trading_policy=tp,
                             max_episode_steps=5)
    env.compress_all_obs()

    obs = env.reset()
    assert obs.values.flatten().shape[-1] == 30
    done = False
    rewards = []
    for _ in range(5):
        action = pd.Series(np.random.rand(obs.shape[0]), index=obs.index)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        assert isinstance(reward, float)
    assert np.unique(rewards).shape[0] == 5
    assert done
