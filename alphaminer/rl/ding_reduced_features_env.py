from typing import Union, List
import copy
import numpy as np
import gym
import pandas as pd
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep, FinalEvalRewardEnv
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

from alphaminer.rl.env import TradingEnv, TradingPolicy, DataSource, TradingRecorder
from alphaminer.rl.ding_env import DingTradingEnv
from alphaminer.rl.reduced_features_env import ReducedFeaturesEnv
from alphaminer.data.handler import AlphaMinerHandler
from qlib.contrib.data.handler import Alpha158, Alpha360
from alphaminer.rl.encoder.vae import VAE, Encoder, Decoder


@ENV_REGISTRY.register('trading_reduced_obs')
class DingReducedFeaturesEnv(DingTradingEnv):
    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        env_id='Trading-v0',
        max_episode_steps=10,
        cash=1000000,
        start_date='2010-01-01',
        end_date='2021-12-31',
        market='csi500',
        strategy=dict(buy_top_n=10, ),
        len_index=None,
        data_handler=dict(
            type='alpha158',
            start_time="2010-01-01",
            end_time="2021-12-31",
            fit_start_time="2010-01-01",
            fit_end_time="2021-12-31",
            infer_processors=[],
            learn_processors=[],
        ),
        action_softmax=False,  # apply softmax to actions array
        model=dict(
            encoder_sizes=[158, 64, 15],
            decoder_sizes=[15, 64, 158],
            type='vae',
            flatten_encode=False,
            load_path=None,
            preprocess_obs=True,
        ),
    )

    def __init__(self, cfg: dict) -> None:
        DingTradingEnv.__init__(self, cfg)

    def _make_env(self):
        if 'type' in self._cfg.data_handler.keys():
            alpha_config = copy.deepcopy(self._cfg.data_handler)
            alpha = alpha_config.pop('type', None)
            if alpha == 'alpha158':
                dh = Alpha158(**alpha_config)
            elif alpha == 'alpha360':
                dh = Alpha360(**alpha_config)
            else:
                dh = AlphaMinerHandler(**alpha_config)
        else:
            dh = AlphaMinerHandler(**self._cfg.data_handler)
        ds = DataSource(start_date=self._cfg.start_date,
                        end_date=self._cfg.end_date,
                        market=self._cfg.market,
                        data_handler=dh)
        tp = TradingPolicy(data_source=ds, **self._cfg.strategy)
        if not self._cfg.max_episode_steps:
            self._cfg.max_episode_steps = len(ds.dates) - 1
        recorder = None
        if self.use_recorder:
            recorder = TradingRecorder(data_source=ds,
                                       dirname=self._cfg.recorder.path)
        e = Encoder(*self._cfg.model.encoder_sizes)
        d = Decoder(*self._cfg.model.decoder_sizes)
        if self._cfg.model.type == 'vae':
            model = VAE(e, d)
        else:
            raise NotImplementedError
        if 'len_index' not in self._cfg.keys():
            self._cfg.len_index = None
        env = ReducedFeaturesEnv(model=model,
                                 load_path=self._cfg.model.load_path,
                                 flatten_obs=self._cfg.model.flatten_encode,
                                 data_source=ds,
                                 trading_policy=tp,
                                 max_episode_steps=self._cfg.max_episode_steps,
                                 len_index=self._cfg.len_index,
                                 cash=self._cfg.cash,
                                 recorder=recorder)
        if self._cfg.model.preprocess_obs:
            env.compress_all_obs()
        env = FinalEvalRewardEnv(env)
        return env

    def __repr__(self) -> str:
        return "Alphaminer Trading Env Reduced Features"
