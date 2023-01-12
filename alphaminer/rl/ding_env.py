from typing import Union, List
import copy
import numpy as np
import gym
import pandas as pd
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

from alphaminer.rl.env import TradingEnv, TradingPolicy, DataSource, TradingRecorder, RandomSampleWrapper, PORTFOLIO_OPTIMISERS, WeeklyEnv
from alphaminer.data.handler import AlphaMinerHandler
from alphaminer.data.alpha518_handler import Alpha518
from qlib.contrib.data.handler import Alpha158, Alpha360
from alphaminer.rl.gtja_env import GTJADataSource
import random

try:
    from ding.envs import EvalEpisodeReturnEnv
    final_env_cls = EvalEpisodeReturnEnv
except ImportError:
    from ding.envs import FinalEvalRewardEnv
    final_env_cls = FinalEvalRewardEnv


@ENV_REGISTRY.register('trading')
class DingTradingEnv(BaseEnv):

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
        exp_name=None,
        strategy=dict(buy_top_n=10, ),
        portfolio_optimizer="topk",
        random_sample=False,
        data_handler=dict(
            start_time="2010-01-01",
            end_time="2021-12-31",
            fit_start_time="2010-01-01",
            fit_end_time="2021-12-31",
            infer_processors=[],
            learn_processors=[],
        ),
        action_softmax=False,  # apply softmax to actions array
        data_path=None,
        freq="daily",
        done_reward="default",
        p_sticky_action=0.,
        p_random_action=0.,
        action_feature=False,
        action_same_prob=False,
    )

    def __init__(self, cfg: dict) -> None:
        self._cfg = EasyDict({**self.config, **cfg})  # Deep merge default config and custom config
        self._replay_path = None  # replay not used in this env
        self.obs_df = None  # store the current observation as Dataframe
        self.use_recorder = 'recorder' in self._cfg.keys()
        self._random_sample = self._cfg.random_sample
        self._freq = self._cfg.freq
        self._env = self._make_env()
        self._env.observation_space.dtype = np.float32  # To unify the format of envs in DI-engine
        self._observation_space = self._env.observation_space
        self._action_space = gym.spaces.Box(low=0., high=1., shape=(self._env.action_space, ), dtype=np.float32)
        self._reward_space = gym.spaces.Box(
            low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
        )
        self.p_sticky_action = self._cfg.p_sticky_action
        self.p_random_action = self._cfg.p_random_action
        self.prev_action = None
        self.action_feature = self._cfg.action_feature
        self.action_same_prob = self._cfg.action_same_prob

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self.seed(self._seed)
        obs = self._env.reset()
        self.prev_action = None
        self.obs_df = obs  # this is because action needs obs.index to be initialized, so we store the obs in df format
        obs = to_ndarray(obs.values).astype('float32')
        return obs.flatten()

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = self.action_to_series(action)
        if self.p_random_action > 0 or self.p_sticky_action > 0:
            action = self.stochastic_action(action)
        obs, rew, done, info = self._env.step(action)
        self.obs_df = obs  # keep a copy of the original df obs
        obs = to_ndarray(obs.values).astype(np.float32)
        rew = to_ndarray([rew]).astype(np.float32)
        self.prev_action = action
        return BaseEnvTimestep(obs.flatten(), rew, done, info)

    def action_to_series(self, action):
        if 'action_norm' in self._cfg.keys():
            action = norm_action(action, self._cfg.action_norm)
        action[action < 0] = 0
        if sum(action) < 1e-6:
            action[:] = 1
        action = action / sum(action)
        return pd.Series(action, index=self.obs_df.index)  # need the original df obs to perform action

    def _make_env(self):
        ds = None
        if 'type' in self._cfg.data_handler.keys():
            dh_config = copy.deepcopy(self._cfg.data_handler)
            dh_type = dh_config.pop('type', None)
            if dh_type == 'alpha158':
                dh = Alpha158(**dh_config)
            elif dh_type == 'alpha360':
                dh = Alpha360(**dh_config)
            elif dh_type == 'alpha518':
                dh = Alpha518(**dh_config)
            elif dh_type == 'guotai':
                ds = GTJADataSource(
                    start_date=self._cfg.start_date, end_date=self._cfg.end_date, data_dir=self._cfg.data_path
                )
            else:
                dh = AlphaMinerHandler(**dh_config)
        else:
            dh = AlphaMinerHandler(**self._cfg.data_handler)
        if not ds:
            ds = DataSource(
                start_date=self._cfg.start_date, end_date=self._cfg.end_date, market=self._cfg.market, data_handler=dh
            )
        po = None
        po_type = self._cfg.get("portfolio_optimizer")
        if po_type is not None:
            assert po_type in PORTFOLIO_OPTIMISERS, "Portfolio optimizer {} do not exist!".format(po_type)
            po_kwargs = {}
            if po_type == "topk" and self._cfg.strategy.get("buy_top_n"):  # For compatibility with old parameters
                po_kwargs["topk"] = int(self._cfg.strategy.get("buy_top_n"))
            po = PORTFOLIO_OPTIMISERS[po_type](**po_kwargs)
        tp = TradingPolicy(data_source=ds, **self._cfg.strategy, portfolio_optimizer=po)
        if not self._cfg.max_episode_steps:
            self._cfg.max_episode_steps = len(ds.dates) - 1
        recorder = None
        if self.use_recorder:
            recorder = TradingRecorder(
                data_source=ds, dirname=self._cfg.recorder.path, filename=self._cfg.recorder.get("exp_name")
            )
        if self._freq == "weekly":
            env = WeeklyEnv(
                data_source=ds,
                trading_policy=tp,
                max_episode_steps=self._cfg.max_episode_steps,
                cash=self._cfg.cash,
                recorder=recorder,
                done_reward=self._cfg.done_reward,
            )
        else:
            env = TradingEnv(
                data_source=ds,
                trading_policy=tp,
                max_episode_steps=self._cfg.max_episode_steps,
                cash=self._cfg.cash,
                recorder=recorder,
                done_reward=self._cfg.done_reward,
            )

        if self._random_sample:
            env = RandomSampleWrapper(env, n_sample=self._random_sample)
        env = final_env_cls(env)
        return env

    def random_action(self) -> pd.Series:
        action = self.action_space.sample()
        action = self.action_to_series(action)
        return action

    def __repr__(self) -> str:
        return "Alphaminer Trading Env"

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_cfg = copy.deepcopy(cfg)
        collector_env_num = collector_cfg.pop('collector_env_num', 1)
        return [collector_cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_env_num = evaluator_cfg.pop('evaluator_env_num', 1)
        return [evaluator_cfg for _ in range(evaluator_env_num)]

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space


@ENV_REGISTRY.register('trading_ma')
class DingMATradingEnv(DingTradingEnv):

    def reset(self) -> np.ndarray:
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self.seed(self._seed)
        raw_obs = self._env.reset()
        self.prev_action = None
        self.obs_df = raw_obs  # this is because action needs obs.index to be initialized, so we store the obs in df format
        agent_obs = to_ndarray(copy.deepcopy(raw_obs.values)).astype('float32')
        if self.action_feature:
            action_agent_obs = self.get_action_feature(agent_obs)
        action_mask = np.ones((500, 1))
        obs = {
            'global_state': agent_obs.flatten(),
            'agent_state': agent_obs if not self.action_feature else action_agent_obs,
            'action_mask': action_mask,
        }
        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        action = to_ndarray(action).astype(np.float32)
        action = self.action_to_series(action)
        if self.p_random_action > 0 or self.p_sticky_action > 0:
            action = self.stochastic_action(action)
        raw_obs, rew, done, info = self._env.step(action)
        self.obs_df = raw_obs  # keep a copy of the original df obs
        agent_obs = to_ndarray(copy.deepcopy(raw_obs.values)).astype('float32')
        action_mask = np.ones((500, 1))
        if self.action_feature:
            action_agent_obs = self.get_action_feature(agent_obs)
        obs = {
            'global_state': agent_obs.flatten(),
            'agent_state': agent_obs if not self.action_feature else action_agent_obs,
            'action_mask': action_mask,
        }
        rew = to_ndarray([rew]).astype(np.float32)
        self.prev_action = action
        return BaseEnvTimestep(obs, rew, done, info)

    def stochastic_action(self, action):
        if self.prev_action is not None:  # don't use stochastic actions in first step
            action_array = action.to_numpy()
            prev_action_array = self.prev_action.to_numpy()
            if self.p_sticky_action > 0:
                if self.action_same_prob and random.random() <= self.p_sticky_action:
                    action = self.prev_action  # all actions are sticky or not
                else:
                    mask = np.random.choice([0, 1], size=len(action),
                                            p=[1 - self.p_sticky_action, self.p_sticky_action])
                    action_array[mask] = prev_action_array
            elif self.p_random_action > 0:
                if self.action_same_prob and random.random() <= self.p_random_action:
                    action = self.random_action()  # all actions are random
                else:
                    mask = np.random.choice([0, 1], size=len(action),
                                            p=[1 - self.p_sticky_action, self.p_sticky_action])
                    action_array[mask] = self.random_action().to_numpy()
            return self.action_to_series(action)
        else:
            return action

    def get_action_feature(self, agent_obs):
        if self.prev_action is None:
            self.prev_action = self.action_to_series(np.zeros_like(self.random_action().to_numpy()))
        return np.concatenate((agent_obs, np.expand_dims(self.prev_action.to_numpy(), -1)), axis=-1)


def norm_action(action, norm_type=None):

    assert norm_type in {None, 'softmax', 'cut_softmax', 'gumbel_softmax', 'topk_softmax'}

    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _top_k(x, k):
        return np.argpartition(x, -k)[-k:]

    def _cut_softmax(x, k=5):
        indexs = _top_k(x, k)
        x[indexs] = np.min(x[indexs])
        return _softmax(x)

    def _gumbel_softmax(x):
        u = np.random.uniform(size=np.shape(x))
        z = -np.log(-np.log(u))
        return _softmax(z)

    def _topk_softmax(x, k=10):
        x = _softmax(x)
        indexs = _top_k(x, k)
        x[indexs] = 0
        return x

    if norm_type == 'softmax':
        action = _softmax(action)

    elif norm_type == "cut_softmax":
        action = _cut_softmax(action)

    elif norm_type == "gumbel_softmax":
        action = _gumbel_softmax(action)

    elif norm_type == 'topk_softmax':
        action = _topk_softmax(action)

    return action
