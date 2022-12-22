import pytest
import numpy as np
from easydict import EasyDict
from os import path as osp

from ding.utils import set_pkg_seed
from alphaminer.rl.ding_env import DingTradingEnv
from alphaminer.rl.tests.test_gtja_env import get_data_path as get_gtja_data_path
import qlib


def get_data_path() -> str:
    dirname = osp.dirname(osp.realpath(__file__))
    return osp.realpath(osp.join(dirname, "data"))


config_test = dict(
    env_id='Trading-v0',
    max_episode_steps=5,
    cash=1000000,
    start_date='2011-11-01',
    end_date='2011-11-08',
    market='csi500',
    random_sample=False,
    strategy=dict(buy_top_n=1, ),
    data_handler=dict(
        start_time='2011-11-01',
        end_time='2011-11-08',
        alphas=None,
        fit_start_time='2011-11-01',
        fit_end_time='2011-11-08',
        infer_processors=[],
        learn_processors=[],
    ),
    action_softmax=True
)

infer_processors = [
    {
        "class": "Fillna",
        "kwargs": {}
    },
    {
        "class": "ZScoreNorm",
        "kwargs": {}
    },
]
#start_time = '2010-01-01'
#end_time = '2010-10-01'
start_time = "2019-01-02"
end_time = "2019-10-10"

qlib_config = dict(
    env_id='Trading-v0',
    max_episode_steps=100,
    cash=1000000,
    start_date=start_time,
    end_date=end_time,
    market='csi500',
    random_sample=False,
    strategy=dict(buy_top_n=10, ),
    data_handler=dict(
        instruments='csi500',
        start_time=start_time,
        end_time=end_time,
        alphas=None,
        fit_start_time=start_time,
        fit_end_time=end_time,
        infer_processors=infer_processors,
        learn_processors=[],
    )
)

alpha_config = dict(
    env_id='Trading-v0',
    max_episode_steps=0,
    cash=1000000,
    start_date=start_time,
    end_date=end_time,
    market='csi500',
    random_sample=False,
    strategy=dict(buy_top_n=10, ),
    data_handler=dict(
        type='alpha158',
        instruments='csi500',
        start_time=start_time,
        end_time=end_time,
        fit_start_time=start_time,
        fit_end_time=end_time,
    )
)


guotai_config = dict(
    env_id='Trading-v0',
    max_episode_steps=None,
    cash=1000000,
    start_date="2019-01-02",
    end_date="2019-01-10",
    market='csi500',
    data_path=get_gtja_data_path(),
    random_sample=50,
    strategy=dict(
        buy_top_n=10,
    ),
    data_handler=dict(
        type='guotai',
    )
)


@pytest.mark.envtest
def test_ding_trading():
    qlib.init(provider_uri=get_data_path(), region="cn")
    set_pkg_seed(1234, use_cuda=False)
    env = DingTradingEnv(EasyDict(config_test))
    env.seed(1234)
    env.reset()
    action_dim = env.action_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        print(timestep)
        final_eval_reward += timestep.reward
        print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['eval_episode_return'], type(timestep.info['eval_episode_return']), final_eval_reward,
                    type(final_eval_reward)
                )
            )
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['eval_episode_return'].item() - final_eval_reward.item()) / \
                abs(timestep.info['eval_episode_return'].item()) < 1e-5
            break


def test_ding_trading_qlib_csi500():
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    set_pkg_seed(1234, use_cuda=False)
    env = DingTradingEnv(EasyDict(qlib_config))
    env.seed(1234)
    env.reset()
    action_dim = env.action_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        assert timestep.obs.shape[0] == 500 * 6
        final_eval_reward += timestep.reward
        #print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['eval_episode_return'], type(timestep.info['eval_episode_return']), final_eval_reward,
                    type(final_eval_reward)
                )
            )
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['eval_episode_return'].item() - final_eval_reward.item()) / \
                abs(timestep.info['eval_episode_return'].item()) < 1e-5
            break


@pytest.mark.parametrize("alpha", ['518', '158', '360'])
def test_ding_trading_alpha_csi500(alpha):
    qlib.init(provider_uri=get_data_path(), region="cn")
    set_pkg_seed(1234, use_cuda=False)
    config = EasyDict(alpha_config)
    config.data_handler.type = 'alpha' + alpha
    env = DingTradingEnv(config)
    env.seed(1234)
    env.reset()
    action_dim = env.action_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        assert timestep.obs.shape[0] == 3 * int(alpha)
        final_eval_reward += timestep.reward
        #print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['eval_episode_return'], type(timestep.info['eval_episode_return']), final_eval_reward,
                    type(final_eval_reward)
                )
            )
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['eval_episode_return'].item() - final_eval_reward.item()) / \
                   abs(timestep.info['eval_episode_return'].item()) < 1e-5
            break


def test_ding_trading_guotai():
    qlib.init()
    set_pkg_seed(1234, use_cuda=False)
    env = DingTradingEnv(EasyDict(guotai_config))
    env.seed(1234)
    obs = env.reset()
    action_dim = env.action_space.shape
    final_eval_reward = np.array([0.], dtype=np.float32)

    while True:
        action = np.random.random(size=action_dim)
        timestep = env.step(action)
        assert timestep.obs.shape[0] == 50 * 10
        final_eval_reward += timestep.reward
        #print("{}(dtype: {})".format(timestep.reward, timestep.reward.dtype))
        if timestep.done:
            print(
                "{}({}), {}({})".format(
                    timestep.info['eval_episode_return'], type(timestep.info['eval_episode_return']), final_eval_reward,
                    type(final_eval_reward)
                )
            )
            # timestep.reward and the cumulative reward in wrapper FinalEvalReward are not the same.
            assert abs(timestep.info['eval_episode_return'].item() - final_eval_reward.item()) / \
                   abs(timestep.info['eval_episode_return'].item()) < 1e-5
            break