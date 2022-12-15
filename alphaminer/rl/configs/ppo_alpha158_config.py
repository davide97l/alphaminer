from easydict import EasyDict
from alphaminer.rl.ding_env import DingTradingEnv

train_start_time = '2008-01-01'
train_end_time = '2016-12-31'
test_start_time = '2017-01-01'
test_end_time = '2018-12-31'
market = 'csi500'
max_episode_steps = 20
cash = 1000000
buy_top_n = 50
num_env = 1  # >1 if using subrocess env manager, however this could cause a warning due to both qlib and ding using
# subprocesses


def make_env_config(start_time, end_time):
    env_cgf = dict(
        env_id='Trading-v0',
        collector_env_num=num_env,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1.,  # stop when the policy doesn't lose money
        manager=dict(shared_memory=False, ),  # True doesn't work
        max_episode_steps=max_episode_steps,
        cash=cash,
        start_date=start_time,
        end_date=end_time,
        market=market,
        strategy=dict(buy_top_n=buy_top_n, ),
        data_handler=dict(
            type='alpha158',
            instruments=market,
            start_time=start_time,
            end_time=end_time,
            fit_start_time=start_time,
            fit_end_time=end_time,
        ),
        action_softmax=True,
    )
    return EasyDict(env_cgf)


trading_ppo_config = dict(
    exp_name='trading_ppo_seed0',
    env=make_env_config(train_start_time, train_end_time),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        # load_path="./trading_ppo_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=500 * 158,
            action_shape=500,
            action_space='continuous',
            encoder_hidden_size_list=[2048, 256, 64],
        ),
        action_space='continuous',
        learn=dict(
            epoch_per_collect=10,
            batch_size=16,
            learning_rate=3e-3,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=128,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    )
)
trading_ppo_config = EasyDict(trading_ppo_config)
main_config = trading_ppo_config

trading_ppo_create_config = dict(
    env=dict(
        type='trading',
        import_names=['alphaminer.rl.ding_env'],
    ),
    env_manager=dict(type='base'),  # 'subprocess' or 'base'. 'subprocess' may generate warnings
    policy=dict(
        type='ppo',
        import_names=['ding.policy.ppo'],
    ),
    replay_buffer=dict(type='naive', ),
)
trading_ppo_create_config = EasyDict(trading_ppo_create_config)
create_config = trading_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c ppo_alpha158_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline_onpolicy
    import qlib

    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    env_setting = [
        DingTradingEnv, [make_env_config(train_start_time, train_end_time)],
        [make_env_config(test_start_time, test_end_time)]
    ]
    serial_pipeline_onpolicy((main_config, create_config), env_setting=env_setting, seed=0)
