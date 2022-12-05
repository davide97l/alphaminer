from easydict import EasyDict

infer_processors = [
    {"class": "Fillna", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
]
start_time = '2022-01-01'
end_time = '2022-06-01'
market = 'csi500'

trading_ppo_config = dict(
    exp_name='trading_ppo_seed0',
    env=dict(
        env_id='Trading-v0',
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1.,  # stop when the policy doesn't lose money
        manager=dict(shared_memory=False, ),  # True doesn't work
        max_episode_steps=20,
        cash=1000000,
        start_date=start_time,
        end_date=end_time,
        market=market,
        strategy=dict(
            buy_top_n=10,
        ),
        data_handler=dict(
            market=market,
            start_time=start_time,
            end_time=end_time,
            alphas=None,
            fit_start_time=start_time,
            fit_end_time=end_time,
            infer_processors=infer_processors,
            learn_processors=[],
        )
        # The path to save the metrics replay
        # metrics_path='./trading_ppo_seed0/metrics',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        # load_path="./trading_ppo_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=500 * 6,  # 3 stocks x 8 features
            action_shape=500,  # 3 stocks
            action_space='continuous',
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
    env_manager=dict(type='base'),  # or 'base'
    policy=dict(
        type='ppo',
        import_names=['ding.policy.ppo'],
    ),
    replay_buffer=dict(type='naive', ),
)
trading_ppo_create_config = EasyDict(trading_ppo_create_config)
create_config = trading_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c test_ppo_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline_onpolicy
    from os import path as osp
    import qlib

    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
