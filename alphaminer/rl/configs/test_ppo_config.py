from easydict import EasyDict

trading_ppo_config = dict(
    exp_name='trading_ppo_seed0',
    env=dict(
        env_id='Trading-v0',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=0.1,  # stop when the policy doesn't lose money
        manager=dict(shared_memory=False, ),  # True doesn't work
        max_episode_steps=5,
        cash=1000000,
        start_date='2020-01-01',
        end_date='2020-02-01',
        market='csi500',
        strategy=dict(
            buy_top_n=1,
        ),
        # The path to save the metrics replay
        # metrics_path='./trading_ppo_seed0/metrics',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        # load_path="./trading_ppo_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=18,  # 3 stocks x 8 features
            action_shape=3,  # 3 stocks
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
    env_manager=dict(type='subprocess'),  # or 'base'
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

    def get_data_path() -> str:
        dirname = osp.dirname(osp.realpath(__file__))
        return osp.realpath(osp.join(dirname, "../tests/data"))

    qlib.init(provider_uri=get_data_path(), region="cn")
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
