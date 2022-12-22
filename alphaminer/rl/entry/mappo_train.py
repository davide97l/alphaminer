import os
import qlib
import argparse
import logging
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import PPOPolicy

from ding.utils import deep_merge_dicts, set_pkg_seed
from alphaminer.rl.ding_env import DingTradingEnv, DingMATradingEnv
from alphaminer.rl.model.mappo import MAVACv1, MAVACv2
from alphaminer.rl.model.utils import DingDataParallel
from alphaminer.rl.entry.utils import default_exp_name
from ding.worker import InteractionSerialEvaluator, BaseLearner, EpisodeSerialCollector, SampleSerialCollector, NaiveReplayBuffer

market = 'csi500'
train_start_time = '2010-01-01'
train_end_time = '2016-12-31'
eval_start_time = '2017-07-01'
eval_end_time = '2018-12-31'

cash = 1000000
stop_value = 1

main_config = dict(
    policy=dict(), env=dict(n_evaluator_episode=8, stop_value=stop_value, manager=dict(shared_memory=False, ))
)
main_config = EasyDict(main_config)


def get_env_config(args, market, start_time, end_time, env_cls=DingTradingEnv):
    env_config = dict(
        max_episode_steps=args.max_episode_steps,
        cash=cash,
        market=market,
        start_date=start_time,
        end_date=end_time,
        random_sample=args.sample_size,
        # only used by env that don't use Qlib data
        data_path=args.data_path,  # 'guotai': '../../../data/guotai_factors/'
        strategy=dict(
            buy_top_n=args.top_n,
            slippage=args.slippage,
            commission=0 if args.no_cost else 0.0003,
            stamp_duty=0 if args.no_cost else 0.001,
        ),
        portfolio_optimizer=args.portfolio_optimizer,
        data_handler=dict(
            type=None,
            instruments=market,
            start_time=start_time,
            end_time=end_time,
        ),
        action_norm=args.action_norm,
    )
    env_config = EasyDict(env_config)
    env_config.data_handler.type = {
        'basic': None,
        '158': 'alpha158',
        '360': 'alpha360',
        '518': 'alpha518',
        'guotai': 'guotai',
    }[args.env_type]

    env_config = deep_merge_dicts(env_cls.default_config(), env_config)
    return env_config


def get_policy_config(args, policy_cls, collector_cls, evaluator_cls, learner_cls, buffer_cls=None):
    sample_size = args.sample_size
    ppo_config = dict(
        cuda=True,
        action_space='continuous',
        learn=dict(
            multi_gpu=False,
            learning_rate=args.learning_rate,
            epoch_per_collect=10,
            batch_size=args.batch_size,
            entropy_weight=0.001,
            ignore_done=True,
            learner=dict(save_ckpt_after_iter=100000, )
        ),
        collect=dict(n_episode=args.collect_env_num, discount_factor=0.999, collector=dict(get_train_sample=True, )),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        model=dict(
            agent_obs_shape={
                'basic': 6,
                '158': 158,
                '360': 360,
                '518': 518,
                'guotai': 10,
            }[args.env_type],
            global_obs_shape={
                'basic': 6 * (500 if not sample_size else sample_size),
                '158': 158 * (500 if not sample_size else sample_size),
                '360': 360 * (500 if not sample_size else sample_size),
                '518': 518 * (500 if not sample_size else sample_size),
                'guotai': 10 * (50 if not sample_size else sample_size),
            }[args.env_type],
            action_shape=1,
            agent_num={
                'basic': 500 if not sample_size else sample_size,
                '158': 500 if not sample_size else sample_size,
                '360': 500 if not sample_size else sample_size,
                '518': 500 if not sample_size else sample_size,
                'guotai': 50 if not sample_size else sample_size,
            }[args.env_type],
        )
    )

    policy_config = EasyDict(ppo_config)

    policy_config = deep_merge_dicts(policy_cls.default_config(), policy_config)
    policy_config.collect.collector = deep_merge_dicts(collector_cls.default_config(), policy_config.collect.collector)
    policy_config.eval.evaluator = deep_merge_dicts(evaluator_cls.default_config(), policy_config.eval.evaluator)
    policy_config.learn.learner = deep_merge_dicts(learner_cls.default_config(), policy_config.learn.learner)
    if buffer_cls is not None:
        policy_config.other.replay_buffer = deep_merge_dicts(
            buffer_cls.default_config(), policy_config.other.replay_buffer
        )

    return policy_config


def main(cfg, args):
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    set_pkg_seed(args.seed)

    collect_env_cfg = get_env_config(args, market, args.train_start_time, args.train_end_time, env_cls=DingMATradingEnv)
    eval_env_cfg = get_env_config(args, market, args.eval_start_time, args.eval_end_time, env_cls=DingMATradingEnv)
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    policy_cfg = get_policy_config(args, PPOPolicy, EpisodeSerialCollector, InteractionSerialEvaluator, BaseLearner)
    policy_cfg.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    policy_cfg.eval.evaluator.stop_value = cfg.env.stop_value
    model = MAVACv2(**policy_cfg.model)
    #_model = DingDataParallel(model)
    policy = PPOPolicy(policy_cfg, model=model)

    # args.max_episode_steps = -1
    # args.env_type = '158'
    # args.top_n = 20

    collect_env_temp = DingMATradingEnv(collect_env_cfg)
    evaluate_env_temp = DingMATradingEnv(eval_env_cfg)
    collector_env = SyncSubprocessEnvManager(
        env_fn=[lambda: collect_env_temp for _ in range(args.collect_env_num)],
        cfg=cfg.env.manager,
    )

    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: evaluate_env_temp for _ in range(args.evaluate_env_num)],
        cfg=cfg.env.manager,
    )
    collector_env.seed(args.seed)
    evaluator_env.seed(args.seed, dynamic_seed=False)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(args.exp_name), 'serial'))
    learner = BaseLearner(policy_cfg.learn.learner, policy.learn_mode, tb_logger, exp_name=args.exp_name)

    collector = EpisodeSerialCollector(
        policy_cfg.collect.collector, collector_env, policy.collect_mode, tb_logger, args.exp_name
    )

    evaluator = InteractionSerialEvaluator(
        policy_cfg.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=args.exp_name
    )

    learner.call_hook('before_run')

    reward_buffer = deque(maxlen=20)
    prev_best = -1e8

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if 'eval_episode_return' in reward[0]:
                reward_value = np.average([r['eval_episode_return'].numpy()[0] for r in reward])
            else:
                reward_value = np.average([r['final_eval_reward'].numpy()[0] for r in reward])
            reward_buffer.append(reward_value)
            avg_reward = np.average(reward_buffer)
            if avg_reward > prev_best:
                learner.save_checkpoint('avg_best_at_{}.pth.tar'.format(learner.train_iter))
                prev_best = avg_reward
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)

    learner.call_hook('after_run')
    collector_env.close()
    evaluator_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trading rl train')
    parser.add_argument('-e', '--env-type', choices=['basic', '158', '360', '518', 'guotai'], default='basic')
    parser.add_argument(
        '-t',
        '--top-n',
        type=int,
        default=20,
    )
    parser.add_argument('-ms', '--max-episode-steps', type=int, default=40)
    parser.add_argument('-cn', '--collect-env-num', type=int, default=1)
    parser.add_argument('-en', '--evaluate-env-num', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4)
    parser.add_argument('--train-start-time', type=str, default='2010-01-01')
    parser.add_argument('--train-end-time', type=str, default='2016-12-31')
    parser.add_argument('--eval-start-time', type=str, default='2017-01-01')
    parser.add_argument('--eval-end-time', type=str, default='2018-12-31')
    parser.add_argument('-ss', '--sample-size', type=int, default=None)
    parser.add_argument('-sl', '--slippage', type=float, default=0.00246)
    parser.add_argument('-nc', '--no-cost', action='store_true')
    parser.add_argument(
        '-an', '--action_norm', choices=['softmax', 'cut_softmax', 'uniform', 'gumbel_softmax', None], default=None
    )
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('-po', '--portfolio-optimizer', type=str, default="topk")
    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = default_exp_name('mappo', args)
    main(main_config, args)
