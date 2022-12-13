import os
import qlib
import argparse
import logging
from tensorboardX import SummaryWriter
from easydict import EasyDict
from torch.nn.parallel import DataParallel

from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import DDPGPolicy
from ding.utils import deep_merge_dicts, set_pkg_seed
from alphaminer.rl.ding_env import DingTradingEnv, DingMATradingEnv
from alphaminer.rl.model.maddpg import MAQACv1, MAQACv2
from ding.worker import InteractionSerialEvaluator, BaseLearner, SampleSerialCollector, NaiveReplayBuffer


market = 'csi500'
train_start_time = '2010-01-01'
train_end_time = '2016-12-31'
eval_start_time = '2017-07-01'
eval_end_time = '2018-12-31'

cash = 1000000
stop_value = 1


main_config = dict(
    policy=dict(),
    env=dict(
        n_evaluator_episode=8,
        stop_value=stop_value,
        manager=dict(
            shared_memory=False,
        )
    )
)
main_config = EasyDict(main_config)


def get_env_config(args, market, start_time, end_time, env_cls=DingTradingEnv):
    env_config = dict(
        max_episode_steps=args.max_episode_steps,
        cash=cash,
        market=market,
        start_date=start_time,
        end_date=end_time,
        random_sample=(args.env_type == 'sample'),
        strategy=dict(
            buy_top_n=args.top_n,
        ),
        data_handler=dict(
            type=None,
            market=market,
            start_time=start_time,
            end_time=end_time,
        ),
    )
    env_config = EasyDict(env_config)
    env_config.data_handler.type = {
        'basic': None,
        '158': 'alpha158',
        'sample': 'alpha158',
    }[args.env_type]

    env_config = deep_merge_dicts(env_cls.default_config(), env_config)
    return env_config


def get_policy_config(args, policy_cls, collector_cls, evaluator_cls, learner_cls, buffer_cls=None):
    ddpg_config = dict(
        cuda=True,
        action_space='continuous',
        learn=dict(
            multi_gpu=False,
            learning_rate_actor=args.learning_rate,
            learning_rate_critic=args.learning_rate,
            update_per_collect=args.collect_env_num * args.max_episode_steps * 4 // args.batch_size,
            batch_size=args.batch_size,
            discount_factor=0.999,
            ignore_done=True,
            learner=dict(
                save_ckpt_after_iter=100000,
            )
        ),
        collect=dict(
            n_sample=args.collect_env_num * args.max_episode_steps,
            collector=dict(),
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=5000,
            )
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=200000,
                periodic_thruput_seconds=300,
            )
        ),
        model=dict(
            agent_obs_shape={
                'basic': 6,
                '158': 158,
                'sample': 158,
            }[args.env_type],
            global_obs_shape={
                'basic': 500*6,
                '158': 500*158,
                'sample': 50*158,
            }[args.env_type],
            action_shape=1,
            agent_num={
                'basic': 500,
                '158': 500,
                'sample': 50,
            }[args.env_type],
        )
    )

    policy_config = EasyDict(ddpg_config)

    policy_config = deep_merge_dicts(policy_cls.default_config(), policy_config)
    policy_config.collect.collector = deep_merge_dicts(collector_cls.default_config(), policy_config.collect.collector)
    policy_config.eval.evaluator = deep_merge_dicts(evaluator_cls.default_config(), policy_config.eval.evaluator)
    policy_config.learn.learner = deep_merge_dicts(learner_cls.default_config(), policy_config.learn.learner)
    if buffer_cls is not None:
        policy_config.other.replay_buffer = deep_merge_dicts(buffer_cls.default_config(), policy_config.other.replay_buffer)

    return policy_config


def main(cfg, args):
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    set_pkg_seed(args.seed)

    collect_env_cfg = get_env_config(args, market, train_start_time, train_end_time, env_cls=DingMATradingEnv)
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    policy_cfg = get_policy_config(args, DDPGPolicy, SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer)
    policy_cfg.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    policy_cfg.eval.evaluator.stop_value = cfg.env.stop_value
    model = MAQACv1(**policy_cfg.model)
    policy = DDPGPolicy(policy_cfg, model=model)
    policy._model = DataParallel(policy._model)
    policy._learn_model._model = policy._model
    policy._target_model._model = policy._model

    # args.max_episode_steps = -1
    # args.env_type = '158'
    # args.top_n = 20
    eval_env_cfg = get_env_config(args, market, eval_start_time, eval_end_time, env_cls=DingMATradingEnv)

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

    collector = SampleSerialCollector(
        policy_cfg.collect.collector, collector_env, policy.collect_mode, tb_logger, args.exp_name)

    evaluator = InteractionSerialEvaluator(
        policy_cfg.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=args.exp_name
    )
    buffer = NaiveReplayBuffer(policy_cfg.other.replay_buffer, tb_logger, exp_name=args.exp_name)

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(policy_cfg.learn.update_per_collect):
        # Learner will train ``update_per_collect`` times in one iteration.
            train_data = buffer.sample(policy_cfg.learn.batch_size, learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            learner.train(train_data, collector.envstep)
        if learner.policy.get_attribute('priority'):
            buffer.update(learner.priority_info)

    learner.call_hook('after_run')
    collector_env.close()
    evaluator_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trading rl train')
    parser.add_argument('-e', '--env-type', choices=['basic', '158', 'sample'], default='basic')
    parser.add_argument('-t', '--top-n', type=int, default=20,)
    parser.add_argument('-ms', '--max-episode-steps', type=int, default=40)
    parser.add_argument('-cn', '--collect-env-num', type=int, default=1)
    parser.add_argument('-en', '--evaluate-env-num', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4)
    parser.add_argument('--exp-name', type=str, default=None)
    args = parser.parse_args()
    main(main_config, args)
