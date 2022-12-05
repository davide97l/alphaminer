import os
import qlib
import argparse
import logging
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import PPOPolicy, DDPGPolicy
from ding.utils import deep_merge_dicts, set_pkg_seed
from alphaminer.rl.ding_env import DingTradingEnv, DingMATradingEnv
from alphaminer.rl.model.ma_model import MAVACv1, MAVACv2
from alphaminer.rl.entry.train import get_env_config, get_policy_config
from ding.worker import InteractionSerialEvaluator, BaseLearner, EpisodeSerialCollector, SampleSerialCollector, NaiveReplayBuffer


market = 'csi500'
train_start_time = '2010-01-01'
train_end_time = '2016-12-31'
eval_start_time = '2018-01-01'
eval_end_time = '2022-06-30'

cash = 1000000
stop_value = 1


main_config = dict(
    policy=dict(),
    env=dict(
        n_evaluator_episode=1,
        stop_value=1,
        manager=dict(
            shared_memory=False,
        ),
        recorder=dict(
            path="./records",
        ),
    )
)
main_config = EasyDict(main_config)


def main(cfg, args):
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region="cn")
    set_pkg_seed(args.seed)
    env_class = {
        'single': DingTradingEnv,
        'multi': DingMATradingEnv,
    }[args.action_type]

    eval_env_cfg = get_env_config(args, market, eval_start_time, eval_end_time, env_cls=env_class)
    eval_env_cfg.recorder = dict(
        path="./records",
    )
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    policy_class, model_class, collector_class, buffer_class = {
        'ppo': (PPOPolicy, None, EpisodeSerialCollector, None),
        'ddpg': (DDPGPolicy, None, SampleSerialCollector, NaiveReplayBuffer),
    }[args.policy]
    if args.action_type == 'multi':
        model_class = MAVACv2

    policy_cfg = get_policy_config(args, policy_class, collector_class, InteractionSerialEvaluator, BaseLearner, buffer_class)
    policy_cfg.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    policy_cfg.eval.evaluator.stop_value = cfg.env.stop_value
    policy_cfg.load_path = args.load_path
    if model_class is not None:
        model = model_class(**policy_cfg.model)
    else:
        model = None
    policy = policy_class(policy_cfg, model=model)

    evaluate_env_temp = env_class(eval_env_cfg)
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: evaluate_env_temp for _ in range(args.evaluate_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env.seed(args.seed, dynamic_seed=False)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(args.exp_name), 'serial'))

    evaluator = InteractionSerialEvaluator(
        policy_cfg.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=args.exp_name
    )

    evaluator.eval()
    evaluator_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trading rl train')
    parser.add_argument('-p', '--policy', choices=['ppo', 'ddpg'], default='ppo')
    parser.add_argument('-e', '--env-type', choices=['basic', '158'], default='basic')
    parser.add_argument('-a', '--action-type', choices=['single', 'multi'], default='single')
    parser.add_argument('-t', '--top-n', type=int, default=20,)  # remember to change this according to the loaded policy
    parser.add_argument('-cn', '--collect-env-num', type=int, default=1)
    parser.add_argument('-en', '--evaluate-env-num', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--exp-name', type=str, default='eval')
    parser.add_argument('--load_path', type=str, default=None)
    # examples load_path:
    # /mnt/lustre/liudavide/trading_exp/sh08/mapponew03_fixorder_158_top50_n8
    # lustre/share_data/chenruobing/trading/sh08/mapponew03_fixorder_158_top50_n8
    args = parser.parse_args()
    args.max_episode_steps = 0
    main(main_config, args)
