args_dict = {
    'batch_size': 'bs',
    'collect_env_num': 'cn',
    'env_type': 'env_',
    'evaluate_env_num': 'en',
    'max_episode_steps': 'ms',
    'no_cost': 'nc',
    'sample_size': 'ss',
    'seed': 's',
    'slippage': 'ss',
    'softmax': 'sm',
    'top_n': 't',
    'critic_size': 'cs'
}

exclude_keys = ['exp_name', 'learning_rate']


def default_exp_name(policy: str, args):
    args.exp_name = policy
    for k, v in vars(args).items():
        if isinstance(v, str) and k != 'env_type':  # don't add str args to avoid too long filename
            continue
        if v in [None, False] or k in exclude_keys:
            continue
        if isinstance(v, bool) and v:
            args.exp_name += '_{}'.format(args_dict[k])
        else:
            args.exp_name += '_{}{}'.format(args_dict[k], v)
    return args.exp_name.replace('.', '')
