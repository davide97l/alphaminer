import pickle
import qlib
import gplearn
import pandas as pd
from datetime import datetime

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from alphaminer.gpmining.my_functions import random_n_functions, n_in_data_functions
from alphaminer.gpmining.my_fitness import top10pct_return_fitness
from alphaminer.gpmining.mining_utils import draw_alpha_graph, draw_statistics, get_data


market = "csi500"
fields = ['$change', '$open', '$close', '$high', '$low']
return_shifts = [5]
start_time = "2014-01-04"
end_time = "2019-01-01"

generations = 4
population_size = 5000
hall_of_fame = 100
init_depth = (2, 6)
tournament_size = 500
const_rangetuple = (1, 10)
p_crossover = 0.4
p_subtree_mutation = 0.01
p_hoist_mutation = 0.01
p_point_mutation = 0.01
p_point_replace = 0.4
parsimony_coefficient = 0.0001
random_state = 2200
feature_names = ['CHANGE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']
default_function = ['add', 'sub', 'mul', 'div', 'abs', 'inv']
ts_function = []
d_list = [1, 2, 3, 5, 8, 10]
verbose = 2

exp_name = 'g{}_nindata_top10new_p{}_d{}_{}_t{}_v0'.format(generations, population_size, init_depth[0], init_depth[1], tournament_size)
cur_time = datetime.today().isoformat('_').split('.')[0].replace(':', '-')
exp_name = exp_name + '_' + cur_time
# exp_name = 'debug'


if __name__ == '__main__':
    qlib.init()  # must init before running all the other commands

    all_data = get_data(market, fields, start_time, end_time, return_shifts)
    label = all_data['return_5']
    train = all_data.drop(columns='return_5')

    # function_set = default_function
    # ts_function = random_n_functions
    # d_ls = d_list

    function_set = default_function + n_in_data_functions
    ts_function = []
    d_ls = [1]
    for d in d_list:
        train[str(d)] = d
        feature_names.append(str(d))
    
    print(train)
    print(label)

    model = SymbolicTransformer(
        population_size=population_size,
        hall_of_fame=hall_of_fame,
        generations=generations,
        tournament_size=tournament_size,
        function_set=function_set,
        ts_function_set=ts_function,
        d_list=d_ls,
        const_range=const_rangetuple,
        init_depth=init_depth,
        stopping_criteria=1e8,
        p_crossover=p_crossover,
        p_hoist_mutation=p_hoist_mutation,
        p_point_mutation=p_point_mutation,
        p_point_replace=p_point_replace,
        p_subtree_mutation=p_subtree_mutation,
        feature_names=feature_names,
        parsimony_coefficient=parsimony_coefficient,
        random_state=random_state,
        n_jobs=196,
        #metric='spearman',
        metric=top10pct_return_fitness,
        verbose=verbose,
        warm_start=True,
    )
    # with open('model_g4_randn_top10_p5000_d2_6_t500_v0_2022-10-26_19-42-54.pkl.bak', 'rb') as f:
    #     model = pickle.load(f)
    # model.stopping_criteria = 1e8
    # model.warm_start = True
    model.fit(train, label)
    
    gp_statistics = pd.DataFrame(model.run_details_)

    print(gp_statistics)
    with open('model_{}.pkl'.format(exp_name), 'wb') as f:
        pickle.dump(model, f)

    draw_statistics(gp_statistics, exp_name)

    best_alphas = model._best_programs
    best_alpha_dict = {}
    for i, a in enumerate(best_alphas):
        alpha_name = 'alpha_' + str(i + 1)
        best_alpha_dict[alpha_name] = {'fitness': a.fitness_, 'expression': a, 'depth': a.depth_, 'length': a.length_}
    
    best_alpha_data = pd.DataFrame(best_alpha_dict).T.sort_values(by='fitness', ascending=False)
    print(best_alpha_data)
    best_alpha_name = best_alpha_data.index[0]
    best_alpha = best_alpha_dict[best_alpha_name]['expression']

    graph = draw_alpha_graph(best_alpha, exp_name)
