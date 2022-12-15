import pickle
import pandas as pd

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from alphaminer.gpmining.mining_utils import draw_alpha_graph

model_path = 'model_debug.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

best_alphas = model._best_programs
best_alpha_dict = {}
for i, a in enumerate(best_alphas):
    alpha_name = 'alpha_' + str(i + 1)
    best_alpha_dict[alpha_name] = {'fitness': a.fitness_, 'expression': a, 'depth': a.depth_, 'length': a.length_}

best_alpha_data = pd.DataFrame(best_alpha_dict).T.sort_values(by='fitness', ascending=False)
print(best_alpha_data)
best_alpha_name = best_alpha_data.index[0]
best_alpha = best_alpha_dict[best_alpha_name]['expression']

graph = draw_alpha_graph(best_alpha)
