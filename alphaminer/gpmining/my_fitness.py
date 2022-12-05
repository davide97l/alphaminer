import pandas as pd
import numpy as np
from gplearn.fitness import make_fitness


# def _top10pct_return_fitness(y, y_pred, w):
#     x1 = pd.Series(y.flatten())
#     x2 = pd.Series(y_pred.flatten())
#     df = pd.concat([x1, x2], axis=1)
#     df.columns = ['y', 'y_pred']
#     df.sort_values(by='y_pred', ascending=True, inplace=True)
#     num = len(df) // 10

#     y_high = df['y'][-num:]
#     y_low = df['y'][:num]
#     value = y_high.mean() / y_low.mean()
#     return value


def _top10pct_return_fitness(y, y_pred, w):
    x1 = pd.Series(y.flatten() + 1)
    x2 = pd.Series(y_pred.flatten())
    df = pd.concat([x1, x2], axis=1)
    df.columns = ['y', 'y_pred']
    df.sort_values(by='y_pred', ascending=True, inplace=True)
    num = len(df) // 10

    y_high = df['y'][-num:]
    y_low = df['y'][:num]
    value = y_high.mean() - y_low.mean()
    return value


top10pct_return_fitness = make_fitness(function=_top10pct_return_fitness, greater_is_better=True)
