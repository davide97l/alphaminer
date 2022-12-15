from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np
import pandas as pd


def r2(x, y):
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    preds = reg.predict(x)
    r2_ = r2_score(y, preds)
    return r2_


def tail_ratio(r):
    r = r.values
    p95 = np.percentile(r, 95)
    p5 = np.percentile(r, 5)
    return p95 / p5


def gain_to_pain_ratio(r):
    r = r.groupby(pd.Grouper(freq='M')).mean().values
    r_pos = sum(r[r > 0])
    r_neg = sum(r[r < 0])
    return r_pos / (-r_neg)


def common_sense_ratio(r):
    return tail_ratio(r) * gain_to_pain_ratio(r)


def beta(r, b):
    cov = np.cov(r.values, b.values)[0][1]
    b_std = b.std()
    return cov / b_std


def ann_cum_ret(ret, years: int):
    return np.power(ret + 1, 1 / years) - 1
