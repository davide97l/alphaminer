import pandas as pd
import numpy as np
from scipy.stats import rankdata, percentileofscore
from gplearn.functions import make_function, _Function
from joblib import wrap_non_picklable_objects


class RandomNFunctions():

    @staticmethod
    def _delay(data, d):
        value = pd.Series(data).shift(d).values
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_delta(data, d):
        value = data - RandomNFunctions._delay(data, d)
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_min(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).min())
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_max(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).max())
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_argmin(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).apply(np.argmin))
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_argmax(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).apply(np.argmax))
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_sum(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).sum())
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_mean(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).mean())
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_stddev(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).std())
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_prod(data, d):
        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).apply(lambda x: np.prod(x)))
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_rank(data, d):

        def _rolling_rank(data):
            return rankdata(data)[-1]

        value = np.array(pd.Series(data).rolling(d, min_periods=d // 2).apply(_rolling_rank).to_list())
        value = np.nan_to_num(value)
        return value

    # def _ts_rank(data, d):
    #     value = pd.Series(data).rolling(d, min_periods=d//2).apply(
    #         lambda x: percentileofscore(x, x[-1]) / 100
    #     )
    #     value = np.nan_to_num(value)
    #     return value

    # def _rank(data):
    #     value = np.array(pd.Series(data.flatten()).rank().tolist())
    #     value = np.nan_to_num(value)
    #     return value

    # def _scale(data):
    #     k = 1
    #     data = pd.Series(data.flatten())
    #     value = data.mul(1).div(np.abs(data).sum())
    #     value = np.nan_to_num(value)
    #     return value

    @staticmethod
    def _ts_corr(data1, data2, d):
        value = np.array(pd.Series(data1).rolling(d, min_periods=d // 2).corr(pd.Series(data2)))
        value = np.nan_to_num(value)
        return value

    @staticmethod
    def _ts_cov(data1, data2, d):
        value = np.array(pd.Series(data1).rolling(d, min_periods=d // 2).cov(pd.Series(data2)))
        value = np.nan_to_num(value)
        return value


class NInDataFunctions():

    @staticmethod
    def _delay(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = pd.Series(data.flatten()).shift(period)
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        # except:
        #     return np.zeros(data.shape)

    @staticmethod
    def _ts_delta(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            # try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                p = np.zeros_like(data)
                p[:] = period
                value = data - NInDataFunctions._delay(data, p)
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #   return np.zeros(data.shape)

    @staticmethod
    def _ts_sum(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(period).sum())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #   print('not right!')
        #   return np.zeros(data.shape)

    @staticmethod
    def _sma(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(period).mean())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #   return np.zeros(data.shape)

    @staticmethod
    def _ts_stddev(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(period).std())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_rank(data, n):

        def _rolling_rank(data):
            return rankdata(data)[-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(period).apply(_rolling_rank))
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_argmin(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data).rolling(period).apply(np.argmin) + 1)
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_argmax(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data).rolling(period).apply(np.argmax) + 1)
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_min(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data).rolling(period).min())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_max(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data).rolling(period).max())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_prod(data, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                value = np.array(pd.Series(data).rolling(period).apply(np.prod))
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape)
        #except:
        #    return np.zeros(data.shape)

    @staticmethod
    def _ts_corr(data1, data2, n):
        with np.errstate(divide='ignore', invalid='ignore'):
            #try:
            if n[0] > 0 and (n == n[0]).all():
                period = int(n[0])
                # x1 = pd.Series(data1.flatten())
                # x2 = pd.Series(data2.faltten())
                # df = pd.concat([x1, x2], axis=1)
                # tmp = pd.Series()
                # for i in range(len(df)):
                #     if i <= period - 2:
                #         tmp[str(i)] = np.nan
                #     else:
                #         df2 = df.iloc(i-period+1:i, :)
                #         tmp[str(i)] = df2.corr('spearman').iloc[1, 0]
                # return np.nan_to_num(tmp)
                value = pd.Series(data1).rolling(period, min_periods=period // 2).corr(pd.Series(data2))
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data1.shape)
        #except:
        #    return np.zeros(data1.shape)


def get_random_n_functions():
    delay = _Function(function=wrap_non_picklable_objects(RandomNFunctions._delay), name='delay', arity=1, is_ts=True)
    ts_delta = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_delta), name='ts_delta', arity=1, is_ts=True
    )
    ts_min = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_min), name='ts_min', arity=1, is_ts=True
    )
    ts_max = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_max), name='ts_max', arity=1, is_ts=True
    )
    ts_argmin = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_argmin), name='ts_argmin', arity=1, is_ts=True
    )
    ts_argmax = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_argmax), name='ts_argmax', arity=1, is_ts=True
    )
    ts_sum = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_sum), name='ts_sum', arity=1, is_ts=True
    )
    ts_mean = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_mean), name='ts_mean', arity=1, is_ts=True
    )
    ts_stddev = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_stddev), name='ts_stddev', arity=1, is_ts=True
    )
    ts_prod = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_prod), name='ts_prod', arity=1, is_ts=True
    )
    ts_rank = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_rank), name='ts_rank', arity=1, is_ts=True
    )
    ts_corr = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_corr), name='ts_corr', arity=2, is_ts=True
    )
    ts_cov = _Function(
        function=wrap_non_picklable_objects(RandomNFunctions._ts_cov), name='ts_cov', arity=2, is_ts=True
    )
    return [delay, ts_delta, ts_rank, ts_min, ts_max, ts_argmin, ts_argmax, ts_mean, ts_stddev, ts_sum, ts_corr, ts_cov]


random_n_functions = get_random_n_functions()


def get_n_in_data_functions():
    delay = make_function(function=NInDataFunctions._delay, name='delay', arity=2)
    ts_delta = make_function(function=NInDataFunctions._ts_delta, name='ts_delta', arity=2)
    sma = make_function(function=NInDataFunctions._sma, name='ts_mean', arity=2)
    ts_stddev = make_function(function=NInDataFunctions._ts_stddev, name='stddev', arity=2)
    ts_sum = make_function(function=NInDataFunctions._ts_sum, name='ts_sum', arity=2)
    ts_min = make_function(function=NInDataFunctions._ts_min, name='ts_min', arity=2)
    ts_max = make_function(function=NInDataFunctions._ts_max, name='ts_max', arity=2)
    ts_argmin = make_function(function=NInDataFunctions._ts_argmin, name='ts_argmin', arity=2)
    ts_argmax = make_function(function=NInDataFunctions._ts_argmax, name='ts_argmax', arity=2)
    ts_rank = make_function(function=NInDataFunctions._ts_rank, name='ts_rank', arity=2)
    ts_corr = make_function(function=NInDataFunctions._ts_corr, name='ts_corr', arity=3)
    ts_prod = make_function(function=NInDataFunctions._ts_prod, name='ts_prod', arity=2)
    return [delay, sma, ts_stddev, ts_sum, ts_delta, ts_min, ts_max, ts_argmin, ts_argmax, ts_corr, ts_rank]


n_in_data_functions = get_n_in_data_functions()
