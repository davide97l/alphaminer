import numpy as np
import pandas as pd


class JoinQuantFunction():
    
    @staticmethod
    def _add(data1, data2):
        x1 = data1.values
        x2 = data2.values
        value = np.add(x1, x2)
        return pd.Series(value, index=data1.index)
    
    @staticmethod
    def _sub(data1, data2):
        x1 = data1.values
        x2 = data2.values
        value = np.subtract(x1, x2)
        return pd.Series(value, index=data1.index)
    
    @staticmethod
    def _mul(data1, data2):
        x1 = data1.values
        x2 = data2.values
        value = np.multiply(x1, x2)
        return pd.Series(value, index=data1.index)
    
    @staticmethod
    def _div(data1, data2):
        x1 = data1.values
        x2 = data2.values
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)
        return pd.Series(value, index=data1.index)
    
    @staticmethod
    def _abs(data):
        x1 = data.values
        value = np.abs(x1)
        return pd.Series(value, index=data.index)
    
    @staticmethod
    def _inv(data):
        x1 = data.values
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.where(np.abs(x1) > 0.001, 1. / x1, 0.)
        return pd.Series(value, index=data.index)
    
    @staticmethod
    def _delay(data, d):
        return data.shift(d).fillna(0)
    
    @staticmethod
    def _ts_delta(data, d):
        return (data - JoinQuantFunction._delay(data, d)).fillna(0)
    
    @staticmethod
    def _ts_min(data, d):
        return data.rolling(d, min_periods=d//2).min().fillna(0)
    
    @staticmethod
    def _ts_max(data, d):
        return data.rolling(d, min_periods=d//2).max().fillna(0)
    
    @staticmethod
    def _ts_argmin(data, d):
        return data.rolling(d, min_periods=d//2).apply(np.argmin).fillna(0)
    
    @staticmethod
    def _ts_argmax(data, d):
        return data.rolling(d, min_periods=d//2).apply(np.argmax).fillna(0)
    
    @staticmethod
    def _ts_mean(data, d):
        return data.rolling(d, min_periods=d//2).mean().fillna(0)
    
    @staticmethod
    def _ts_stddev(data, d):
        return data.rolling(d, min_periods=d//2).std().fillna(0)
    
    @staticmethod
    def _ts_corr(data1, data2, d):
        return data1.rolling(d, min_periods=d//2).corr(data2).fillna(0)
    
    @staticmethod
    def _ts_cov(data1, data2, d):
        return data1.rolling(d, min_periods=d//2).cov(data2).fillna(0)
