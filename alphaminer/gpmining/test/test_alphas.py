import pandas as pd
import numpy as np
from alphaminer.gpmining.my_functions import RandomNFunctions as randF
from alphaminer.gpmining.my_functions import NInDataFunctions as nindF
from alphaminer.gpmining.joinquant_functions import JoinQuantFunction as jqF
import gplearn.functions as gpF
from qlib.data.dataset.loader import QlibDataLoader
import qlib
import pytest


@pytest.fixture(scope="function", autouse=True)
def init_tests():
    qlib.init()


def random_dataset(days: int = 252):
    # NOT USED
    date = pd.date_range(start='1/1/2010', periods=days)
    close = np.random.uniform(95, 105, size=days)
    open = np.random.uniform(95, 105, size=days)
    df = pd.DataFrame(
        {'open': open,
         'close': close,
         'date': date
         })
    df.set_index('date', inplace=True)
    return df


def qlib_dataset(config=None, instruments=None, start_time='20190101', end_time='20190630'):
    if instruments is None:
        instruments = ['sh000300']
    if config is None:
        config = ([["$close", "$open", "$high", "$low", "$volume"], ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']])
    qdl = QlibDataLoader(
        config=config)
    df = qdl.load(instruments=instruments, start_time=start_time, end_time=end_time)
    return df


@pytest.mark.unittest
class TestFunction:

    @pytest.mark.parametrize("d", [1, 5, 10])
    def test_randn_functions(self, d):
        df = qlib_dataset()
        configs = [
            [randF._delay(df['OPEN'].values, d), f'Ref($open, {d})'],  # Qlib can read values outside of the scope of the dataset if available
            [gpF.add2(df['OPEN'].values, df['HIGH'].values), 'Add($open, $high)'],
            [gpF.mul2(df['OPEN'].values, df['HIGH'].values), 'Mul($open, $high)'],
            [randF._ts_stddev(df['OPEN'].values, d), f'Std($open, {d})'],
            [randF._ts_sum(df['OPEN'].values, d), f'Sum($open, {d})'],
            [randF._ts_mean(df['OPEN'].values, d), f'Mean($open, {d})'],
            # [randF._ts_rank(df['OPEN'].values), f'Rank($open, {d})'],  # alpha_ql = alpha_gp * 4
            [randF._ts_argmax(df['OPEN'].values, d) + 1, f'IdxMax($open, {d})'],
            [randF._ts_sum(randF._ts_stddev(randF._ts_mean(gpF.add2(df['OPEN'].values, df['HIGH'].values), d), d), d), f'Sum(Std(Mean(Add($open, $high), {d}), {d}), {d})'],
            [randF._ts_sum(randF._delay(randF._ts_argmax(df['LOW'].values, d) + 1, d), d), f"Sum(Ref(IdxMax($low, {d}), {d}), {d})"],
        ]
        for config in configs:
            alpha_gp = config[0]
            alpha_ql = qlib_dataset(config=[[config[1]], ['ALPHA']])['ALPHA'].values
            alpha_ql = np.nan_to_num(alpha_ql)
            assert np.allclose(alpha_gp[d*4:], alpha_ql[d*4:])


    @pytest.mark.parametrize("d", [1, 5, 10])
    def test_nindata_functions(self, d):
        df = qlib_dataset()
        df[str(d)] = d
        configs = [
            [nindF._delay(df['OPEN'].values, df[str(d)]), f'Ref($open, {d})'],  # Qlib can read values outside of the scope of the dataset if available
            [gpF.add2(df['OPEN'].values, df['HIGH'].values), 'Add($open, $high)'],
            [gpF.mul2(df['OPEN'].values, df['HIGH'].values), 'Mul($open, $high)'],
            [nindF._ts_stddev(df['OPEN'].values, df[str(d)]), f'Std($open, {d})'],
            [nindF._ts_sum(df['OPEN'].values, df[str(d)]), f'Sum($open, {d})'],
            [nindF._sma(df['OPEN'].values, df[str(d)]), f'Mean($open, {d})'],
            # [randF._ts_rank(df['OPEN'].values), f'Rank($open, {d})'],  # alpha_ql = alpha_gp * 4
            [nindF._ts_argmax(df['OPEN'].values, df[str(d)]), f'IdxMax($open, {d})'],
            [nindF._ts_sum(nindF._ts_stddev(nindF._sma(gpF.add2(df['OPEN'].values, df['HIGH'].values), df[str(d)]), df[str(d)]), df[str(d)]), f'Sum(Std(Mean(Add($open, $high), {d}), {d}), {d})'],
            [nindF._ts_sum(nindF._delay(nindF._ts_argmax(df['LOW'].values, df[str(d)]), df[str(d)]), df[str(d)]), f"Sum(Ref(IdxMax($low, {d}), {d}), {d})"],
        ]
        for config in configs:
            alpha_gp = config[0]
            alpha_ql = qlib_dataset(config=[[config[1]], ['ALPHA']])['ALPHA'].values
            alpha_ql = np.nan_to_num(alpha_ql)
            assert np.allclose(alpha_gp[d*4:], alpha_ql[d*4:])
    
    @pytest.mark.parametrize("d", [1, 5, 10])
    def test_joinquant_functions(self, d):
        df = qlib_dataset()
        pairs = [
            [gpF.add2(df['OPEN'].values, df['HIGH'].values), jqF._add(df['OPEN'], df['HIGH']).values],
            [gpF.sub2(df['OPEN'].values, df['HIGH'].values), jqF._sub(df['OPEN'], df['HIGH']).values],
            [gpF.mul2(df['OPEN'].values, df['HIGH'].values), jqF._mul(df['OPEN'], df['HIGH']).values],
            [gpF.div2(df['OPEN'].values, df['HIGH'].values), jqF._div(df['OPEN'], df['HIGH']).values],
            [gpF.abs1(df['OPEN'].values), jqF._abs(df['OPEN']).values],
            [gpF.inv1(df['OPEN'].values), jqF._inv(df['OPEN']).values],
            [randF._delay(df['OPEN'].values, d), jqF._delay(df['OPEN'], d).values],
            [randF._ts_delta(df['OPEN'].values, d), jqF._ts_delta(df['OPEN'], d).values],
            [randF._ts_argmax(df['OPEN'].values, d), jqF._ts_argmax(df['OPEN'], d)],
            [randF._ts_argmin(df['OPEN'].values, d), jqF._ts_argmin(df['OPEN'], d)],
            [randF._ts_max(df['OPEN'].values, d), jqF._ts_max(df['OPEN'], d).values],
            [randF._ts_min(df['OPEN'].values, d), jqF._ts_min(df['OPEN'], d).values],
            [randF._ts_mean(df['OPEN'].values, d), jqF._ts_mean(df['OPEN'], d).values],
            [randF._ts_stddev(df['OPEN'].values, d), jqF._ts_stddev(df['OPEN'], d).values],
            [randF._ts_corr(df['OPEN'].values, df['HIGH'].values, d), jqF._ts_corr(df['OPEN'], df['HIGH'], d).values],
            [randF._ts_cov(df['OPEN'].values, df['HIGH'].values, d), jqF._ts_cov(df['OPEN'], df['HIGH'], d).values],
        ]
        for p in pairs:
            randf_res = p[0]
            jqf_res = p[1]
            assert np.allclose(randf_res[d:], jqf_res[d:])
