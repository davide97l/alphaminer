from alphaminer.data.handler import AlphaMinerHandler
from os import path as osp
import pandas as pd
import numpy as np
import qlib
from qlib.contrib.data.handler import Alpha158, Alpha360


def get_data_path() -> str:
    dirname = osp.dirname(osp.realpath(__file__))
    return osp.realpath(osp.join(dirname, "../../rl/tests/data"))


qlib.init(provider_uri=get_data_path(), region="cn")


def test_data_handler():
    alphas = [
        ['(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close',
         '$high - $low'],
        ['MACD', 'DELTA']
    ]
    dh = AlphaMinerHandler(
        start_time="2010-01-01",
        end_time="2020-01-01",
        alphas=alphas,
        fit_start_time="2010-01-01",
        fit_end_time="2020-01-01",
        infer_processors=[],
        learn_processors=[],
    )
    data = dh.fetch()
    assert 'MACD' in data.columns
    assert 'DELTA' in data.columns
    assert (data['$high'] - data['$low']).equals(data['DELTA'])

    infer_processors = [
        {"class": "Fillna", "kwargs": {}},
        {"class": "ZScoreNorm", "kwargs": {}},
    ]

    dh = AlphaMinerHandler(
        start_time="2010-01-01",
        end_time="2020-01-01",
        fit_start_time="2010-01-01",
        fit_end_time="2020-01-01",
        infer_processors=infer_processors,
        learn_processors=[],
    )
    data = dh.fetch()
    # check fillna is performed
    assert data.isna().sum().sum() == 0
    # check Z-score normalization is performed
    assert np.isclose([data['$close'].mean()], [0.], atol=1e-03)
    assert np.isclose([data['$close'].std()], [1.], atol=1e-03)


def test_alpha158():
    dh = Alpha158(
        start_time="2010-01-01",
        end_time="2011-01-01",
        fit_start_time="2010-01-01",
        fit_end_time="2011-01-01",
    )
    data = dh.fetch()
    assert len(data.columns) == 159  # 158 alphas + 1 label


def test_alpha360():
    dh = Alpha360(
        start_time="2008-01-01",
        end_time="2012-01-01",
        fit_start_time="2010-01-01",
        fit_end_time="2010-02-01",
    )
    data = dh.fetch()
    assert len(data.columns) == 361  # 360 alphas + 1 label
