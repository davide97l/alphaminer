# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Text, Union
from qlib.data.dataset.weight import Reweighter
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class CAPModel(Model):
    """CAPModel"""

    def __init__(self):
        self.rf = 0
        self.alpha = {}
        self.beta = {}
        self.rm = None
        self.market_daily_return = None

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if df_train.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        if reweighter is not None:
            w: pd.Series = reweighter.reweight(df_train)
            w = w.values
        else:
            w = None
        if w is not None:
            raise NotImplementedError("TODO: support CAPM with weight")  # TODO
        df_label = df_train['label']['LABEL0']
        all_codes = list(df_label.index.get_level_values(1).unique())
        self.market_daily_return = df_label.groupby('datetime').mean().values
        self.rm = self.market_daily_return.mean() * 252  # this is the expected return of the market
        df_label = df_label.droplevel(0)
        for code in all_codes:
            df_code = df_label[df_label.index == code]
            diff = len(self.market_daily_return) - len(df_code)
            if diff > 0:  # TODO fill missing values before computing alpha and beta
                self.beta[code] = 1
                self.alpha[code] = 0
                continue
            df_code.fillna(0., inplace=True)
            b, a = np.polyfit(self.market_daily_return, df_code.values, 1)
            self.beta[code] = b
            self.alpha[code] = a
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.rm is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        predictions = []
        all_codes = list(x_test.index.get_level_values(1))
        for code in all_codes:
            if code not in self.beta.keys():  # TODO compute alpha and beta of new stocks
                p = 0.
            else:
                p = self.rf + (self.beta[code] * (self.rm - self.rf))
            predictions.append(p)
        return pd.Series(predictions, index=x_test.index)
