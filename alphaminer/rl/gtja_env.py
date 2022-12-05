from .env import DataSource
from typing import Union, Dict, List
import pandas as pd
from os import path as osp


class GTJADataSource(DataSource):
    """
    Use datasource from gtja.
    """
    def __init__(self, start_date: Union[str, pd.Timestamp],
                 end_date: Union[str, pd.Timestamp], data_dir: str) -> None:
        self._data_dir = data_dir
        self._start_date = start_date
        self._end_date = end_date
        self._feasible = self._load_feasible()
        super().__init__(start_date=start_date,
                         end_date=end_date,
                         market="all",
                         data_handler=None)
        self._trading_data = self._postprocess_trading_data(self._trading_data)

    def _load_feasible(self) -> pd.DataFrame:
        feasible = pd.read_csv(osp.join(self._data_dir, "feasible.csv"),
                               index_col=0)
        feasible.index = pd.to_datetime(feasible.index)
        feasible = feasible[(feasible.index >= self._start_date)
                            & (feasible.index <= self._end_date)].copy()
        return feasible

    def _load_obs_data(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        factor_list = [
            "ACCA", "BBIC", "BIAS10", "BIAS5", "MA10Close",
            "MA10RegressCoeff6", "REVS5", "REVS5Indu1", "ROC6", "SwingIndex"
        ]
        factors = {}
        for factor in factor_list:
            df = pd.read_csv(osp.join(self._data_dir,
                                      "StockFactor/{}.csv".format(factor)),
                             index_col=0)
            df.index = pd.to_datetime(df.index)
            factors[factor] = df
        factors = pd.concat(factors)
        factors = factors.reset_index(level=0)

        # Reindex by date
        factor_dict = {}
        for date in self._dates:
            df = factors.loc[date]
            assert df.shape[0] == len(
                factor_list
            ), "Factors on date {} do not meet the shape of all factors".format(
                date)
            df = df.set_index("level_0").transpose()[factor_list]
            df.index = df.index.str.slice(
                0, 6)  # Only keep the number part of stock code.
            factor_dict[date] = df
        return factor_dict

    def _postprocess_trading_data(
        self, trading_data: Dict[pd.Timestamp, pd.DataFrame]
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Rename stock code of each dataframe. e.g. SH000006 -> 000006
        """
        for td in trading_data.values():
            td.index = td.index.str.slice(2, 8)
        return trading_data

    def _load_instruments(self) -> Dict[pd.Timestamp, List[str]]:
        instruments = {}
        for date in self._dates:
            codes = self._feasible.loc[date].dropna().index.tolist()
            instruments[date] = codes
        return instruments
