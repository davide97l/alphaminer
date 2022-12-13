import gym
import pandas as pd
import numpy as np
import logging
import os
from os import path as osp
from typing import List, Optional, Union, Dict, Tuple, Any
from qlib.data import D
from qlib.data.dataset import DataHandler
from time import time
from datetime import datetime
from abc import ABC, abstractmethod


class DataSource:
    """
    A proxy between qlib and alphaminer rl framework.
    Please first download data from https://github.com/chenditc/investment_data,
    and init qlib by `qlib.init` before use this package.
    For fundamental data, use PIT data https://github.com/microsoft/qlib/tree/main/scripts/data_collector/pit/.
    """
    def __init__(self, start_date: Union[str, pd.Timestamp],
                 end_date: Union[str, pd.Timestamp], market: str,
                 data_handler: DataHandler) -> None:
        self._label_in_obs = False
        self._dates: List[pd.Timestamp] = D.calendar(
            start_time=start_date, end_time=end_date,
            freq='day').tolist()  # type: ignore
        self._market = market
        self._instruments = self._load_instruments()
        self._len_index = len(self.instruments(self._dates[0]))
        self._dh = data_handler
        self._obs_data = self._load_obs_data()
        self._trading_data = self._load_trading_data()
        self._benchmark_price = self._load_benchmark_price()

    def query_obs(self, date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """
        Get observations from data handler
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        instruments = self.instruments(date)
        data = self._obs_data[date]
        df = data[data.index.isin(instruments)]
        # Reindex and fill missing values with 0
        miss_indexes = set(instruments) - set(df.index)
        for miss_ind in miss_indexes:
            logging.warning("Code {} {} is missing in obs!".format(
                miss_ind, date))
        df = df.reindex(instruments).fillna(0)
        return df

    def query_trading_data(
            self,
            date: Union[str, pd.Timestamp],
            instruments: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Overview:
            Query trading data of the current day.
        Arguments:
            - date: the date.
            - instruments: the code in the query, if the parameter is None,
                will use the constituent stocks of the date.
            - fields: fields in list.
            - ffill: whether ffill the data when feature is none (useful in calculate nav).
        Example:
                           close      open    factor   close_1  suspended
            instrument
            SH600006    1.557522  1.587765  0.504052  1.582596      False
            SH600021    1.169540  1.220501  0.254802  1.205460      False
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        if instruments is None:
            instruments = self.instruments(date)
        if len(instruments) == 0:
            df = pd.DataFrame(columns=self._trading_columns)
        else:
            data = self._trading_data[date]
            df = data[data.index.get_level_values(0).isin(instruments)]
        return df

    def _load_trading_data(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Load data that is necessary for trading.
        """
        start = time()
        feature_map = {
            "$close": "close",
            "$open": "open",
            "$factor": "factor",
            "Ref($close,1)": "prev_close",
            "$close/$factor": "real_price",
        }
        codes = D.list_instruments(D.instruments(self._market),
                                   start_time=self._dates[0],
                                   end_time=self._dates[-1],
                                   as_list=True)
        # Need all the prev data to avoid suspended code in the market during the time window
        df = D.features(codes,
                        list(feature_map.keys()),
                        freq="day",
                        end_time=self._dates[-1])
        df.rename(feature_map, axis=1, inplace=True)

        # Filter by chosen dates
        def processing_each_stock(df):
            code = df.index.get_level_values(0).unique()[0]
            df = df.loc[code]
            complete_dates = list(set(self._dates + list(df.index)))
            complete_dates.sort()
            # Append missing dates
            df = df.reindex(complete_dates)
            # If close is not in the dataframe, we think it is suspended or
            df["suspended"] = df["close"].isnull()
            df.fillna(method="ffill", inplace=True)
            # Trim into selected dates
            df = df[(df.index >= self._dates[0])
                    & (df.index <= self._dates[-1])]
            return df

        df = df.groupby(
            df.index.get_level_values(0)).apply(processing_each_stock)
        # Reindex by date
        data = {}
        for date in df.index.get_level_values(1).unique():
            data[date] = df[df.index.get_level_values(1) == date].reset_index(
                level=1, drop=True)
        logging.warning(
            "Time cost: {:.4f}s | Init trading data Done".format(time() -
                                                                 start))
        self._trading_columns = data[list(data.keys())[0]].columns
        return data

    def _load_benchmark_price(self) -> pd.DataFrame:
        benchmark_map = {
            "csi500": "SH000905",
            "csi300": "SH000300",
            "all": "SH000300"
        }
        benchmark = benchmark_map[self._market]
        feature_map = {"$close": "close", "Ref($close,1)": "prev_close"}
        df = D.features([benchmark],
                        list(feature_map.keys()),
                        freq="day",
                        start_time=self._dates[0],
                        end_time=self._dates[-1])
        df.rename(feature_map, axis=1, inplace=True)
        df["log_change"] = np.log(df["close"] / df["prev_close"])
        df.reset_index(level=0, drop=True, inplace=True)
        return df

    def _load_obs_data(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        data = {}
        obs = self._dh.fetch()
        if not self._label_in_obs:
            labels = [col for col in obs.columns if 'LABEL' in col.upper()]
            obs = obs.drop(labels, axis=1)
        for date in obs.index.get_level_values(0).unique():
            data[date] = obs.loc[date]
        return data

    def _load_instruments(self) -> Dict[pd.Timestamp, List[str]]:
        start = time()
        instruments = {}
        for date in self._dates:
            codes = D.list_instruments(D.instruments(self._market),
                                       start_time=date,
                                       end_time=date,
                                       as_list=True)
            codes.sort()
            instruments[date] = codes
        logging.info(
            "Time cost: {:.4f}s | Load instruments Done".format(time() -
                                                                start))
        return instruments

    def query_benchmark(self, date: Union[str, pd.Timestamp]) -> pd.Series:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self._benchmark_price.loc[date]

    def instruments(self, date: pd.Timestamp) -> List[str]:
        """
        Overview:
            Get instruments in the index.
        Arguments:
            - date: the date.
        """
        return self._instruments[date]

    @property
    def dates(self) -> List[pd.Timestamp]:
        return self._dates

    def next_date(self, date: Union[str, pd.Timestamp]) -> pd.Timestamp:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self._dates[self._dates.index(date) + 1]

    def prev_date(self, date: Union[str, pd.Timestamp]) -> pd.Timestamp:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return self._dates[self._dates.index(date) - 1]


class Portfolio:
    """
    The portfolio contains positions and cash.
    """
    def __init__(self, cash: float = 1000000) -> None:
        self.cash = cash
        self.positions = pd.Series(dtype=np.float64)

    def nav(self, price: pd.Series) -> float:
        """
        Get nav of current portfolio.
        """
        assert not price.isnull().values.any(
        ), "Price contains null value when calculating nav, {}".format(price)

        miss_codes = set(self.positions.index) - set(price.index)
        if len(miss_codes) > 0:
            logging.warning(
                "Codes {} are missing in price when calculating nav.".format(
                    miss_codes))

        nav = (self.positions * price).sum() + self.cash
        return nav

    def __repr__(self) -> str:
        return "Cash: {:.2f}; Positions: {}".format(self.cash,
                                                    self.positions.to_dict())


class PortfolioOptimizer(ABC):
    """
    Portfolio optimizer
    """
    @abstractmethod
    def get_weight(self, action: pd.Series) -> pd.Series:
        """
        Get weighted buy stocks from action list.
        """
        raise NotImplementedError


class TopkOptimizer(PortfolioOptimizer):
    def __init__(self, topk: int, equal_weight: bool = True) -> None:
        super().__init__()
        self._topk = topk
        self._equal_weight = equal_weight

    def get_weight(self, action: pd.Series) -> pd.Series:
        action = action.sort_values(ascending=False)
        action = action[:self._topk]
        if self._equal_weight:
            action = pd.Series(1 / action.shape[0], index=action.index)
            return action
        else:
            action = action / action.sum()
            return action


PORTFOLIO_OPTIMISERS = {"Topk": TopkOptimizer}


class TradingPolicy:
    """
    A super naive policy which will buy the top 10 stocks
    in the action list, and sell the other stocks.
    This class will also focus on some trading rules like:
    - The cost of buy and sell.
    - The stop limit of buy and sell.
    - Round the number of shares to the board lot size.
    """
    def __init__(
            self,
            data_source: DataSource,
            buy_top_n: int = 50,
            use_benchmark: bool = True,
            portfolio_optimizer: Optional[PortfolioOptimizer] = None,
            slippage: Optional[int] = 0.00246,
            commission: Optional[int] = 0.0003,
            stamp_duty: Optional[int] = 0.001) -> None:
        self._ds = data_source
        self._buy_top_n = buy_top_n
        self._stamp_duty = stamp_duty  # Charged only when sold.
        self._commission = commission  # Charged at both side.
        self._slippage = slippage
        self._use_benchmark = use_benchmark  # Use excess income to calculate reward.
        if portfolio_optimizer is None:
            portfolio_optimizer = TopkOptimizer(self._buy_top_n,
                                                equal_weight=True)
        self._portfolio_optimizer = portfolio_optimizer

    def take_step(self, date: Union[str, pd.Timestamp], action: pd.Series,
                  portfolio: Portfolio) -> Tuple[Portfolio, float]:
        """
        Overview:
            Take step, update portfolio and get reward (the change of nav).
            The default policy is buy the top 10 stocks from the action and sell the others
            at the open price of the t+1 day, then calculate the change of nav by the close
            price of the t+1 day.
        Arguments:
            - date: the date to take action.
            - action: the action.
        Returns:
            - portfolio, log_change: the newest portfolio and returns.
        """
        buy_stocks_weight = self._portfolio_optimizer.get_weight(action)
        buy_stocks = buy_stocks_weight.index.tolist()
        prev_date = self._ds.prev_date(date)
        prev_price = self._ds.query_trading_data(
            prev_date, portfolio.positions.index.tolist())["close"]
        prev_nav = portfolio.nav(price=prev_price)  # type: ignore

        # Sell stocks
        for code in portfolio.positions.keys():
            if code in buy_stocks:
                continue
            self.order_target_value(date, code, 0, portfolio)  # type: ignore

        # Buy stocks
        if len(buy_stocks) > 0:
            open_price = self._ds.query_trading_data(
                date, portfolio.positions.index.tolist())["open"]
            current_nav = portfolio.nav(open_price)
            buy_value = buy_stocks_weight * current_nav
            for code, value in buy_value.iteritems():
                self.order_target_value(date, code, value,
                                        portfolio)  # type: ignore

        # Calculate reward
        future_price = self._ds.query_trading_data(
            date, portfolio.positions.index.tolist())["close"]
        nav = portfolio.nav(price=future_price)  # type: ignore
        log_change = np.log(nav / prev_nav)

        if self._use_benchmark:
            benchmark = self._ds.query_benchmark(date=date)
            benchmark_change = benchmark.loc["log_change"]
            log_change -= benchmark_change
        return portfolio, log_change

    def order_target_value(self, date: Union[str, pd.Timestamp], code: str,
                           value: float, portfolio: Portfolio) -> Portfolio:
        """
        Overview:
            Set an order into the market, will calculate the cost of trading.
        Arguments:
            - date: the date of order.
            - code: stock code.
            - value: value of cash.
            - hold: hold volume in current portfolio.
        Returns:
            - value, hold: change of cash and hold volume
        """
        # Sell or buy at the open price
        data = self._ds.query_trading_data(date, [code]).loc[code]
        open_price, factor, suspended = data.loc["open"], data.loc[
            "factor"], data.loc["suspended"]
        if suspended:
            return portfolio
        hold = portfolio.positions.loc[
            code] if code in portfolio.positions else 0
        # Trim volume by real open price, then adjust by factor
        volume = self._round_lot(code, value, open_price / factor) / factor
        # type: ignore
        if hold > volume:  # Sell
            if self._available_to_sell(date, code):
                portfolio.cash += open_price * (1 - self._slippage / 2) * (
                    hold - volume) * (1 - self._stamp_duty - self._commission
                                      )  # type: ignore
                if volume == 0:
                    if code in portfolio.positions:
                        portfolio.positions.drop(code, inplace=True)
                else:
                    portfolio.positions.loc[code] = volume
            else:
                logging.warning("Stock {} {} is not available to sell.".format(
                    code, date))
        else:  # Buy
            if self._available_to_buy(date, code):
                need_cash = open_price * (1 + self._slippage / 2) * (
                    volume - hold) * (1 + self._commission)  # type: ignore
                if need_cash > portfolio.cash:
                    logging.warning(
                        "Insufficient cash to buy stock {} {}, need {:.0f}, have {:.0f}"
                        .format(code, date, need_cash, portfolio.cash))
                    # Only buy a part of stock. In order to avoid the amount being negative, use floor to round up.
                    part_volume = self._round_lot(code,
                                                  portfolio.cash,
                                                  open_price / factor,
                                                  round_type="floor") / factor
                    volume = hold + part_volume
                portfolio.cash -= open_price * (1 + self._slippage / 2) * (
                    volume - hold) * (1 + self._commission)
                portfolio.positions.loc[code] = volume
            else:
                logging.warning("Stock {} {} is not available to buy.".format(
                    code, date))
        return portfolio

    def _available_to_buy(self, date: Union[str, pd.Timestamp],
                          code: str) -> bool:
        """
        Overview:
            Check if it is available to buy the stock.
            Possible reasons include suspension, non-trading days and others.
        """
        data = self._ds.query_trading_data(date, [code]).loc[code]
        open_price, suspended, prev_close = data.loc["open"], data.loc[
            "suspended"], data.loc["prev_close"]
        if suspended:
            return False
        if open_price / prev_close > (1 + self._stop_limit(code)):
            return False
        return True

    def _available_to_sell(self, date: Union[str, pd.Timestamp],
                           code: str) -> bool:
        data = self._ds.query_trading_data(date, [code]).loc[code]
        open_price, suspended, prev_close = data.loc["open"], data.loc[
            "suspended"], data.loc["prev_close"]
        if suspended:
            return False
        if open_price / prev_close < (1 - self._stop_limit(code)):
            return False
        return True

    def _round_lot(self,
                   code: str,
                   value: float,
                   real_price: float,
                   round_type: str = "round") -> int:
        """
        Overview:
            Round the volume by broad lot.
        Arguments:
            - code: stock code.
            - value: buy value.
            - real_price: real price of stock.
            - round_type: round or floor.
        -
        """
        if code[2:5] == "688":
            if round_type == "floor":
                volume = int(value // real_price)
            else:
                volume = round(value / real_price)
            if volume < 200:
                volume = 0
        else:
            if round_type == "floor":
                volume = int(value // (real_price * 100) * 100)
            else:
                volume = round(value / (real_price * 100)) * 100
        return volume

    def _stop_limit(self, code: str) -> float:
        if code[2:5] == "688" or code[2] == "3":
            return 0.195
        else:
            return 0.095


class TradingRecorder:
    def __init__(self,
                 data_source: DataSource,
                 dirname: str = "./records",
                 filename: str = None) -> None:
        self._dirname = dirname
        self._ds = data_source
        self.filename = filename
        self.reset()

    def record(self, date: pd.Timestamp, action: pd.Series,
               portfolio: Portfolio) -> None:
        self._records["date"].append(date)
        self._records["action"].append(action)
        self._records["cash"].append(portfolio.cash)
        self._records["position"].append(portfolio.positions.copy())
        price = self._ds.query_trading_data(
            date, portfolio.positions.index.tolist())["close"]
        self._records["nav"].append(portfolio.nav(price))

    def dump(self) -> None:
        """
        Overview:
            Dump the reconstructed data into csv or pickle object.
            In some case, Using pickle will be unavailable due to software version issues.
            By default the data will be saved as csv.
        """
        if not osp.exists(self._dirname):
            os.makedirs(self._dirname)

        data = self.get_df()
        if data is None:
            return
        if self.filename is None:
            self.filename = "trading_record_{}.csv".format(
                datetime.now().strftime("%y%m%d_%H%M%S"))
        file_path = osp.join(self._dirname, self.filename)
        data.to_csv(file_path)
        logging.info('Record dumped at {}'.format(file_path))

    @property
    def records(self) -> Optional[Dict[str, Any]]:
        """
        Reconstruct records into dict.
        """
        if len(self._records["date"]) == 0:
            return
        date = self._records["date"]
        # Action dataframe
        action = pd.concat(self._records["action"], axis=1,
                           keys=date).transpose()
        # Position dataframe
        position = pd.concat(self._records["position"], axis=1,
                             keys=date).transpose()

        # Nav dataframe
        nav = pd.DataFrame(self._records["nav"], index=date, columns=["nav"])
        # Cash dataframe
        cash = pd.DataFrame(self._records["cash"],
                            index=date,
                            columns=["cash"])

        # Join together
        data = {
            "date": date,
            "action": action,
            "position": position,
            "nav": nav,
            "cash": cash
        }

        return data

    def get_df(self) -> Optional[pd.DataFrame]:
        """
        Reconstruct records into dataframe.
        """
        if len(self._records["date"]) == 0:
            return
        date = self._records["date"]
        # Action dataframe
        action = pd.concat(self._records["action"], axis=1,
                           keys=date).transpose()
        col_map = dict(zip(action.columns, action.columns + "_A"))
        action.rename(col_map, axis=1, inplace=True)
        # Position dataframe
        position = pd.concat(self._records["position"], axis=1,
                             keys=date).transpose()
        col_map = dict(zip(position.columns, position.columns + "_P"))
        position.rename(col_map, axis=1, inplace=True)
        # Nav dataframe
        nav = pd.DataFrame(self._records["nav"], index=date, columns=["nav"])
        # Cash dataframe
        cash = pd.DataFrame(self._records["cash"],
                            index=date,
                            columns=["cash"])
        # Join together
        df = pd.concat([nav, cash, position, action], axis=1)
        return df

    def reset(self):
        self._records = {
            "date": [],
            "action": [],
            "cash": [],
            "position": [],
            "nav": []
        }


class TradingEnv(gym.Env):
    """
    Simulate all the information of the trading day.
    """
    def __init__(self,
                 data_source: DataSource,
                 trading_policy: TradingPolicy,
                 max_episode_steps: int = 20,
                 cash: float = 1000000,
                 recorder: Optional[TradingRecorder] = None) -> None:
        super().__init__()
        self._ds = data_source
        if max_episode_steps == -1:
            max_episode_steps = len(self._ds.dates) - 1
        self.max_episode_steps = max_episode_steps
        assert len(
            self._ds.dates
        ) > max_episode_steps, "Max episode step ({}) should be less than effective trading days ({}).".format(
            max_episode_steps, len(self._ds.dates))
        self._trading_policy = trading_policy
        self._cash = cash
        self.observation_space = np.array(
            self._ds.query_obs(
                date=self._ds.dates[0]).values.shape)  # type: ignore
        self.action_space = self.observation_space[0]  # number of instruments
        self.reward_range = (-np.inf, np.inf)
        self._recorder = recorder
        self._obs_index: List[str] = []
        self._reset()

    def step(
        self, action: pd.Series
    ) -> Tuple[pd.DataFrame, float, bool, Dict[Any, Any]]:
        next_date = self._ds.next_date(self._today)
        self._portfolio, reward = self._trading_policy.take_step(
            next_date, action=action, portfolio=self._portfolio)
        obs = self._ds.query_obs(date=next_date)
        obs = self._reindex_obs(obs)
        self._step += 1
        done = True if self._step >= self.max_episode_steps else False
        self._today = next_date
        if self._recorder:
            self._recorder.record(self._today, action, self._portfolio)
        return obs, reward, done, {}

    def reset(self) -> pd.DataFrame:
        """
        Reset states and return the reset obs.
        """
        self._reset()
        if self._recorder:
            self._recorder.dump()
            self._recorder.reset()
        obs = self._ds.query_obs(self._today)
        self._obs_index = obs.index.tolist()
        return obs

    def _reset(self) -> None:
        """
        Reset states.
        """
        self._today = np.random.choice(
            self._ds.dates[:-self.max_episode_steps])  # type: ignore
        self._step = 0
        self._portfolio = Portfolio(cash=self._cash)

    def close(self) -> None:
        pass

    def _reindex_obs(self, obs: pd.DataFrame) -> pd.DataFrame:
        """
        Keep the original order of stocks when an index changes it's constituent stocks.
        """
        new = obs.index.tolist()
        old = self._obs_index
        swap_in = set(new) - set(old)
        swap_out = set(old) - set(new)
        index = []
        for code in old:
            if code in swap_out:
                index.append(swap_in.pop())
            else:
                index.append(code)
        self._obs_index = index
        return obs.reindex(index)


class RandomSampleEnv(TradingEnv):
    """
    Only randomly sample a subset from the stock pool as a observation space for training.
    """
    def __init__(self, *args, n_sample: int = 50, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._n_sample = n_sample
        self.observation_space[0] = n_sample  # type: ignore
        self.action_space = n_sample

    def reset(self) -> pd.DataFrame:
        """
        Reset states and return the reset obs.
        """
        obs = super().reset()
        self._obs_index_mask = np.random.choice(range(len(self._obs_index)),
                                                size=self._n_sample,
                                                replace=False)
        self._obs_index = [self._obs_index[i] for i in self._obs_index_mask]
        obs = obs.loc[self._obs_index]
        return obs
