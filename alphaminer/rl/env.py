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

    def __init__(
            self, start_date: Union[str, pd.Timestamp], end_date: Union[str, pd.Timestamp], market: str,
            data_handler: DataHandler
    ) -> None:
        self._label_in_obs = False
        self._dates: List[pd.Timestamp] = D.calendar(
            start_time=start_date, end_time=end_date, freq='day'
        ).tolist()  # type: ignore
        self._market = market
        self._instruments = self._load_instruments()
        self._len_index = len(self.instruments(self._dates[0]))
        self._dh = data_handler
        self._obs_data = self._load_obs_data()
        self._trading_data = self._load_trading_data()

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
            logging.debug("Code {} {} is missing in obs!".format(miss_ind, date))
        df = df.reindex(instruments).fillna(0)
        return df

    def query_trading_data(
            self, date: Union[str, pd.Timestamp], instruments: Optional[List[str]] = None
    ) -> pd.DataFrame:
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
        codes = D.list_instruments(
            D.instruments(self._market), start_time=self._dates[0], end_time=self._dates[-1], as_list=True
        )
        # Need all the prev data to avoid suspended code in the market during the time window
        df = D.features(codes, list(feature_map.keys()), freq="day", end_time=self._dates[-1])
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
            df["change"] = df["close"] / df["prev_close"]  # Calculate change after ffill
            # Trim into selected dates
            df = df[(df.index >= self._dates[0]) & (df.index <= self._dates[-1])]
            return df

        df = df.groupby(df.index.get_level_values(0)).apply(processing_each_stock)
        # Reindex by date
        data = {}
        for date in df.index.get_level_values(1).unique():
            data[date] = df[df.index.get_level_values(1) == date].reset_index(level=1, drop=True)
        logging.info("Time cost: {:.4f}s | Init trading data Done".format(time() - start))
        self._trading_columns = data[list(data.keys())[0]].columns
        return data

    def load_benchmark_price(self, benchmark: str) -> pd.DataFrame:
        """
        Overview:
            Construct the index of benchmark price.
        Arguments:
            - benchmark: SH000905(csi500) or SH000300(csi300).
        """
        feature_map = {"$close": "close", "Ref($close,1)": "prev_close"}
        df = D.features(
            [benchmark], list(feature_map.keys()), freq="day", start_time=self._dates[0], end_time=self._dates[-1]
        )
        df.rename(feature_map, axis=1, inplace=True)
        df["log_change"] = np.log(df["close"] / df["prev_close"])
        df.reset_index(level=0, drop=True, inplace=True)
        self._benchmark_price = df
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
            codes = D.list_instruments(D.instruments(self._market), start_time=date, end_time=date, as_list=True)
            codes.sort()
            instruments[date] = codes
        logging.info("Time cost: {:.4f}s | Load instruments Done".format(time() - start))
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
        assert not price.isnull().values.any(), "Price contains null value when calculating nav, {}".format(price)

        miss_codes = set(self.positions.index) - set(price.index)
        if len(miss_codes) > 0:
            logging.debug("Codes {} are missing in price when calculating nav.".format(miss_codes))

        nav = (self.positions * price).sum() + self.cash
        return nav

    def __repr__(self) -> str:
        return "Cash: {:.2f}; Positions: {}".format(self.cash, self.positions.to_dict())


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


class WeightedOptimizer(PortfolioOptimizer):
    """
    Use action as weight directly.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_weight(self, action: pd.Series) -> pd.Series:
        return action / action.sum()


PORTFOLIO_OPTIMISERS = {"topk": TopkOptimizer, "weighted": WeightedOptimizer}


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
            portfolio_optimizer: Optional[PortfolioOptimizer] = None,
            slippage: float = 0.00246,
            commission: float = 0.0003,
            stamp_duty: float = 0.001
    ) -> None:
        """
        Overview:
            Use a trading policy the buy/sell stocks, and calculate the reward of
            actions after `take_step`.
        Arguments:
            - data_source: the datasource instance
            - buy_top_n: top n in the polcy
            - use_benchmark: use relative returns to calculate the reward
            - portfolio_optimizer: choose how to weight the portfolio
            - slippage: slippage rate
            - commission: commission charged by both side
            - stamp_duty: tax charged on sold
            - benchmark_index: the index used for benchmark, if none, use average returns of all stocks
        """
        self._ds = data_source
        self._buy_top_n = buy_top_n
        self._stamp_duty = stamp_duty  # Charged only when sold.
        self._commission = commission  # Charged at both side.
        self._slippage = slippage
        if portfolio_optimizer is None:
            portfolio_optimizer = TopkOptimizer(self._buy_top_n, equal_weight=True)
        self._portfolio_optimizer = portfolio_optimizer

    def take_step(self, date: Union[str, pd.Timestamp], action: pd.Series, portfolio: Portfolio) -> Portfolio:
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
        buy_stocks_weight = buy_stocks_weight[buy_stocks_weight > 0]
        buy_stocks = buy_stocks_weight.index.tolist()

        # Sell stocks
        for code in portfolio.positions.keys():
            if code in buy_stocks:
                continue
            self.order_target_value(date, code, 0, portfolio)  # type: ignore

        # Buy stocks
        if len(buy_stocks) > 0:
            open_price = self._ds.query_trading_data(date, portfolio.positions.index.tolist())["open"]
            current_nav = portfolio.nav(open_price)
            buy_value = buy_stocks_weight * current_nav
            for code, value in buy_value.items():
                self.order_target_value(date, code, value, portfolio)  # type: ignore

        return portfolio

    def order_target_value(
            self, date: Union[str, pd.Timestamp], code: str, value: float, portfolio: Portfolio
    ) -> Portfolio:
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
        open_price, factor, suspended = data.loc["open"], data.loc["factor"], data.loc["suspended"]
        if suspended:
            return portfolio
        hold = portfolio.positions.loc[code] if code in portfolio.positions else 0
        # Trim volume by real open price, then adjust by factor
        volume = self._round_lot(code, value, open_price / factor) / factor
        # type: ignore
        if hold > volume:  # Sell
            if self._available_to_sell(date, code):
                portfolio.cash += open_price * (1 - self._slippage / 2) * (hold - volume) * (
                    1 - self._stamp_duty - self._commission
                )  # type: ignore
                if volume == 0:
                    if code in portfolio.positions:
                        portfolio.positions.drop(code, inplace=True)
                else:
                    portfolio.positions.loc[code] = volume
            else:
                logging.debug("Stock {} {} is not available to sell.".format(code, date))
        else:  # Buy
            if self._available_to_buy(date, code):
                need_cash = open_price * (1 + self._slippage / 2) * (volume -
                                                                     hold) * (1 + self._commission)  # type: ignore
                if need_cash > portfolio.cash:
                    logging.debug(
                        "Insufficient cash to buy stock {} {}, need {:.0f}, have {:.0f}".format(
                            code, date, need_cash, portfolio.cash
                        )
                    )
                    # Only buy a part of stock. In order to avoid the amount being negative, use floor to round up.
                    part_volume = self._round_lot(
                        code, portfolio.cash, open_price / factor, round_type="floor"
                    ) / factor
                    volume = hold + part_volume
                portfolio.cash -= open_price * (1 + self._slippage / 2) * (volume - hold) * (1 + self._commission)
                if volume > 0:
                    portfolio.positions.loc[code] = volume
            else:
                logging.debug("Stock {} {} is not available to buy.".format(code, date))
        return portfolio

    def _available_to_buy(self, date: Union[str, pd.Timestamp], code: str) -> bool:
        """
        Overview:
            Check if it is available to buy the stock.
            Possible reasons include suspension, non-trading days and others.
        """
        data = self._ds.query_trading_data(date, [code]).loc[code]
        open_price, suspended, prev_close = data.loc["open"], data.loc["suspended"], data.loc["prev_close"]
        if suspended:
            return False
        if open_price / prev_close > (1 + self._stop_limit(code)):
            return False
        return True

    def _available_to_sell(self, date: Union[str, pd.Timestamp], code: str) -> bool:
        data = self._ds.query_trading_data(date, [code]).loc[code]
        open_price, suspended, prev_close = data.loc["open"], data.loc["suspended"], data.loc["prev_close"]
        if suspended:
            return False
        if open_price / prev_close < (1 - self._stop_limit(code)):
            return False
        return True

    def _round_lot(self, code: str, value: float, real_price: float, round_type: str = "round") -> int:
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

    def __init__(self, data_source: DataSource, dirname: str = "./records", filename: str = None) -> None:
        self._dirname = dirname
        self._ds = data_source
        self.filename = filename
        self.reset()

    def record(self, date: pd.Timestamp, action: pd.Series, portfolio: Portfolio) -> None:
        """
        Arguments:
            - date: date to take step
            - action: action before take step
            - portfolio: portfolio after take step
        """
        self._records["date"].append(date)
        self._records["action"].append(action)
        self._records["cash"].append(portfolio.cash)
        self._records["position"].append(portfolio.positions.copy())
        price = self._ds.query_trading_data(date, portfolio.positions.index.tolist())["close"]
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
            self.filename = "trading_record_{}.csv".format(datetime.now().strftime("%y%m%d_%H%M%S"))
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
        action = pd.concat(self._records["action"], axis=1, keys=date).transpose()
        # Position dataframe
        position = pd.concat(self._records["position"], axis=1, keys=date).transpose()

        # Nav dataframe
        nav = pd.Series(self._records["nav"], index=date)
        # Cash dataframe
        cash = pd.Series(self._records["cash"], index=date)

        # Join together
        data = {"date": date, "action": action, "position": position, "nav": nav, "cash": cash}

        return data

    def get_df(self) -> Optional[pd.DataFrame]:
        """
        Reconstruct records into dataframe.
        """
        if len(self._records["date"]) == 0:
            return
        date = self._records["date"]
        # Action dataframe
        action = pd.concat(self._records["action"], axis=1, keys=date).transpose()
        col_map = dict(zip(action.columns, action.columns + "_A"))
        action.rename(col_map, axis=1, inplace=True)
        # Position dataframe
        position = pd.concat(self._records["position"], axis=1, keys=date).transpose()
        col_map = dict(zip(position.columns, position.columns + "_P"))
        position.rename(col_map, axis=1, inplace=True)
        # Nav dataframe
        nav = pd.DataFrame(self._records["nav"], index=date, columns=["nav"])
        # Cash dataframe
        cash = pd.DataFrame(self._records["cash"], index=date, columns=["cash"])
        # Join together
        df = pd.concat([nav, cash, position, action], axis=1)
        return df

    def reset(self):
        self._records = {"date": [], "action": [], "cash": [], "position": [], "nav": []}


class TradingEnv(gym.Env):
    """
    Simulate all the information of the trading day.
    """

    def __init__(
            self,
            data_source: DataSource,
            trading_policy: TradingPolicy,
            max_episode_steps: int = 20,
            cash: float = 1000000,
            recorder: Optional[TradingRecorder] = None,
            use_benchmark: bool = True,
            benchmark_index: Optional[str] = None,
            done_reward: str = "default",
    ) -> None:
        """
        Overview:
            Trading env.
        Arguments:
            - data_source: the data source instance.
            - trading_policy: trading policy instance.
            - max_episode_steps: max steps to finish an episode.
            - cash: initial cash in portfolio.
            - recorder: the recorder instance.
            - use_benchmark: whether subtract the reward of the benchmark, default is True.
            - benchmark_index: whether use index price as benchmark, default is None.
            - done_reward: how to calculate the reward of the final step.
        """
        super().__init__()
        self.ds = data_source
        self.date_steps = self._get_date_steps()
        if max_episode_steps == -1:
            max_episode_steps = len(self.date_steps) - 1
        self.max_episode_steps = max_episode_steps
        assert len(
            self.date_steps
        ) > max_episode_steps, "Max episode step ({}) should be less than effective trading days ({}).".format(
            max_episode_steps, len(self.date_steps)
        )
        self._done_reward = done_reward
        self._reward_history = []
        self._trading_policy = trading_policy
        self._cash = cash
        obs, _ = self._query_obs(trading_date=self.date_steps[0])
        self.observation_space = np.array(obs.values.shape)  # type: ignore
        self.action_space: int = self.observation_space[0]  # number of instruments
        self.reward_range = (-np.inf, np.inf)
        self._recorder = recorder
        self.obs_index: List[str] = []
        self._use_benchmark = use_benchmark
        self._benchmark_index = benchmark_index
        if benchmark_index is not None:
            self.ds.load_benchmark_price(benchmark_index)
        self._reset()

    def step(self, action: pd.Series) -> Tuple[pd.DataFrame, float, bool, Dict[Any, Any]]:
        # Trading date is next date after the last day of obs
        trading_date = self.date_steps[self._step_idx + 1]
        prev_reward_date = self.ds.prev_date(trading_date)
        prev_nav = self._get_nav(prev_reward_date)
        self._portfolio = self._trading_policy.take_step(trading_date, action=action, portfolio=self._portfolio)
        # Reward date is the last date before next trading date, or the last date of current obs
        obs, reward_date = self._query_obs(trading_date=trading_date)
        obs = self._reindex_obs(obs)
        reward = self._get_reward(
            prev_reward_date=prev_reward_date, reward_date=reward_date, prev_nav=prev_nav, action=action
        )
        self._reward_history.append(reward)

        # Change date
        self._step += 1
        self._step_idx += 1
        self._today = self.date_steps[self._step_idx]  # Set today to trading date
        done = True if self._step >= self.max_episode_steps else False
        if done and self._done_reward == "sharpe":
            reward = self._get_sharpe_reward()
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
            self._recorder.record(self._today, pd.Series(), self._portfolio)
        obs, _ = self._query_obs(self._today)
        self.obs_index = obs.index.tolist()
        return obs

    def _reset(self) -> None:
        """
        Reset states.
        """
        self._step_idx = np.random.choice(len(self.date_steps[:-self.max_episode_steps]))
        self._today = self.date_steps[self._step_idx]
        self._step = 0
        self._portfolio = Portfolio(cash=self._cash)
        self._reward_history = []

    def close(self) -> None:
        pass

    def _reindex_obs(self, obs: pd.DataFrame) -> pd.DataFrame:
        """
        Keep the original order of stocks when an index changes it's constituent stocks.
        """
        new = obs.index.tolist()
        old = self.obs_index
        swap_in = set(new) - set(old)
        swap_out = set(old) - set(new)
        index = []
        if len(swap_in) < len(swap_out):
            logging.warning(
                "Not enough data fill in the missing index! Swap in: {}; Swap out: {}".format(swap_in, swap_out)
            )
        for code in old:
            if code not in swap_out or len(swap_in) == 0:
                index.append(code)
            else:
                index.append(swap_in.pop())
        self.obs_index = index
        return obs.reindex(index)

    def _get_nav(self, date: pd.Timestamp) -> float:
        portfolio = self._portfolio
        price = self.ds.query_trading_data(date, portfolio.positions.index.tolist())["close"]
        nav = portfolio.nav(price=price)
        return nav

    def _get_reward(
            self, prev_reward_date: pd.Timestamp, reward_date: pd.Timestamp, prev_nav: float, action: pd.Series
    ) -> float:
        """
        Overview:
            Reward is calculated after take step, so the current date should be the trading date.
            Nav will be calculated by the close price of the day before current trading date (prev_reward_date)
            divide the close price of the day before the next trading date (reward_date).
        """
        # Calculate reward
        nav = self._get_nav(reward_date)
        log_change = np.log(nav / prev_nav)

        if self._use_benchmark:
            if self._benchmark_index is None:
                # Use average log change of all the stocks in the universe
                prev_close = self.ds.query_trading_data(prev_reward_date, action.index.tolist())["close"]
                current_close = self.ds.query_trading_data(reward_date, action.index.tolist())["close"]
                benchmark_change = (current_close / prev_close).dropna()
                benchmark_change = benchmark_change.sum() / benchmark_change.shape[0]
                benchmark_change = np.log(benchmark_change)
            else:
                prev_close = self.ds.query_benchmark(date=prev_reward_date).loc["close"]
                current_close = self.ds.query_benchmark(date=reward_date).loc["close"]
                benchmark_change = np.log(current_close / prev_close)
            log_change -= benchmark_change
        return log_change

    def _get_sharpe_reward(self) -> float:
        """
        Overview:
            Calculate the sharpe ratio of reward history.
        """
        if len(self._reward_history) < 2:
            return 0
        return np.mean(self._reward_history) / np.std(self._reward_history)

    def _get_date_steps(self) -> List[pd.Timestamp]:
        """
        Overview:
            Get date of each step.
        """
        return self.ds.dates

    def _query_obs(self, trading_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Overview:
            Get obs of each step.
        """
        return self.ds.query_obs(date=trading_date), trading_date


class RandomSampleWrapper(gym.Wrapper):

    def __init__(self, env: TradingEnv, n_sample: int = 50, new_step_api: bool = False):
        """
        Only randomly sample a subset from the stock pool as a observation space for training.
        """
        super().__init__(env, new_step_api)
        self._n_sample = n_sample
        self.env = env
        self.env.observation_space[0] = n_sample  # type: ignore
        self.env.action_space = n_sample

    def reset(self) -> pd.DataFrame:
        """
        Reset states and return the reset obs.
        """
        obs = self.env.reset()
        obs_index_mask = np.random.choice(range(len(self.env.obs_index)), size=self._n_sample, replace=False)
        self.env.obs_index = [self.env.obs_index[i] for i in obs_index_mask]
        obs = obs.loc[self.env.obs_index]
        return obs


class WeeklyEnv(TradingEnv):

    def __init__(self, *args, weekday: int = 1, **kwargs) -> None:
        """
        Overview:
            Sample data in weeks.
        Arguments:
            - weekday: The nth trading day of the week (1-5). If weekday is 1, it means Monday most of the time.
                But if the market is closed on Monday, the first trading day after the market opening will be selected.
                If weekday exceeds the number of trading days in the week, use the last day as the trading date.
                The obs is a fixed five-day period before the trading day.
        """
        self._weekday = weekday
        super().__init__(*args, **kwargs)

    def _get_date_steps(self) -> List[pd.Timestamp]:
        """
        Overview:
            Get date of each step, one day a week.
        """
        dates = pd.DataFrame({"date": self.ds.dates}, index=self.ds.dates)
        dates["week"] = dates["date"].dt.strftime("%Y-%W")
        # Discard the first incomplete weeks.
        edge_week = dates["week"].unique()[0]
        if dates[dates["week"] == edge_week].shape[0] < 5:
            dates = dates[dates["week"] != edge_week]
        # Convert to series
        dates = dates["date"]
        date_steps = dates.groupby(pd.Grouper(freq="W")).head(self._weekday)
        date_steps = date_steps.groupby(pd.Grouper(freq="W")).tail(1)
        return date_steps.tolist()

    def _query_obs(self, trading_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Overview:
            Find the next reward date and query previous 5 day's obs.
        Arguments:
            - date: the next trading date which will not include in obs
        """
        step_idx = self.date_steps.index(trading_date)
        obs_dates = []
        if step_idx == len(self.date_steps) - 1:
            # If in the last trading date, return the finally 5 days of data source as obs.
            obs_dates = self.ds.dates[-5:]
        else:
            # Five days before next trading date
            date = self.date_steps[step_idx + 1]
            for _ in range(5):
                date = self.ds.prev_date(date)
                obs_dates.insert(0, date)
        reward_date = obs_dates[-1]
        obs_data = []
        for date in obs_dates:
            obs = self.ds.query_obs(date=date)
            obs_data.append(obs)
        obs_data = pd.concat(obs_data, axis=1).fillna(0)
        return obs_data, reward_date
