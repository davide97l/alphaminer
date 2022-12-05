import logging
from tqdm import tqdm
import numpy as np
from alphaminer.rl.env import TradingEnv
import pandas as pd
from typing import List, Optional, Union, Dict, Tuple, Any
import torch
import torch.nn.functional as F


class ReducedFeaturesEnv(TradingEnv):
    def __init__(self, model, load_path: str = None, flatten_obs: bool = False, len_index: int = None,
                 **kwargs):
        TradingEnv.__init__(self, **kwargs)
        self.model = model
        self.flatten_obs = flatten_obs
        self.len_index = len_index
        if load_path:
            self.model.load_state_dict(torch.load(load_path, map_location=torch.device(
                'cuda' if torch.cuda.is_available() else'cpu')))
        self.observation_space = np.array(
            self.compress_obs(self._ds.query_obs(
                date=self._ds.dates[0])).values.shape)  # type: ignore
        self.model.eval()
        self.obs_compressed = False

    def reset(self) -> pd.DataFrame:
        """
        Reset states and return the reset obs.
        """
        obs = super().reset()
        if not self.obs_compressed:
            reduced_obs = self.compress_obs(obs)
        else:
            reduced_obs = obs
        return reduced_obs

    def step(
        self, action: pd.Series
    ) -> Tuple[pd.DataFrame, float, bool, Dict[Any, Any]]:
        obs, reward, done, info = super().step(action)
        if not self.obs_compressed:
            reduced_obs = self.compress_obs(obs)
        else:
            reduced_obs = obs
        return reduced_obs, reward, done, info

    def compress_obs(self, obs: np.array):
        obs_index = obs.index
        tensor_obs = torch.Tensor(obs.values)
        if self.flatten_obs:
            assert self.len_index is not None
            if len(obs_index) < self.len_index:
                logging.warning('Incomplete index ({} < {}): zero padding missing data'.format(len(obs_index),
                                                                                               self.len_index))
                tensor_obs = F.pad(input=tensor_obs, pad=(0, 0, 0, self.len_index - len(obs_index)),
                                   mode='constant', value=0)
            elif len(obs_index) > self.len_index:
                logging.warning('Index too long ({} < {}): removing excess data'.format(len(obs_index), self.len_index))
                tensor_obs = tensor_obs[: self.len_index]
            tensor_obs = torch.reshape(tensor_obs, (-1,))
            assert tensor_obs.shape[0] % self.len_index == 0
        reduced_obs = self.model(tensor_obs)[2]
        reduced_obs = reduced_obs.detach().cpu().numpy()
        cols = ["feature_{}".format(i) for i in range(reduced_obs.shape[-1])]
        if self.flatten_obs:
            reduced_obs = reduced_obs.reshape((self.len_index, -1))
            cols = ["feature_{}".format(i) for i in range(reduced_obs.shape[-1])]
        obs = pd.DataFrame(data=reduced_obs, index=obs_index, columns=cols)
        return obs

    def compress_all_obs(self):
        logging.info('compressing all obs...')
        for date in tqdm(self._ds._obs_data.keys()):
            obs = self._ds.query_obs(date=date)
            obs = self.compress_obs(obs.fillna(0))
            self._ds._obs_data[date] = obs
        self.obs_compressed = True
