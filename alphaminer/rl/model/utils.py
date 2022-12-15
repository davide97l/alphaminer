from typing import Union, Dict, Optional
from torch.nn.parallel import DataParallel
import torch
import torch.nn as nn

from ding.torch_utils import MLP


class DingDataParrallel(DataParallel):

    def __getattr__(self, key: str):
        return getattr(self.module, key)


class ReducedRegressionHead(nn.Module):
    """
        Overview:
            The ``RegressionHead`` used to output actions Q-value.
            Input is a (:obj:`torch.Tensor`) of shape ``(B, N)`` and returns a (:obj:`Dict`) containing \
            output ``pred``.
        Interfaces:
            ``__init__``, ``forward``.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            layer_num: int = 2,
            final_tanh: Optional[bool] = False,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(ReducedRegressionHead, self).__init__()
        self.main = MLP(input_size, hidden_size, hidden_size, layer_num, activation=activation, norm_type=norm_type)
        self.last = nn.Linear(hidden_size, output_size)  # for convenience of special initialization
        self.final_tanh = final_tanh
        if self.final_tanh:
            self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Dict:
        x = self.main(x)
        x = self.last(x)
        if self.final_tanh:
            x = self.tanh(x)
        if x.shape[-1] == 1 and len(x.shape) > 1:
            x = x.squeeze(-1)
        return {'pred': x}
