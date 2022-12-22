from typing import Union, Dict, Optional
from torch.nn.parallel import DataParallel
import torch
import torch.nn as nn
from itertools import chain

from ding.torch_utils import MLP


class DingDataParallel(DataParallel):

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.module.__getattr__(name)
    
    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        max_device_num = min(self._get_input_size(inputs), len(self.device_ids))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids[:max_device_num])
        if max_device_num == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    
    def _get_input_size(self, inputs):
        if isinstance(inputs, dict):
            return self._get_input_size(inputs[list(inputs.keys())[0]])
        elif isinstance(inputs, tuple):
            return self._get_input_size(inputs[0])
        elif isinstance(inputs, torch.Tensor):
            return inputs.shape[0]


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
