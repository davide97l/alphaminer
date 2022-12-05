import torch
import torch.nn as nn
from typing import Union, Dict, Optional

from ding.utils import SequenceType, squeeze
from ding.model.common import ReparameterizationHead, RegressionHead, FCEncoder
from ding.torch_utils import MLP


class MAVACv1(nn.Module):

    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            actor_encoder_hidden_size_list: SequenceType = [128, 128, 64],
            critic_encoder_hidden_size_list: SequenceType = [128, 128, 64],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            sigma_type: Optional[str] = 'independent',
            bound_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape

        self.actor_encoder = FCEncoder(
            obs_shape=agent_obs_shape,
            hidden_size_list=actor_encoder_hidden_size_list,
            activation=activation,
            norm_type=norm_type
        )

        self.critic_encoder = FCEncoder(
            obs_shape=global_obs_shape,
            hidden_size_list=critic_encoder_hidden_size_list,
            activation=activation,
            norm_type=norm_type
        )

        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )

        self.actor_head = ReparameterizationHead(
            actor_head_hidden_size,
            action_shape,
            actor_head_layer_num,
            sigma_type=sigma_type,
            activation=activation,
            norm_type=norm_type,
            bound_type=bound_type
        )
        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)
    
    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)
    
    def compute_actor(self, x: torch.Tensor) -> Dict:
        x = x['agent_state']
        x = self.actor_encoder(x)
        x = self.actor_head(x)
        for k in x.keys():
            x[k] = x[k].squeeze(-1)
        return {'logit': x}
    
    def compute_critic(self, x: Dict) -> Dict:
        x = self.critic_encoder(x['global_state'])
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: Dict) -> Dict:
        logit = self.compute_actor(x)['logit']
        value = self.compute_critic(x)['value']
        return {'logit': logit, 'value': value}


class MAVACv2(nn.Module):

    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            agent_num: int = 500,
            encoder_hidden_size_list: SequenceType = [128, 64, 32],
            actor_head_hidden_size: int = 32,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
            sigma_type: Optional[str] = 'independent',
            bound_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape

        self.encoder = FCEncoder(
            obs_shape=agent_obs_shape,
            hidden_size_list=encoder_hidden_size_list,
            activation=activation,
            norm_type=norm_type
        )

        self.critic_head = ReducedRegressionHead(
            encoder_hidden_size_list[-1] * agent_num, critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
        )

        self.actor_head = ReparameterizationHead(
            actor_head_hidden_size,
            action_shape,
            actor_head_layer_num,
            sigma_type=sigma_type,
            activation=activation,
            norm_type=norm_type,
            bound_type=bound_type
        )
        self.actor = [self.encoder, self.actor_head]
        self.critic = [self.encoder, self.critic_head]
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)
    
    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)
    
    def compute_actor(self, x: torch.Tensor) -> Dict:
        x = x['agent_state']
        x = self.encoder(x)
        x = self.actor_head(x)
        for k in x.keys():
            x[k] = x[k].squeeze(-1)
        return {'logit': x}
    
    def compute_critic(self, x: Dict) -> Dict:
        x = x['agent_state']
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: Dict) -> Dict:
        x = x['agent_state']
        x = self.encoder(x)
        x1 = self.actor_head(x)
        for k in x1.keys():
            x1[k] = x1[k].squeeze(-1)
        x2 = x.reshape(x.shape[0], -1)
        x2 = self.critic_head(x2)
        return {'logit': x1, 'value': x2['pred']}


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
