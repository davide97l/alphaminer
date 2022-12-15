import torch
import torch.nn as nn
from typing import Union, Dict, Optional

from ding.utils import SequenceType, squeeze
from ding.model.common import RegressionHead, FCEncoder

from .utils import ReducedRegressionHead


class MAQACv1(nn.Module):

    mode = ['compute_actor', 'compute_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        agent_num: int,
        twin_critic: bool = False,
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

        self.actor_head = RegressionHead(
            actor_head_hidden_size,
            action_shape,
            actor_head_layer_num,
            final_tanh=True,
            activation=activation,
            norm_type=norm_type
        )

        critic_input_size = global_obs_shape + action_shape * agent_num
        self.critic_encoder = FCEncoder(
            obs_shape=critic_input_size,
            hidden_size_list=critic_encoder_hidden_size_list,
            activation=activation,
            norm_type=norm_type
        )

        self.critic_head = RegressionHead(
            critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
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
        # for k in x.keys():
        #     x[k] = x[k].squeeze(-1)
        return {'action': x['pred']}

    def compute_critic(self, inputs: Dict) -> Dict:
        obs, action = inputs['obs']['global_state'], inputs['action']
        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=-1)
        x = self.critic_encoder(x)
        x = self.critic_head(x)['pred']
        return {'q_value': x}


class MAQACv2(nn.Module):

    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            agent_obs_shape: Union[int, SequenceType],
            global_obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            agent_num: int = 500,
            twin_critic: bool = False,
            encoder_hidden_size_list: SequenceType = [128, 64, 32],
            actor_head_hidden_size: int = 32,
            actor_head_layer_num: int = 1,
            critic_head_hidden_size: int = 64,
            critic_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
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
            (encoder_hidden_size_list[-1] + action_shape) * agent_num,
            critic_head_hidden_size,
            1,
            critic_head_layer_num,
            activation=activation,
            norm_type=norm_type
        )

        self.actor_head = RegressionHead(
            actor_head_hidden_size,
            action_shape,
            actor_head_layer_num,
            final_tanh=True,
            activation=activation,
            norm_type=norm_type
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
        # for k in x.keys():
        #     x[k] = x[k].squeeze(-1)
        return {'action': x['pred']}

    def compute_critic(self, inputs: Dict) -> Dict:
        obs, action = inputs['obs']['agent_state'], inputs['action']
        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = self.encoder(obs)
        action = self.actor_head(x)['pred']
        x = torch.cat([x.reshape(x.shape[0], -1), action], dim=-1)
        x = self.critic_head(x)['pred']
        return {'q_value': x}
