import torch
from torch.distributions import Categorical

from algorithms.common.abstract.loss import Loss
from algorithms.common.models.base import BaseModel
from omegaconf import DictConfig

class REINFORCELoss(Loss):
    def __init__(self, hyper_params: DictConfig, use_cuda: bool):
        super().__init__(hyper_params, use_cuda)

    def __call__(self, networks: BaseModel, data: tuple) -> torch.Tensor:
        actor = networks
        states, actions, rewards = data

        # Compute discounted returns
        returns = torch.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.hyper_params.gamma * G
            returns[t] = G
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

        policy_dists = actor.forward(states)
        dist = Categorical(policy_dists)
        log_probs = dist.log_prob(actions.squeeze())

        loss = -(log_probs * returns).mean()
        return loss