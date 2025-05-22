import torch
from torch.optim import Adam

from algorithms.common.abstract.learner import Learner
from algorithms.build import build_model, build_loss
from omegaconf import DictConfig

class REINFORCELearner(Learner):
    def __init__(self, experiment_info, hyper_params, model_cfg: DictConfig):
        super().__init__(experiment_info, hyper_params, model_cfg)
        self._initialize()

    def _initialize(self):
        self.model_cfg.actor.params.model_cfg.state_dim = self.experiment_info.env.state_dim
        self.model_cfg.actor.params.model_cfg.action_dim = self.experiment_info.env.action_dim

        self.actor = build_model(self.model_cfg.actor, self.use_cuda)
        self.optimizer = Adam(self.actor.parameters(), lr=self.hyper_params.actor_learning_rate)
        self.loss_fn = build_loss(self.experiment_info.actor_loss, self.hyper_params, self.use_cuda)

    def update_model(self, trajectories):
        states, actions, rewards = trajectories[0]  # Single trajectory for REINFORCE
        loss = self.loss_fn(self.actor, (states, actions, rewards))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (loss.item(),)  # for logging

    def get_policy(self, to_cuda: bool):
        return self.actor.cuda() if to_cuda else self.actor.cpu()

    def save_params(self):
        torch.save(self.actor.state_dict(), self.ckpt_path + f"/reinforce.pt")