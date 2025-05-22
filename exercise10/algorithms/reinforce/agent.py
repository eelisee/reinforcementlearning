import numpy as np
from algorithms.common.abstract.agent import Agent
from algorithms.build import build_learner, build_action_selector
from omegaconf import DictConfig, OmegaConf
from algorithms.common.utils.common_utils import np2tensor
from algorithms.common.utils.logger import Logger

class REINFORCEAgent(Agent):
    def __init__(self, experiment_info: DictConfig, hyper_params: DictConfig, model_cfg: DictConfig):
        super().__init__(experiment_info, hyper_params, model_cfg)
        self._initialize()

    def _initialize(self):
        self.experiment_info.env.state_dim = self.env.observation_space.shape[0]
        self.experiment_info.env.action_dim = self.env.action_space.n

        self.learner = build_learner(self.experiment_info, self.hyper_params, self.model_cfg)
        self.action_selector = build_action_selector(self.experiment_info, self.use_cuda)

        if self.experiment_info.log_wandb:
            cfg = OmegaConf.create(dict(
                experiment_info=self.experiment_info,
                hyper_params=self.hyper_params,
                model=self.learner.model_cfg,
            ))
            self.logger = Logger(cfg)

    def train(self):
        step = 0
        while step < self.experiment_info.max_update_steps:
            states, actions, rewards = self._collect_trajectory()
            trajectory = self._preprocess_trajectory((states, actions, rewards))
            loss_info = self.learner.update_model([trajectory])
            step += 1

            if self.experiment_info.log_wandb:
                self.logger.write_log({"reinforce_loss": loss_info[0]})

    def _collect_trajectory(self):
        state = self.env.reset()
        done = False
        states, actions, rewards = [], [], []

        while not done:
            action = self.action_selector(self.learner.actor, state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        return np.array(states), np.array(actions), np.array(rewards)

    def _preprocess_trajectory(self, trajectory):
        states, actions, rewards = trajectory
        states = np2tensor(states, self.use_cuda)
        actions = np2tensor(actions.reshape(-1, 1), self.use_cuda).long()
        rewards = np2tensor(rewards.reshape(-1, 1), self.use_cuda)
        return (states, actions, rewards)