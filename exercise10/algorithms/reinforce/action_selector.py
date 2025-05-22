import numpy as np
from torch.distributions import Categorical

from algorithms.common.abstract.action_selector import ActionSelector
from algorithms.common.models.base import BaseModel
from algorithms.common.utils.common_utils import np2tensor

class REINFORCEDiscreteActionSelector(ActionSelector):
    """Action selector for REINFORCE with discrete action space"""

    def __init__(self, use_cuda: bool):
        super().__init__(use_cuda)

    def __call__(self, policy: BaseModel, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        state = np2tensor(state, self.use_cuda)
        dist = policy.forward(state)
        categorical = Categorical(dist)
        action = categorical.sample()
        return action.item()