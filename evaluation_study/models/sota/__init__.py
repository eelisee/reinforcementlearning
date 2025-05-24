from .ars_wrapper import train as ars_train
from .a2c_wrapper import train as a2c_train
from .ddpg_wrapper import train as ddpg_train
from .ppo_wrapper import train as ppo_train
from .sac_wrapper import train as sac_train
from .td3_wrapper import train as td3_train
from .tqc_wrapper import train as tqc_train
from .trpo_wrapper import train as trpo_train

class AlgoWrapper:
    def __init__(self, train_fn):
        self.train = train_fn

ALL_SOTA_ALGOS = {
    "ars": AlgoWrapper(ars_train),
    "a2c": AlgoWrapper(a2c_train),
    "ddpg": AlgoWrapper(ddpg_train),
    "ppo": AlgoWrapper(ppo_train),
    "sac": AlgoWrapper(sac_train),
    "td3": AlgoWrapper(td3_train),
    "tqc": AlgoWrapper(tqc_train),
    "trpo": AlgoWrapper(trpo_train),
}