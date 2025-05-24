from concurrent.futures import ProcessPoolExecutor
from utils.config_loader import load_all_configs
from models.reinforce import reinforce
from models.reinforce import minibatch
from models.sota import ALL_SOTA_ALGOS

def run_experiment(config):
    algo = config["algorithm"]
    if algo == "reinforce":
        reinforce.train(config)
    elif algo == "minibatch":
        minibatch.train(config)
    elif algo in ALL_SOTA_ALGOS:
        ALL_SOTA_ALGOS[algo].train(config)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

def run_all_experiments():
    configs = load_all_configs()
    with ProcessPoolExecutor() as executor:
        executor.map(run_experiment, configs)