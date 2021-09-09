from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from megastep.demo.envs.search_and_rescue import SearchAndRescueBase
from rebar import recording, arrdict

SAVE_DIR = "recordings"
CONFIG_DIR = "../configs"
CONFIG_FILE = "baseconfig.yaml"

if __name__ == '__main__':
    configfile = Path(CONFIG_DIR) / CONFIG_FILE
    with open(configfile, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as yerror:
            print("Please provide valid config file (*.yaml)")
            print(yerror)

    seed = config['seed']
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = SearchAndRescueBase(config['ENV'], rng=rng)

    # number of processes to use for encoding. None -> 1/2 of CPUs
    N = None
    # length of recording. If None, record until first reset)
    length = 120
    # which environment to plot
    d = 0

    world = env.reset()

    steps = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    with recording.ParallelEncoder(env.plot_state, N=N) as encoder, \
            tqdm(total=length) as pbar:
        while True:
            decision = arrdict.arrdict(actions=torch.randint(9, size=(config['ENV']['n_envs'],
                                                                      config['ENV']['n_agents'])))
            world = env.step(decision)
            steps += 1
            pbar.update(1)
            if length is None and world.reset.any():
                break
            state = env.state(d)

            fig = env.display(0, config['ENV']['n_agents'])
            plt.show()

            encoder(arrdict.numpyify(arrdict.arrdict(**state, decision=decision)), config['ENV']['n_agents'])
            if steps == length:
                break

    save_dir = Path().absolute() / SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(save_dir / f"{timestamp}_record.mp4")
