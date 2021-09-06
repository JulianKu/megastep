from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from tqdm.auto import tqdm

from megastep.demo.envs.search_and_rescue import SearchAndRescueBase
from rebar import recording, arrdict

SAVE_DIR = "recordings"

if __name__ == '__main__':
    seed = 10
    n_envs = 5
    n_agents = 3
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = SearchAndRescueBase(n_envs, n_agents=n_agents, rng=rng)

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
            decision = arrdict.arrdict(actions=torch.randint(9, size=(n_envs, n_agents)))
            world = env.step(decision)
            steps += 1
            pbar.update(1)
            if length is None and world.reset.any():
                break
            state = env.state(d)

            fig = env.display(0)
            # plt.show()

            encoder(arrdict.numpyify(arrdict.arrdict(**state, decision=decision)))
            if steps == length:
                break

    save_dir = Path().absolute() / SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(save_dir / f"{timestamp}_record.mp4")
