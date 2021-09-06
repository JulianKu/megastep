import matplotlib.pyplot as plt
import torch
from matplotlib import patches as mpl_patches
from numpy import concatenate as np_concat, stack as np_stack, arange as np_arange

from megastep import modules, core, plotting, scene, cubicasa
from rebar import arrdict, dotdict

CLEARANCE = 1.


class SearchAndRescueBase:

    def __init__(self, n_envs, *args, **kwargs):
        geometries = cubicasa.sample(n_envs, **kwargs)
        scenery = scene.scenery(geometries, **kwargs)
        self.core = core.Core(scenery, *args, res=256, fov=179, **kwargs)
        self._battery = modules.BatteryLevel(self.core)
        self._laser = modules.Laser(self.core, subsample=1)
        self._depth = modules.Depth(self.core, subsample=1)
        self._mover = modules.MomentumMovementOnOff(self.core)
        self._imu = modules.IMU(self.core)
        self._respawner = modules.RandomSpawns(geometries, self.core)

        self.action_space = self._mover.space
        self.obs_space = dotdict.dotdict(
            bat=self._battery.space,
            las=self._laser.space,
            dep=self._depth.space,
            imu=self._imu.space)

        self._bounds = arrdict.torchify(np_stack([g.masks.shape * g.res for g in geometries])).to(self.core.device)
        self._tex_to_env = self.core.scenery.lines.inverse[self.core.scenery.textures.inverse.long()].long()
        self._seen = torch.full_like(self._tex_to_env, False)
        self._potential = self.core.env_full(0.)

        self._lengths = torch.zeros(self.core.n_envs, device=self.core.device, dtype=torch.int)

        self.device = self.core.device

    def _tex_indices(self, aux):
        scenery = self.core.scenery
        mask = aux.indices >= 0
        result = torch.full_like(aux.indices, -1, dtype=torch.long)
        tex_n = (scenery.lines.starts[:, None, None, None] + aux.indices)[mask]
        tex_w = scenery.textures.widths[tex_n.to(torch.long)]
        tex_i = torch.min(torch.floor(tex_w.to(torch.float) * aux.locations[mask]), tex_w.to(torch.float) - 1)
        tex_s = scenery.textures.starts[tex_n.to(torch.long)]
        result[mask] = tex_s.to(torch.long) + tex_i.to(torch.long)
        return result.unsqueeze(2)

    def _reward(self, r, reset):
        texindices = self._tex_indices(r)
        self._seen[texindices] = True

        potential = torch.zeros_like(self._potential)
        potential.scatter_add_(0, self._tex_to_env, self._seen.float())

        reward = (potential - self._potential) / self.core.res
        self._potential = potential

        # Should I render twice so that the last reward is accurate?
        reward[reset] = 0.

        return reward

    def _observe(self, reset):
        r = modules.render(self.core)
        obs = arrdict.arrdict(
            bat=self._battery(r),
            las=self._laser(r),
            dep=self._depth(r),
            imu=self._imu())
        reward = self._reward(r, reset)
        return obs, reward

    def _reset(self, reset=None):
        self._respawner(reset.unsqueeze(-1))
        self._seen[reset[self._tex_to_env]] = False
        self._potential[reset] = 0
        self._lengths[reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = self.core.env_full(True)
        self._reset(reset)
        obs, reward = self._observe(reset)
        return arrdict.arrdict(
            obs=obs,
            reset=reset,
            reward=reward)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)

        self._lengths += 1

        reset = (self._lengths >= self._potential + 200)
        self._reset(reset)
        obs, reward = self._observe(reset)
        return arrdict.arrdict(
            obs=obs,
            reset=reset,
            reward=reward)

    def state(self, e=0):
        seen = self._seen[self._tex_to_env == e]
        return arrdict.arrdict(
            core=self.core.state(e),
            bat=self._battery.state(e),
            las=self._laser.state(e),
            dep=self._depth.state(e),
            potential=self._potential[e].clone(),
            seen=seen.clone(),
            length=self._lengths[e].clone(),
            max_length=self._potential[e].add(200).clone(),
            bounds=self._bounds[e].clone())

    @classmethod
    def plot_state(cls, state):
        n_agents = state.core.n_agents

        fig = plt.figure()
        gs = plt.GridSpec(n_agents, 2, fig)

        colors = [f'C{i}' for i in range(n_agents)]

        alpha = .1 + .9 * state.seen.astype(float)
        # modifying this in place will bite me eventually. o for a lens
        state.core.scenery.textures.vals = np_concat([state.core.scenery.textures.vals, alpha[:, None]], 1)
        plan = core.Core.plot_state(state.core, plt.subplot(gs[:-1, :-1]))

        # Add bounding box
        size = state.bounds[::-1] + 2 * CLEARANCE
        bounds = mpl_patches.Rectangle(
            (-CLEARANCE, -CLEARANCE), *size,
            linewidth=1, edgecolor='k', facecolor=(0., 0., 0., 0.))
        plan.add_artist(bounds)

        images = {'dep': state.dep}
        plotting.plot_images(images, [plt.subplot(gs[i, -1]) for i in range(n_agents)])

        s = (f'length: {state.length:d}/{state.max_length:.0f}\n'
             f'potential: {state.potential:.0f}')
        plan.annotate(s, (5., 5.), xycoords='axes points')

        ax = plt.subplot(gs[-1, 0])
        ax.barh(np_arange(n_agents), state.bat, color=colors)
        ax.set_ylabel('battery')
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)

        return fig

    def display(self, d=0):
        return self.plot_state(arrdict.numpyify(self.state(d)))
