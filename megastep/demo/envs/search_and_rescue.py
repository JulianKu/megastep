import matplotlib.pyplot as plt
import torch
from matplotlib import patches as mpl_patches
from numpy import concatenate as np_concat, stack as np_stack, arange as np_arange

from megastep import modules, core, plotting, scene, cubicasa
from rebar import arrdict, dotdict

CLEARANCE = 1.


class SearchAndRescueBase:

    def __init__(self, config, *args, **kwargs):
        n_envs = config['n_envs']
        n_agents = config['n_agents']
        n_search_objects = config['n_search_objects']
        n_all_entities = n_agents + n_search_objects
        geometries = cubicasa.sample(n_envs, **kwargs)
        scenery = scene.scenery(geometries, n_all_entities, **kwargs)
        self.core = core.Core(scenery, *args, **config, **kwargs)
        self.n_controllable_agents = n_agents
        self.n_search_objects = n_search_objects
        self.n_all_entities = n_all_entities
        self._mover = modules.MomentumMovementOnOff(self.core, n_agents=n_agents)
        max_vel = self._mover.get_maximum_velocities()
        self._battery = modules.BatteryLevel(self.core, n_agents=n_agents, velocity_scales=max_vel, **kwargs)
        self._laser = modules.Laser(self.core, n_agents=n_agents, subsample=config['subsample'])
        self._depth = modules.Depth(self.core, n_agents=n_agents, subsample=config['subsample'])
        self._imu = modules.IMU(self.core)
        self._respawner = modules.RandomSpawns(geometries, self.core)

        self.action_space = self._mover.space
        self.obs_space = dotdict.dotdict(
            bat=self._battery.space,
            las=self._laser.space,
            dep=self._depth.space,
            imu=self._imu.space)

        self._bounds = arrdict.torchify(np_stack([g.masks.shape * g.res for g in geometries])).to(self.core.device)
        self._tex_to_env = self.core.scenery.lines.inverse[self.core.scenery.textures.inverse.long()].long()[:, None] \
            .repeat((1, n_agents))
        self._seen = torch.full_like(self._tex_to_env, False)
        agt_objs_tensor_size = (n_envs, n_agents, n_search_objects)
        self._found_search_objects = torch.zeros(size=agt_objs_tensor_size, device=self.core.device,
                                                 dtype=torch.bool)
        self._tracked_search_objects = torch.zeros(size=agt_objs_tensor_size, device=self.core.device,
                                                   dtype=torch.int)
        self._time_to_detect = config['time_to_detect']
        self._potential = torch.zeros((n_envs, n_agents), device=self.core.device, dtype=torch.float32)

        self._lengths = torch.zeros((n_envs, n_agents), device=self.core.device, dtype=torch.int)

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

    def _search_objects_in_view(self, aux):
        line_idxs = modules.downsample(aux.indices[:, :self.n_controllable_agents],
                                       self._laser.subsample)[..., self._laser.subsample // 2]
        obj_idxs = line_idxs // len(self.core.scenery.model)
        mask = (0 <= line_idxs) & (obj_idxs >= self.n_controllable_agents) & (obj_idxs < self.n_search_objects)
        objects_in_view = obj_idxs.where(mask, torch.full_like(line_idxs, -1))
        search_objects = torch.arange(start=self.n_controllable_agents, end=self.n_all_entities,
                                      device=self.core.device)
        obj_in_view = (objects_in_view[:, :, None] == search_objects[None, None, :, None, None]).any(-1).squeeze(-1)
        new_objs_in_view = obj_in_view & torch.logical_not(self._found_search_objects)

        return new_objs_in_view

    def _reward(self, r, reset):
        texindices = self._tex_indices(r)
        self._seen[texindices] = True

        potential = torch.zeros_like(self._potential)
        potential.scatter_add_(0, self._tex_to_env, self._seen.float())

        new_objects_in_view = self._search_objects_in_view(r)
        self._tracked_search_objects.masked_fill_(torch.logical_not(new_objects_in_view), 0)
        self._tracked_search_objects[new_objects_in_view] += 1
        new_found_objects = self._tracked_search_objects >= self._time_to_detect
        self._found_search_objects |= new_found_objects

        reward = (potential - self._potential) / self.core.res
        self._potential = potential

        # Should I render twice so that the last reward is accurate?
        reward[reset] = 0.

        return reward

    def _observe(self, reset):
        r = modules.render(self.core)[:, :self.n_controllable_agents]

        obs = arrdict.arrdict(
            bat=self._battery(r),
            las=self._laser(r),
            dep=self._depth(r),
            imu=self._imu())
        reward = self._reward(r, reset)
        return obs, reward

    def _reset(self, reset=None):
        self._respawner(reset)
        self._battery.reset(reset)
        controllable_agt_reset = reset[:, :self.n_controllable_agents]
        self._seen[controllable_agt_reset[self._tex_to_env][:, 0]] = False
        self._potential[controllable_agt_reset] = 0
        self._lengths[controllable_agt_reset] = 0

    @torch.no_grad()
    def reset(self):
        reset = self.core.agent_full(True)
        self._reset(reset)
        obs, reward = self._observe(reset[:, :self.n_controllable_agents])
        return arrdict.arrdict(
            obs=obs,
            reset=reset,
            reward=reward)

    @torch.no_grad()
    def step(self, decision):
        self._mover(decision)

        self._lengths += 1

        reset = (self._lengths >= self._potential + 200) | (self._battery.get_battery_level() <= 0)
        self._reset(reset)
        obs, reward = self._observe(reset)
        return arrdict.arrdict(
            obs=obs,
            reset=reset,
            reward=reward)

    def state(self, e=0):
        seen = self._seen.any(-1)[self._tex_to_env[:, 0] == e]
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
    def plot_state(cls, state, n_agents, dpi=None):
        fig = plt.figure()
        if dpi is not None:
            fig.set_dpi(dpi)

        gs = plt.GridSpec(max(2, n_agents), 2, fig)

        colors = [f'C{i}' for i in range(n_agents)]

        alpha = .1 + .9 * state.seen.astype(float)
        # modifying this in place will bite me eventually. o for a lens
        state.core.scenery.textures.vals = np_concat([state.core.scenery.textures.vals, alpha[:, None]], 1)
        plan = core.Core.plot_state(state.core, n_agents, plt.subplot(gs[:-1, :-1]))

        # Add bounding box
        size = state.bounds[::-1] + 2 * CLEARANCE
        bounds = mpl_patches.Rectangle(
            (-CLEARANCE, -CLEARANCE), *size,
            linewidth=1, edgecolor='k', facecolor=(0., 0., 0., 0.))
        plan.add_artist(bounds)

        images = {'dep': state.dep[:n_agents]}
        plotting.plot_images(images, [plt.subplot(gs[i, -1]) for i in range(n_agents)])

        s = ' '.join(['length:'] +
                     [f'{stat_len}/{state.max_length[i]:.0f}' for i, stat_len in enumerate(state.length)] +
                     ['\npotential:'] +
                     [f'{pot:.0f}' for pot in state.potential])
        plan.annotate(s, (5., 5.), xycoords='axes points')

        ax = plt.subplot(gs[-1, 0])
        ax.barh(np_arange(n_agents), state.bat[:n_agents], color=colors)
        ax.set_ylabel('battery')
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)

        return fig

    def display(self, d=0, n_agents=None, dpi=None):
        return self.plot_state(arrdict.numpyify(self.state(d)), n_agents or self.core.n_agents, dpi)
