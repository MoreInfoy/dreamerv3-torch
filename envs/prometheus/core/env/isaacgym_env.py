#  Copyright (c) 2024 CLEAR Lab
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

from envs.prometheus.core.env.env_base import EnvBase
from envs.prometheus.core.task.task_base import TaskBase
from envs.prometheus.core.sim.isaacsim import IsaacSim
from typing import Tuple, Dict, Union, Any
from omegaconf import DictConfig
import torch
import time


class IsaacgymEnv(EnvBase):
    sim: IsaacSim
    task: TaskBase
    cfg: DictConfig

    def __init__(self, sim: IsaacSim, task: TaskBase, cfg: DictConfig):
        super().__init__(sim, cfg, task)
        termination_contact_names = []
        if hasattr(self.cfg, "terminate_after_contacts_on"):
            for name in self.cfg.terminate_after_contacts_on:
                termination_contact_names.extend(
                    [s for s in self.sim.body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.sim.gym.find_actor_rigid_body_handle(self.sim.envs[0],
                                                                                            self.sim.actor_handles[0],
                                                                                            termination_contact_names[
                                                                                                i])
        self._init_buf()

    def _init_buf(self):
        super()._alloc_buffer()
        self.noise_scale_vec = self.get_noise_scale_vec()
        if self.cfg.observation.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.push_robot = self.cfg.external.push_robot
        self.push_interval = int(
            self.cfg.external.push_interval_s / self.cfg.dt)
        self.max_push_vel_xy = self.cfg.external.max_push_vel_xy

    def step(self, actions):
        actions = actions["action"]
        delay = torch.rand((self.num_envs, 1), device=self.device)
        # delay = 0.0
        actions_f = (1.0 - delay) * actions + delay * self.sim.actions
        clip_actions = self.cfg.clip.clip_actions
        actions_clipped = torch.clip(
            actions_f, -clip_actions, clip_actions).to(self.device)
        for i in range(self.substeps):
            self.sim.step(actions_clipped)
        self._post_physics_step()

        obs = self.get_observation()
        self.extras["discount"] = torch.ones(self.num_envs, dtype=self.sim.dtype).to(self.device)
        return obs, self.rew_buf.clone(), self.reset_buf.clone(), self.extras

    def _post_physics_step(self):
        self.episode_length_buf += 1
        self.check_termination()
        # if not self.cfg.test:
        self.compute_reward()
        self._post_physics_step_callback()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.task.update()
        self.compute_observations()

        if self.sim.viewer and self.sim.enable_viewer_sync:
            self.draw_debug_vis()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # fill extras
        self.extras["log"] = {}
        for key in self.task.episode_reward_sums.keys():
            self.extras["log"]['rew_' + key] = torch.mean(
                self.task.episode_reward_sums[key] / self.episode_length_buf.clip(min=1))

        # reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.terminated_buf[env_ids] = 0
        self.is_first_buf[env_ids] = 0
        self.sim.reset_idx(env_ids)
        for i in range(self.obs_history_buf.maxlen):
            self.obs_history_buf[i][env_ids] *= 0.

        if self.cfg.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.task.reset_idx(env_ids)

    def _init_height_points(self):
        obs_cfg = self.cfg.observation
        y = torch.tensor(obs_cfg.terrain.measured_points_y,
                         device=self.device, requires_grad=False)
        x = torch.tensor(obs_cfg.terrain.measured_points_x,
                         device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points,
                             3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _post_physics_step_callback(self):
        if self.cfg.observation.terrain.measure_heights:
            self.measured_heights = self.get_heights()
        if self.push_robot and (self.sim.common_step_counter % self.push_interval == 0):
            self.sim.push_robots(self.max_push_vel_xy)

    def check_termination(self):
        self.is_first_buf[:] = self.reset_buf[:]
        self.terminated_buf[:] = torch.any(
            torch.norm(self.sim.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.terminated_buf |= (self.sim.root_states[:, 2] < 0.4)
        # no terminal reward for time-outs
        self.time_out_buf[:] = self.episode_length_buf > self.max_episode_length
        self.reset_buf = torch.logical_or(self.time_out_buf, self.terminated_buf)

    def compute_reward(self):
        self.rew_buf[:] = self.cfg.dt * self.task.reward_eval()
        if self.cfg.clip.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.task.has_termination_reward:
            self.rew_buf[:] += self.cfg.dt * self.task.reward_termination(self.terminated_buf)

    def get_noise_scale_vec(self):
        raise NotImplementedError

    def compute_observations(self):
        raise NotImplementedError

    def compute_proprioception_obs(self):
        raise NotImplementedError

    def draw_debug_vis(self):
        raise NotImplementedError

    def get_heights(self, env_ids=None):
        raise NotImplementedError
