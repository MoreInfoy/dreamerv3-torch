#   Copyright (c) 2024 CLEAR Lab
#  #
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#  #
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#  #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from typing import Dict
from envs.prometheus.core.task.task_base import TaskBase
from envs.prometheus.core.sim.isaacsim import IsaacSim
from envs.prometheus.core.env.isaacgym_env import IsaacgymEnv
from omegaconf import DictConfig
from envs.prometheus.utils.math import quat_apply_yaw
import torch


class HumanoidTaskEnv(IsaacgymEnv):
    def __init__(self, sim: IsaacSim, task: TaskBase, cfg: DictConfig):
        super().__init__(sim, task, cfg)
        self.height_samples = None
        self.reset()

    def get_heights(self, env_ids=None):
        if self.sim.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.sim.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")
        if env_ids:
            points = quat_apply_yaw(self.sim.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.sim.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.sim.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.sim.root_states[:, :3]).unsqueeze(1)

        points += self.sim.cfg.terrain.border_size
        points = (points / self.sim.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.sim.cfg.terrain.vertical_scale

    def get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.compute_proprioception_obs()[0])
        noise = self.cfg.observation.noise
        self.add_noise = noise.add_noise
        noise_scales = noise.noise_scales
        noise_level = noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.cfg.observation.normalization.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:24] = noise_scales.dof_pos * noise_level * self.cfg.observation.normalization.dof_pos
        noise_vec[24:34] = noise_scales.dof_vel_leg * noise_level * self.cfg.observation.normalization.dof_vel
        noise_vec[34:42] = noise_scales.dof_vel_arm * noise_level * self.cfg.observation.normalization.dof_vel
        # noise_vec[42:70] = 0.  # previous actions
        return noise_vec

    def draw_debug_vis(self):
        self.task.draw_debug_vis()

    def compute_proprioception_obs(self):
        obs_scales = self.cfg.observation.normalization
        return torch.cat((
            self.sim.base_ang_vel * obs_scales.ang_vel,
            self.sim.projected_gravity,
            (self.sim.dof_pos - self.sim.default_dof_pos) *
            obs_scales.dof_pos,
            self.sim.dof_vel * obs_scales.dof_vel,
            self.sim.actions,
        ), dim=-1)

    def compute_observations(self):
        clip_obs = self.cfg.clip.clip_observations
        obs_scales = self.cfg.observation.normalization
        privileged_obs = torch.cat((self.sim.base_lin_vel * obs_scales.lin_vel,
                                    self.sim.base_ang_vel * obs_scales.ang_vel,
                                    self.sim.projected_gravity,
                                    (self.sim.dof_pos - self.sim.default_dof_pos) * obs_scales.dof_pos,
                                    self.sim.dof_vel * obs_scales.dof_vel,
                                    self.sim.actions,
                                    self.task.get_task_privileged_obs()), dim=-1)

        proprioception_obs = self.compute_proprioception_obs()
        # add noise if needed
        if self.cfg.observation.noise.add_noise:
            self.obs_buf = proprioception_obs + (2 * torch.rand_like(proprioception_obs) - 1) * self.noise_scale_vec
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        self.obs_history_buf.append(proprioception_obs)
        obs_buf_all = torch.stack([self.obs_history_buf[i]
                                   for i in range(self.obs_history_buf.maxlen)], dim=1)  # N,T,K
        self.obs_history = torch.clip(obs_buf_all.reshape(self.num_envs, -1), -clip_obs, clip_obs)

        env_params = torch.cat((self.sim.friction_coef.unsqueeze(-1),
                                self.sim.restitution_coef.unsqueeze(-1),
                                self.sim.base_add_mass.unsqueeze(-1),
                                self.sim.base_com_shift,
                                self.sim.p_gains_scale,
                                self.sim.d_gains_scale,
                                self.sim.torques_scale,
                                self.sim.default_dof_pos,
                                self.sim.action_delay.unsqueeze(-1),
                                self.sim.hand_add_mass,
                                self.sim.hand_com_shift
                                ),
                               dim=-1)  # 87

        # dims = 63 + 34 + 87 = 184
        # self.privileged_obs_buf = torch.clip(torch.cat((privileged_obs, env_params), dim=-1), -clip_obs, clip_obs)
        self.privileged_obs_buf = privileged_obs
        self.commands = self.task.get_task_obs()
