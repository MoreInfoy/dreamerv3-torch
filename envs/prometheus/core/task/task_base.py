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

from omegaconf import DictConfig, OmegaConf
from envs.prometheus.core.sim.isaacsim import IsaacSim
from abc import ABC, abstractmethod
from typing import Dict
import torch


class TaskBase(ABC):
    sim: IsaacSim
    cfg: DictConfig
    device: str
    has_termination_reward: bool
    episode_reward_sums: Dict[str, torch.Tensor]
    commands: torch.Tensor

    def __init__(self, sim: IsaacSim, task_cfg: DictConfig):
        self.num_envs = sim.num_envs
        self.sim = sim
        self.cfg = task_cfg
        self.device = sim.device

        self.rew_buf = torch.zeros(self.num_envs, dtype=self.sim.dtype, device=self.device, requires_grad=False)
        self.iter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.has_termination_reward = task_cfg.has_termination_reward

        self.commands = None

        self._prepare_reward_function()

        print(f'create task {task_cfg.name}')

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = OmegaConf.to_container(self.cfg.rewards.scales)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        # reward episode sums
        self.episode_reward_sums = {
            name: torch.zeros(self.num_envs, dtype=self.sim.dtype, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def reward_eval(self) -> torch.Tensor:
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_reward_sums[name] += rew
        return self.rew_buf

    def reward_termination(self, terminated_buf) -> torch.Tensor:
        rew = self.reward_scales["termination"] * terminated_buf
        self.episode_reward_sums["termination"] += rew
        return rew

    def reset_idx(self, env_ids):
        if len(env_ids) > 0:
            self.iter[env_ids] = 0
            for key in self.episode_reward_sums.keys():
                self.episode_reward_sums[key][env_ids] = 0.
            self.sample_commands(env_ids)

    def draw_debug_vis(self):
        pass

    @abstractmethod
    def get_task_obs(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_task_privileged_obs(self) -> torch.Tensor:
        return NotImplementedError

    @abstractmethod
    def sample_commands(self, env_ids):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError
