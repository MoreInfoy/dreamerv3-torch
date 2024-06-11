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
import uuid
from datetime import datetime

from envs.prometheus.core.sim.isaacsim import IsaacSim
from omegaconf import DictConfig
from envs.prometheus.utils.utils import set_seed
from envs.prometheus.core.task import task_base
from collections import deque
import torch
import gym
import numpy as np


class EnvBase:
    num_envs: int
    num_obs: int
    num_commands: int
    num_proprioception_obs: int
    obs_history_length: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    is_first_buf: torch.Tensor
    time_out_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # current episode duration
    extras: dict
    device: str
    task: task_base.TaskBase

    def __init__(self, sim: IsaacSim, cfg: DictConfig, task: task_base.TaskBase):
        self.sim = sim
        self.cfg = cfg
        self.num_envs = sim.num_envs
        self.num_obs = cfg.num_proprioception_obs
        self.num_privileged_obs = cfg.num_privileged_obs
        self.num_actions = cfg.num_actions
        self.obs_history_length = cfg.obs_history_length
        self.num_commands = cfg.num_commands

        self.max_episode_length = int(cfg.max_episode_length_s / cfg.dt)
        self.device = sim.device
        self.substeps = max(int(cfg.dt / self.sim.time_step), 1)
        self.num_proprioception_obs = cfg.num_proprioception_obs
        self.num_actions = cfg.num_actions
        self.dtype = sim.dtype
        self.extras = {}
        self.task = task

        set_seed(self.cfg.seed)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def _alloc_buffer(self):
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=self.dtype)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=self.dtype)
        self.commands = torch.zeros(self.num_envs, self.num_commands, device=self.device, dtype=self.dtype)
        self.obs_history = torch.zeros(self.num_envs, self.num_proprioception_obs * self.obs_history_length,
                                       device=self.device, dtype=self.dtype)
        self.obs_history_buf = deque(maxlen=self.obs_history_length)
        for _ in range(self.obs_history_length):
            self.obs_history_buf.append(torch.zeros(
                self.num_envs, self.num_proprioception_obs, dtype=self.dtype, device=self.device))
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=self.dtype)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.terminated_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.is_first_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

    def get_observation(self):
        # obs = {"obs": self.obs_buf.clone(), "obs_history": self.obs_history.clone(),
        #        "privileged_obs_buf": self.privileged_obs_buf.clone(), "commands": self.commands.clone(),
        #        "is_terminal": self.terminated_buf.clone(), "is_first": self.is_first_buf.clone()}
        return {"obs": self.privileged_obs_buf.clone(), "commands": self.commands.clone(),
                "is_terminal": self.reset_buf.clone(), "is_first": self.is_first_buf.clone()}

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.init_at_random_ep_len:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self.step({"action": torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)})
        return self.get_observation()

    def step(self, actions):
        raise NotImplementedError

    @property
    def observation_space(self):
        spaces = {}
        # spaces["obs"] = gym.spaces.Box(-np.inf, np.inf, [self.num_obs, ], dtype=np.float32)
        # spaces["obs_history"] = gym.spaces.Box(-np.inf, np.inf, [self.num_obs * self.obs_history_length, ],
        #                                        dtype=np.float32)
        # spaces["privileged_obs_buf"] = gym.spaces.Box(-np.inf, np.inf, [self.num_privileged_obs, ], dtype=np.float32)
        spaces["obs"] = gym.spaces.Box(-np.inf, np.inf, [self.num_privileged_obs, ], dtype=np.float32)
        spaces["commands"] = gym.spaces.Box(-np.inf, np.inf, [self.num_commands, ], dtype=np.float32)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Box(-10.0, 10.0, [self.num_actions, ], dtype=np.float32)
