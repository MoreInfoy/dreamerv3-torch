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

from isaacgym.torch_utils import torch_rand_float, quat_rotate_inverse, quat_apply
from isaacgym import gymutil, gymapi
from envs.prometheus.core.task.task_base import TaskBase
from envs.prometheus.core.sim.isaacsim import IsaacSim
from omegaconf import DictConfig
import torch


class Locomotion(TaskBase):
    def __init__(self, sim: IsaacSim, task_cfg: DictConfig):
        super().__init__(sim, task_cfg)

        # v_x, v_y, v_yaw
        self.commands = torch.zeros(self.num_envs, 10, dtype=self.sim.dtype, device=self.device, requires_grad=False)
        self.commands[:, 3] = 0.78
        self.commands[:, 4] = 0.2722
        self.commands[:, 5] = 0.1752
        self.commands[:, 6] = 0.1390
        self.commands[:, 7] = 0.2722
        self.commands[:, 8] = -0.1752
        self.commands[:, 9] = 0.1390
        self.commands_resampling_time = int(self.cfg.commands_resampling_time_s / self.cfg.dt)

        penalized_contact_names = []
        for name in self.cfg.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.sim.body_names if name in s])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.sim.gym.find_actor_rigid_body_handle(self.sim.envs[0],
                                                                                          self.sim.actor_handles[0],
                                                                                          penalized_contact_names[i])

        self.dof_pos_ref = torch.zeros_like(self.sim.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.sim.dof_vel)
        self.last_actions = torch.zeros_like(self.sim.actions)
        self.last_last_actions = torch.zeros_like(self.sim.actions)
        self.last_contacts = torch.zeros(self.num_envs, len(self.sim.feet_indices), dtype=torch.bool,
                                         device=self.device,
                                         requires_grad=False)
        self.last_root_vel = torch.zeros_like(self.sim.root_states[:, 7:13])
        self.feet_air_time = torch.zeros((self.num_envs, len(self.sim.feet_indices)), dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.gait_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.feet_height = torch.zeros((self.num_envs, len(self.sim.feet_indices)), dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.last_feet_z = 0.05 * torch.ones_like(self.feet_height)

        self.need_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.delta_lh = torch.zeros((self.num_envs, 5), dtype=self.sim.dtype, device=self.device,
                                    requires_grad=False)
        self.delta_rh = torch.zeros((self.num_envs, 5), dtype=self.sim.dtype, device=self.device,
                                    requires_grad=False)
        self.vel_mag = torch.zeros((self.num_envs, 3), dtype=self.sim.dtype, device=self.device,
                                   requires_grad=False)
        self.err_lh = torch.zeros((self.num_envs, 3), dtype=self.sim.dtype, device=self.device,
                                  requires_grad=False)
        self.err_rh = torch.zeros((self.num_envs, 3), dtype=self.sim.dtype, device=self.device,
                                  requires_grad=False)
        self.lhand_shift_local = torch.tensor([0.2722, 0.1752, 0.1390], device=self.device, dtype=self.sim.dtype,
                                              requires_grad=False)
        self.rhand_shift_local = torch.tensor([0.2722, -0.1752, 0.1390], device=self.device, dtype=self.sim.dtype,
                                              requires_grad=False)
        self.ee_rew = torch.zeros(self.num_envs, dtype=self.sim.dtype, device=self.device, requires_grad=False)

        self.lin_vel_rew = torch.zeros(self.num_envs, dtype=self.sim.dtype, device=self.device, requires_grad=False)

        self.default_dof_rew = torch.zeros(self.num_envs, dtype=self.sim.dtype, device=self.device, requires_grad=False)

        self.feet_clearance_rew = torch.zeros(self.num_envs, dtype=self.sim.dtype, device=self.device,
                                              requires_grad=False)

    def draw_debug_vis(self):
        # draw height lines
        if not self.cfg.debug_vis:
            return
        self.sim.gym.clear_lines(self.sim.viewer)
        self.sim.gym.refresh_rigid_body_state_tensor(self.sim.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 30, 16, None, color=(1, 0, 0))

        base_quat = self.sim.base_quat
        root_states = self.sim.root_states
        for i in range(self.num_envs):
            cmds = root_states[i, :3] + quat_apply(base_quat[i], self.commands[i, 4:7])
            pos = cmds.cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.sim.gym, self.sim.viewer, self.sim.envs[i], sphere_pose)

            cmds = root_states[i, :3] + quat_apply(base_quat[i], self.commands[i, 7:10])
            pos = cmds.cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.sim.gym, self.sim.viewer, self.sim.envs[i], sphere_pose)

        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 30, 16, None, color=(0, 0, 1))
        for i in range(self.num_envs):
            pos = self.sim.rigid_state[i, self.sim.hand_indices[0], :3].cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.sim.gym, self.sim.viewer, self.sim.envs[i], sphere_pose)

            pos = self.sim.rigid_state[i, self.sim.hand_indices[1], :3].cpu().numpy()
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.sim.gym, self.sim.viewer, self.sim.envs[i], sphere_pose)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.dof_pos_ref[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_contacts[env_ids] = 1
        self.last_root_vel[env_ids] = 0.
        self.feet_height[env_ids] = 0.
        self.last_feet_z[env_ids] = 0.
        self.gait_length_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0.
        self.need_step[env_ids] = 0
        self.delta_lh[env_ids] = 0.
        self.delta_rh[env_ids] = 0.
        self.vel_mag[env_ids] = 0.
        self.err_lh[env_ids] = 0.
        self.err_rh[env_ids] = 0.
        self.ee_rew[env_ids] = 0.
        self.lin_vel_rew[env_ids] = 0.
        self.default_dof_rew[env_ids] = 0.
        self.feet_clearance_rew[env_ids] = 0.
        self.sample_commands(env_ids)

    def get_task_obs(self) -> torch.Tensor:  # dims: 10
        return self.commands.clone()

    def get_task_privileged_obs(self) -> torch.Tensor:  # dims: 2 + 7 + 2 + 18 + 2 + 2 + 1= 34
        stance_mask = self._get_gait_phase()
        contact_mask = self.sim.contact_forces[:, self.sim.feet_indices, 2] > 5.
        diff = self.sim.dof_pos - self.dof_pos_ref
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        root_states = self.sim.root_states
        task_obs = torch.cat((
            self.sim.base_rpy[:, :2],
            2.0 * self.err_lh,
            2.0 * self.err_rh,
            root_states[:, 2].unsqueeze(-1),
            sin_pos,
            cos_pos,
            diff,
            stance_mask,
            contact_mask,
            self.need_step.unsqueeze(1)), dim=-1)
        return task_obs

    def update(self):
        self.iter += 1
        self.gait_length_buf += 1
        self.gait_length_buf *= (1 * self.need_step)

        resample_ids = (self.iter % self.commands_resampling_time == 0).nonzero(as_tuple=False).flatten()
        self.sample_commands(resample_ids)
        self.update_commands()

        self.compute_dof_ref()
        self.last_last_actions = self.last_actions.clone()
        self.last_actions = self.sim.actions.clone()
        self.last_dof_vel = self.sim.dof_vel.clone()
        self.last_root_vel = self.sim.root_states[:, 7:13].clone()

    def sample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, :] = 0.0
        self.commands[env_ids, 0] = torch_rand_float(self.cfg.command_ranges.lin_vel_x[0],
                                                     self.cfg.command_ranges.lin_vel_x[1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.cfg.command_ranges.lin_vel_y[0],
                                                     self.cfg.command_ranges.lin_vel_y[1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.cfg.command_ranges.ang_vel_yaw[0],
                                                     self.cfg.command_ranges.ang_vel_yaw[1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.vel_mag[env_ids, 0] = torch_rand_float(self.cfg.command_ranges.lin_vel_x[0],
                                                    self.cfg.command_ranges.lin_vel_x[1], (len(env_ids), 1),
                                                    device=self.device).squeeze(1)
        self.vel_mag[env_ids, 1] = torch_rand_float(self.cfg.command_ranges.lin_vel_y[0],
                                                    self.cfg.command_ranges.lin_vel_y[1], (len(env_ids), 1),
                                                    device=self.device).squeeze(1)
        self.vel_mag[env_ids, 2] = torch_rand_float(self.cfg.command_ranges.ang_vel_yaw[0],
                                                    self.cfg.command_ranges.ang_vel_yaw[1], (len(env_ids), 1),
                                                    device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(0.55, 0.8, (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, 3] = 0.75

        self.delta_lh[env_ids, 0] = torch_rand_float(0., self.cfg.hands_cmds_range["delta_x"], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_lh[env_ids, 1] = torch_rand_float(0., self.cfg.hands_cmds_range["delta_y"], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_lh[env_ids, 2] = torch_rand_float(0., self.cfg.hands_cmds_range["delta_z"], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_lh[env_ids, 3] = torch_rand_float(self.cfg.hands_cmds_range["freq"][0],
                                                     self.cfg.hands_cmds_range["freq"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_lh[env_ids, 4] = torch_rand_float(self.cfg.hands_cmds_range["phase_shift"][0],
                                                     self.cfg.hands_cmds_range["phase_shift"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

        self.delta_rh[env_ids, 0] = torch_rand_float(0., self.cfg.hands_cmds_range["delta_x"], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_rh[env_ids, 1] = -torch_rand_float(0., self.cfg.hands_cmds_range["delta_y"], (len(env_ids), 1),
                                                      device=self.device).squeeze(1)
        self.delta_rh[env_ids, 2] = torch_rand_float(0., self.cfg.hands_cmds_range["delta_z"], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_rh[env_ids, 3] = torch_rand_float(self.cfg.hands_cmds_range["freq"][0],
                                                     self.cfg.hands_cmds_range["freq"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.delta_rh[env_ids, 4] = torch_rand_float(self.cfg.hands_cmds_range["phase_shift"][0],
                                                     self.cfg.hands_cmds_range["phase_shift"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

    def update_commands(self):
        seg = int(self.num_envs / 5)

        time = self.iter * self.cfg.dt
        phase = torch.pi * time * self.delta_lh[:, 3] + self.delta_lh[:, 4]
        self.commands[:, 4:7] = torch.sin(phase).unsqueeze(-1) * self.delta_lh[:, :3] + self.lhand_shift_local
        phase = torch.pi * time * self.delta_rh[:, 3] + self.delta_rh[:, 4]
        self.commands[:, 7:10] = torch.sin(phase).unsqueeze(-1) * self.delta_rh[:, :3] + self.rhand_shift_local

        # variant vel
        phase = torch.pi * time[-seg:] * (self.delta_lh[-seg:, 3] + self.delta_rh[-seg:, 3])
        self.commands[-seg:, :3] = torch.sin(phase).unsqueeze(-1) * self.vel_mag[-seg:]

        # fixed
        self.commands[:seg, 4] = 0.2722
        self.commands[:seg, 5] = 0.1752
        self.commands[:seg, 6] = 0.1390
        self.commands[:seg, 7] = 0.2722
        self.commands[:seg, 8] = -0.1752
        self.commands[:seg, 9] = 0.1390

        # symmetric
        self.commands[seg:2 * seg, 7] = self.commands[seg:2 * seg, 4]
        self.commands[seg:2 * seg, 8] = -self.commands[seg:2 * seg, 5]
        self.commands[seg:2 * seg, 9] = self.commands[seg:2 * seg, 6]

        # no step
        self.commands[2 * seg:3 * seg, :3] = 0.
        self.commands[2 * seg:3 * seg, 3] = 0.65 + 0.15 * torch.sin(phase)

        base_vel = self.sim.base_lin_vel
        self.need_step[:] = torch.logical_or(torch.norm(self.commands[:, :3], dim=1) > 0.1,
                                             torch.norm(base_vel, dim=1) > 0.45)
        self.commands[:, :3] *= self.need_step.unsqueeze(1)  # set small commands to zero

    def compute_dof_ref(self):
        phase = self._get_phase()
        sin_p = torch.sin(2 * torch.pi * phase)
        sin_p_l = sin_p.clone()
        sin_p_r = sin_p.clone()
        self.dof_pos_ref[:] = 0
        sin_p_l[sin_p_l > 0] = 0
        self.dof_pos_ref[:, 2] = -2.0 * sin_p_r * self.cfg.rewards.dof_ref_scale
        self.dof_pos_ref[:, 3] = 3.0 * sin_p_r * self.cfg.rewards.dof_ref_scale
        self.dof_pos_ref[:, 4] = -sin_p_r * self.cfg.rewards.dof_ref_scale
        sin_p_r[sin_p_r < 0] = 0
        self.dof_pos_ref[:, 7] = -2.0 * sin_p_r * self.cfg.rewards.dof_ref_scale
        self.dof_pos_ref[:, 8] = 3.0 * sin_p_r * self.cfg.rewards.dof_ref_scale
        self.dof_pos_ref[:, 9] = -sin_p_r * self.cfg.rewards.dof_ref_scale
        self.dof_pos_ref[torch.abs(sin_p) < 0.05, :10] = 0.

        damping = torch.eye(4, device=self.device) * 1e-2
        jac_l = self.sim.jacobian[:, self.sim.hand_indices[0] - 1, :3, 16:20]
        jac_l_T = torch.transpose(jac_l, 1, 2)
        u1 = (torch.inverse(jac_l_T @ jac_l + damping)
              @ (jac_l_T @ self.err_lh.unsqueeze(-1) - 1e-2 * self.sim.dof_pos[:, 10:14].unsqueeze(
                    -1))).view(self.num_envs, 4)

        jac_r = self.sim.jacobian[:, self.sim.hand_indices[1] - 1, :3, 20:24]
        jac_r_T = torch.transpose(jac_r, 1, 2)
        u2 = (torch.inverse(jac_r_T @ jac_r + damping)
              @ (jac_r_T @ self.err_rh.unsqueeze(-1) - 1e-2 * self.sim.dof_pos[:, 14:18].unsqueeze(
                    -1))).view(self.num_envs, 4)
        self.dof_pos_ref[:, 10:] = torch.cat((u1, u2), dim=-1) + self.sim.dof_pos[:, 10:]

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.gait_length_buf * self.cfg.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # c_mask=1 is stance, c_mask=0 is swing
        phase = self._get_phase()
        sin_p = torch.sin(2 * torch.pi * phase)
        c_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        c_mask[:, 0] = sin_p >= 0
        # right foot stance
        c_mask[:, 1] = sin_p < 0
        # double support
        c_mask[torch.abs(sin_p) < 0.2] = 1
        return c_mask

    # ------------ reward functions----------------
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.sim.dof_pos.clone()
        pos_target = self.dof_pos_ref.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.sim.rigid_state[:, self.sim.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.sim.rigid_state[:, self.sim.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.sim.contact_forces[:, self.sim.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.sim.rigid_state[:, self.sim.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.sim.contact_forces[:, self.sim.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.cfg.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.sim.contact_forces[:, self.sim.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.sim.base_rpy[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.sim.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.sim.contact_forces[:, self.sim.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.sim.dof_pos - self.sim.default_dof_pos
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        rew_new = 0.5 * torch.exp(-yaw_roll * 100.0) + 0.5 * torch.exp(-yaw_roll * 50.0) - 0.01 * torch.norm(
            joint_diff, dim=1)
        pbrs = 5.0 * torch.clip(rew_new - self.default_dof_rew, min=0.0)
        self.default_dof_rew[:] = rew_new[:]
        return rew_new + pbrs

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.sim.rigid_state[:, self.sim.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.sim.root_states[:, 2] - (measured_heights - 0.05)
        err = (torch.abs(base_height - self.commands[:, 3]) - 0.02)
        return torch.exp(-err * 100.0) - torch.abs(base_height - self.commands[:, 3])

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.sim.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.sim.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.sim.base_ang_vel[:, :2], dim=1) * 5.)
        c_update = (lin_mismatch + ang_mismatch) / 2.
        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.sim.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.sim.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.sim.base_lin_vel[:, :2]), dim=1)
        rew_new = torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)
        pbrs = torch.clip(rew_new - self.lin_vel_rew, min=0.0)
        self.lin_vel_rew[:] = rew_new[:]
        return rew_new + pbrs

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.sim.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.sim.contact_forces[:, self.sim.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.sim.rigid_state[:, self.sim.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

        rew_process = self.feet_height - self.cfg.rewards.target_feet_height
        rew_process = torch.sum(rew_process * swing_mask, dim=1)

        pbrs = 5.0 * torch.clip(rew_process - self.feet_clearance_rew, min=0.0)

        self.feet_clearance_rew[:] = rew_process[:]
        self.feet_height *= ~contact

        return rew_pos + pbrs

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.sim.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.sim.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.sim.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.sim.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.sim.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.sim.dof_vel) / self.cfg.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1. * (torch.norm(self.sim.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.sim.actions), dim=1)
        term_2 = torch.sum(torch.square(self.sim.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.sim.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_fix_arm(self):
        arm_joint_pos = self.sim.dof_pos[:, 10:].clone()
        arm_joint_vel = self.sim.dof_vel[:, 10:].clone()
        return torch.norm(arm_joint_pos, dim=1) + 0.2 * torch.norm(arm_joint_vel, dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.sim.base_ang_vel[:, :2]), dim=1)

    def _reward_power(self):
        index1 = [0, 1, 4, 5, 6, 9]
        index2 = [2, 3, 7, 8]
        power1 = torch.sum(torch.abs(self.sim.torques[:, index1] * self.sim.dof_vel[:, index1]), dim=1)
        power2 = torch.sum(torch.abs(self.sim.torques[:, index2] * self.sim.dof_vel[:, index2]), dim=1)
        power3 = torch.sum(torch.abs(self.sim.torques[:, 10:] * self.sim.dof_vel[:, 10:]), dim=1)
        return 0.4 * power1 + 0.2 * power2 + power3

    def _reward_ee_ref(self):
        base_quat = self.sim.base_quat
        rigid_state = self.sim.rigid_state
        root_states = self.sim.root_states
        lh_cmd = root_states[:, :3] + quat_apply(base_quat, self.commands[:, 4:7])
        self.err_lh[:] = lh_cmd - rigid_state[:, self.sim.hand_indices[0], :3]
        rh_cmd = root_states[:, :3] + quat_apply(base_quat, self.commands[:, 7:10])
        self.err_rh[:] = rh_cmd - rigid_state[:, self.sim.hand_indices[1], :3]
        err = torch.norm(self.err_lh, dim=1) + torch.norm(self.err_rh, dim=1)
        rew = torch.exp(-20 * err) + torch.exp(-5 * err) - 0.2 * err
        pbrs = (rew - self.ee_rew).clip(min=-1.0)
        self.ee_rew[:] = rew[:]
        return rew + pbrs
