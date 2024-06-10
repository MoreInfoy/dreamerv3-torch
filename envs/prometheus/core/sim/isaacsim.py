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

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import to_torch, torch_rand_float, get_axis_params, quat_rotate_inverse, get_euler_xyz
from omegaconf import DictConfig, OmegaConf
from envs.prometheus.utils.utils import parse_sim_params
from envs.prometheus import SOURCE_DIR

import torch
import os
import sys
import numpy as np
from collections import deque


class IsaacSim(object):
    def __init__(self, sim_cfg: DictConfig):
        self.num_envs = sim_cfg.num_envs
        self.num_actions = sim_cfg.num_sim_actions
        self.device = ("cuda:0" if torch.cuda else "cpu")
        self.dtype = torch.float
        self.time_step = sim_cfg.sim.dt * sim_cfg.control.decimation

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.cfg = sim_cfg
        self._create_sim()
        self._alloc_buffer()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def step(self, actions: torch.Tensor):
        self.actions[:] = actions.to(self.device)
        self._render()
        for i in range(self.cfg.control.decimation):
            self.action_torque_buffer[:, 1:, :] = self.action_torque_buffer[:, :self.action_len_buffer - 1, :]
            self.action_torque_buffer[:, 0, :] = self.actions[:]
            self.torques[:] = self._compute_torques(
                self.action_torque_buffer[torch.arange(self.num_envs), self.action_delay, :].squeeze(1)).view(
                self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.common_step_counter += 1
        # prepare quantities
        self.base_lin_vel_last[:] = self.base_lin_vel[:]
        self.base_quat[:] = self.root_states[:, 3:7]

        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        roll[roll > torch.pi] -= 2 * torch.pi
        roll[roll < -torch.pi] += 2 * torch.pi
        pitch[pitch > torch.pi] -= 2 * torch.pi
        pitch[pitch < -torch.pi] += 2 * torch.pi
        yaw[yaw > torch.pi] -= 2 * torch.pi
        yaw[yaw < -torch.pi] += 2 * torch.pi
        self.base_rpy[:, 0] = roll
        self.base_rpy[:, 1] = pitch
        self.base_rpy[:, 2] = yaw

        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, (self.root_states[:, :3] - self.base_pos_last) / self.time_step)
        self.base_lin_acc[:] = (self.base_lin_vel - self.base_lin_vel_last) / self.time_step
        self.base_pos_last[:] = self.root_states[:, :3]

        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # reset robot states
        self.actions[env_ids] = 0.
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.action_torque_buffer[env_ids] = 0

    def push_robots(self, max_push_vel_xy: float):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        self.root_states[:, 7:9] += torch_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2),
                                                     device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids, :] = self.default_dof_pos[env_ids, :]
        self.dof_vel[env_ids, :] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
        #                                                    device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.base_pos_last[env_ids, :] = self.root_states[env_ids, :3]
        self.base_lin_vel[env_ids, :] = quat_rotate_inverse(self.root_states[env_ids, 3:7],
                                                            self.root_states[env_ids, 7:10])

    def _create_sim(self):
        self.gym = gymapi.acquire_gym()
        sim_params = {"sim": OmegaConf.to_container(
            self.cfg.sim, resolve=True)}
        sim_params = parse_sim_params(sim_params)

        _, sim_device_id = gymutil.parse_device_str(self.device)

        # graphics device for rendering, -1 for no rendering
        graphics_device_id = sim_device_id
        if self.cfg.headless:
            graphics_device_id = -1

        self.sim = self.gym.create_sim(
            sim_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # add ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

        # load
        self._load_isaac_sim()

        # prepare vis
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.cfg.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            # self.gym.subscribe_viewer_keyboard_event(
            #     self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # set camera
        if not self.cfg.headless:
            position = self.cfg.viewer.pos
            lookat = self.cfg.viewer.lookat
            cam_pos = gymapi.Vec3(position[0], position[1], position[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def _load_isaac_sim(self):

        asset = self.cfg.asset

        asset_path = os.path.join(SOURCE_DIR, os.path.dirname(asset.urdf))
        asset_file = os.path.basename(asset.urdf)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = asset.flip_visual_attachments
        asset_options.fix_base_link = asset.fix_base_link
        asset_options.density = asset.density
        asset_options.angular_damping = asset.angular_damping
        asset_options.linear_damping = asset.linear_damping
        asset_options.max_angular_velocity = asset.max_angular_velocity
        asset_options.max_linear_velocity = asset.max_linear_velocity
        asset_options.armature = asset.armature
        asset_options.thickness = asset.thickness
        asset_options.disable_gravity = asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_path, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print(self.dof_names)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        # initial state
        init_state = self.cfg.init_state
        base_init_state_list = init_state.pos + init_state.quat + \
                               init_state.lin_vel + init_state.ang_vel
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.friction_coef = torch.zeros(
            self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.restitution_coef = torch.zeros(
            self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.base_add_mass = torch.zeros(
            self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.hand_add_mass = torch.zeros(
            (self.num_envs, 2), dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.base_com_shift = torch.zeros(
            self.num_envs, 3, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.hand_com_shift = torch.zeros(
            self.num_envs, 6, dtype=self.dtype, device=self.device, requires_grad=False
        )

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1.,
                                        (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.name, i,
                                                 asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle)

            hand_names = [s for s in self.body_names if self.cfg.asset.hand_name in s]
            hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
            for k in range(len(hand_names)):
                hand_indices[k] = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, hand_names[k])

            body_props = self._process_rigid_body_props(body_props, i, hand_indices)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                         self.actor_handles[0],
                                                                         feet_names[i])

        knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                         self.actor_handles[0],
                                                                         knee_names[i])

        hand_names = [s for s in self.body_names if self.cfg.asset.hand_name in s]
        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                         self.actor_handles[0],
                                                                         hand_names[i])

    def _get_env_origins(self):
        terrain_cfg = self.cfg.terrain
        if terrain_cfg.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = terrain_cfg.max_init_terrain_level
            if not terrain_cfg.curriculum:
                max_init_level = terrain_cfg.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / terrain_cfg.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = terrain_cfg.num_rows
            self.terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self.device).to(self.dtype)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols))
            spacing = self.cfg.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coef[:] = friction_buckets[bucket_ids].squeeze()

            for s in range(len(props)):
                props[s].friction = self.friction_coef[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                (
                    min_restitution,
                    max_restitution,
                ) = self.cfg.domain_rand.restitution_range
                self.restitution_coef = (
                        torch.rand(
                            self.num_envs,
                            dtype=self.dtype,
                            device=self.device,
                            requires_grad=False,
                        )
                        * (max_restitution - min_restitution)
                        + min_restitution
                )
            for s in range(len(props)):
                props[s].restitution = self.restitution_coef[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=self.dtype, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=self.dtype, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=self.dtype, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
        return props

    def _process_rigid_body_props(self, props, env_id, hand_indices):
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                inertia_scale = np.random.uniform(low_bound, high_bound)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale

        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                self.base_add_mass = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (max_add_mass - min_add_mass) + min_add_mass
                )
            props[0].mass += self.base_add_mass[env_id]

        if self.cfg.domain_rand.randomize_hand_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_hand_mass_range
                self.hand_add_mass = (
                        torch.rand((self.num_envs, 2), dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (max_add_mass - min_add_mass) + min_add_mass
                )
            props[hand_indices[0]].mass += self.hand_add_mass[env_id, 0]
            props[hand_indices[1]].mass += self.hand_add_mass[env_id, 1]

        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                self.base_com_shift[:, 0] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_x * 2) - com_x
                )
                self.base_com_shift[:, 1] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_y * 2) - com_y
                )
                self.base_com_shift[:, 2] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_z * 2) - com_z
                )
            props[0].com.x += self.base_com_shift[env_id, 0]
            props[0].com.y += self.base_com_shift[env_id, 1]
            props[0].com.z += self.base_com_shift[env_id, 2]

        if self.cfg.domain_rand.randomize_hand_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_hand_com_vec
                self.hand_com_shift[:, 0] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_x * 2) - com_x
                )
                self.hand_com_shift[:, 1] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_y * 2) - com_y
                )
                self.hand_com_shift[:, 2] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_z * 2) - com_z
                )

                self.hand_com_shift[:, 3] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_x * 2) - com_x
                )
                self.hand_com_shift[:, 4] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_y * 2) - com_y
                )
                self.hand_com_shift[:, 5] = (
                        torch.rand(self.num_envs, dtype=self.dtype, device=self.device, requires_grad=False,
                                   ) * (com_z * 2) - com_z
                )
            props[hand_indices[0]].com.x += self.hand_com_shift[env_id, 0]
            props[hand_indices[0]].com.y += self.hand_com_shift[env_id, 1]
            props[hand_indices[0]].com.z += self.hand_com_shift[env_id, 2]
            props[hand_indices[1]].com.x += self.hand_com_shift[env_id, 3]
            props[hand_indices[1]].com.y += self.hand_com_shift[env_id, 4]
            props[hand_indices[1]].com.z += self.hand_com_shift[env_id, 5]

        return props

    def _alloc_buffer(self):
        self.gravity_vec = to_torch(get_axis_params(-1., 2), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch(
            [1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        self.common_step_counter = 0

        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=self.dtype, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.name)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(
            rigid_body_state).view(self.num_envs, -1, 13)

        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=self.dtype,
                                   device=self.device, requires_grad=False)
        self.torques_scale = torch.ones(self.num_envs, self.num_dof, dtype=self.dtype, device=self.device,
                                        requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=self.dtype,
                                   device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        self.base_rpy = torch.zeros_like(self.base_lin_vel)
        self.base_pos_last = torch.zeros_like(self.base_lin_vel)
        self.base_lin_vel_last = torch.zeros_like(self.base_lin_vel)

        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)

        self.p_gains = torch.zeros(self.num_envs, self.num_dof, dtype=self.dtype, device=self.device,
                                   requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_dof, dtype=self.dtype, device=self.device,
                                   requires_grad=False)
        self.p_gains_scale = torch.zeros(self.num_envs, self.num_dof, dtype=self.dtype, device=self.device,
                                         requires_grad=False)
        self.d_gains_scale = torch.zeros(self.num_envs, self.num_dof, dtype=self.dtype, device=self.device,
                                         requires_grad=False)
        self.action_scale = torch.zeros(self.num_envs, self.num_dof, dtype=self.dtype, device=self.device,
                                        requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    self.action_scale[:, i] = self.cfg.control.action_scale[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                self.action_scale[:, i] = 1.
                if self.cfg.env.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero")

        if self.cfg.domain_rand.randomize_Kp:
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains_scale = torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.p_gains.shape,
                device=self.device,
            )
            self.p_gains *= self.p_gains_scale
        if self.cfg.domain_rand.randomize_Kd:
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains_scale = torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
            self.d_gains *= self.d_gains_scale

        if self.cfg.domain_rand.randomize_motor_torque:
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )

        if self.cfg.domain_rand.randomize_motor_response_delay:
            self.action_delay = torch.randint(
                int(self.cfg.domain_rand.randomize_motor_response_delay_range[0] / self.cfg.sim.dt),
                int(self.cfg.domain_rand.randomize_motor_response_delay_range[1] / self.cfg.sim.dt),
                (self.num_envs,),
                device=self.device,
            )
            self.action_len_buffer = int(
                self.cfg.domain_rand.randomize_motor_response_delay_range[1] / self.cfg.sim.dt)
            self.action_torque_buffer = torch.zeros(
                (self.num_envs, self.action_len_buffer, self.num_dof),
                device=self.device, dtype=self.dtype)
        else:
            self.action_delay = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
            self.action_len_buffer = 2
            self.action_torque_buffer = torch.zeros(
                (self.num_envs, self.action_len_buffer, self.num_dof),
                device=self.device, dtype=self.dtype)

    def _compute_torques(self, actions):
        # pd controller
        actions_scaled = actions * self.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits)
