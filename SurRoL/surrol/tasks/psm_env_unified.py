import os
import time
import numpy as np
import pybullet as p

from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle, reset_camera
from surrol.const import ASSET_DIR_PATH

def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class UnifiedPsmEnv(PsmEnv):
    """
    A unified environment that can operate in multiple modes.
    The supported task types in this example are:
      - "NeedleReach"
      - "NeedlePick"
      - "GauzeRetrieve"
      
    The unified action space is fixed to 5 dimensions:
       [dx, dy, dz, d_yaw_or_d_pitch, jaw]
    (Depending on the task, some components will be zeroed out.)
    
    The observation is the concatenation of the robot state (tip position,
    orientation, jaw status) plus object-related states.
    """
    
    # You might want to override the SCALING or WORKSPACE_LIMITS if needed.
    SCALING = 5.0

    def __init__(self, task_type="NeedleReach", render_mode=None):
        self.task_type = task_type  # e.g. "NeedleReach", "NeedlePick", "GauzeRetrieve"
        super(UnifiedPsmEnv, self).__init__(render_mode=render_mode)
        # Optionally, set the scaling for simulation here:
        self.SCALING = UnifiedPsmEnv.SCALING

    def _env_setup(self):
        """
        Common environment setup. This calls the base class _env_setup()
        then dispatches to a task-specific setup.
        """
        super(UnifiedPsmEnv, self)._env_setup()
        if self.task_type == "NeedleReach":
            self._setup_needle_reach()
        elif self.task_type == "NeedlePick":
            self._setup_needle_pick()
        elif self.task_type == "GauzeRetrieve":
            self._setup_gauze_retrieve()
        else:
            raise ValueError(f"Unknown task_type {self.task_type}")

    # -------------------------------------------------------------------------
    # Task-specific setup routines
    # (These routines are adapted from your original implementations.)
    # -------------------------------------------------------------------------

    def _setup_needle_reach(self):
        # ----- NeedleReach Setup (adapted from your NeedleReach class) -----
        self.has_object = False

        # Robot: set initial position.
        ws = self.workspace_limits1  # from PsmEnv; already scaled
        pos = (ws[0][0], ws[1][1], ws[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = True

        # Load tray pad.
        tray_urdf = os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf')
        tray_pose = (np.array((0.55, 0, 0.6751)) * self.SCALING, p.getQuaternionFromEuler((0, 0, 0)))
        obj_id = p.loadURDF(tray_urdf, tray_pose[0], tray_pose[1], globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # add tray as a fixed object

        # Load needle.
        needle_urdf = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        ws_limits = self.workspace_limits1
        needle_pos = (ws_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                      ws_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                      ws_limits[2][0] + 0.01)
        yaw = (np.random.rand() - 0.5) * np.pi
        needle_quat = p.getQuaternionFromEuler((0, 0, yaw))
        obj_id = p.loadURDF(needle_urdf, needle_pos, needle_quat, useFixedBase=False, globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)
        # Save the needle object and the link used for goal sampling.
        self.obj_id = self.obj_ids['rigid'][0]
        self.obj_link1 = 1

    def _setup_needle_pick(self):
        # ----- NeedlePick Setup (adapted from your NeedlePick class) -----
        self.has_object = True
        self._waypoint_goal = True
        ws = self.workspace_limits1
        pos = (ws[0][0], ws[1][1], (ws[2][1] + ws[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        self._contact_approx = False

        # Load tray pad.
        tray_urdf = os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf')
        tray_pose = (np.array((0.55, 0, 0.6751)) * self.SCALING, p.getQuaternionFromEuler((0, 0, 0)))
        obj_id = p.loadURDF(tray_urdf, tray_pose[0], tray_pose[1], globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)

        # Load needle.
        needle_urdf = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        needle_pos = (ws[0].mean() + (np.random.rand() - 0.5) * 0.1,
                      ws[1].mean() + (np.random.rand() - 0.5) * 0.1,
                      ws[2][0] + 0.01)
        yaw = (np.random.rand() - 0.5) * np.pi
        needle_quat = p.getQuaternionFromEuler((0, 0, yaw))
        obj_id = p.loadURDF(needle_urdf, needle_pos, needle_quat, useFixedBase=False, globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)
        self.obj_id = self.obj_ids['rigid'][0]
        self.obj_link1 = 1

    def _setup_gauze_retrieve(self):
        # ----- GauzeRetrieve Setup (adapted from your GauzeRetrieve class) -----
        self.POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0)) 
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))

        # gauze
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

    # -------------------------------------------------------------------------
    # Goal sampling and callback (for visualization)
    # -------------------------------------------------------------------------

    def _sample_goal(self) -> np.ndarray:
        """
        Dispatch to task-specific goal sampling.
        """
        if self.task_type == "NeedleReach":
            # Sample goal: use the needle’s waypoint (shifted upward)
            pos, _ = get_link_pose(self.obj_id, self.obj_link1)
            goal = np.array([pos[0], pos[1], pos[2] + 0.005 * self.SCALING])
        elif self.task_type == "NeedlePick":
            # Sample goal for NeedlePick (example logic)
            ws = self.workspace_limits1
            goal = np.array([ws[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                             ws[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                             ws[2][1] - 0.04 * self.SCALING])
        elif self.task_type == "GauzeRetrieve":
            workspace_limits = self.workspace_limits1
            goal = np.array([workspace_limits[0].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.03 * self.SCALING])
        else:
            raise ValueError("Unknown task type for goal sampling")
        return goal.copy()

    def _sample_goal_callback(self):
        """
        Visualize the goal (by moving the red sphere).
        For tasks with waypoints, you might also set them here.
        """
        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal, (0, 0, 0, 1))
        # For tasks that use waypoints, you could initialize self._waypoints here.
        if self.task_type == "NeedlePick":
            # Adapted from NeedlePick: define a list of four waypoints.
            self._waypoints = [None, None, None, None]
            pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
            self._waypoint_z_init = pos_obj[2]
            orn = p.getEulerFromQuaternion(orn_obj)
            # Here we use the same yaw logic as before.
            eef_orn = p.getEulerFromQuaternion(get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1])
            yaw = orn[2] if abs(wrap_angle(orn[2] - eef_orn[2])) < abs(wrap_angle(orn[2] + np.pi - eef_orn[2])) else wrap_angle(orn[2] + np.pi)
            self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                           pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])
            self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                           pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])
            self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                           pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])
            self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                           self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])
        elif self.task_type == "GauzeRetrieve":
            # Adapted from GauzeRetrieve: define five waypoints.
            super()._sample_goal_callback()
            self._waypoints = [None, None, None, None, None]  # five waypoints
            pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
            self._waypoint_z_init = pos_obj[2]

            self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                        pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., 0.5])  # approach
            self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                        pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])  # approach
            self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                        pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])  # grasp
            self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                        pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., -0.5])  # grasp
            self._waypoints[4] = np.array([self.goal[0], self.goal[1],
                                        self.goal[2] + 0.0102 * self.SCALING, 0., -0.5])  # lift up


    # -------------------------------------------------------------------------
    # Action setting (unified)
    # -------------------------------------------------------------------------

    def _set_action(self, action: np.ndarray):
        """
        Unified action setting. The action is assumed to be a 5D vector:
          [dx, dy, dz, d_theta, jaw]
        For tasks like NeedleReach and GauzeRetrieve, we force d_theta=0.
        """
        action = action.copy()  # prevent in-place modification
        if self.task_type in ["NeedleReach", "GauzeRetrieve"]:
            action[3] = 0  # no yaw/pitch change for these tasks
        super(UnifiedPsmEnv, self)._set_action(action)

    # -------------------------------------------------------------------------
    # Oracle policy (scripted expert) dispatch
    # -------------------------------------------------------------------------

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Dispatch to task-specific oracle policies.
        """
        if self.task_type == "NeedleReach":
            # Oracle for NeedleReach (adapted from your NeedleReach.get_oracle_action)
            delta_pos = (obs['desired_goal'] - obs['achieved_goal']) / 0.01
            if np.linalg.norm(delta_pos) < 1.5:
                delta_pos.fill(0)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            delta_pos *= 0.3
            return np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., 0.])
        elif self.task_type == "NeedlePick":
            # Oracle for NeedlePick (using the waypoint sequence)
            action = np.zeros(5)
            action[4] = -0.5
            for i, waypoint in enumerate(self._waypoints):
                if waypoint is None:
                    continue
                delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
                delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
                if np.abs(delta_pos).max() > 1:
                    delta_pos /= np.abs(delta_pos).max()
                scale_factor = 0.4
                delta_pos *= scale_factor
                action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
                # If we are close enough, mark this waypoint as done.
                if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                    self._waypoints[i] = None
                break
            return action
        elif self.task_type == "GauzeRetrieve":
            # Oracle for GauzeRetrieve.
            # four waypoints executed in sequential order
            action = np.zeros(5)
            action[4] = -0.5
            for i, waypoint in enumerate(self._waypoints):
                if waypoint is None:
                    continue
                delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
                if np.abs(delta_pos).max() > 1:
                    delta_pos /= np.abs(delta_pos).max()
                scale_factor = 0.6
                delta_pos *= scale_factor
                action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., waypoint[4]])
                if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                    self._waypoints[i] = None
                break
            return action
        else:
            return np.zeros(5)
    def _meet_contact_constraint_requirement(self):
        if self.task_type == "GauzeRetrieve":
        
            # add a contact constraint to the grasped object to make it stable
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.0025 * self.SCALING
        return True  # mimic the dVRL setting
    
if __name__ == "__main__":
    # Example: run unified environment with a chosen task.
    env = UnifiedPsmEnv(task_type="GauzeRetrieve", render_mode='human')
    # Optionally test the environment loop:
    env.test()
    env.close()
    time.sleep(2)
