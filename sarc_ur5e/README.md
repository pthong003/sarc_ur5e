Example of one trajectory https://drive.google.com/file/d/19W-Ewk-hg1nxgqV2UiqZmmTHHEblrUqY/view?usp=sharing

The dataset is collected with a UR5e robot and consists of 232 trajectories in total:

    traj0 to traj100: Pick up the green cylinder block.
    traj101 to traj120: Pick up the yellow cube block.
    traj121 to traj220: Pick up the yellow cube block and place it on top of the orange rectangular block.
    traj221 to traj232: Pick up the green cylinder block and place it on top of the orange rectangular block.

The data format is as follows:

    robot_state: np.ndarray((L, 15))
        This stores the robot state at each timestep.
        Format: [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked]
            x, y, z, qx, qy, qz, qw: End-effector pose expressed in the robot base frame.
            gripper_is_closed: Binary value (0 = fully open; 1 = fully closed).
            action_blocked: Binary value (1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise).

    action: np.ndarray((L, 8))
        This stores the expert action input at each timestep.
        Format: [x, y, z, roll, pitch, yaw, delta_gripper_closed, terminate]
            x, y, z: Delta changes to the robot base frame.
            roll, pitch, yaw: Delta changes to orientation.
            delta_gripper_closed: Ternary value (1 if gripper closing is triggered, -1 if opening is triggered, 0 otherwise). Note that this is different from the gripper_is_closed state in robot_state.
            terminate: Indicates whether the trajectory should be terminated.

    image: np.ndarray((L, 480, 640, 3))
        Image captured from the robot workspace.

    task: np.ndarray((L, 1))
        Stores the name of the task in natural language.

    other:
        "hand_image": np.ndarray((L, 480, 640, 3))
        "third_person_image": np.ndarray((L, 480, 640, 4))
            The first 3 channels are the same as "image", and the last channel is depth.

The dataset format follows the structure outlined in the reference:

@misc{BerkeleyUR5Website,
title = {Berkeley {UR5} Demonstration Dataset},
author = {Lawrence Yunliang Chen and Simeon Adebola and Ken Goldberg},
howpublished = {\url{https://sites.google.com/view/berkeley-ur5/home}},
}
