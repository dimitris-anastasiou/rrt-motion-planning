# RRT Motion Planning for UR5 Robotic Arm

## Objective
This project focused on implementing the Rapidly-Exploring Random Tree (RRT) algorithm for sampling-based motion planning to control a UR5 robotic arm. Utilizing the MoveIt! library in a ROS2 environment, the goal was to develop a robust motion planning solution capable of navigating the robotic arm through various environments, avoiding obstacles, and reaching specified end-effector poses.

| Video 1 | Video 2 |
|---|---|
| [![YouTube Video 1](https://img.youtube.com/vi/MTEMFD5bnlk/0.jpg)](https://www.youtube.com/watch?v=MTEMFD5bnlk) | [![YouTube Video 2](https://img.youtube.com/vi/iff-jk5FrEs/0.jpg)](https://www.youtube.com/watch?v=iff-jk5FrEs) |

## Key Features

1. **RRT Algorithm Implementation:**
   - RRT algorithm to compute joint trajectories, ensuring the arm's end-effector could reach desired positions while avoiding obstacles.

2. **Collision Avoidance:**
   - Collision detection and avoidance using data from an `/obstacles` topic, ensuring all planned paths were free from collisions.

3. **Inverse Kinematics (IK):**
   - IK functions to convert end-effector goals into joint space configurations, essential for effective motion planning.

4. **Path Optimization:**
   - Path shortcutting and resampling techniques to refine the planned trajectories, improving execution efficiency and ensuring smooth movement.

## Project Structure
- `rrt_motion_planning.py`: Main script for implementing RRT-based motion planning for the UR5 robotic arm.
