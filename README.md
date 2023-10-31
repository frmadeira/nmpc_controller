# nmpc_controller
**Overview:**

This repository contains the source code for a Nonlinear Model Predictive Control (NMPC) controller designed for trajectory tracking and obstacle avoidance. The controller can perform avoiding action from both static and moving obstacles. It is developed using Python 3.10.11. and is intended for use as a local planner for a differential drive robot in a factory environment, though adaptable for use with different scenarios and vehicle dynamics.

**Features:**

The developed controller can accurately track a feasible trajectory and avoid both static and moving obstacles. It is agnostic to the shape and dimension of the obstacles and needs no _a priori_ knowledge of its trajectory. Further, it is easily tunable to better accommodate the specifics of the use case considered.

**Installation:**

To use this NMPC controller, clone this repository to your local machine:
```
git clone https://github.com/frmadeira/nmpc_controller.git
cd nmpc_controller
```

**Usage and configuration:**

To use the code considering a differential drive vehicle, please only change the variables in the function variable_definition(). These variables are as follows: controller's operating frequency (f); prediction horizon length (N); cost function weighting coefficient (lambda); vehicle's wheel-base (L); vehicle's maximum velocity (v_max); vehicle's maximum acceleration (a_max); safety distance (d_safe); voxel cell's dimensions (voxel_size); maximum range for occupancy map (max_range); minimum value for view-window of occupancy map (data_min); maximum value for view-window of occupancy map (data_max) and distance from reference that triggers replanning (d_replan)

**Contact:**

For any questions regarding the project, its usage or any suggestions, feel free to contact: francisco.raposo.madeira@tecnico.ulisboa.pt

**Acknowledgements:**

This project was developed as part of a Master's thesis for Instituto Superior Técnico, University of Lisbon.
