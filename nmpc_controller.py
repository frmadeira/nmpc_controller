# !/usr/bin/env python3

# Global imports
import casadi as ca
import numpy as np
import time
from scipy import interpolate
import math
import statistics as stat
import matplotlib.pyplot as plt
import os
import json

# ROS imports
import rospy
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
import tf, tf2_ros

def variable_definition(t_traj):
    # Frequency and time-step 
    f = 5                             # Controller frequency [Hz]
    h = 1/f                           # Time-step [s]
    # ROS rate
    r = rospy.Rate(f)

    # Vehicle dimensions and constraints
    L = 0.633                       # Wheel-base [m]
    v_max = 0.8                     # Maximum velocity [m/s]
    a_max = 0.5                     # Maximum acceleration [m/s^2]

    # NMPC variables
    # Number of steps in NMPC
    step_tot = int(t_traj[-1]/h)
    defined_N = 20                  # Prediction horizon length
    if step_tot < defined_N:
        N = step_tot
    else:
        N = defined_N
    # Weighting coefficient
    lambda_1 = 0.25
    # Counter
    step = 0

    # Obstacle avoidance variables
    d_safe = 0.8       # Safety distance from obstacles [m]
    voxel_size = 0.5    # Voxel cell size [m]
    max_range = 3.5       # Maximum range for occupancy map [m]

    # Replanning variable
    d_replan = 10        # Distance from reference that triggers trajectory replanning [m]

    return f, h, r, L, v_max, a_max, step_tot, N, lambda_1, step, d_safe, voxel_size, max_range, d_replan

def quaternion_to_yaw(orientation):
    # Convert quaternion orientation data to yaw angle of robot
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    theta = math.atan2(2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2))

    return theta

def trajectory_plot(x_ref, y_ref, theta_ref):
    # Plot trajectory to RViz
    target_traj_msg = Path()
    target_traj_msg.header.stamp = rospy.Time.now()
    target_traj_msg.header.frame_id = "map"
    for i in range(len(x_ref)):
        pose = PoseStamped()
        pose.pose.position.x = x_ref[i]
        pose.pose.position.y = y_ref[i]
        q = tf.transformations.quaternion_from_euler(0, 0, theta_ref[i])
        pose.pose.orientation = Quaternion(*q)
        target_traj_msg.poses.append(pose)

    return target_traj_msg

def occupancy_callback(data):
    # Get occupancy gridmap through callback function
    global occupancy_message
    occupancy_message = data

class NMPCController():
    def __init__(self,h,N,L,v_max,a_max,lambda_1,d_safe,voxel_size, max_range):
        # --- Initialization of all needed variables ---
        # NMPC related variables
        self.h = h
        self.N = N
        self.lambda_1 = lambda_1

        # Vehicle related variables
        self.L = L
        self.v_max = v_max
        self.a_max = a_max

        # Obstacle avoidance related variables
        self.d_safe = d_safe
        self.voxel_size = voxel_size
        self.max_range = max_range

        # Calling solver setup
        self.solver_setup()
   
    def solver_setup(self):
        # --- State and control variables ---
        # Variables
        # X = [x0, y0, theta0, vr0, vl0, ar0, al0, (...), xN, yN, thetaN, vrN, vlN, arN, alN]
        self.n = 7
        self.X = ca.MX.sym('X',self.N*self.n)
       
        # Bounds on variables
        lbx = [-np.inf,-np.inf,-np.inf,-self.v_max,-self.v_max,-self.a_max,-self.a_max]*self.N
        self.lbx = np.array(lbx)
        ubx = [np.inf,np.inf,np.inf,self.v_max,self.v_max,self.a_max,self.a_max]*self.N
        self.ubx = np.array(ubx)

        # --- Constraints ---
        # Vehicle dynamics constraints
        # Position (x, y, theta)
        gx = self.X[0::self.n][1:] - self.X[0::self.n][:-1] - 0.5*self.h*(((self.X[3::self.n][1:] + self.X[4::self.n][1:])/2)*np.cos(self.X[2::self.n][1:]) + ((self.X[3::self.n][:-1] + self.X[4::self.n][:-1])/2)*np.cos(self.X[2::self.n][:-1]))
        gy = self.X[1::self.n][1:] - self.X[1::self.n][:-1] - 0.5*self.h*(((self.X[3::self.n][1:] + self.X[4::self.n][1:])/2)*np.sin(self.X[2::self.n][1:]) + ((self.X[3::self.n][:-1] + self.X[4::self.n][:-1])/2)*np.sin(self.X[2::self.n][:-1]))
        gtheta = self.X[2::self.n][1:] - self.X[2::self.n][:-1] - 0.5*self.h*(((self.X[3::self.n][1:] - self.X[4::self.n][1:])/self.L) + ((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L))
        # Velocity (v_r, v_l)
        gv_r = self.X[3::self.n][1:] - self.X[3::self.n][:-1] - 0.5*self.h*(self.X[5::self.n][1:] + self.X[5::self.n][:-1])
        gv_l = self.X[4::self.n][1:] - self.X[4::self.n][:-1] - 0.5*self.h*(self.X[6::self.n][1:] + self.X[6::self.n][:-1])
        # Positive linear velocity
        gv = self.X[3::self.n][:-1] + self.X[4::self.n][:-1] - 0.02  
        # Minimum angular velocity
        gw_min = (((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L)*((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L)) - 0.0025
        # Maximum angular velocity
        gw_max = 0.81 - (((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L)*((self.X[3::self.n][:-1] - self.X[4::self.n][:-1])/self.L))

        self.g = ca.vertcat(gx, gy, gtheta, gv_r, gv_l, gv, gw_min, gw_max)

    def main_NMPC(self,step,step_tot,x_ref,y_ref,theta_ref,vr_ref,vl_ref,ar_ref,al_ref,X0,occupancy_message,x_obstacles_all,y_obstacles_all,position_pub):
        # --- Constraints ---
        data_time_start = time.time()
        # Get laser data message 
        data = np.array(occupancy_message.data)
        data[data != 99] = 0

        map_origin_x = occupancy_message.info.origin.position.x
        map_origin_y = occupancy_message.info.origin.position.y
        map_resolution = occupancy_message.info.resolution
        map_width = occupancy_message.info.width
        map_height = occupancy_message.info.height
        
        # Reshape data into a grid
        grid = np.reshape(data,(map_width,map_height))
        
        # Reshaping of the view window to include greater distance ahead
        heading_angle = X0[2] 
        normalized_heading_angle = heading_angle % (2*np.pi)
        
        # print('Heading angle = ', heading_angle*(180/np.pi))
        # print('Normalized heading angle = ', normalized_heading_angle*(180/np.pi))

        data_min = 0.2
        data_max = 1 - data_min

        # Moving primarily in positive-x direction
        if ((15*np.pi/8) <= normalized_heading_angle < 2*np.pi) or (0 <= normalized_heading_angle < (np.pi/8)):
            a_min = round((map_width/2)-((data_min*self.max_range)/map_resolution))
            a_max = round((map_width/2)+((data_max*self.max_range)/map_resolution)+1)
            b_min = round((map_height/2)-(self.max_range/(2*map_resolution)))
            b_max = round((map_height/2)+(self.max_range/(2*map_resolution))+1)

        # Moving in positive-x and negative-y directions
        elif ((np.pi/8) <= normalized_heading_angle < (3*np.pi/8)):
            a_min = round((map_width/2)-((data_min*self.max_range)/map_resolution))
            a_max = round((map_width/2)+((data_max*self.max_range)/map_resolution)+1)
            b_min = round((map_height/2)-((data_min*self.max_range)/map_resolution))
            b_max = round((map_height/2)+((data_max*self.max_range)/map_resolution)+1)

        # Moving primarily in negative-y direction
        elif (3*np.pi/8) <= normalized_heading_angle < (5*np.pi/8):
            a_min = round((map_width/2)-(self.max_range/(2*map_resolution)))
            a_max = round((map_width/2)+(self.max_range/(2*map_resolution))+1)
            b_min = round((map_height/2)-((data_min*self.max_range)/map_resolution))
            b_max = round((map_height/2)+((data_max*self.max_range)/map_resolution)+1)

        # Moving primarily in negative-x and negative-y direction
        elif (5*np.pi/8) <= normalized_heading_angle < (7*np.pi/8):
            a_min = round((map_width/2)-((data_max*self.max_range)/map_resolution))
            a_max = round((map_width/2)+((data_min*self.max_range)/map_resolution)+1)
            b_min = round((map_height/2)-((data_min*self.max_range)/map_resolution))
            b_max = round((map_height/2)+((data_max*self.max_range)/map_resolution)+1)

        # Moving primarily in negative-x direction
        elif (7*np.pi/8) <= normalized_heading_angle < (9*np.pi/8):
            a_min = round((map_width/2)-((data_max*self.max_range)/map_resolution))
            a_max = round((map_width/2)+((data_min*self.max_range)/map_resolution)+1)
            b_min = round((map_height/2)-(self.max_range/(2*map_resolution)))
            b_max = round((map_height/2)+(self.max_range/(2*map_resolution))+1)

        # Moving primarily in negative-x positive-y direction
        elif (9*np.pi/8) <= normalized_heading_angle < (11*np.pi/8):
            a_min = round((map_width/2)-((data_max*self.max_range)/map_resolution))
            a_max = round((map_width/2)+((data_min*self.max_range)/map_resolution)+1)
            b_min = round((map_height/2)-((data_max*self.max_range)/map_resolution))
            b_max = round((map_height/2)+((data_min*self.max_range)/map_resolution)+1)

        # Moving primarily in positive-y direction
        elif (11*np.pi/8) <= normalized_heading_angle < (13*np.pi/8):
            a_min = round((map_width/2)-(self.max_range/(2*map_resolution)))
            a_max = round((map_width/2)+(self.max_range/(2*map_resolution))+1)
            b_min = round((map_height/2)-((data_max*self.max_range)/map_resolution))
            b_max = round((map_height/2)+((data_min*self.max_range)/map_resolution)+1)

        # Moving primarily in positive-x positive-y direction
        elif (13*np.pi/8) <= normalized_heading_angle < (15*np.pi/8):
            a_min = round((map_width/2)-((data_min*self.max_range)/map_resolution))
            a_max = round((map_width/2)+((data_max*self.max_range)/map_resolution)+1)
            b_min = round((map_height/2)-((data_max*self.max_range)/map_resolution))
            b_max = round((map_height/2)+((data_min*self.max_range)/map_resolution)+1)

        # Simple square
        else:
            a_min = round((map_width/2)-(self.max_range/(2*map_resolution)))
            a_max = round((map_width/2)+(self.max_range/(2*map_resolution))+1)
            b_min = round((map_height/2)-(self.max_range/(2*map_resolution)))
            b_max = round((map_height/2)+(self.max_range/(2*map_resolution))+1)

        grid = grid[b_min:b_max,a_min:a_max]

        # Get obstacle coordinates and define obstacle avoidance constraints over prediction horizon
        i_coords, j_coords = np.where(grid)

        if i_coords != []:
            x_obstacles = map_origin_x + (a_min + j_coords) * map_resolution
            y_obstacles = map_origin_y + (b_min + i_coords) * map_resolution

            # Boundaries of the point cloud
            min_x = np.min(x_obstacles)
            max_x = np.max(x_obstacles)
            min_y = np.min(y_obstacles)
            max_y = np.max(y_obstacles)

            # Calculate the number of voxels in each dimension
            num_voxels_x = int(np.ceil((max_x - min_x) / self.voxel_size))
            num_voxels_y = int(np.ceil((max_y - min_y) / self.voxel_size))

            # Calculate the indices of obstacle positions in terms of i and j
            i_indices = np.round((x_obstacles[:] - min_x) / self.voxel_size).astype(int)
            j_indices = np.round((y_obstacles[:] - min_y) / self.voxel_size).astype(int)

            # Create a boolean grid with the appropriate shape
            grid = np.zeros((num_voxels_x+1, num_voxels_y+1), dtype=bool)

            # Set the grid values at the obstacle indices to True
            grid[i_indices, j_indices] = 1

            # Calculate the midpoints of voxel cells where grid is True
            i_coords, j_coords = np.where(grid)
            midpoints = np.column_stack(((i_coords + 0.5) * self.voxel_size + min_x, (j_coords + 0.5) * self.voxel_size + min_y))
            voxel_midpoints = midpoints
            # print(voxel_midpoints)
            # print('voxel_midpoints = ', len(voxel_midpoints))

            # Calculate distances to each obstacle point for every point in the prediction horizon
            obstacle_constraints = ca.MX.zeros((self.N*len(voxel_midpoints),1))
            k = 0
            for i in range(self.N):
                for voxel_midpoint in voxel_midpoints:
                    obstacle_constraints[k] = (self.X[0::self.n][i]-voxel_midpoint[0])*(self.X[0::self.n][i]-voxel_midpoint[0]) + (self.X[1::self.n][i]-voxel_midpoint[1])*(self.X[1::self.n][i]-voxel_midpoint[1]) - (self.d_safe*self.d_safe)
                    k += 1
            # print(obstacle_constraints) 

            # Add the points to the marker
            marker_id_ = 0
            marker = Marker()
            marker.ns = "obstacle"
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            
            for voxel_midpoint in voxel_midpoints:
                marker.pose.position.x = voxel_midpoint[0]
                marker.pose.position.y = voxel_midpoint[1]
                marker.pose.orientation.w = 0
                marker.id = marker_id_
                marker_id_ += 1
                position_pub.publish(marker)

        else:
            # Case that no obstacle point is detected in the range
            obstacle_constraints = ca.MX.zeros((self.N,1))
       
        data_time_end = time.time()
        data_time = data_time_end - data_time_start

        # obstacle_constraints = ca.MX.zeros((self.N,1))
        # data_time = 0

        # Bounds on constraints
        multiple_constraints = 8
        single_constraints = 5
        l_obs = obstacle_constraints.size()[0]
        self.lbg = np.zeros((self.N-1)*multiple_constraints + single_constraints + l_obs)
        self.ubg = np.zeros((self.N-1)*multiple_constraints + single_constraints + l_obs)
        self.ubg[(-l_obs-5-(3*(self.N-1))):(-l_obs-5)] = np.inf     # Positive velocity constraints
        self.ubg[(-l_obs):] = np.inf                                # Obstacle avoidance constraints

        # Set the constraints that vary at each time-step
        if step == 0:
            self.g = ca.vertcat(self.g, self.X[0::self.n][0] - X0[0], self.X[1::self.n][0] - X0[1], self.X[2::self.n][0] - X0[2], self.X[3::self.n][0] - X0[3], self.X[4::self.n][0] - X0[4], obstacle_constraints) # self.X[0::self.n][-1] - x_ref[step+(self.N-1)], self.X[1::self.n][-1] - y_ref[step+(self.N-1)])
        else:
            self.g = self.g[:(multiple_constraints*(self.N-1))]
            self.g = ca.vertcat(self.g, self.X[0::self.n][0] - X0[0], self.X[1::self.n][0] - X0[1], self.X[2::self.n][0] - X0[2], self.X[3::self.n][0] - X0[3], self.X[4::self.n][0] - X0[4], obstacle_constraints)

        # --- Cost function ---
        J = 0
        for i in range(self.N):
            # Reference following error cost
            ref_follow_error = ((self.X[0::self.n][i] - x_ref[i+step])*(self.X[0::self.n][i] - x_ref[i+step])) + ((self.X[1::self.n][i] - y_ref[i+step])*(self.X[1::self.n][i] - y_ref[i+step]))
            
            # Successive control cost
            if i != (self.N-1):
                successive_error = ((self.X[5::self.n][i+1]-self.X[5::self.n][i])*(self.X[5::self.n][i+1]-self.X[5::self.n][i]))+((self.X[6::self.n][i+1]-self.X[6::self.n][i])*(self.X[6::self.n][i+1]-self.X[6::self.n][i]))
            successive_weight = self.lambda_1

            # Cost function calculation
            J += (ref_follow_error + successive_weight*successive_error)
       
        # --- Initial guess ---
        init_guess = []
        if step == 0:
            init_guess = [[x_ref[:self.N][i], y_ref[:self.N][i], theta_ref[:self.N][i], vr_ref[:self.N][i], vl_ref[:self.N][i], ar_ref[:self.N][i], al_ref[:self.N][i]] for i in range(len(x_ref[:self.N]))]
            init_guess = ca.vertcat(*init_guess)
        else:
            init_guess = ca.vertcat(self.opt_states[7:], x_ref[step+self.N-1], y_ref[step+self.N-1], theta_ref[step+self.N-1], vr_ref[step+self.N-1], vl_ref[step+self.N-1], ar_ref[step+self.N-1], al_ref[step+self.N-1])
        
        # --- Solver ---
        solver_time_start = time.time()
        # Setup
        opts = {'ipopt.print_level': 0,
                'print_time': 0,
                'expand': 1}
        prob = {'f': J, 'x': self.X, 'g': self.g}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Solution
        solution = solver(x0=init_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        
        solver_time_end = time.time()
        solver_time = solver_time_end - solver_time_start
        
        # Optimal control retrieval
        if step == step_tot-self.N:
            vr_opt = solution['x'][3::self.n]
            vl_opt = solution['x'][4::self.n]
        else:
            vr_opt = solution['x'][3::self.n][1]
            vl_opt = solution['x'][4::self.n][1]

        # Initial guess for next steps
        self.opt_states = solution['x']

        return vr_opt, vl_opt, x_obstacles_all, y_obstacles_all, solver_time, data_time

def controller_node(x_traj, y_traj, theta_traj, t_traj, vr_traj, vl_traj, ar_traj, al_traj):
    # --- Variable definition for scenario considered ---
    f, h, r, L, v_max, a_max, step_tot, N, lambda_1, step, d_safe, voxel_size, max_range, d_replan = variable_definition(t_traj)

    # omega = np.zeros(len(vr_traj))
    # for i in range(len(omega)):
    #     omega[i] = np.sqrt(((vr_traj[i]-vl_traj[i])/L)*((vr_traj[i]-vl_traj[i])/L))
    # print('Maximum omega = ', max(omega))
    # for i in range(len(omega)):
    #     if max(omega) == omega[i]:
    #         print('At maximum omega, we have vr = {} m/s and vl = {} m/s.'.format(vr_traj[i],vl_traj[i]))

    # --- Initialization ---
    # Location from tf
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # Publishers
    vel_pub = rospy.Publisher("/robot0/cmd_vel", Twist, queue_size = 1)
    traj_pub = rospy.Publisher("/traj_ines", Path, queue_size = 10)
    position_pub = rospy.Publisher("/actual_position", Marker, queue_size = 10)

    # Subscribers
    costmap_sub = rospy.Subscriber("/robot0/move_base/local_costmap/costmap", OccupancyGrid, occupancy_callback)

    # --- Reference trajectory ---
    interp_x = interpolate.interp1d(t_traj, x_traj, kind='linear')
    interp_y = interpolate.interp1d(t_traj, y_traj, kind='linear')
    interp_theta = interpolate.interp1d(t_traj, theta_traj, kind='linear')
    interp_vr = interpolate.interp1d(t_traj, vr_traj, kind='linear')
    interp_vl = interpolate.interp1d(t_traj, vl_traj, kind='linear')
    interp_ar = interpolate.interp1d(t_traj, ar_traj, kind='linear')
    interp_al = interpolate.interp1d(t_traj, al_traj, kind='linear')

    # Get interpolated data
    t_ref = np.arange(t_traj[0], t_traj[-1], h)
    x_ref = interp_x(t_ref)
    y_ref = interp_y(t_ref)
    theta_ref = interp_theta(t_ref)
    vr_ref = interp_vr(t_ref)
    vl_ref = interp_vl(t_ref)
    ar_ref = interp_ar(t_ref)
    al_ref = interp_al(t_ref)
    
    # --- Initialization ---
    nmpc = NMPCController(h,N,L,v_max,a_max,lambda_1,d_safe,voxel_size,max_range)

    # --- Setting variables for post-processing or plots to RViz ---
    target_traj_msg = trajectory_plot(x_ref, y_ref, theta_ref)
    cycle_times = []
    solver_times = []
    data_times = []
    marker_id = 0
    x_error = np.zeros((step_tot,),dtype=float)
    y_error = np.zeros((step_tot,),dtype=float)
    theta_error = np.zeros((step_tot,),dtype=float)
    X0_all = []

    # --- Setting variables for obstacle avoidance ---
    x_obstacles_all = []
    y_obstacles_all = []

    # Wait until all needed data is available
    time.sleep(1)

    # --- Completing the given trajectory ---
    while step <= step_tot and not rospy.is_shutdown():
        print('\nstep = ', step)

        # Final commands given in open-loop
        if step >= step_tot-(N-1):
            # Count time in each cycle to guarantee that h is large enough for online implementation
            start_time = time.time()

            # --- Set current state as initial condition X(n) = X0 ---
            # X0 = [x0, y0, theta0, vr0, vl0, ar0, al0]
            try:
                pose = tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Exception while getting robot pose: {e}")
                rospy.sleep(0.1)
            X0 = [pose.transform.translation.x, pose.transform.translation.y, quaternion_to_yaw(pose.transform.rotation), vr_opt[N-(step_tot-step)-1], vl_opt[N-(step_tot-step)-1]]

            X0_all.append(X0[:3])
            
            # --- Apply optimal controls to robot ---
            if step == step_tot:
                v_opt = 0
                w_opt = 0

                vel_msg = Twist() 
                vel_msg.linear.x = v_opt
                vel_msg.angular.z = w_opt
                vel_pub.publish(vel_msg)

                step += 1

            else: 
                vr_robot = (X0[3]+vr_opt[N-(step_tot-step)])/2
                vl_robot = (X0[4]+vl_opt[N-(step_tot-step)])/2

                v_opt = (vr_robot+vl_robot)/2
                w_opt = (vr_robot-vl_robot)/L

                vel_msg = Twist() 
                vel_msg.linear.x = v_opt
                vel_msg.angular.z = w_opt
                vel_pub.publish(vel_msg)

                # --- Collection of data for error plots ---
                x_error[step] = x_ref[step] - X0[0]
                y_error[step] = y_ref[step] - X0[1]
                theta_error[step] = (-theta_ref[step]) - X0[2]

                # --- Trajectory advancement ---
                step += 1

                # --- Plots to RViz ---
                # Plot trajectory to RViz
                traj_pub.publish(target_traj_msg)

                # Publish actual position and trajectory reference point to RViz
                if step != step_tot:
                    marker = Marker()
                    marker.ns = "following"
                    marker.header.frame_id = "map"
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.id = marker_id
                    marker_id += 1

                    marker.pose.position.x = X0[0]
                    marker.pose.position.y = X0[1]
                    marker.pose.orientation.w = 0

                    position_pub.publish(marker)
                
                    marker = Marker()
                    marker.ns = "trajectory"
                    marker.header.frame_id = "map"
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.id = marker_id
                    marker_id += 1

                    marker.pose.position.x = x_ref[step]
                    marker.pose.position.y = y_ref[step]
                    marker.pose.orientation.w = 0
            
                    position_pub.publish(marker)

                r.sleep()

        # Normal closed-loop commands
        else:

            # Count time in each cycle to guarantee that h is large enough for online implementation
            start_time = time.time()

            # --- Set current state as initial condition X(n) = X0 ---
            # X0 = [x0, y0, theta0, vr0, vl0, ar0, al0]
            flag = 1
            while flag:
                try:
                    pose = tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"Exception while getting robot pose: {e}")
                else: 
                    flag = 0
            
            if step == 0:
                # Initial position from reference trajectory
                X0 = [pose.transform.translation.x, pose.transform.translation.y, quaternion_to_yaw(pose.transform.rotation), 0, 0]
            else:
                X0 = [pose.transform.translation.x, pose.transform.translation.y, quaternion_to_yaw(pose.transform.rotation), vr_opt, vl_opt]

            X0_all.append(X0[:3])
            
            # --- Solve the optimal control problem and get optimum controls for next sampling period ---
            # Solve the optimal control problem
            vr_opt, vl_opt, x_obstacles_all, y_obstacles_all, solver_time, data_time = nmpc.main_NMPC(step,step_tot,x_ref,y_ref,theta_ref,vr_ref,vl_ref,ar_ref,al_ref,X0,occupancy_message,x_obstacles_all,y_obstacles_all,position_pub)

            # --- Apply optimal controls to robot ---
            if step == step_tot-N:
                vr_robot = (X0[3]+vr_opt[0])/2
                vl_robot = (X0[4]+vl_opt[0])/2

                v_opt = (vr_robot+vl_robot)/2
                w_opt = (vr_robot-vl_robot)/L
            else:
                vr_robot = (X0[3]+vr_opt)/2
                vl_robot = (X0[4]+vl_opt)/2

                v_opt = (vr_robot+vl_robot)/2
                w_opt = (vr_robot-vl_robot)/L

            vel_msg = Twist() 
            vel_msg.linear.x = v_opt
            vel_msg.angular.z = w_opt
            vel_pub.publish(vel_msg)

            # --- Collection of data for error plots ---
            x_error[step] = x_ref[step] - X0[0]
            y_error[step] = y_ref[step] - X0[1]
            theta_error[step] = (-theta_ref[step]) - X0[2]

            # --- Trajectory advancement ---
            step += 1

            end_time = time.time()
            # print('Loop time = ', end_time-start_time)
            cycle_times.append(end_time-start_time)
            solver_times.append(solver_time)
            data_times.append(data_time)

            # --- Plots to RViz ---
            # Plot trajectory to RViz
            traj_pub.publish(target_traj_msg)

            # Publish actual position and trajectory reference point to RViz
            marker = Marker()
            marker.ns = "following"
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.id = marker_id
            marker_id += 1

            marker.pose.position.x = X0[0]
            marker.pose.position.y = X0[1]
            marker.pose.orientation.w = 0
            position_pub.publish(marker)
        
            marker = Marker()
            marker.ns = "trajectory"
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.id = marker_id
            marker_id += 1

            marker.pose.position.x = x_ref[step]
            marker.pose.position.y = y_ref[step]
            marker.pose.orientation.w = 0
    
            position_pub.publish(marker)
            
            r.sleep()

        # --- Calculate need for trajectory replanning ---
        # Replanning done when distance to reference is greater than safety
        if step <= step_tot - (N-1):
            dist = np.sqrt(((X0[0]-x_ref[step])*(X0[0]-x_ref[step]))+((X0[1]-y_ref[step])*(X0[1]-y_ref[step])))
            if dist > d_replan:
                print('Trajectory replanning has been deployed!')
                break
    
    print('\nDone!\n')

    # Print all data related to the scenario to the console
    data = {
        "N": N,
        "f": f,
        "v_max": v_max,
        "a_max": a_max,
        "lambda": lambda_1,
        "d_safe": d_safe,
        "d_replan": d_replan,
        "max_range": max_range,
        "voxel_size": voxel_size,
        "step_tot": step_tot,
        "t_ref": list(t_ref),
        "x_ref": list(x_ref),
        "y_ref": list(y_ref),
        "theta_ref": list(theta_ref),
        "vr_ref": list(vr_ref),
        "vl_ref": list(vl_ref),
        "ar_ref": list(ar_ref),
        "al_ref": list(al_ref),
        "X0_all": list(X0_all),
        "x_error": list(x_error),
        "y_error": list(y_error),
        "theta_error": list(theta_error),
        "X0_last": list(X0[:3]),
        "last_x_ref": x_ref[-1],
        "last_y_ref": y_ref[-1],
        "last_theta_ref": theta_ref[-1],
        "cycle_times": cycle_times,
        "solver_times": solver_times,
        "data_times": data_times,
        "avg_loop_time": stat.mean(cycle_times),
        "max_loop_time": max(cycle_times)}
            
    print(data)

if __name__ == '__main__':
    try:
        controller_node()
    except rospy.ROSInterruptException:
        pass