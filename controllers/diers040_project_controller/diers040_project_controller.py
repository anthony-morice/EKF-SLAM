"""
Anthony Dierssen-Morice (diers040)
project_controller
"""
from controller import Supervisor 
from controller import CameraRecognitionObject
from controller import Robot, Motor, DistanceSensor
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
import numpy as np

'''
***********************************************************************
Constants
***********************************************************************
'''

TIME_LIMIT = 125 # stop simulation after 125 seconds even if robot is not at goal 
NUM_LANDMARKS = 200 # number of landmarks in world
# robot
AXEL_LENGTH = 0.057 # 57 mm
WHEEL_RADIUS = 0.020 # 20 mm
MAX_WHEEL_SPEED = 6.28 # < 6.28 rad/s
# lidar
LIDAR_FOV = (np.pi / 2, -1 * np.pi / 2) # rad
LIDAR_MIN, LIDAR_MAX = 0.05, 3 # in meters
# plant noise
STD_M = 0.05 # std deviation in x and y of relative position measurments
SIGMA_M = np.diag([STD_M ** 2] * 2)
STD_V = 0.0033 # std deviation in translational velocity control signal
# STD_V has been reduced to improve localization performance for the EKF based SLAM
STD_OMEGA = np.pi / 60. # std in angular velocity control signal
SIGMA_U = np.diag([STD_V ** 2, STD_OMEGA ** 2])
# controller parameters
UPDATE_FREQ = 200 # update with visible landmark measurements every 100 timesteps
PLOT_FREQ = 100 # update plot every 50 timesteps
D_THRESH = 0.02 # at goal when within 2 cm
T_THRESH = 0.015 # correct orientation when within 0.015 rad
MAX_SPEED = np.pi * WHEEL_RADIUS * 2 - 0.002 # < 3.14 rad/s
MAX_ROTATION_SPEED = 0.6 # rad/s
ACCEL_RATE_ANGULAR = 0.075
ACCEL_RATE_TRANS = 0.0075
FRONT_PADDING = 0.1 # stop if obstacle inside of 10 cm of robot
SIDE_PADDING = 0.075 # maintain 7.5 cm of side padding robot
TURN_TIGHT = 1.25 # \in [1,) control how tight robot hugs corners (bigger == tighter) 

'''
***********************************************************************
Initial Conditions
***********************************************************************
'''

G_p_goal = np.array([-2.89, 0], dtype=float) # goal position
x_hat_t = np.array([2.89, 0, 0], dtype=float) # robot starting state
Sigma_x_t = np.zeros((3,3), dtype=float) # certain about starting state
u = np.array([0, 0], dtype=float) # initial control signals

# augment state vector to include landmark coords
x_hat_t = np.hstack((x_hat_t, np.zeros((NUM_LANDMARKS * 2,))))
Sigma_x_t = np.block([[Sigma_x_t, np.zeros((3, NUM_LANDMARKS * 2))],\
                      [np.zeros((NUM_LANDMARKS * 2, 3)),\
                       np.eye(NUM_LANDMARKS * 2) * -1]])
# when landmark covariance is -1 then we have yet to see that landmark
landmarks_seen = 0
landmarks = {} # empty because no landmarks have been seen
'''
***********************************************************************
EKF functions
***********************************************************************
'''

'''
Calculate the robot state estimation and variance for the next timestep
@param x_hat_t # robot position and orientation
@param Sigma_x_t # estimation uncertainty
@param u # control signals
@param Sigma_u # uncertainty in control signals
@param dt # timestep
'''
def EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_u, dt):
  v, omega = u
  x, y, theta = x_hat_t[:3] # only update the robot pose
  ct, st = np.cos(theta), np.sin(theta)
  # propagate robot state
  x_hat_t[:3] += dt * np.array([v * ct, v * st, omega])
  # jacobian w.r.t position and pose
  J_fx = np.array([[1, 0, -1 * v * dt * st],\
                   [0, 1, v * dt * ct],\
                   [0, 0, 1]], dtype=float)
  G = np.eye(2 * NUM_LANDMARKS + 3)
  G[:3,:3] = J_fx
  # jacobian w.r.t control signals 
  J_fu = np.array([[dt * ct, 0],\
                   [dt * st, 0],\
                   [0, dt]], dtype=float)
  # progagate variance
  Sigma_x_t = G @ Sigma_x_t @ G.T
  Sigma_x_t[:3,:3] += J_fu @ Sigma_u @ J_fu.T # incorporate control noise
  return x_hat_t, Sigma_x_t

'''
Update the robot state estimation and variance based on the received measurement
@param x_hat_t # robot position and orientation
@param Sigma_x_t # estimation uncertainty
@param z # measurements
@param Sigma_m # measurements' uncertainty
@param L_index # \in [0, NUM_LANDMARKS) landmarks position in state and covariance matrix 
@param dt # timestep
'''
def EKFRelPosUpdate(x_hat_t, Sigma_x_t, z, Sigma_m, L_index, dt):
  x, y, theta = x_hat_t[:3]
  ct, st = np.cos(theta), np.sin(theta)
  # calculate measurement residual
  Rot = np.array([[ct, -1 * st], [st, ct]], dtype=float)
  start = 3 + L_index * 2
  diff = x_hat_t[start:start+2, np.newaxis] - x_hat_t[:2, np.newaxis]
  z_hat = Rot.T @ diff 
  r = z[:,np.newaxis] - z_hat
  # calculate residual covariance
  J_xyt = np.array([[-1 * ct, -1 * st, -1 * st * diff[0,0] + ct * diff[1,0]],\
                [st, -1 * ct, -1 * ct * diff[0,0] - st * diff[1,0]]], dtype=float)
  J_LxLy = np.array([[ct, st],[-1 * st, ct]], dtype=float)
  H = np.zeros((2, 3 + 2 * NUM_LANDMARKS), dtype=float)
  H[:, :3] = J_xyt
  H[:, start:start+2] = J_LxLy
  Sigma_m = np.array(Sigma_m, dtype=float)
  S = H @ Sigma_x_t @ H.T + Sigma_m
  S_inv = np.linalg.inv(S)
  # calculate Kalman gain
  K = Sigma_x_t @ H.T @ S_inv 
  # update state and covariance
  x_hat_t += (K @ r).reshape(x_hat_t.shape)
  Sigma_x_t -= Sigma_x_t @ H.T @ S_inv @ H @ Sigma_x_t
  return x_hat_t, Sigma_x_t
  
'''
***********************************************************************
Visualization Functions
***********************************************************************
'''
plt.style.use('ggplot')

def plot_cov(cov_mat, prob=0.95, num_pts=50):
  conf = chi2.ppf(0.95, df=2)
  L, V = np.linalg.eig(cov_mat)
  s1 = np.sqrt(conf*L[0])
  s2 = np.sqrt(conf*L[1])
  thetas = np.linspace(0, 2*np.pi, num_pts)
  xs = np.cos(thetas)
  ys = np.sin(thetas)
  standard_norm = np.vstack((xs, ys))
  S = np.array([[s1, 0],[0, s2]])
  scaled = np.matmul(S, standard_norm)
  R= V
  rotated = np.matmul(R, scaled)
  return rotated 

def update_plot(x_hat_t, Sigma_x_t, G_p_R):
  pts = plot_cov(Sigma_x_t[0:2,0:2])
  pts[0] += x_hat_t[0]
  pts[1] += x_hat_t[1]
  plt.scatter([pts[0,:]], [pts[1,:]])
  plt.scatter(x_hat_t[0],x_hat_t[1])
  plt.scatter(G_p_R[0],G_p_R[1], color='c')
  plt.axis('equal')

'''
***********************************************************************
Path Planning Functions
***********************************************************************
'''

'''
base class that any navigation class should extend
'''
class Controller:
  def __init__(self, robot):
    self.robot = robot
    # configure lidar
    self.lidar = robot.getDevice('lidar')
    self.lidar.enable(1)
    self.lidar.enablePointCloud()
    # configure motors
    self.motor_l = self.robot.getDevice('left wheel motor')
    self.motor_r = self.robot.getDevice('right wheel motor')
    self.motor_l.setPosition(float('inf'))
    self.motor_r.setPosition(float('inf'))
    # set initial velocity to 0
    self.trans_vel, self.cur_trans_vel = 0, 0
    self.angular_vel, self.cur_angular_vel = 0, 0
    self.updateVelocity()

  def updateVelocity(self):
    # limit acceleration
    angular_sign = np.sign(self.angular_vel - self.cur_angular_vel)
    trans_sign = np.sign(self.trans_vel - self.cur_trans_vel)
    self.cur_angular_vel += angular_sign * ACCEL_RATE_ANGULAR
    self.cur_trans_vel += trans_sign * ACCEL_RATE_TRANS
    self.cur_angular_vel = min(self.cur_angular_vel, self.angular_vel) if angular_sign == 1 else \
                           max(self.cur_angular_vel, self.angular_vel) 
    self.cur_trans_vel = min(self.cur_trans_vel, self.trans_vel) if trans_sign == 1 else \
                         max(self.cur_trans_vel, self.trans_vel)
    # convert current angular and translational vel into motor commands 
    # i.e. rotational velocity of each wheel
    # also add control noise
    correction = (self.cur_angular_vel + np.random.normal(0, STD_OMEGA)) * AXEL_LENGTH * 0.5
    noisy_trans_vel = self.cur_trans_vel + np.random.normal(0, STD_V)
    left_v = (noisy_trans_vel - correction) / WHEEL_RADIUS 
    right_v = (noisy_trans_vel + correction) / WHEEL_RADIUS
    # ensure the motor command is within the robot's control limits
    left_v, right_v = np.clip([left_v, right_v], -1 * MAX_WHEEL_SPEED, MAX_WHEEL_SPEED) 
    # set motor speeds
    self.motor_l.setVelocity(left_v)
    self.motor_r.setVelocity(right_v)

'''
A hybrid left/right turning BugZero based path planning controller
'''
class BugZero(Controller):
  def __init__(self, robot, initial_pose, G_p_goal):
    super().__init__(robot)
    self.goal = G_p_goal
    self.status = "go to goal"
    # print('Go to goal')
    self.pose = initial_pose
    self.right_turning = False 

  def getLidarReading(self):
    dists = np.array(self.lidar.getRangeImage()) # 180 degree sweep
    n = dists.shape[0]
    t_inc = np.abs(LIDAR_FOV[1] - LIDAR_FOV[0]) / n
    # convert dists to array of [d, theta] pairs
    points = np.array([np.array([dists[i], np.pi / 2 - t_inc * i]) for i in range(n)])
    return points

  def distanceToGoal(self):
    return np.linalg.norm(self.goal - self.pose[:2])

  def angleOfGoal(self):
    theta = np.arctan2(self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
    return 2 * np.pi + theta if theta < 0 else theta

  def arrived(self):
    if self.status == 'arrived at goal' or self.distanceToGoal() < D_THRESH:
      if self.status != 'arrived at goal':
        self.status = 'arrived at goal'
        print('Arrived at goal!')
      self.trans_vel = 0
      self.angular_vel = 0
      return True
    return False

  def turnTo(self, theta):
    cur_angle = self.pose[2]
    if np.abs(cur_angle - theta) >= T_THRESH:
      # determine best direction to rotate
      ccw, cw = None, None
      if cur_angle < theta:
        ccw = theta - cur_angle
        cw = cur_angle + 2 * np.pi - theta 
      else:
        ccw = 2 * np.pi - cur_angle + theta
        cw = cur_angle - theta
      vel_scale = -1 if cw < ccw else 1
      self.angular_vel = vel_scale * MAX_ROTATION_SPEED
      self.trans_vel = 0
      self.updateVelocity()
      return False
    else: # current angle == theta
      self.angular_vel = 0
      self.updateVelocity()
      return True

  def frontClear(self):
    points = self.getLidarReading()
    # determine which slice of lidar scan to consider
    fov_theta = np.arctan((AXEL_LENGTH / 2 + SIDE_PADDING) / FRONT_PADDING)
    start = np.argmin(np.abs(points[:,1] - fov_theta)) 
    stop = points.shape[0] - start
    # see if any obstacles are obstructing movement
    for dist, theta in points[start:stop]:
      if dist * np.cos(theta) <= FRONT_PADDING: 
        if self.status != 'follow obstacle':
          # determine turn direction
          points = self.getLidarReading()
          d_left = np.mean(np.clip(points[:points.shape[0]//2, 0], LIDAR_MIN, LIDAR_MAX))
          d_right = np.mean(np.clip(points[points.shape[0]//2:, 0], LIDAR_MIN, LIDAR_MAX))
          direction = 1 if d_left > d_right else -1
          self.right_turning = direction == 1
          self.status = 'follow obstacle'
          # print('follow obstacle')
          # print('right_turning: ', self.right_turning)
        return False
    return True

  def toGoalClear(self):
    cur_angle = self.pose[2]
    goal_angle = self.angleOfGoal()
    # transform goal_angle to robot frame
    ccw, cw = None, None
    if cur_angle < goal_angle:
      ccw = goal_angle - cur_angle
      cw = cur_angle + 2 * np.pi - goal_angle
    else:
      ccw = 2 * np.pi - cur_angle + goal_angle
      cw = cur_angle - goal_angle
    r_goal_angle = -1 * cw if cw < ccw else ccw
    # determine if path is clear in goal direction
    if LIDAR_FOV[1] < r_goal_angle < LIDAR_FOV[0]:
      points = self.getLidarReading()
      d = points[np.argmin(np.abs(points[:,1] - r_goal_angle)), 0] 
      if d > 1.5 * FRONT_PADDING or self.distanceToGoal() <= 1.5 * FRONT_PADDING:
        return True
    return False

  def leftClear(self):
    # check if immediate left of robot is clear
    points = self.getLidarReading()
    d = np.mean(points[0:4,0]) # use mean of small slice to average out lidar noise
    return d > SIDE_PADDING * 2

  def rightClear(self):
    # check if immediate right of robot is clear
    points = self.getLidarReading()
    d = np.mean(points[-4:,0]) # use mean of small slice to average out lidar noise
    return d > SIDE_PADDING * 2

  def spinInPlace(self, rate):
    self.trans_vel = 0
    self.angular_vel = rate
    self.updateVelocity()

  def moveForwards(self):
    if self.frontClear():
      self.trans_vel = MAX_SPEED
    else:
      self.trans_vel = 0
    self.updateVelocity()

  def followObstacle(self):
    lined_up = np.abs(self.angleOfGoal() - self.pose[2]) < T_THRESH 
    if self.toGoalClear() and lined_up and self.frontClear():
      self.status = 'go to goal'
      # print('Go to goal')
    elif not self.frontClear(): # turn until front is clear
      direction = 1 if self.right_turning else -1
      self.spinInPlace(direction * MAX_ROTATION_SPEED)
    elif self.right_turning:
      if self.rightClear(): # make a right turn
        self.angular_vel = -1 * MAX_ROTATION_SPEED
        self.trans_vel = MAX_SPEED / TURN_TIGHT
      else: # front is clear and right is wall
        self.angular_vel = 0
        self.trans_vel = MAX_SPEED
      self.updateVelocity()
    else: # left turning
      if self.leftClear(): # make a left turn
        self.angular_vel = MAX_ROTATION_SPEED
        self.trans_vel = MAX_SPEED / TURN_TIGHT
      else: # front is clear and left is wall
        self.angular_vel = 0
        self.trans_vel = MAX_SPEED
      self.updateVelocity()

  def update(self, x_hat_t):
    self.pose = x_hat_t
    if self.status == 'go to goal':
      # point towards goal
      if self.turnTo(self.angleOfGoal()):
        self.moveForwards()
    elif self.status == 'follow obstacle':
      self.followObstacle()
    elif self.status == 'arrived at goal':
      self.updateVelocity() # needed to decelerate robot to complete stop
    else:
      print('Should never print')
    return np.array([self.cur_trans_vel, self.cur_angular_vel], dtype=float)

'''
***********************************************************************
Robot Setup
***********************************************************************
'''
# create the Robot instance.
robot = Supervisor()
robot_node = robot.getSelf()
# configure camera
camera = robot.getDevice('camera')
camera.enable(1)
if camera.hasRecognition():
  camera.recognitionEnable(1)
  camera.enableRecognitionSegmentation()
else:
  print("Robot camera does not have recognition")
timestep = int(robot.getBasicTimeStep()) # in ms
dt = float(timestep) / 1000
# create controller
controller = BugZero(robot, x_hat_t[:3], G_p_goal)

'''
***********************************************************************
Main Loop
***********************************************************************
'''

count, arrived_count, timer = 0, 0, 0 # timing related counters
while robot.step(timestep) != -1:
  timer += timestep / 1000
  # check if robot is at goal position
  if controller.arrived() or timer > TIME_LIMIT:
    arrived_count += 1
    if arrived_count > 100:
      break
  # EKF Propagate
  x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, SIGMA_U, dt)
  x_hat_t[2] %= 2 * np.pi # rescale orientation to range [0, 2pi)
  # EKF Update
  if count % UPDATE_FREQ == 0:
    #print('EKF update')
    # extract landmark information from camera
    rec_objs = camera.getRecognitionObjects()
    num_rec_objs = camera.getRecognitionNumberOfObjects()
    for i in range(num_rec_objs):
      landmark_i = robot.getFromId(rec_objs[i].get_id())
      R_pose_L = landmark_i.getPose(robot_node)
      R_p_Lx, R_p_Ly = R_pose_L[3], R_pose_L[7]
      # corrupt relative position with measurement noise
      z_i = np.array([R_p_Lx + np.random.normal(0, STD_M), R_p_Ly + np.random.normal(0,STD_M)])
      if landmark_i.getId() not in landmarks.keys(): # first time seeing this landmark 
        # add entry to landmarks lookup table
        landmarks[landmark_i.getId()] = landmarks_seen 
        landmarks_seen += 1
        # initialize landmark position in state vector with 
        # observed position in global reference frame
        start = 3 + landmarks[landmark_i.getId()] * 2
        ct, st = np.cos(x_hat_t[2]), np.sin(x_hat_t[2])
        Rot = np.array([[ct, -1 * st], [st, ct]], dtype=float)
        x_hat_t[start:start+2] = Rot @ z_i + x_hat_t[:2]
        uncertainty = 2 * (np.diag(Sigma_x_t[:2,:2]) + np.diag(SIGMA_M))
        Sigma_x_t[start:start+2, start:start+2] = np.diag(uncertainty)
      L_index = landmarks[landmark_i.getId()]
      # update robot state and error
      x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, z_i, SIGMA_M, L_index, dt)
  if count % PLOT_FREQ == 0:
    update_plot(x_hat_t, Sigma_x_t, robot_node.getPosition())
  # update control signals
  u = controller.update(x_hat_t[:3])
  count += 1

'''
***********************************************************************
Cleanup and display statistics
***********************************************************************
'''
# ensure robot is fully stopped
motor_l = robot.getDevice('left wheel motor')
motor_r = robot.getDevice('right wheel motor')
motor_l.setVelocity(0)
motor_r.setVelocity(0)
# print some results
G_p_R = robot_node.getPosition()
print('final values: ')
print('  G_p_R_t', G_p_R, flush=True)
print('  x_hat_t', x_hat_t[:3], flush=True)
plt.show()
