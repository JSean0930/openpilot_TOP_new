from collections import deque
import math
import numpy as np

from cereal import log
from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]
LOW_SPEED_Y_NN = [12, 3, 1, 0]

LAT_PLAN_MIN_IDX = 5
LATERAL_LAG_MOD = 0.1

def get_predicted_lateral_jerk(lat_accels, t_diffs):
  # compute finite difference between subsequent model_data.acceleration.y values
  # this is just two calls of np.diff followed by an element-wise division
  lat_accel_diffs = np.diff(lat_accels)
  lat_jerk = lat_accel_diffs / t_diffs
  # return as python list
  return lat_jerk.tolist()

def sign(x):
  return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)

def get_lookahead_value(future_vals, current_val):
  if len(future_vals) == 0:
    return current_val

  same_sign_vals = [v for v in future_vals if sign(v) == sign(current_val)]

  # if any future val has opposite sign of current val, return 0
  if len(same_sign_vals) < len(future_vals):
    return 0.0

  # otherwise return the value with minimum absolute value
  min_val = min(same_sign_vals + [current_val], key=lambda x: abs(x))
  return min_val

# At a given roll, if pitch magnitude increases, the
# gravitational acceleration component starts pointing
# in the longitudinal direction, decreasing the lateral
# acceleration component. Here we do the same thing
# to the roll value itself, then passed to nnff.
def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)

class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki,
                             k_f=self.torque_params.kf)
    self.update_limits()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    # Twilsonco's Lateral Neural Network Feedforward
    self.use_nn = CI.has_lateral_torque_nn
    self.use_lateral_jerk = False  # self.param_s.get_bool("TorqueLateralJerk")

    # These are only used if use_nn or use_lateral_jerk is True,
    # but they're defined here since use_lateral_jerk can be toggled while onroad

    # Instantaneous lateral jerk changes very rapidly, making it not useful on its own,
    # however, we can "look ahead" to the future planned lateral jerk in order to guage
    # whether the current desired lateral jerk will persist into the future, i.e.
    # whether it's "deliberate" or not. This lets us simply ignore short-lived jerk.
    # Note that LAT_PLAN_MIN_IDX is defined above and is used in order to prevent
    # using a "future" value that is actually planned to occur before the "current" desired
    # value, which is offset by the steerActuatorDelay.
    self.friction_look_ahead_v = [1.4, 2.0]  # how many seconds in the future to look ahead in [0, ~2.1] in 0.1 increments
    self.friction_look_ahead_bp = [9.0, 30.0]  # corresponding speeds in m/s in [0, ~40] in 1.0 increments
    # precompute time differences between ModelConstants.T_IDXS
    self.t_diffs = np.diff(ModelConstants.T_IDXS)
    self.desired_lat_jerk_time = CP.steerActuatorDelay + LATERAL_LAG_MOD

    if self.use_nn or self.use_lateral_jerk:
      # Scaling the lateral acceleration "friction response" could be helpful for some.
      # Increase for a stronger response, decrease for a weaker response.
      nnff_lateral_jerk_factor = 1.0  # TODO-SP: replace with ---> float(self.param_s.get("NNFFLateralJerkFactor", encoding="utf8"))
      nnff_lateral_jerk_factor = max(0.0, min(1.0, nnff_lateral_jerk_factor))
      self.lat_jerk_friction_factor = 0.4 * nnff_lateral_jerk_factor
      # Increasing lat accel friction factor to account for any decrease of the lat jerk friction factor from default
      self.lat_accel_friction_factor = 0.7 + (0.3 * (1.0 - nnff_lateral_jerk_factor))  # in [0, 3], in 0.05 increments. 3 is arbitrary safety limit

    if self.use_nn:
      self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
      # NN model takes current v_ego, lateral_accel, lat accel/jerk error, roll, and past/future/planned data
      # of lat accel and roll
      # Past value is computed using previous desired lat accel and observed roll
      self.torque_from_nn = CI.get_ff_nn
      self.nn_friction_override = CI.lat_torque_nn_model.friction_override

      # setup future time offsets
      self.future_times = [0.3, 0.6, 1.0, 1.5]  # seconds in the future
      self.nn_future_times = [i + self.desired_lat_jerk_time for i in self.future_times]
      self.nn_future_times_np = np.array(self.nn_future_times)

      # setup past time offsets
      self.past_times = [-0.3, -0.2, -0.1]
      history_check_frames = [int(abs(i)*100) for i in self.past_times]
      self.history_frame_offsets = [history_check_frames[0] - i for i in history_check_frames]
      self.lateral_accel_desired_deque = deque(maxlen=history_check_frames[0])
      self.roll_deque = deque(maxlen=history_check_frames[0])
      self.past_future_len = len(self.past_times) + len(self.nn_future_times)

  def update_lateral_lag(self, lag):
    self.desired_lat_jerk_time = max(0.01, lag) + LATERAL_LAG_MOD

    if self.use_nn:
      self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]
      self.nn_future_times_np = np.array(self.nn_future_times)

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params),
                        self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, model_data=None):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    nn_log = None

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
      roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY
      actual_curvature = actual_curvature_vm
      curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))

      desired_lateral_accel = desired_curvature * CS.vEgo ** 2

      # desired rate is the desired rate of change in the setpoint, not the absolute desired curvature
      # desired_lateral_jerk = desired_curvature_rate * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      low_speed_factor = np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y if not self.use_nn else LOW_SPEED_Y_NN)**2
      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature

      lookahead_lateral_jerk = 0

      model_good = model_data is not None and len(list(model_data.orientation.x)) >= CONTROL_N
      if model_good and (self.use_nn or self.use_lateral_jerk):
        # prepare "look-ahead" desired lateral jerk
        lookahead = np.interp(CS.vEgo, self.friction_look_ahead_bp, self.friction_look_ahead_v)
        friction_upper_idx = next((i for i, val in enumerate(ModelConstants.T_IDXS) if val > lookahead), 16)
        predicted_lateral_jerk = get_predicted_lateral_jerk(model_data.acceleration.y, self.t_diffs)
        desired_lateral_jerk = (np.interp(self.desired_lat_jerk_time, ModelConstants.T_IDXS, model_data.acceleration.y) - desired_lateral_accel) / self.desired_lat_jerk_time
        lookahead_lateral_jerk = get_lookahead_value(predicted_lateral_jerk[LAT_PLAN_MIN_IDX:friction_upper_idx], desired_lateral_jerk)

      lat_accel_friction_factor = self.lat_accel_friction_factor if lookahead_lateral_jerk != 0.0 else 1.0

      gravity_adjusted_lateral_accel = desired_lateral_accel - roll_compensation

      if self.use_nn and model_good:
        pitch = 0.0
        roll = params.roll
        if model_data is not None and hasattr(model_data.orientation, 'y') and len(model_data.orientation.y) > 0:
          pitch = self.pitch.update(model_data.orientation.y[0])
          roll = roll_pitch_adjust(roll, pitch)
        self.roll_deque.append(roll)
        self.lateral_accel_desired_deque.append(desired_lateral_accel)

        adjusted_future_times = [t + 0.5*CS.aEgo*(t/max(CS.vEgo, 1.0)) for t in self.nn_future_times]
        past_rolls = [self.roll_deque[min(len(self.roll_deque)-1, i)] for i in self.history_frame_offsets]
        future_rolls = [roll_pitch_adjust(np.interp(t, ModelConstants.T_IDXS, model_data.orientation.x) + roll,
          np.interp(t, ModelConstants.T_IDXS, model_data.orientation.y) + pitch) for t in adjusted_future_times]
        past_lateral_accels_desired = [self.lateral_accel_desired_deque[min(len(self.lateral_accel_desired_deque)-1, i)]
          for i in self.history_frame_offsets]
        T_IDXS = [float(x) for x in ModelConstants.T_IDXS[:CONTROL_N]]
        lateral_accels = [float(y) for y in list(model_data.acceleration.y)[:CONTROL_N]]
        future_planned_lateral_accels = [np.interp(float(t), T_IDXS, lateral_accels) for t in adjusted_future_times]

        error = setpoint - measurement
        friction_input = lat_accel_friction_factor * error + self.lat_jerk_friction_factor * lookahead_lateral_jerk
        nn_input = [CS.vEgo, desired_lateral_accel, friction_input, roll] + \
          past_lateral_accels_desired + future_planned_lateral_accels + \
          past_rolls + future_rolls
        nn_torque = self.torque_from_nn(nn_input)
        nn_lat_accel = self.lateral_accel_from_torque(nn_torque, self.torque_params)

        if self.nn_friction_override:
          ff = nn_lat_accel
        else:
          friction_torque = get_friction(friction_input, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)
          friction_lat_accel = self.lateral_accel_from_torque(friction_torque, self.torque_params)
          ff = nn_lat_accel + friction_lat_accel

        ff -= self.torque_params.latAccelOffset

        pid_log.error = float(setpoint - measurement)

        nn_log = nn_input
      else:
        ff = gravity_adjusted_lateral_accel
        ff -= self.torque_params.latAccelOffset

        if self.use_lateral_jerk:
          friction_input = lat_accel_friction_factor * (desired_lateral_accel - actual_lateral_accel) + self.lat_jerk_friction_factor * lookahead_lateral_jerk
          friction_torque = get_friction(friction_input, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)
          ff += self.lateral_accel_from_torque(friction_torque, self.torque_params)
        else:
          ff += get_friction(desired_lateral_accel - actual_lateral_accel, lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

        pid_log.error = float(setpoint - measurement)

      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_lataccel = self.pid.update(pid_log.error,
                                        feedforward=ff,
                                        speed=CS.vEgo,
                                        freeze_integrator=freeze_integrator)
      output_torque = self.torque_from_lateral_accel(output_lataccel, self.torque_params)

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.d = float(self.pid.d)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(-output_torque)
      pid_log.actualLateralAccel = float(actual_lateral_accel)
      pid_log.desiredLateralAccel = float(desired_lateral_accel)
      pid_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))
      if nn_log is not None:
        pid_log.nnLog = [float(x) for x in nn_log] if isinstance(nn_log, (list, tuple)) else [float(nn_log)]
      else:
        pid_log.nnLog = [0.0]

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
