#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LongitudinalPlanner（E2E-only 可切換 + 簡易 FCW + 起步 Boost；檔頭布林開關）
- 保留：ACC/Blended、SNG 紅燈偵測、VTSC、油門允許、轉彎限縮
- Toyota 上/下緣：低速靈敏、高速平緩；封頂 130 km/h；線性分段
- 檔頭布林開關：
  * E2E_ONLY_ENABLED：純 E2E 主導，不混用 MPC 的 a_target
  * SNG_E2E_ENABLED：紅燈/停等自動切 ExperimentalMode
  * LAUNCH_BOOST_ENABLED：由靜止起步 0.6s 內 +0.25 m/s²（受夾限/轉彎限縮保護）
"""

# ====== 檔頭布林開關（直接改 True/False 即可）==========================================
E2E_ONLY_ENABLED     = True   # True：完全以 modelV2 主導；False：維持 ACC/Blended
SNG_E2E_ENABLED      = False    # True：紅燈/停等自動切 ExperimentalMode
LAUNCH_BOOST_ENABLED = True    # True：啟用起步瞬間 Boost

# ====== 起步 Boost 參數 ===============================================================
LAUNCH_BOOST_A = 0.25           # 每循環加上的加速度（m/s^2），建議 0.20～0.30
LAUNCH_BOOST_DURATION_S = 0.6   # 持續時間（秒），建議 0.4～0.8
LAUNCH_BOOST_V_MAX = 6.0        # 低速門檻（m/s），超過不再施加

import math
import numpy as np
from openpilot.common.params import Params
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.conversions import Conversions as CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog
from openpilot.top.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerTOP

# PFEIFER - VTSC {{
from openpilot.selfdrive.controls.vtsc import vtsc
# }} PFEIFER - VTSC

# ====== Planner/Vehicle Constants =========================================================
LON_MPC_STEP = 0.2  # 第一個規劃步長 0.2s（MPC horizon 初段時間解析度）

# 通用車型（非 Toyota）的加速上限表：縱向 a 上界 vs v_ego（m/s^2 對 m/s）
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]

# ====== Toyota：速度斷點（m/s），封頂至 130 km/h = 36.11 m/s ==========================
#   - 0~8.33 m/s（≤30 km/h）：起步靈敏（高 a_max）
#   - 8.33~25 m/s（30~90 km/h）：線性下降到 0.35（平順）
#   - 25~36.11 m/s（90~130 km/h）：線性下降到 0.20（高速更舒適）
A_CRUISE_MAX_BP_TOYOTA = [0.0, 1.0, 3.0, 6.0, 8.33, 11.0, 15.0, 20.0, 25.0, 30.0, 36.11]

# 對應最大縱向加速度（m/s^2）
A_CRUISE_MAX_VALS_TOYOTA = [
  2.20,  # 0.00
  2.09,  # 1.00
  1.88,  # 3.00
  1.55,  # 6.00
  1.30,  # 8.33 ≈ 30 km/h
  1.15,  # 11.00
  0.92,  # 15.00
  0.63,  # 20.00
  0.35,  # 25.00 = 90 km/h
  0.283, # 30.00 ≈ 108 km/h
  0.20   # 36.11 = 130 km/h（封頂）
]

# ====== Toyota：減速（煞車）下限（m/s^2，負值）同 BP，低速柔和→高速逐步加強 =========
#   - 0~8.33 m/s：-1.20 → -1.40（都會低速跟車不點頭）
#   - 8.33~25 m/s：線性加強至 -2.00（中速穩）
#   - 25~36.11 m/s：線性加強至 -2.40（高速穩定/安全）
A_CRUISE_MIN_VALS_TOYOTA = [
  -1.20,  # 0.00
  -1.25,  # 1.00
  -1.30,  # 3.00
  -1.35,  # 6.00
  -1.40,  # 8.33 ≈ 30 km/h
  -1.55,  # 11.00
  -1.70,  # 15.00
  -1.85,  # 20.00
  -2.00,  # 25.00 = 90 km/h
  -2.20,  # 30.00 ≈ 108 km/h
  -2.40   # 36.11 = 130 km/h
]

# 控制輸出長度（下游控制環長度）的時間座標（與 ModelConstants 對齊的前綴）
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

# 模型油門允許判斷（更寬鬆的 gating）
ALLOW_THROTTLE_THRESHOLD = 0.3   # 模型判斷踩油門的機率門檻（允許節氣門）
MIN_ALLOW_THROTTLE_SPEED = 4.0   # 低速例外（<= 此速時放寬油門允許）

# 轉彎時的總加速度限制（sqrt(ax^2 + ay^2) <= a_total_max），以速度分段給定
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

# 紅燈偵測濾波門檻與參數（依模型路徑長度）
THRESHOLD = 0.5
CRUISING_SPEED = 5.0  # m/s
PLANNER_TIME = 10.0   # s（→ 50m 門檻）

# ====== Helper lookups ===================================================================
def get_max_accel(v_ego: float) -> float:
  """通用車型：依 v_ego 線性插值取縱向加速上限（m/s^2）。"""
  return float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS))

def get_max_accel_toyota(v_ego: float) -> float:
  """Toyota：速度對應的加速上緣（m/s^2），分段線性，封頂 130 km/h。"""
  return float(np.interp(v_ego, A_CRUISE_MAX_BP_TOYOTA, A_CRUISE_MAX_VALS_TOYOTA))

def get_min_accel_toyota(v_ego: float) -> float:
  """Toyota：速度對應的減速下緣（m/s^2，負值），分段線性，封頂 130 km/h。"""
  return float(np.interp(v_ego, A_CRUISE_MAX_BP_TOYOTA, A_CRUISE_MIN_VALS_TOYOTA))

def get_coast_accel(pitch: float) -> float:
  """
  估算「不給油時」的滑行加速度（含重力坡度項與滾阻/風阻近似項）。
  pitch 單位為弧度；回傳 m/s^2，為負號代表減速。
  """
  return np.sin(pitch) * -5.65 - 0.3  # 以實數據擬合（工具：compute_coast_accel.py）

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  依現有側向加速度（藉由方向盤角近似）限制縱向加速度上界，避免轉彎中過度加速。
  備註：此 ay 估算方法為簡化，理想應使用 VehicleModel。
  回傳：與輸入 a_target 同格式 [a_min, a_max]，但 a_max 可能被限縮。
  """
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)  # 速度→總加速度上限
  # ay ≈ v^2 * yaw_rate ≈ v^2 * (steer_angle / steer_ratio / wheelbase)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
  return [a_target[0], min(a_target[1], a_x_allowed)]

# ====== E2E-only: 簡易 FCW（TTC + 距離 + 逼近速度，含遲滯） ===========================
def _ttc(v_rel: float, d_rel: float, eps: float = 1e-3) -> float:
  """相對速度為負(逼近)且距離正時，回傳 TTC 秒；否則回傳 +inf。"""
  if v_rel < -eps and d_rel > eps:
    return d_rel / (-v_rel)
  return float('inf')

def simple_fcw_e2e(lead, v_ego: float, prev_on: bool) -> bool:
  """
  E2E-only 下的簡易 FCW：
  觸發條件（任一成立）：
    - TTC < 1.2s 且 距離 < 18m 且 v_rel < -0.5 m/s
    - TTC < 0.9s
  解除條件（遲滯）：
    - TTC > 1.6s 或 距離 > 25m 或 v_rel > -0.2 m/s
  """
  if not getattr(lead, "status", False) or v_ego < 1.0:
    return False

  d = float(lead.dRel)
  vrel = float(lead.vRel)
  ttc = _ttc(vrel, d)

  on = False
  if (ttc < 1.2 and d < 18.0 and vrel < -0.5) or (ttc < 0.9):
    on = True

  # 遲滯：已經亮著時，需要更安全才關
  if prev_on and not on:
    if (ttc > 1.6) or (d > 25.0) or (vrel > -0.2):
      on = False
    else:
      on = True

  return on

# ====== Planner Class ====================================================================
class LongitudinalPlanner(LongitudinalPlannerTOP):
  """
  長控規劃器（縱向）：
  - 管理 ACC / Blended (Experimental) / E2E-only 模式
  - 整合雷達前車、模型軌跡（x/v/a/j）、VTSC 目標速
  - 控制輸出：速度/加速度/jerk 序列、是否停車、FCW 標誌等
  """

  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(CP, dt=dt)
    self.mpc.mode = 'acc'           # MPC 內部初始為 ACC，與 experimental 切換分離
    LongitudinalPlannerTOP.__init__(self)

    # 狀態變數
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    # 規劃目標狀態（濾波速度、欲望加速度等）
    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    # 完整軌跡（供下游控制器使用）
    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0

    # 個性化跟車/權重等（保留原功能）
    self.params = Params()
    self.dynamic_follow = self.params.get_bool("Dynamic_Follow")

    # === 開關狀態（直接使用檔頭布林） ===
    self.e2e_only = bool(E2E_ONLY_ENABLED)
    self.sng_e2e  = bool(SNG_E2E_ENABLED)
    self.fcw_e2e_prev = False  # E2E-only FCW 遲滯狀態

    # === 起步 Boost（與 SNG 無關，獨立運作） ===
    self.launch_boost_enabled = bool(LAUNCH_BOOST_ENABLED)
    self.launch_boost_frames_total = max(1, int(LAUNCH_BOOST_DURATION_S / self.dt))
    self.launch_boost_frames_left = 0
    self._prev_standstill = False  # 全域起步偵測，不依賴 SNG

    # ==== StandstillMode（僅在 sng_e2e = True 時啟用） ====
    if self.sng_e2e:
      # 停等狀態跟蹤與邏輯防抖（過渡計數）
      self.standstill_prev = False
      self.standstill_current = False
      self.standstill_transit_counter = 0
      self.STANDSTILL_TRANSIT_FRAMES = 10

      # 兩種開啟實驗模式的來源旗標
      self.experimental_mode_active_by_standstill = False
      self.experimental_mode_active_by_traffic_light = False

      # 前車/紅燈判定用的閾值
      self.LEAD_DISTANCE_THRESHOLD = 5.0
      self.LEAD_SPEED_THRESHOLD = 3.0 * CV.KPH_TO_MS

      # 紅燈偵測濾波器（以模型前視距離是否近乎 0 代表將停止）
      self.red_light_filter = FirstOrderFilter(0, 1, self.dt)
      self.red_light_detected = False

      # 模型路徑長度與領航車狀態快取
      self.model_length = 0
      self.tracking_lead = False
      self.lead_distance = float('inf')
      self.lead_moving = False
      self.lead_velocity = 0.0

  def detect_traffic_light(self, sm, v_ego):
    """
    以模型路徑長度（最後一點 x）推估是否「需要停」（紅燈/阻擋）：
    - 若 model_length < CRUISING_SPEED * PLANNER_TIME（=50m），視為「模型預期停止」
    - 當沒有可靠前車或前車在模型視線之外，才啟用紅燈偵測（避免與前車邏輯互相干擾）
    - 使用一階濾波 + 閾值 THRESHOLD 提升穩定性
    """
    if len(sm['modelV2'].position.x) > 0:
      self.model_length = sm['modelV2'].position.x[-1]

    model_stopped = self.model_length < CRUISING_SPEED * PLANNER_TIME

    # 取用雷達前車（主/次）狀態
    lead_one = sm['radarState'].leadOne
    lead_two = sm['radarState'].leadTwo
    has_lead_primary = lead_one.status
    has_lead_secondary = lead_two.status
    self.tracking_lead = has_lead_primary or has_lead_secondary
    active_lead = lead_one if has_lead_primary else (lead_two if has_lead_secondary else None)

    # 快取前車的距離/速度/移動性（vRel > -0.5 視為在移動）
    if active_lead is not None:
      self.lead_distance = active_lead.dRel
      self.lead_moving = active_lead.vRel > -0.5
      self.lead_velocity = active_lead.vLead
    else:
      self.lead_distance = float('inf')
      self.lead_moving = False
      self.lead_velocity = 0.0

    # 僅當「沒有前車」或「前車在模型視距之外」時才考慮紅燈
    prev_red_light_detected = self.red_light_detected
    consider_red_light = not self.tracking_lead or self.lead_distance > self.model_length
    if consider_red_light:
      self.red_light_filter.update(model_stopped)
      self.red_light_detected = self.red_light_filter.x >= THRESHOLD
    else:
      self.red_light_filter.x = 0
      self.red_light_detected = False

    # 簡單日誌
    if self.red_light_detected != prev_red_light_detected:
      light_status = "RED LIGHT" if self.red_light_detected else "GREEN LIGHT"
      print(f"Traffic light status changed: {light_status}, filter value: {self.red_light_filter.x:.2f}, model length: {self.model_length:.2f}m")

    # 每 10 迭代打印一次狀態（debug）
    if hasattr(self, 'update_counter'):
      self.update_counter += 1
      if self.update_counter >= 10:
        light_status = "RED LIGHT" if self.red_light_detected else "GREEN LIGHT"
        lead_info = f"(dist: {self.lead_distance:.1f}m, v: {self.lead_velocity:.1f}m/s, moving: {self.lead_moving})" if self.tracking_lead else "(no lead)"
        print(f"Traffic light status: {light_status}, filter: {self.red_light_filter.x:.2f}, model length: {self.model_length:.2f}m, lead: {lead_info}")
        self.update_counter = 0
    else:
      self.update_counter = 0

    return self.red_light_detected

  @staticmethod
  def parse_model(model_msg):
    """
    將模型輸出（position/velocity/acceleration）插值到 MPC 時間格點 T_IDXS_MPC。
    若資料不完整，回傳零陣列；同時取出 gasPressProbs[1] 作為油門允許的依據。
    """
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    """
    規劃循環：
    1) 讀取車輛/模型/雷達/模式
    2) SNG 紅燈/前車邏輯（可選）
    3) 設定 accel_clip（Toyota 上下緣 + 轉彎限縮 + 個性化上限）
    4) 濾波 v_ego、解析模型、油門允許
    5) VTSC 融合、MPC 解（保留診斷）
    6) 輸出：E2E-only 或 ACC/Blended；起步 Boost；夾限
    7) FCW 設定、軌跡與狀態輸出
    """
    LongitudinalPlannerTOP.update(self, sm)

    # ====== SNG 紅燈/前車 → ExperimentalMode 自動切換（可選） ======
    v_ego = sm['carState'].vEgo
    red_light_detected = self.detect_traffic_light(sm, v_ego) if self.sng_e2e else False

    # —— 起步 Boost 的下降沿觸發（不依賴 SNG）——
    prev_standstill_global = self._prev_standstill
    current_standstill = sm['carState'].standstill
    self._prev_standstill = current_standstill  # 更新全域起步狀態

    if self.sng_e2e:
      self.standstill_current = current_standstill
      lead_one = sm['radarState'].leadOne
      has_lead = lead_one.status
      lead_dist = lead_one.dRel if has_lead else float('inf')
      lead_speed = lead_one.vRel + sm['carState'].vEgo if has_lead else 0.0
      lead_moving_away = has_lead and lead_dist > self.LEAD_DISTANCE_THRESHOLD and lead_speed > self.LEAD_SPEED_THRESHOLD

      if lead_moving_away and self.experimental_mode_active_by_standstill:
        self.params.put_bool_nonblocking("ExperimentalMode", False)
        self.experimental_mode_active_by_standstill = False
        print(f"Lead vehicle moving away: dist={lead_dist:.1f}m, speed={lead_speed*3.6:.1f}km/h, disabling ExperimentalMode")

      elif not red_light_detected and self.experimental_mode_active_by_traffic_light:
        self.params.put_bool_nonblocking("ExperimentalMode", False)
        self.experimental_mode_active_by_traffic_light = False
        print("Traffic light turned green: disabling ExperimentalMode")

      if self.standstill_current != getattr(self, "standstill_prev", False):
        self.standstill_transit_counter = self.STANDSTILL_TRANSIT_FRAMES
        print(f"Standstill state change: {getattr(self, 'standstill_prev', False)} -> {self.standstill_current}")

      if self.standstill_transit_counter > 0:
        self.standstill_transit_counter -= 1

        if self.standstill_transit_counter == 0:
          if self.standstill_current:
            should_enable_experimental = False
            if has_lead:
              should_enable_experimental = True
              self.experimental_mode_active_by_standstill = True
              self.experimental_mode_active_by_traffic_light = False
              print("Entering standstill with lead vehicle: Enabling ExperimentalMode")
            elif red_light_detected:
              should_enable_experimental = True
              self.experimental_mode_active_by_standstill = False
              self.experimental_mode_active_by_traffic_light = True
              print("Entering standstill at red light: Enabling ExperimentalMode")
            if should_enable_experimental:
              self.params.put_bool_nonblocking("ExperimentalMode", True)
          else:
            if self.experimental_mode_active_by_standstill or self.experimental_mode_active_by_traffic_light:
              self.params.put_bool_nonblocking("ExperimentalMode", False)
              self.experimental_mode_active_by_standstill = False
              self.experimental_mode_active_by_traffic_light = False
              print("Leaving standstill: Disabling ExperimentalMode")

      self.standstill_prev = self.standstill_current

    # 模式來源：selfdriveState.experimentalMode（true→blended/e2e, false→acc）
    self.mode = 'blended' if sm['selfdriveState'].experimentalMode else 'acc'

    # ====== 姿態（pitch）估算滑行加速度，缺資料時回退 ======
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])  # pitch 弧度
      pitch_rad = sm['carControl'].orientationNED[1]
    else:
      accel_coast = ACCEL_MAX
      pitch_rad = 0.0

    # ====== 巡航速度/開關/約束 ======
    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    reset_state = reset_state or not v_cruise_initialized
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # ====== 設定加速度夾限（acc 模式使用表格與轉彎限制；blended 放寬到 PID 上限）======
    if self.mode == 'acc':
      if self.CP.brand == "toyota":
        accel_clip = [get_min_accel_toyota(v_ego), get_max_accel_toyota(v_ego)]
      else:
        accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]

      steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
      accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)
    else:
      accel_clip = [ACCEL_MIN, ACCEL_MAX]

    if self.accel_controller.is_personality_enabled:
      max_limit = self.accel_controller._get_max_accel_for_speed(v_ego)
      if self.mode == 'acc':
        accel_clip = [accel_clip[0], max_limit]
        steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
        accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)
      else:
        accel_clip = [ACCEL_MIN, ACCEL_MAX]

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # 解析模型輸出與油門允許判斷（E2E 基礎資料）
    x, v, a, j, throttle_prob = self.parse_model(sm['modelV2'])
    e2e_v_traj = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, v) if len(v) else np.zeros_like(CONTROL_N_T_IDX)
    e2e_a_traj = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, a) if len(a) else np.zeros_like(CONTROL_N_T_IDX)
    e2e_j_traj = np.zeros_like(CONTROL_N_T_IDX)
    e2e_a_target = float(sm['modelV2'].action.desiredAcceleration)
    e2e_should_stop = bool(sm['modelV2'].action.shouldStop)

    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2],
                                             [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    # ====== VTSC 融合（若啟用且 vtsc 給的目標更低，採用較低值）======
    vtsc.update(prev_accel_constraint, v_ego, sm)
    if vtsc.active and v_cruise > vtsc.v_target:
      v_cruise = vtsc.v_target

    # ====== MPC 保留運行（診斷用途） ======
    lead_xv_0 = self.mpc.process_lead(sm['radarState'].leadOne)
    lead_xv_1 = self.mpc.process_lead(sm['radarState'].leadTwo)
    v_lead0 = lead_xv_0[0,1]
    v_lead1 = lead_xv_1[0,1]

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality,
                         v_lead0=v_lead0, v_lead1=v_lead1)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, x, v, a, j,
                    personality=sm['selfdriveState'].personality,
                    dynamic_follow=self.dynamic_follow,
                    pitch_rad=pitch_rad)

    # MPC 解（供 ACC/Blended）
    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    fcw_from_mpc = self.mpc.crash_cnt > 2 and not sm['carState'].standstill

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(
      self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
      action_t=action_t, vEgoStopping=self.CP.vEgoStopping
    )
    output_a_target_e2e = e2e_a_target
    output_should_stop_e2e = e2e_should_stop

    # ====== 輸出選擇：E2E-only 或 原 ACC/Blended ======
    if self.e2e_only:
      output_a_target = output_a_target_e2e
      self.output_should_stop = output_should_stop_e2e
      self.v_desired_trajectory = e2e_v_traj
      self.a_desired_trajectory = e2e_a_traj
      self.j_desired_trajectory = e2e_j_traj
      self.mpc.source = 'e2e'
    else:
      if self.mode == 'acc':
        output_a_target = output_a_target_mpc
        self.output_should_stop = output_should_stop_mpc
      else:
        output_a_target = min(output_a_target_mpc, output_a_target_e2e)
        self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc

    # ====== 起步瞬間 Boost（下降沿觸發；條件：允許油門 & 低速）========================
    if self.launch_boost_enabled:
      # 觸發：上一循環靜止、這一循環非靜止 → 啟動倒數
      if prev_standstill_global and (not current_standstill):
        self.launch_boost_frames_left = self.launch_boost_frames_total

      # 施加 Boost：在倒數期間、低速、允許油門時才加
      if self.launch_boost_frames_left > 0 and self.allow_throttle and v_ego < LAUNCH_BOOST_V_MAX:
        output_a_target += LAUNCH_BOOST_A
        self.launch_boost_frames_left -= 1
      elif self.launch_boost_frames_left > 0:
        # 條件不符也要結算倒數（避免卡住）
        self.launch_boost_frames_left -= 1

    # ====== 平滑夾限並裁剪輸出加速度 ======
    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx],
                                self.prev_accel_clip[idx] - 0.05,
                                self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

    # ====== 更新濾波初值（供下一輪）：Trapezoid 積分平滑 v；a_desired = 0.05s 後值 ======
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    # ====== FCW ：E2E-only 用簡易版本，否則用 mpc.crash_cnt ======
    if self.e2e_only:
      active_lead = sm['radarState'].leadOne if sm['radarState'].leadOne.status else sm['radarState'].leadTwo
      self.fcw = simple_fcw_e2e(active_lead, v_ego, self.fcw_e2e_prev)
      self.fcw_e2e_prev = self.fcw
    else:
      self.fcw = fcw_from_mpc
      if self.fcw:
        cloudlog.info("FCW triggered")

  def publish(self, sm, pm):
    """
    發布 longitudinalPlan：
    - 軌跡：speeds/accels/jerks
    - 來源：mpc.source（'acc' 或 'e2e' / 'blended'）
    - FCW、是否應停、油門允許
    """
    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
    self.publish_longitudinal_plan_top(sm, pm)
