#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from typing import Any

import capnp
from cereal import messaging, log, car
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D


# ===============================================================
# 視覺優先（可切換模式）版本：維持原訊息結構、最小改動
# 模式支援：
#   - vision_only    : 幾乎純視覺（保留極低速安全覆蓋）
#   - hybrid (default): 視覺為主、雷達小幅混合（嚴格匹配，高速不混合）
#   - radar_strong   : 偏雷達（允許較強覆蓋與高速混合）
# 透過 Params() 鍵 "RadarVisionMode" 設定；未設則為 "hybrid"
# ===============================================================


# ======== 視覺採用與雷達覆蓋的「基線」門檻（hybrid 預設） ========
VISION_PROB_MIN_BASE = 0.30       # 視覺最低採用門檻（> 即生成視覺 lead）
RADAR_OVERRIDE_PROB_BASE = 0.45   # 視覺強於此值就不允許雷達覆蓋（hybrid）

_LEAD_ACCEL_TAU = 1.5
SPEED, ACCEL = 0, 1
V_EGO_STATIONARY = 4.0
RADAR_TO_CENTER = 2.7
RADAR_TO_CAMERA = 1.52


def _get_mode_from_params() -> str:
  """
  從 Params 讀取模式（RadarVisionMode），允許：
    "vision_only" / "hybrid" / "radar_strong"
  若未設定或不合法，回退 "hybrid"
  """
  try:
    val = Params().get("RadarVisionMode")
    if val is None:
      return "hybrid"
    s = val.decode("utf-8").strip().lower()
    if s in ("vision_only", "hybrid", "radar_strong"):
      return s
  except Exception:
    pass
  return "hybrid"


def _mode_thresholds(mode: str):
  """
  依模式回傳 (VISION_PROB_MIN, RADAR_OVERRIDE_PROB, allow_high_speed_refine)
   - VISION_PROB_MIN: 視覺採用門檻
   - RADAR_OVERRIDE_PROB: 視覺強於此值不覆蓋（雷達覆蓋/混合的上限）
   - allow_high_speed_refine: 是否允許在高速也做雷達混合
  """
  if mode == "vision_only":
    # 視覺容易採用、幾乎不讓雷達覆蓋；不在高速混合
    return (0.28, 0.35, False)
  if mode == "radar_strong":
    # 視覺要稍高才用；允許雷達在視覺弱時更常覆蓋；高速亦可混合
    return (0.35, 0.70, True)
  # hybrid（預設）
  return (VISION_PROB_MIN_BASE, RADAR_OVERRIDE_PROB_BASE, False)


def _alpha_by_speed(mode: str, v_ego: float) -> float:
  """
  雷達混合權重 alpha（0~1，越小越偏視覺）
   - vision_only  : 更偏視覺，低速 0.25 → 高速 0.0
   - hybrid       : 低速 0.30 → 中速 0.22 → 高速 0.0（不允許高速混合）
   - radar_strong : 低速 0.45 → 高速 0.20（高速仍允許混合）
  """
  if mode == "vision_only":
    return float(np.interp(v_ego, [0.0, 10.0, 20.0], [0.25, 0.18, 0.0]))
  if mode == "radar_strong":
    return float(np.interp(v_ego, [0.0, 15.0, 35.0], [0.45, 0.33, 0.20]))
  # hybrid
  return float(np.interp(v_ego, [0.0, 10.0, 20.0], [0.30, 0.22, 0.0]))


class KalmanParams:
  def __init__(self, dt: float):
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float):
    self.dRel = d_rel
    self.yRel = y_rel
    self.vRel = v_rel
    self.vLead = v_lead
    self.measured = measured

    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": self.is_potential_fcw(model_prob),
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  # 收窄低速保守覆蓋：減少雷達主導概率，但保留安全網
  def potential_low_speed_lead(self, v_ego: float):
    return (
      abs(self.yRel) < 0.8 and
      (v_ego < 2.5) and
      (1.0 < self.dRel < 22.0) and
      bool(self.measured)
    )

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x - mu) / b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track]):
  """
  更嚴格的合理性檢查（距離允差 20%/≥3m；速度允差 7m/s 或 v_ego+v_rel>4m/s），
  減少錯配導致的雷達干擾。
  """
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])
    return prob_d * prob_y * prob_v

  track = max(tracks.values(), key=prob)

  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist) * 0.20, 3.0])
  vel_sane = (abs(track.vRel + v_ego - lead.v[0]) < 7.0) or (v_ego + track.vRel > 4.0)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_msg.prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track],
             lead_msg: capnp._DynamicStructReader, model_v_ego: float,
             low_speed_override: bool = True) -> dict[str, Any]:
  """
  視覺優先決策（可切換模式）：
    1) 視覺機率 > VISION_PROB_MIN：先用視覺
       - 若機率 <= RADAR_OVERRIDE_PROB 且符合匹配條件，依模式進行「小幅混合」
       - 其中 hybrid/vision_only：高速可禁止混合（allow_high_speed_refine=False）
    2) 備援：維持原版（雷達匹配→純視覺）
    3) 低速保守覆蓋：條件收窄，僅作安全網
  """
  mode = _get_mode_from_params()
  VISION_PROB_MIN, RADAR_OVERRIDE_PROB, allow_high_speed_refine = _mode_thresholds(mode)
  lead_dict = {'status': False}

  # ---- 1. 視覺優先 ----
  if ready and (lead_msg.prob > VISION_PROB_MIN):
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)

    # 視覺較弱才允許混合
    can_refine_prob = (lead_msg.prob <= RADAR_OVERRIDE_PROB)
    can_refine_speed = allow_high_speed_refine or (v_ego < 20.0)   # hybrid/vision_only 高速不混合
    if len(tracks) > 0 and can_refine_prob and can_refine_speed:
      trk = match_vision_to_track(v_ego, lead_msg, tracks)
      if trk is not None:
        alpha = _alpha_by_speed(mode, v_ego)
        # 視覺為主的小幅混合（僅調 dRel/vRel/aLeadK）
        lead_dict["dRel"] = float((1 - alpha) * lead_dict["dRel"] + alpha * trk.dRel)
        lead_dict["vRel"] = float((1 - alpha) * lead_dict["vRel"] + alpha * trk.vRel)
        lead_dict["vLead"] = float(v_ego + lead_dict["vRel"])
        lead_dict["vLeadK"] = lead_dict["vLead"]
        if abs(trk.aLeadK) < 6.0:
          lead_dict["aLeadK"] = float((1 - alpha) * lead_dict["aLeadK"] + alpha * trk.aLeadK)

  else:
    # ---- 2. 備援路徑（與原版一致） ----
    if len(tracks) > 0 and ready and lead_msg.prob > .5:
      track = match_vision_to_track(v_ego, lead_msg, tracks)
    else:
      track = None

    if track is not None:
      lead_dict = track.get_RadarState(lead_msg.prob)
    elif (track is None) and ready and (lead_msg.prob > .5):
      lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)

  # ---- 3. 低速保守覆蓋（安全網，條件已收窄） ----
  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  return lead_dict


class RadarD:
  def __init__(self, delay: float = 0.0):
    self.current_time = 0.0
    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)
    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL)) + 1)
    self.last_v_ego_frame = -1
    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False
    self.ready = False

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9 * max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    # 更新/維護 tracks
    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in rr.points}

    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    for ids, rpt in ar_pts.items():
      v_lead = rpt[2] + self.v_ego_hist[0]
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    # 準備輸出
    self.radar_state_valid = sm.all_checks()
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    # 取得 model v_ego（若缺回退 carState）
    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego

    # 視覺 lead 融合
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      self.radar_state.leadOne = get_lead(self.v_ego, self.ready, self.tracks,
                                          leads_v3[0], model_v_ego, low_speed_override=True)
      self.radar_state.leadTwo = get_lead(self.v_ego, self.ready, self.tracks,
                                          leads_v3[1], model_v_ego, low_speed_override=False)

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None
    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  sm = messaging.SubMaster(['modelV2', 'carState', 'liveTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])
  RD = RadarD(CP.radarDelay)

  while 1:
    sm.update()
    RD.update(sm, sm['liveTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
