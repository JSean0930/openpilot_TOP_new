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


# ===================== 視覺優先的主要開關與門檻 =============================
VISION_PROB_MIN = 0.35     # 視覺最小採用門檻：> 立即用純視覺生成 lead。調低→更容易採用視覺（可能增加誤檢）；調高→更保守，可能在視覺弱時出現空檔
VISION_PROB_STRONG = 0.60  # 視覺較強時才嘗試用雷達做微調（refine），避免弱視覺被過度修正。調高→更少使用雷達微調；調低→更常以雷達細修（但仍以視覺為主）。
LOW_SPEED_PROB_CUTOFF = 0.20  # 視覺很弱且低速時，才允許雷達低速補強覆蓋。調高→更容易觸發低速雷達覆蓋；調低→更倚賴視覺。

# === 其他參數（保留原始語意） ===============================================
_LEAD_ACCEL_TAU = 1.5
SPEED, ACCEL = 0, 1
V_EGO_STATIONARY = 4.0
RADAR_TO_CENTER = 2.7
RADAR_TO_CAMERA = 1.52


class KalmanParams:
  """Kalman 濾波器參數：對 DT_MDL 用插值取得 K 增益"""
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
  """單一雷達追蹤軌跡（保留，用於速度/距離的輔助修正與低速保守覆蓋）"""
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

    # 小加速度→回到較長時間常數；大加速度→快速衰減
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
      "fcw": model_prob > .9,
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float):
    # 僅在極低速與視覺弱時當作保守覆蓋條件
    return abs(self.yRel) < 1.0 and (v_ego < V_EGO_STATIONARY) and (0.75 < self.dRel < 25)


# ===================== 視覺領先：生成與（可選）微調 ============================
def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def _sane_match(v_ego: float, lead: capnp._DynamicStructReader, trk: Track) -> bool:
  """合理性檢查：距離&速度近似一致"""
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA
  dist_sane = abs(trk.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  vel_sane = (abs(trk.vRel + v_ego - lead.v[0]) < 10) or (v_ego + trk.vRel > 3)
  return dist_sane and vel_sane


def pick_best_track_for_lead(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track]) -> Track | None:
  """以（距離/橫向/速度）的拉普拉斯似然挑最佳 track；若不合理則回 None"""
  if not tracks:
    return None
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c: Track):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])
    return prob_d * prob_y * prob_v

  candidate = max(tracks.values(), key=prob)
  return candidate if _sane_match(v_ego, lead, candidate) else None


def vision_lead_dict(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float) -> dict[str, Any]:
  """純視覺生成 lead 字典（視覺優先的主角）"""
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - RADAR_TO_CAMERA),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": lead_msg.prob > 0.9,   # 可替換為 TTC 邏輯
    "modelProb": float(lead_msg.prob),
    "status": True,
    "radar": False,               # 標示來源為視覺
    "radarTrackId": -1,
    "radarRefined": False,        # 是否曾用雷達微調
  }


def refine_with_radar(base: dict[str, Any], v_ego: float, lead_msg: capnp._DynamicStructReader, trk: Track) -> dict[str, Any]:
  """
  用雷達做「輕量微調」而不改變「視覺為主」的事實：
  - 僅當視覺 prob 較高且配對合理時，微調 dRel / vRel / vLead
  - 保留 base['radar']=False，但標註 radarRefined=True 以利下游除錯
  """
  # 以雷達距離/相對速度做小幅混合，視覺為主、雷達為輔（alpha 可調）
  alpha = 0.25  # 越小→越相信視覺
  v_rel_radar = trk.vRel
  d_rel_radar = trk.dRel

  # 混合距離與相對速度
  base["dRel"] = float((1 - alpha) * base["dRel"] + alpha * d_rel_radar)
  base["vRel"] = float((1 - alpha) * base["vRel"] + alpha * v_rel_radar)
  base["vLead"] = float(v_ego + base["vRel"])
  base["vLeadK"] = base["vLead"]

  # 若雷達加速度估計可信（幅度合理），適度帶入
  if abs(trk.aLeadK) < 6.0:
    base["aLeadK"] = float((1 - alpha) * base["aLeadK"] + alpha * trk.aLeadK)

  base["radarRefined"] = True
  base["radarTrackId"] = trk.identifier
  return base


def get_lead_vision_first(v_ego: float, tracks: dict[int, Track],
                          lead_msg: capnp._DynamicStructReader, model_v_ego: float) -> dict[str, Any]:
  """
  視覺優先的單 lead 融合步驟：
  1) 只要視覺 prob > VISION_PROB_MIN，就直接生成視覺 lead（主體）。
  2) 視覺較強（> VISION_PROB_STRONG）且找到合理雷達 track，才以雷達做「微調」。
  3) 若視覺很弱（<= LOW_SPEED_PROB_CUTOFF）且低速，才允許雷達低速覆蓋。
  """
  prob = float(lead_msg.prob)
  # Case A：常態—用視覺
  if prob > VISION_PROB_MIN:
    base = vision_lead_dict(lead_msg, v_ego, model_v_ego)
    if prob > VISION_PROB_STRONG:
      trk = pick_best_track_for_lead(v_ego, lead_msg, tracks)
      if trk is not None:
        base = refine_with_radar(base, v_ego, lead_msg, trk)
    return base

  # Case B：視覺很弱 → 僅在低速時允許雷達做保守覆蓋
  if prob <= LOW_SPEED_PROB_CUTOFF and v_ego < V_EGO_STATIONARY and len(tracks) > 0:
    low_speed_tracks = [c for c in tracks.values() if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)
      # 以雷達直接產生（此時標示 radar=True、來源確為雷達）
      return closest_track.get_RadarState(model_prob=prob)

  # Case C：什麼都沒有
  return {"status": False}


# ===================== RadarD 主體（以視覺為主） ============================
class RadarD:
  def __init__(self, delay: float = 0.0):
    self.current_time = 0.0
    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    # 是否收得到 modelV2 不再作為採用視覺的硬條件（但沒有就沒資料）
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    # 更新 tracks
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

    # 取 model v_ego（若缺則回退 carState）
    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego

    # 以「視覺優先」計算 leadOne/leadTwo
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 0:
      # leadOne：允許雷達微調
      self.radar_state.leadOne = get_lead_vision_first(self.v_ego, self.tracks, leads_v3[0], model_v_ego)
      # leadTwo：若存在第二個視覺 lead，亦使用視覺優先（也可選擇不微調，以做對照）
      if len(leads_v3) > 1:
        self.radar_state.leadTwo = get_lead_vision_first(self.v_ego, self.tracks, leads_v3[1], model_v_ego)
      else:
        self.radar_state.leadTwo = {"status": False}
    else:
      # 沒有視覺 lead → 僅在低速且視覺缺失時考慮雷達保守覆蓋
      low_speed_tracks = [c for c in self.tracks.values() if c.potential_low_speed_lead(self.v_ego)]
      if len(low_speed_tracks) > 0 and self.v_ego < V_EGO_STATIONARY:
        closest_track = min(low_speed_tracks, key=lambda c: c.dRel)
        self.radar_state.leadOne = closest_track.get_RadarState(model_prob=0.0)
      else:
        self.radar_state.leadOne = {"status": False}
      self.radar_state.leadTwo = {"status": False}

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