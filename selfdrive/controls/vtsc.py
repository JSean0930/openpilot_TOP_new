# PFEIFER - VTSC Optimized

# Acknowledgements:
# Based on original VTSC implementation from move-fast and sunnypilot.

import numpy as np
from time import time
from collections import deque
from openpilot.common.params import Params

params = Params()

# 彎道控制參數
TARGET_LAT_A = 3.5  # 最大橫向加速度限制 (m/s^2) 2.5
MIN_TARGET_V = 10.0  # 最低彎道速度限制 (m/s)
HISTORY_LENGTH = 10  # 曲率歷史長度
SMOOTHING_ALPHA = 0.3  # 目標速度平滑因子

# 提前預測參數
PREDICT_SEC_AHEAD = 2.5  # 預測未來曲率的時間視野 (秒)
WEIGHT_HISTORY = 0.4  # 歷史曲率權重
WEIGHT_FUTURE = 0.6   # 未來曲率權重

class VisionTurnController:
    def __init__(self):
        # 控制啟用狀態
        self.op_enabled = False
        self.gas_pressed = False
        self.enabled = params.get_bool("TurnVisionControl")  # 從參數控制是否啟用入彎控制
        self.last_params_update = 0.0

        # 初始目標速度
        self.v_target = MIN_TARGET_V
        self.smoothed_v_target = MIN_TARGET_V

        # 曲率歷史緩衝區
        self.curvature_history = deque(maxlen=HISTORY_LENGTH)
        self.last_time = time()  # 用於計算更新時間間隔

    @property
    def active(self):
        # 判斷控制是否應該啟動（系統啟動、未踩油門、控制開啟）
        return self.op_enabled and not self.gas_pressed and self.enabled

    def update_params(self):
        # 每 5 秒重新讀取一次開關參數
        t = time()
        if t > self.last_params_update + 5.0:
            self.enabled = params.get_bool("TurnVisionControl")
            self.last_params_update = t

    def update(self, op_enabled: bool, v_ego: float, sm: dict):
        # 計算 dt，用於時間同步
        current_time = time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.last_time = current_time

        # 更新系統狀態與控制啟用標記
        self.update_params()
        self.op_enabled = op_enabled
        self.gas_pressed = sm['carState'].gasPressed

        # 從模型輸出中提取預測的偏航率與速度
        rate_plan = np.abs(np.array(sm['modelV2'].orientationRate.z))
        vel_plan = np.array(sm['modelV2'].velocity.x)

        # 若預測數據無效則中止更新
        if rate_plan.size == 0 or vel_plan.size == 0:
            return

        # 計算每一預測點的曲率 = 偏航率 / 車速
        predicted_curvatures = rate_plan / np.maximum(vel_plan, 0.1)

        # 取 90% 分位數的曲率，表示即將到來彎道的嚴重程度
        max_curve_sample = np.percentile(predicted_curvatures, 90)
        self.curvature_history.append(max_curve_sample)

        # 根據未來預測時間提前提取未來曲率最大值
        dt_model = 0.05  # 預測時間步長
        future_idx = min(int(PREDICT_SEC_AHEAD / dt_model), len(predicted_curvatures) - 1)
        max_future_curve = np.max(predicted_curvatures[:future_idx])

        # 歷史平均曲率
        curve_array = np.array(self.curvature_history)
        mean_history_curve = np.mean(curve_array) if len(curve_array) > 0 else 0.0

        # 加權平均曲率：歷史與未來曲率融合，提高提前預判能力
        avg_curve = (
            WEIGHT_HISTORY * mean_history_curve +
            WEIGHT_FUTURE * max_future_curve
        )

        # 根據曲率與橫向加速度限制，反推安全目標速度
        v_ego_safe = max(v_ego, 0.1)
        if avg_curve > 0:
            v_target_raw = np.sqrt(TARGET_LAT_A / avg_curve)
        else:
            v_target_raw = MIN_TARGET_V
        v_target_raw = max(v_target_raw, MIN_TARGET_V)

        # 平滑速度變化，避免突兀減速
        self.smoothed_v_target = (
            SMOOTHING_ALPHA * self.smoothed_v_target +
            (1 - SMOOTHING_ALPHA) * v_target_raw
        )

        # 最終速度限制
        self.v_target = max(self.smoothed_v_target, MIN_TARGET_V)

# 建立控制器實例
vtsc = VisionTurnController()
