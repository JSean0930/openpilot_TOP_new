import numpy as np

# ========= 小工具：km/h <-> m/s =========
def kph(k):  # 轉成 m/s，讓錨點可用直覺的 km/h
  return float(k) / 3.6

# ========= 1) 用錨點定義“人性化、近線性”的目標 =========
# 你可針對 eco/normal/sport 分別調整這些錨點。
# 建議：低速(<=30 km/h)較大加速，高速(>=90 km/h)較小加速，130 km/h 封頂。
MAX_ANCHORS = {
  "eco": [
    (kph(0),   3.0),
    (kph(30),  1.2),
    (kph(90),  0.55),
    (kph(130), 0.40),
  ],
  "normal": [
    (kph(0),   3.0),
    (kph(30),  1.5),
    (kph(90),  0.70),
    (kph(130), 0.50),
  ],
  "sport": [
    (kph(0),   3.0),
    (kph(30),  1.8),
    (kph(90),  0.90),
    (kph(130), 0.65),
  ],
}

# 最小加速度（負值；越負代表越強的煞車）
# 低速允許大一點（跟停順手），高速收斂較溫和
MIN_ANCHORS = {
  "eco": [
    (kph(0),   -1.4),
    (kph(50),  -1.2),
    (kph(130), -0.9),
  ],
  "normal": [
    (kph(0),   -1.5),
    (kph(50),  -1.2),
    (kph(130), -1.0),
  ],
  "sport": [
    (kph(0),   -1.6),
    (kph(50),  -1.3),
    (kph(130), -1.1),
  ],
  # 保留 stock 作比較
  "stock": [
    (kph(0),   -1.2),
    (kph(130), -1.2),
  ],
}

# ========= 2) 由錨點生成“近線性”的密集 breakpoints =========
def build_linear_profile(anchors, dense_step=1.0):
  """
  anchors: [(x0, y0), (x1, y1), ...] with x strictly increasing (單位 m/s)
  dense_step: 以多少 m/s 產生一個節點（1.0 m/s 已很夠用）
  回傳: xp, yp  (等間距 xp，段內純線性補足的 yp)
  """
  xs, ys = zip(*anchors)
  x_min, x_max = xs[0], xs[-1]
  xp = np.arange(x_min, x_max + 1e-6, dense_step, dtype=float)
  yp = np.interp(xp, xs, ys)  # 分段線性
  return xp.tolist(), yp.tolist()

# ========= 3) 單調 Hermite 斜率（PCHIP/Fritsch–Carlson）=========
def compute_monotone_slopes(x, y):
  """
  產生單調保護的節點斜率，避免 Hermite 造成局部過衝與波紋。
  參考 Fritsch–Carlson 方法：對於單調區段，斜率取為加權調和平均；對於非單調，置零避免過衝。
  """
  n = len(x)
  if n < 2:
    raise ValueError("Need at least two points")

  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  h = np.diff(x)
  delta = np.diff(y) / h

  m = np.zeros(n, dtype=float)
  m[0] = delta[0]
  m[-1] = delta[-1]

  for i in range(1, n-1):
    if delta[i-1] * delta[i] <= 0.0:
      m[i] = 0.0
    else:
      w1 = 2*h[i] + h[i-1]
      w2 = h[i] + 2*h[i-1]
      m[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i])

  # 斜率限幅（可選）：抑制太尖銳的變化，讓“線性感”更穩
  slope_limit = 4.0  # 依需要調
  m = np.clip(m, -slope_limit, slope_limit)
  return m

# ========= 4) Hermite 插值 =========
def hermite_interpolate_scalar(x, xp, yp, m):
  # 夾到定義域
  x = float(np.clip(x, xp[0], xp[-1]))
  idx = np.searchsorted(xp, x) - 1
  idx = int(np.clip(idx, 0, len(xp) - 2))

  x0, x1 = xp[idx], xp[idx+1]
  y0, y1 = yp[idx], yp[idx+1]
  m0, m1 = m[idx], m[idx+1]

  t = (x - x0) / (x1 - x0)
  h00 =  2*t**3 - 3*t**2 + 1
  h10 =      t**3 - 2*t**2 + t
  h01 = -2*t**3 + 3*t**2
  h11 =      t**3 -    t**2

  return (h00*y0 + h10*(x1-x0)*m0 + h01*y1 + h11*(x1-x0)*m1)

# ========= 5) 建構每個模式的 profile 與斜率（一次性）=========
class _Profiles:
  def __init__(self, max_anchors, min_anchors, dense_step=1.0):
    self.max_x = {}
    self.max_y = {}
    self.max_m = {}
    self.min_x = {}
    self.min_y = {}
    self.min_m = {}

    for mode, anchors in max_anchors.items():
      xp, yp = build_linear_profile(anchors, dense_step=dense_step)
      m = compute_monotone_slopes(xp, yp)
      self.max_x[mode], self.max_y[mode], self.max_m[mode] = xp, yp, m

    for mode, anchors in min_anchors.items():
      xp, yp = build_linear_profile(anchors, dense_step=dense_step)
      m = compute_monotone_slopes(xp, yp)
      self.min_x[mode], self.min_y[mode], self.min_m[mode] = xp, yp, m

  def get_max(self, v, mode):
    xp, yp, m = self.max_x[mode], self.max_y[mode], self.max_m[mode]
    return float(hermite_interpolate_scalar(v, xp, yp, m))

  def get_min(self, v, mode):
    xp, yp, m = self.min_x[mode], self.min_y[mode], self.min_m[mode]
    return float(hermite_interpolate_scalar(v, xp, yp, m))

_PROFILES = _Profiles(MAX_ANCHORS, MIN_ANCHORS, dense_step=1.0)  # 1 m/s 節點密度

# ========= 6) 對外 API（維持你的呼叫介面）=========
def get_max_accel_hermite(v_ego: float, mode: str = "normal") -> float:
  # 非法模式保護：fallback 到 normal
  if mode not in _PROFILES.max_x:
    mode = "normal"
  return _PROFILES.get_max(v_ego, mode)

def get_min_accel_hermite(v_ego: float, mode: str = "normal") -> float:
  if mode not in _PROFILES.min_x:
    mode = "normal"
  return _PROFILES.get_min(v_ego, mode)
