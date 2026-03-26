import numpy as np
import matplotlib.pyplot as plt

# 1. 设置画布大小
plt.figure(figsize=(8, 6))

# --- 部分 A：绘制连续信号 (余弦波) ---
# 定义时间范围：从 0 到 1 秒，取 500 个点，这样看起来就是连续的平滑曲线
t = np.linspace(0, 1, 500) 
f = 5  # 频率 5Hz
y1 = np.cos(2 * np.pi * f * t)

plt.subplot(2, 1, 1)    # 开启第一个子图
plt.plot(t, y1, 'b-')   # 使用 plot 函数绘制连续线段
plt.title('Continuous Signal: Cosine Wave')
plt.grid(True)

# --- 部分 B：绘制离散信号 (阶跃信号) ---
# 定义离散点序号：从 -5 到 10 的整数
n = np.arange(-5, 11)
# 阶跃逻辑：n 大于等于 0 时为 1，否则为 0
y2 = np.where(n >= 0, 1, 0)

plt.subplot(2, 1, 2)    # 开启第二个子图
plt.stem(n, y2)         # 使用 stem 函数绘制离散“火柴棒”图
plt.title('Discrete Signal: Unit Step Sequence')
plt.grid(True)

# 自动调整间距并显示
plt.tight_layout()
plt.show()