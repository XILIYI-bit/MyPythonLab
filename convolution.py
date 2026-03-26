import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons

# ================= 全局配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

# ================= 信号定义 =================
# 连续时间轴 [-5, 10]
t = np.linspace(-5, 10, 500)
dt = t[1] - t[0]
x_c = np.where((t >= 0) & (t <= 2), 1, 0)
h_c_base = np.where((t >= 0) & (t <= 3), np.exp(-t), 0)

# 离散时间轴 [-5, 14]
n = np.arange(-5, 15)
x_d = np.where((n >= 0) & (n <= 3), 1, 0)
h_d_base = np.where((n >= 0) & (n <= 4), 0.8**n, 0)

# ================= GUI 初始化 =================
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('小g的卷积动态演示系统 - 核心逻辑修正版')
plt.subplots_adjust(left=0.25, hspace=0.4)

ax_sig = plt.subplot(311)
ax_prod = plt.subplot(312)
ax_conv = plt.subplot(313)

mode = 'Continuous'

def format_axes():
    """统一定义坐标轴的样式和边界"""
    if mode == 'Continuous':
        ax_sig.set(xlim=(-5, 10), ylim=(-0.5, 1.5), title="1. 信号翻折与平移 x(τ) & h(t-τ)")
        ax_prod.set(xlim=(-5, 10), ylim=(-0.5, 1.5), title="2. 信号重叠与相乘")
        ax_conv.set(xlim=(-5, 10), ylim=(-0.5, 2.5), title="3. 积分求和得出卷积结果 y(t)")
    else:
        ax_sig.set(xlim=(-5, 15), ylim=(-0.5, 1.5), title="1. 序列翻折与平移 x[k] & h[n-k]")
        ax_prod.set(xlim=(-5, 15), ylim=(-0.5, 1.5), title="2. 序列重叠与相乘")
        ax_conv.set(xlim=(-5, 15), ylim=(-0.5, 3.5), title="3. 离散求和得出卷积结果 y[n]")
    
    for ax in (ax_sig, ax_prod, ax_conv):
        ax.grid(True, alpha=0.3)

def update(frame):
    """动画更新逻辑"""
    ax_sig.clear()
    ax_prod.clear()
    ax_conv.clear()
    
    if mode == 'Continuous':
        shift_t = -2 + frame * 0.05
        
        # 1. 动态生成当前时刻的翻折平移信号
        h_shifted = np.where((t >= shift_t - 3) & (t <= shift_t), np.exp(-(shift_t - t)), 0)
        product = x_c * h_shifted
        y_val = np.sum(product) * dt
        
        ax_sig.plot(t, x_c, 'b-', lw=2, label='x(τ)')
        ax_sig.plot(t, h_shifted, 'r-', lw=2, label=f'h({shift_t:.1f}-τ)')
        ax_sig.legend(loc='upper right')
        
        ax_prod.plot(t, product, 'g-', lw=2, label='相乘结果')
        ax_prod.fill_between(t, product, color='g', alpha=0.3)
        ax_prod.legend(loc='upper right')
        
        # 2. 修正：使用 convolve 并计算正确的物理时间轴
        y_hist = np.convolve(x_c, h_c_base, mode='full') * dt
        # 两个 [-5, 10] 的信号卷积，结果区间为 [-10, 20]
        t_hist = np.linspace(-10, 20, len(y_hist))  
        
        valid_idx = t_hist <= shift_t
        ax_conv.plot(t_hist[valid_idx], y_hist[valid_idx], 'k-', lw=2, label='y(t)')
        ax_conv.plot(shift_t, y_val, 'ro')
        ax_conv.legend(loc='upper right')
        
    else:
        shift_n = -2 + frame
        
        h_shifted = np.where((n >= shift_n - 4) & (n <= shift_n), 0.8**(shift_n - n), 0)
        product = x_d * h_shifted
        y_val = np.sum(product)
        
        ax_sig.stem(n, x_d, linefmt='b-', markerfmt='bo', basefmt='k-', label='x[k]')
        ax_sig.stem(n, h_shifted, linefmt='r-', markerfmt='ro', basefmt='k-', label=f'h[{shift_n}-k]')
        ax_sig.legend(loc='upper right')
        
        ax_prod.stem(n, product, linefmt='g-', markerfmt='go', basefmt='k-', label='相乘结果')
        ax_prod.legend(loc='upper right')
        
        # 修正离散时间轴映射
        y_hist = np.convolve(x_d, h_d_base, mode='full')
        # n 的起点是 -5，卷积后序列起点是 -5 + (-5) = -10
        n_hist = np.arange(-10, -10 + len(y_hist)) 
        
        valid_idx = n_hist <= shift_n
        if np.any(valid_idx):
            ax_conv.stem(n_hist[valid_idx], y_hist[valid_idx], linefmt='k-', markerfmt='ko', basefmt='k-', label='y[n]')
        ax_conv.plot(shift_n, y_val, 'ro', markersize=8)
        ax_conv.legend(loc='upper right')
    
    format_axes()

# ================= 交互控制 =================
radio_ax = plt.axes([0.02, 0.5, 0.15, 0.15], facecolor='lightgoldenrodyellow')
radio = RadioButtons(radio_ax, ('Continuous', 'Discrete'))

anim = None

def switch_mode(label):
    global mode, anim
    mode = label
    if anim:
        anim.event_source.stop()
    frames = 200 if mode == 'Continuous' else 15
    interval = 50 if mode == 'Continuous' else 800
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=True)
    fig.canvas.draw_idle()

radio.on_clicked(switch_mode)

switch_mode('Continuous')
plt.show()