import numpy as np
from scipy.io import wavfile
import os
import winsound

# 1. 配置存放领地
# 这里直接指向你截图中的桌面文件夹
TARGET_DIR = r"C:\Users\XiLYi\Desktop\MyPythonLab"

# 2. DTMF 标准频率映射表
DTMF_TABLE = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
}

LOW_FREQS = [697, 770, 852, 941]
HIGH_FREQS = [1209, 1336, 1477, 1633]

def goertzel_algorithm(samples, target_freq, fs):
    """核心算法：计算特定频点的能量"""
    N = len(samples)
    k = int(0.5 + (N * target_freq) / fs)
    w = (2.0 * np.pi / N) * k
    cosine = np.cos(w)
    s_prev, s_prev2 = 0.0, 0.0
    for x in samples:
        s = x + 2.0 * cosine * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return s_prev2**2 + s_prev**2 - 2.0 * cosine * s_prev * s_prev22

def run_lab():
    # 确保文件夹存在
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 已自动创建文件夹: {TARGET_DIR}")

    print("--- 老大的拨号音实验室 ---")
    digit = input("请输入想测试的按键 (0-9, *, #): ").strip().upper()
    
    if digit not in DTMF_TABLE:
        print("❌ 输入无效，请输入拨号盘字符。")
        return

    # --- 第一步：生产音频 ---
    fs = 8000
    duration = 0.3
    filename = os.path.join(TARGET_DIR, f"dtmf_{digit}.wav")
    
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f_l, f_h = DTMF_TABLE[digit]
    # 生成双音信号并加一点点噪声
    signal = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
    signal += np.random.normal(0, 0.02, len(t))
    
    # 保存为 16-bit WAV
    scaled_signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
    wavfile.write(filename, fs, scaled_signal)
    
    print(f"\n✅ 音频已存入: {filename}")
    
    # --- 第二步：播放音频 ---
    print(f"🎵 正在为您播放数字 【 {digit} 】 的声音...")
    winsound.PlaySound(filename, winsound.SND_FILENAME)

    # --- 第三步：读取并识别 ---
    print("🔍 正在读取刚才生成的文件进行验证...")
    fs_read, data = wavfile.read(filename)
    samples = data.astype(float) / 32767.0
    
    # 寻找能量最大的频率对
    best_low = max(LOW_FREQS, key=lambda f: goertzel_algorithm(samples, f, fs_read))
    best_high = max(HIGH_FREQS, key=lambda f: goertzel_algorithm(samples, f, fs_read))
    
    # 匹配结果
    recognized = None
    for k, (l, h) in DTMF_TABLE.items():
        if l == best_low and h == best_high:
            recognized = k
            break
            
    if recognized == digit:
        print(f"🎉 识别成功！确认按键为: 【 {recognized} 】")
    else:
        print(f"⚠️ 识别偏差。预期 {digit}，识别为 {recognized}")

if __name__ == "__main__":
    try:
        run_lab()
    except Exception as e:
        print(f"💥 哎呀，出错了: {e}")