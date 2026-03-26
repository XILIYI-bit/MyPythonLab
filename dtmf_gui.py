import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.io import wavfile

# === 理论基础：DTMF 标准频率映射表 ===
DTMF_TABLE = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
}

LOW_FREQS = [697, 770, 852, 941]
HIGH_FREQS = [1209, 1336, 1477, 1633]

# === 核心算法模块 ===
def goertzel_algorithm(samples, target_freq, fs):
    """
    Goertzel 算法实现：计算特定频率在当前样本中的能量。
    代码结构紧凑，避免调用冗余的傅里叶变换库。
    """
    N = len(samples)
    k = int(0.5 + (N * target_freq) / fs)
    w = (2.0 * np.pi / N) * k
    cosine = np.cos(w)
    
    s_prev = 0.0
    s_prev2 = 0.0
    
    for x in samples:
        s = x + 2.0 * cosine * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
        
    power = s_prev2**2 + s_prev**2 - 2.0 * cosine * s_prev * s_prev2
    return power

def recognize_dtmf_from_file(filepath):
    """
    通过滑动窗口处理长音频，识别出连续的拨号号码
    """
    fs, data = wavfile.read(filepath)
    
    # 统一处理为单声道并归一化，防止不同录音音量差异导致门限失效
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(float)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    # 设定滑动窗口参数（40ms 窗口，20ms 步进）
    window_time = 0.04 
    step_time = 0.02
    window_size = int(fs * window_time)
    step_size = int(fs * step_time)

    recognized_number = ""
    last_detected_digit = None

    # 让窗口在音频数据上滑动遍历
    for i in range(0, len(data) - window_size, step_size):
        window_samples = data[i : i + window_size]
        
        # 分别计算低频和高频组的能量
        low_energies = {f: goertzel_algorithm(window_samples, f, fs) for f in LOW_FREQS}
        high_energies = {f: goertzel_algorithm(window_samples, f, fs) for f in HIGH_FREQS}
        
        best_low = max(low_energies, key=low_energies.get)
        best_high = max(high_energies, key=high_energies.get)
        
        # 设定能量门限，排除环境白噪声
        # 归一化后的有效信号能量通常远大于 100
        threshold = 150.0 
        
        if low_energies[best_low] > threshold and high_energies[best_high] > threshold:
            # 在表中匹配对应的字符
            current_digit = None
            for char, freqs in DTMF_TABLE.items():
                if freqs == (best_low, best_high):
                    current_digit = char
                    break
            
            # 状态机防抖：确保同一个长按键只被记录一次
            if current_digit is not None and current_digit != last_detected_digit:
                recognized_number += current_digit
                last_detected_digit = current_digit
        else:
            # 能量低于门限，判定为静音间隔，重置状态准备迎接下一个按键
            last_detected_digit = None

    return recognized_number

# === GUI 界面模块 ===
def select_file():
    """打开文件对话框选择 WAV 文件"""
    filepath = filedialog.askopenfilename(
        title="选择拨号音文件",
        filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
    )
    if filepath:
        file_path_var.set(filepath)
        result_var.set("等待识别...")

def start_recognition():
    """触发识别流程并更新界面结果"""
    filepath = file_path_var.get()
    if not filepath or filepath == "请先选择一个 .wav 文件":
        messagebox.showwarning("提示", "老大，还没选文件呢！")
        return
    
    try:
        result_var.set("正在分析音频，请稍候...")
        window.update() # 强制刷新界面状态
        
        number = recognize_dtmf_from_file(filepath)
        
        if number:
            result_var.set(f"{number}")
        else:
            result_var.set("未识别到有效的拨号音")
            
    except Exception as e:
        messagebox.showerror("错误", f"处理文件时发生异常：\n{str(e)}")
        result_var.set("识别失败")

# --- 主程序入口，构建界面 ---
window = tk.Tk()
window.title("拨号音 (DTMF) 识别系统")
window.geometry("500x350")
window.configure(padx=20, pady=20)

# 变量绑定
file_path_var = tk.StringVar(value="请先选择一个 .wav 文件")
result_var = tk.StringVar(value="-")

# UI 布局排版
tk.Label(window, text="1. 选择音频源文件", font=("微软雅黑", 12, "bold")).pack(anchor="w", pady=(0, 5))
path_label = tk.Label(window, textvariable=file_path_var, fg="gray", wraplength=450, justify="left")
path_label.pack(anchor="w", pady=(0, 10))

btn_select = tk.Button(window, text="浏览文件...", command=select_file, width=15)
btn_select.pack(anchor="w", pady=(0, 20))

tk.Label(window, text="2. 执行 Goertzel 算法识别", font=("微软雅黑", 12, "bold")).pack(anchor="w", pady=(0, 5))
btn_run = tk.Button(window, text="开始识别", command=start_recognition, width=15, bg="#4CAF50", fg="white", font=("微软雅黑", 10, "bold"))
btn_run.pack(anchor="w", pady=(0, 20))

tk.Label(window, text="识别结果：", font=("微软雅黑", 12, "bold")).pack(anchor="w", pady=(0, 5))
result_display = tk.Label(window, textvariable=result_var, font=("Consolas", 24, "bold"), fg="#D32F2F")
result_display.pack(pady=10)

window.mainloop()