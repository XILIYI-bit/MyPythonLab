import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.io import wavfile
import os
import winsound
import time

# === 核心参数与映射表 ===
DTMF_TABLE = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
}

LOW_FREQS = [697, 770, 852, 941]
HIGH_FREQS = [1209, 1336, 1477, 1633]
FS = 8000 # 标准电话采样率

# === 算法层：Goertzel 频率检测 ===
def goertzel_algorithm(samples, target_freq, fs):
    N = len(samples)
    k = int(0.5 + (N * target_freq) / fs)
    w = (2.0 * np.pi / N) * k
    cosine = np.cos(w)
    s_prev, s_prev2 = 0.0, 0.0
    for x in samples:
        s = x + 2.0 * cosine * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return s_prev2**2 + s_prev**2 - 2.0 * cosine * s_prev * s_prev2

def recognize_audio(filepath):
    fs, data = wavfile.read(filepath)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(float)
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    window_size = int(fs * 0.04)  # 40ms 滑动窗口
    step_size = int(fs * 0.02)    # 20ms 步进
    recognized_number = ""
    last_detected = None

    for i in range(0, len(data) - window_size, step_size):
        window = data[i : i + window_size]
        low_energies = {f: goertzel_algorithm(window, f, fs) for f in LOW_FREQS}
        high_energies = {f: goertzel_algorithm(window, f, fs) for f in HIGH_FREQS}
        
        best_low = max(low_energies, key=low_energies.get)
        best_high = max(high_energies, key=high_energies.get)
        
        # 动态门限，排除静音段
        if low_energies[best_low] > 100.0 and high_energies[best_high] > 100.0:
            current_digit = next((char for char, freqs in DTMF_TABLE.items() if freqs == (best_low, best_high)), None)
            if current_digit and current_digit != last_detected:
                recognized_number += current_digit
                last_detected = current_digit
        else:
            last_detected = None

    return recognized_number

# === 业务层：生成与播放逻辑 ===
def generate_dtmf_sequence(sequence, filepath):
    """根据字符串生成带有停顿的连续拨号音"""
    tone_duration = 0.3  # 每个按键持续 0.3 秒
    pause_duration = 0.1 # 按键之间停顿 0.1 秒
    
    t_tone = np.linspace(0, tone_duration, int(FS * tone_duration), endpoint=False)
    pause_signal = np.zeros(int(FS * pause_duration))
    
    full_signal = np.array([])
    for char in sequence:
        if char in DTMF_TABLE:
            f_l, f_h = DTMF_TABLE[char]
            tone = 0.5 * np.sin(2 * np.pi * f_l * t_tone) + 0.5 * np.sin(2 * np.pi * f_h * t_tone)
            full_signal = np.concatenate((full_signal, tone, pause_signal))
            
    # 加一点模拟线路的底噪
    full_signal += np.random.normal(0, 0.03, len(full_signal))
    scaled_signal = (full_signal / np.max(np.abs(full_signal)) * 32767).astype(np.int16)
    wavfile.write(filepath, FS, scaled_signal)

# === GUI 界面类 ===
class DTMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("拨号音通信分析系统 (DTMF Lab)")
        self.root.geometry("650x450")
        
        self.target_file = os.path.join(os.getcwd(), "generated_test.wav")
        self.seq_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="未选择任何文件")
        self.result_var = tk.StringVar(value=" 等待识别... ")
        
        self.setup_ui()

    def setup_ui(self):
        # 左右分栏设计
        left_frame = tk.Frame(self.root, width=300, relief="groove", borderwidth=2)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        right_frame = tk.Frame(self.root, width=300, relief="groove", borderwidth=2)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- 左侧：虚拟拨号盘与生成器 ---
        tk.Label(left_frame, text="📞 信号源合成器", font=("微软雅黑", 12, "bold"), bg="#e0e0e0").pack(fill="x", pady=5)
        tk.Label(left_frame, textvariable=self.seq_var, font=("Consolas", 16, "bold"), fg="blue", bg="white", width=15, relief="sunken").pack(pady=10)
        
        keypad_frame = tk.Frame(left_frame)
        keypad_frame.pack()
        
        keys = [
            ('1', '2', '3', 'A'),
            ('4', '5', '6', 'B'),
            ('7', '8', '9', 'C'),
            ('*', '0', '#', 'D')
        ]
        for r, row in enumerate(keys):
            for c, key in enumerate(row):
                btn = tk.Button(keypad_frame, text=key, font=("微软雅黑", 14, "bold"), width=4, height=1,
                                command=lambda k=key: self.press_key(k))
                btn.grid(row=r, column=c, padx=3, pady=3)
                
        tk.Button(left_frame, text="生成测试音频 (WAV)", command=self.build_audio, bg="#2196F3", fg="white").pack(fill="x", padx=20, pady=15)
        tk.Button(left_frame, text="清空输入", command=lambda: self.seq_var.set("")).pack(fill="x", padx=20)

        # --- 右侧：信号识别分析器 ---
        tk.Label(right_frame, text="📡 信号特征识别", font=("微软雅黑", 12, "bold"), bg="#e0e0e0").pack(fill="x", pady=5)
        
        tk.Label(right_frame, text="当前待测文件:").pack(anchor="w", padx=10, pady=(15, 0))
        tk.Label(right_frame, textvariable=self.file_var, fg="gray", wraplength=250, justify="left").pack(anchor="w", padx=10)
        
        btn_frame = tk.Frame(right_frame)
        btn_frame.pack(fill="x", padx=10, pady=10)
        tk.Button(btn_frame, text="📂 浏览本地文件", command=self.load_file).pack(side="left", padx=5)
        tk.Button(btn_frame, text="▶️ 试听音频", command=self.play_audio).pack(side="left", padx=5)
        
        tk.Button(right_frame, text="执行 Goertzel 算法识别", font=("微软雅黑", 11, "bold"), bg="#4CAF50", fg="white", 
                  command=self.run_recognition).pack(fill="x", padx=20, pady=25)
                  
        tk.Label(right_frame, text="算法解码结果：", font=("微软雅黑", 10)).pack(pady=5)
        tk.Label(right_frame, textvariable=self.result_var, font=("Consolas", 26, "bold"), fg="#D32F2F", bg="#ffebee", width=12).pack(pady=10)

    def press_key(self, key):
        """按下按键：发声并记录"""
        self.seq_var.set(self.seq_var.get() + key)
        # 生成一个极短的单音用于实时反馈
        f_l, f_h = DTMF_TABLE[key]
        t = np.linspace(0, 0.15, int(FS * 0.15), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
        scaled = (sig / np.max(np.abs(sig)) * 32767).astype(np.int16)
        tmp_file = "temp_beep.wav"
        wavfile.write(tmp_file, FS, scaled)
        winsound.PlaySound(tmp_file, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def build_audio(self):
        seq = self.seq_var.get()
        if not seq:
            messagebox.showinfo("提示", "请先在拨号盘上输入一串号码！")
            return
        generate_dtmf_sequence(seq, self.target_file)
        self.file_var.set(self.target_file) # 自动加载刚才生成的文件
        messagebox.showinfo("成功", f"音频已生成并自动加载！\n包含 {len(seq)} 个按键。")

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV 音频", "*.wav")])
        if filepath:
            self.file_var.set(filepath)
            self.result_var.set(" 等待识别... ")

    def play_audio(self):
        path = self.file_var.get()
        if os.path.exists(path):
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            messagebox.showwarning("错误", "找不到音频文件，请确认路径。")

    def run_recognition(self):
        path = self.file_var.get()
        if not os.path.exists(path):
            messagebox.showwarning("错误", "请先生成或加载一个音频文件。")
            return
        self.result_var.set(" 分析中... ")
        self.root.update()
        time.sleep(0.3) # 增加一点仪式感的停顿
        res = recognize_audio(path)
        self.result_var.set(res if res else "未能识别")

if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()