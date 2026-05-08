import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.io import wavfile
import os
import winsound
import time

# --- 新增的绘图库 ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === 核心参数与映射表 ===
DTMF_TABLE = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
}

LOW_FREQS = [697, 770, 852, 941]
HIGH_FREQS = [1209, 1336, 1477, 1633]
FS = 8000 

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
    if len(data.shape) > 1: data = data[:, 0]
    data = data.astype(float)
    if np.max(np.abs(data)) > 0: data = data / np.max(np.abs(data))
    window_size, step_size = int(fs * 0.04), int(fs * 0.02)
    recognized_number, last_detected = "", None
    for i in range(0, len(data) - window_size, step_size):
        window = data[i : i + window_size]
        low_energies = {f: goertzel_algorithm(window, f, fs) for f in LOW_FREQS}
        high_energies = {f: goertzel_algorithm(window, f, fs) for f in HIGH_FREQS}
        best_low, best_high = max(low_energies, key=low_energies.get), max(high_energies, key=high_energies.get)
        if low_energies[best_low] > 100.0 and high_energies[best_high] > 100.0:
            current_digit = next((char for char, freqs in DTMF_TABLE.items() if freqs == (best_low, best_high)), None)
            if current_digit and current_digit != last_detected:
                recognized_number += current_digit
                last_detected = current_digit
        else: last_detected = None
    return recognized_number

class DTMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("拨号音分析系统 (Mobile Lab + Oscilloscope)")
        self.root.geometry("750x760") # 再次拉高，给波形图腾出空间
        self.root.configure(bg="#F4F5F7")
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_file = os.path.join(self.base_dir, "generated_test.wav")
        self.audio_cache_dir = os.path.join(self.base_dir, "dtmf_cache")
        
        self.seq_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="未选择任何文件")
        self.result_var = tk.StringVar(value=" 等待识别... ")
        
        os.makedirs(self.audio_cache_dir, exist_ok=True)
        self.pre_generate_tones()
        self.seq_var.trace_add("write", self.update_display)
        self.setup_ui()

    def pre_generate_tones(self):
        for key, (f_l, f_h) in DTMF_TABLE.items():
            safe_key = {'*': 'star', '#': 'hash'}.get(key, key)
            path = os.path.join(self.audio_cache_dir, f"{safe_key}.wav")
            if not os.path.exists(path):
                t = np.linspace(0, 0.12, int(FS * 0.12), endpoint=False)
                sig = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
                wavfile.write(path, FS, (sig*32767).astype(np.int16))

    def update_display(self, *args):
        if hasattr(self, 'display_text_id'):
            self.canvas.itemconfig(self.display_text_id, text=self.seq_var.get())

    def setup_ui(self):
        left_frame = tk.Frame(self.root, width=350, bg="#F4F5F7")
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        right_frame = tk.Frame(self.root, width=350, relief="flat", bg="white", highlightbackground="#E0E0E0", highlightthickness=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # ---------------- 左侧：拨号盘 ----------------
        tk.Label(left_frame, text="📞 手机仿真拨号盘", font=("微软雅黑", 14, "bold"), bg="#F4F5F7").pack(pady=5)
        self.canvas = tk.Canvas(left_frame, width=310, height=560, bg="#F4F5F7", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_rectangle(15, 10, 295, 70, fill="white", outline="#E0E0E0", width=1)
        self.display_text_id = self.canvas.create_text(155, 40, text="", font=("Consolas", 26, "bold"), fill="#1976D2")
        
        def create_btn(x, y, r, t1, t2, val):
            tag = f"btn_{val}"
            self.canvas.create_oval(x-r+1, y-r+2, x+r+1, y+r+2, fill="#D6D8DC", outline="", tags=tag)
            c = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="#E8EAF6", tags=tag)
            self.canvas.create_text(x, y-8, text=t1, font=("微软雅黑", 22, "bold"), tags=tag)
            self.canvas.create_text(x, y+14, text=t2, font=("微软雅黑", 8, "bold"), fill="gray", tags=tag)
            self.canvas.tag_bind(tag, "<Button-1>", lambda e, v=val: self.press_key(v))

        keys = [('1','','1'),('2','ABC','2'),('3','DEF','3'),
                ('4','GHI','4'),('5','JKL','5'),('6','MNO','6'),
                ('7','PQRS','7'),('8','TUV','8'),('9','WXYZ','9'),
                ('*','','*'),('0','+','0'),('#','','#')]
        
        for i, (t1, t2, v) in enumerate(keys):
            row, col = i // 3, i % 3
            create_btn(65 + col*90, 125 + row*85, 36, t1, t2, v)

        # 左侧底部操作区
        call_x, call_y, call_r = 155, 485, 40
        tag_call = "call_btn"
        self.canvas.create_oval(call_x-call_r+1, call_y-call_r+2, call_x+call_r+1, call_y+call_r+2, fill="#a0c0a0", outline="", tags=tag_call)
        self.canvas.create_oval(call_x-call_r, call_y-call_r, call_x+call_r, call_y+call_r, fill="#4CAF50", outline="", tags=tag_call)
        self.canvas.create_text(call_x, call_y, text="CALL", fill="white", font=("微软雅黑", 12, "bold"), tags=tag_call)
        self.canvas.tag_bind(tag_call, "<Button-1>", lambda e: self.build_audio())

        clear_btn = self.canvas.create_text(65, 485, text="清空", font=("微软雅黑", 11, "bold"), fill="#757575")
        self.canvas.tag_bind(clear_btn, "<Button-1>", lambda e: self.seq_var.set(""))
        back_btn = self.canvas.create_text(245, 485, text="⌫", font=("微软雅黑", 18), fill="#757575")
        self.canvas.tag_bind(back_btn, "<Button-1>", lambda e: self.delete_key())

        # ---------------- 右侧：分析面板 ----------------
        header = tk.Frame(right_frame, bg="#3F51B5", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="🔬 信号解析终端", font=("微软雅黑", 12, "bold"), bg="#3F51B5", fg="white").pack()
        
        tk.Label(right_frame, text="待检测文件:", bg="white", font=("微软雅黑", 9, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
        tk.Label(right_frame, textvariable=self.file_var, fg="#616161", wraplength=300, bg="#F5F5F5", justify="left").pack(fill="x", padx=15, ipady=3)
        
        btn_box = tk.Frame(right_frame, bg="white")
        btn_box.pack(fill="x", padx=15, pady=10)
        tk.Button(btn_box, text="📂 载入音频", command=self.load_file, relief="groove").pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(btn_box, text="▶️ 试听", command=self.play_audio, relief="groove").pack(side="right", expand=True, fill="x", padx=2)
        
        tk.Button(right_frame, text="🚀 启动算法识别号码", font=("微软雅黑", 12, "bold"), bg="#FF9800", fg="white", relief="flat", command=self.run_recognition).pack(fill="x", padx=20, pady=(15, 5), ipady=6)
        
        tk.Label(right_frame, textvariable=self.result_var, font=("Consolas", 32, "bold"), fg="#D32F2F", bg="#FFEBEE").pack(pady=5, fill="x", padx=15, ipady=10)

        # === 新增：实时示波器区域 ===
        tk.Label(right_frame, text="📈 实时时域波形 (Oscilloscope)", bg="white", font=("微软雅黑", 10, "bold")).pack(pady=(10, 0))
        
        # 初始化 Matplotlib 画布
        self.fig, self.ax = plt.subplots(figsize=(4, 1.8), dpi=80)
        self.fig.patch.set_facecolor('#F5F5F5')
        self.ax.set_facecolor('#000000') # 示波器黑底
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) # 隐藏坐标轴刻度
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=(5, 15))
        
        # 画一条初始的平直线
        self.line, = self.ax.plot(np.zeros(200), color="#00FF00", linewidth=1.5)
        self.ax.set_ylim(-1.2, 1.2)
        self.fig.tight_layout(pad=0.5)

    # --- 业务逻辑 ---
    def update_plot(self, signal):
        """核心：更新波形图"""
        # 老大注意：我们只截取前 200 个采样点 (约 25ms)，这样才能看清波形的起伏细节。
        # 如果把整个波形画上去，会密密麻麻变成一个色块。
        view_window = signal[:200] 
        self.line.set_ydata(view_window)
        self.canvas_plot.draw_idle() # 高效刷新画布

    def press_key(self, key):
        self.seq_var.set(self.seq_var.get() + key)
        f_l, f_h = DTMF_TABLE[key]
        
        # 实时生成时域信号
        t = np.linspace(0, 0.15, int(FS * 0.15), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
        
        # ⚡ 触发示波器刷新 ⚡
        self.update_plot(sig)

        safe_key = {'*': 'star', '#': 'hash'}.get(key, key)
        path = os.path.join(self.audio_cache_dir, f"{safe_key}.wav")
        if os.path.exists(path): winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def delete_key(self):
        s = self.seq_var.get()
        if s: self.seq_var.set(s[:-1])
        # 删除时让示波器归零
        self.update_plot(np.zeros(200))

    def build_audio(self):
        seq = self.seq_var.get()
        if not seq: return
        t_tone, pause = np.linspace(0, 0.25, int(FS * 0.25), endpoint=False), np.zeros(int(FS * 0.08))
        full = np.array([])
        for char in seq:
            f_l, f_h = DTMF_TABLE[char]
            tone = 0.5 * np.sin(2 * np.pi * f_l * t_tone) + 0.5 * np.sin(2 * np.pi * f_h * t_tone)
            full = np.concatenate((full, tone, pause))
        wavfile.write(self.target_file, FS, (full*32767).astype(np.int16))
        self.file_var.set(self.target_file)
        messagebox.showinfo("成功", "号码已录入！请点击右侧「启动算法」")

    def load_file(self):
        f = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")], initialdir=self.base_dir)
        if f: self.file_var.set(f)

    def play_audio(self):
        if os.path.exists(self.file_var.get()):
            winsound.PlaySound(self.file_var.get(), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def run_recognition(self):
        p = self.file_var.get()
        if not os.path.exists(p) or p == "未选择任何文件": return
        self.result_var.set("识别中...")
        self.root.update()
        time.sleep(0.5)
        res = recognize_audio(p)
        self.result_var.set(res if res else "无结果")

if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()