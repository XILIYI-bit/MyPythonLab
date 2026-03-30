import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.io import wavfile
import os
import winsound
import time

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

# === 核心算法模块 ===
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

# === GUI 界面类 ===
class DTMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("拨号音通信分析系统 (Mobile Lab)")
        self.root.geometry("680x600")
        self.root.configure(bg="#F4F5F7") # 更现代的背景灰
        
        self.target_file = os.path.join(os.getcwd(), "generated_test.wav")
        self.seq_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="未选择任何文件")
        self.result_var = tk.StringVar(value=" 等待识别... ")
        
        # 预生成按键音，彻底解决连续按键导致的文件冲突卡顿问题
        self.audio_cache_dir = os.path.join(os.getcwd(), "dtmf_cache")
        os.makedirs(self.audio_cache_dir, exist_ok=True)
        self.pre_generate_tones()

        self.seq_var.trace_add("write", self.update_display)
        self.setup_ui()

    def pre_generate_tones(self):
        """一次性生成所有按键的短音，加速响应"""
        for key, (f_l, f_h) in DTMF_TABLE.items():
            safe_key = {'*': 'star', '#': 'hash'}.get(key, key) # 规避 Windows 非法文件名
            path = os.path.join(self.audio_cache_dir, f"{safe_key}.wav")
            if not os.path.exists(path):
                t = np.linspace(0, 0.12, int(FS * 0.12), endpoint=False)
                sig = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
                wavfile.write(path, FS, (sig*32767).astype(np.int16))

    def update_display(self, *args):
        if hasattr(self, 'display_text_id'):
            self.canvas.itemconfig(self.display_text_id, text=self.seq_var.get())

    def setup_ui(self):
        # 左侧：虚拟拨号盘
        left_frame = tk.Frame(self.root, width=320, bg="#F4F5F7")
        left_frame.pack(side="left", fill="y", padx=15, pady=15)
        
        # 右侧：分析面板
        right_frame = tk.Frame(self.root, width=320, relief="flat", bg="white", highlightbackground="#E0E0E0", highlightthickness=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        # ----------------- 绘制拨号盘 -----------------
        tk.Label(left_frame, text="📞 信号合成终端", font=("微软雅黑", 14, "bold"), bg="#F4F5F7", fg="#333333").pack(pady=(0, 10))
        
        self.canvas = tk.Canvas(left_frame, width=310, height=520, bg="#F4F5F7", highlightthickness=0)
        self.canvas.pack()
        
        # 输入显示屏区
        self.canvas.create_rectangle(15, 10, 295, 60, fill="white", outline="#E0E0E0", width=1)
        self.display_text_id = self.canvas.create_text(155, 35, text="", font=("Consolas", 24, "bold"), fill="#1976D2")
        
        # 退格删除按钮
        del_btn = self.canvas.create_text(275, 35, text="⌫", font=("微软雅黑", 18), fill="#9E9E9E")
        self.canvas.tag_bind(del_btn, "<Button-1>", lambda e: self.delete_key())
        self.canvas.tag_bind(del_btn, "<Enter>", lambda e: self.canvas.itemconfig(del_btn, fill="#F44336"))
        self.canvas.tag_bind(del_btn, "<Leave>", lambda e: self.canvas.itemconfig(del_btn, fill="#9E9E9E"))

        # 按键绘制工厂
        def create_btn(x, y, r, t1, t2, val):
            tag_name = f"btn_{val}" # 核心：将同一个按钮的所有元素打上统一标签
            
            # 阴影与本体
            self.canvas.create_oval(x-r+1, y-r+2, x+r+1, y+r+2, fill="#D6D8DC", outline="", tags=tag_name)
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="#E8EAF6", tags=tag_name)
            
            # 文本
            self.canvas.create_text(x, y-8, text=t1, font=("微软雅黑", 22, "bold"), fill="#212121", tags=tag_name)
            if t2:
                self.canvas.create_text(x, y+14, text=t2, font=("微软雅黑", 8, "bold"), fill="#9E9E9E", tags=tag_name)
            
            # 事件全部绑定到统一的 Tag 上，点哪里都不会失灵
            self.canvas.tag_bind(tag_name, "<Button-1>", lambda e, v=val: self.press_key(v))

        # 九宫格布局
        keys = [('1','','1'),('2','ABC','2'),('3','DEF','3'),
                ('4','GHI','4'),('5','JKL','5'),('6','MNO','6'),
                ('7','PQRS','7'),('8','TUV','8'),('9','WXYZ','9'),
                ('*','','*'),('0','+','0'),('#','','#')]
        
        for i, (t1, t2, v) in enumerate(keys):
            row, col = i // 3, i % 3
            create_btn(65 + col*90, 115 + row*85, 36, t1, t2, v)

        # 底部操作区 (CALL 与 清除)
        call_btn_tag = "call_btn"
        self.canvas.create_oval(115, 455, 195, 535, fill="#4CAF50", outline="", tags=call_btn_tag)
        self.canvas.create_text(155, 495, text="CALL", fill="white", font=("微软雅黑", 12, "bold"), tags=call_btn_tag)
        self.canvas.tag_bind(call_btn_tag, "<Button-1>", lambda e: self.build_audio())
        
        clear_btn = self.canvas.create_text(55, 495, text="清空", font=("微软雅黑", 11, "bold"), fill="#757575")
        self.canvas.tag_bind(clear_btn, "<Button-1>", lambda e: self.seq_var.set(""))

        # ----------------- 右侧分析面板 -----------------
        header_frame = tk.Frame(right_frame, bg="#3F51B5", pady=10)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="🔬 信号解码器", font=("微软雅黑", 12, "bold"), bg="#3F51B5", fg="white").pack()
        
        tk.Label(right_frame, text="待测音频路径:", bg="white", font=("微软雅黑", 9, "bold")).pack(anchor="w", padx=15, pady=(15, 5))
        tk.Label(right_frame, textvariable=self.file_var, fg="#616161", wraplength=260, bg="#F5F5F5", justify="left", relief="flat").pack(fill="x", padx=15, ipady=5)
        
        btn_frame = tk.Frame(right_frame, bg="white")
        btn_frame.pack(fill="x", padx=15, pady=15)
        tk.Button(btn_frame, text="📂 载入音频", command=self.load_file, relief="groove", bg="white").pack(side="left", expand=True, fill="x", padx=(0, 5))
        tk.Button(btn_frame, text="▶️ 试听信号", command=self.play_audio, relief="groove", bg="white").pack(side="right", expand=True, fill="x", padx=(5, 0))
        
        tk.Button(right_frame, text="🚀 启动算法提取特征", font=("微软雅黑", 12, "bold"), bg="#FF9800", fg="white", 
                  activebackground="#F57C00", activeforeground="white", relief="flat", command=self.run_recognition).pack(fill="x", padx=20, pady=25, ipady=5)
        
        tk.Label(right_frame, text="智能解码结果", bg="white", font=("微软雅黑", 10)).pack()
        tk.Label(right_frame, textvariable=self.result_var, font=("Consolas", 32, "bold"), fg="#D32F2F", bg="#FFEBEE").pack(pady=10, fill="x", padx=15, ipady=10)

    # --- 业务逻辑 ---
    def press_key(self, key):
        self.seq_var.set(self.seq_var.get() + key)
        # 播放预加载的音频，告别卡顿
        safe_key = {'*': 'star', '#': 'hash'}.get(key, key)
        path = os.path.join(self.audio_cache_dir, f"{safe_key}.wav")
        if os.path.exists(path):
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

    def delete_key(self):
        """退格删除逻辑"""
        seq = self.seq_var.get()
        if seq:
            self.seq_var.set(seq[:-1])

    def build_audio(self):
        seq = self.seq_var.get()
        if not seq: 
            messagebox.showwarning("警告", "拨号盘为空，无法呼叫！")
            return
            
        t_tone, pause = np.linspace(0, 0.25, int(FS * 0.25), endpoint=False), np.zeros(int(FS * 0.08))
        full = np.array([])
        for char in seq:
            f_l, f_h = DTMF_TABLE[char]
            tone = 0.5 * np.sin(2 * np.pi * f_l * t_tone) + 0.5 * np.sin(2 * np.pi * f_h * t_tone)
            full = np.concatenate((full, tone, pause))
            
        wavfile.write(self.target_file, FS, (full*32767).astype(np.int16))
        self.file_var.set(self.target_file)
        messagebox.showinfo("成功", f"号码流合成完毕！\n请点击右侧「启动算法」进行验证。")

    def load_file(self):
        f = filedialog.askopenfilename(filetypes=[("WAV 音频", "*.wav")])
        if f: self.file_var.set(f)

    def play_audio(self):
        if os.path.exists(self.file_var.get()):
            winsound.PlaySound(self.file_var.get(), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def run_recognition(self):
        if not os.path.exists(self.file_var.get()) or self.file_var.get() == "未选择任何文件": return
        self.result_var.set("分析中...")
        self.root.update()
        time.sleep(0.4)
        res = recognize_audio(self.file_var.get())
        self.result_var.set(res if res else "无结果")

if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()