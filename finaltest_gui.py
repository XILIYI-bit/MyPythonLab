# -*- coding: utf-8 -*-
"""
DTMF 拨号音识别系统

本程序用于“根据音频拨号音识别当前所拨号码”的课程大作业。
核心思想并不是做通用语音识别，而是利用 DTMF 的标准频率表进行定点频率检测：

1. 每个按键音都由两个正弦波叠加而成，一个来自低频组，一个来自高频组；
2. 程序读取音频后，将音频统一转换为单声道浮点序列，便于后续计算；
3. 使用滑动窗口把长音频切成若干短片段，逐段判断当前是否存在 DTMF 按键；
4. 对每个窗口使用 Goertzel 算法计算目标频点能量，而不是计算完整 FFT 频谱；
5. 找到能量最大的低频点和高频点后，通过 DTMF_TABLE 查表得到对应数字；
6. 连续窗口识别到同一个数字时只保留一次，避免一个按键被重复输出。

支持格式说明：
- WAV、AIFF、AU/SND 使用 Python 标准库或 scipy 读取；
- MP3、OGG、FLAC、M4A 等压缩音频通过 imageio_ffmpeg 自带的 ffmpeg 解码；
- 所有格式最终都会转成同一种“采样率 + 单声道浮点数组”形式，再进入同一套识别算法。
"""

import aifc
import os
import subprocess
import sunau
import time
import wave
import winsound
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile


# DTMF 键盘频率映射表。
# 表中每个键的值都是一个二元组：(低频, 高频)。
# 例如数字 5 的音频由 770 Hz 和 1336 Hz 两个正弦分量相加得到。
# 识别时只要检测到这两个频点最强，就可以判断当前按键是 5。
DTMF_TABLE = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
}

# DTMF 标准频点。
# 低频组决定键盘行，高频组决定键盘列。
# 普通电话键盘只使用 1209、1336、1477 三个高频列；
# 1633 Hz 属于扩展列，本项目虽然不输出 A/B/C/D 键，但保留该频点可以说明标准更完整。
LOW_FREQS = [697, 770, 852, 941]
HIGH_FREQS = [1209, 1336, 1477, 1633]

# 程序生成拨号音时使用的采样率。
# 8000 Hz 是电话语音处理中常见的采样率，已经足够覆盖 DTMF 的最高频点 1633 Hz。
# 根据奈奎斯特采样定理，采样率只要大于最高频率的 2 倍即可避免频率混叠。
FS = 8000

# GUI 文件选择器中展示的格式。
# 这里的列表只影响“文件选择窗口能看到哪些扩展名”，真正能否读取由 load_audio_file 决定。
SUPPORTED_FILETYPES = [
    ("支持的音频文件", "*.wav *.wave *.aif *.aiff *.aifc *.au *.snd *.mp3 *.ogg *.flac *.m4a"),
    ("WAV 文件", "*.wav *.wave"),
    ("AIFF 文件", "*.aif *.aiff *.aifc"),
    ("AU/SND 文件", "*.au *.snd"),
    ("可选扩展格式", "*.mp3 *.ogg *.flac *.m4a"),
    ("所有文件", "*.*"),
]


def normalize_audio(data):
    """
    把音频数据转为 float，并归一化到 -1 到 1 附近。

    不同音频文件的采样值范围可能不同：
    - 16 bit PCM 通常在 -32768 到 32767；
    - 8 bit PCM 可能是 0 到 255；
    - ffmpeg 解码出的浮点数据可能已经接近 -1 到 1。

    归一化的作用是让后面的能量阈值更稳定，避免同一个拨号音因为音量不同而被误判。
    """
    data = np.asarray(data, dtype=np.float64)
    peak = np.max(np.abs(data)) if data.size else 0
    if peak > 0:
        data = data / peak
    return data


def read_pcm_frames(frames, sample_width, channels, byteorder="little", unsigned_8bit=False):
    """
    将未压缩 PCM 字节流转换为 numpy 数组。

    参数说明：
    - frames：音频原始字节；
    - sample_width：每个采样点占用的字节数；
    - channels：声道数；
    - byteorder：多字节采样的字节序，WAV 通常是 little，AIFF 通常是 big；
    - unsigned_8bit：WAV 的 8 bit PCM 通常是无符号数据，需要减去 128。

    这个函数解决的是“音频封装格式”和“算法输入格式”之间的差异：
    无论原文件是 WAV、AIFF 还是 AU，只要能提取出 PCM 采样点，
    后续算法都只需要处理一维浮点数组。
    """
    if sample_width == 1:
        dtype = np.uint8 if unsigned_8bit else np.int8
        data = np.frombuffer(frames, dtype=dtype).astype(np.float64)
        if unsigned_8bit:
            data -= 128
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=f"{'<' if byteorder == 'little' else '>'}i2").astype(np.float64)
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        if byteorder == "little":
            values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        else:
            values = raw[:, 2].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 0].astype(np.int32) << 16)
        data = np.where(values >= 2 ** 23, values - 2 ** 24, values).astype(np.float64)
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=f"{'<' if byteorder == 'little' else '>'}i4").astype(np.float64)
    else:
        raise ValueError(f"暂不支持 {sample_width * 8} bit PCM 音频。")

    # 多声道音频按帧重排后取平均值。
    # 这样可以把左右声道合并为单声道，避免只取一个声道导致某些文件信息丢失。
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return normalize_audio(data)


def load_wav_audio(filepath):
    """
    读取 WAV 文件。

    优先使用 scipy.io.wavfile，因为它能直接返回采样率和数组；
    如果遇到特殊 WAV 文件导致 scipy 读取失败，就回退到 Python 标准库 wave。
    这样做可以提高兼容性，也便于课堂演示时应对不同来源的 WAV 文件。
    """
    try:
        fs, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return fs, normalize_audio(data)
    except Exception:
        with wave.open(filepath, "rb") as wf:
            fs = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        return fs, read_pcm_frames(frames, sample_width, channels, byteorder="little", unsigned_8bit=(sample_width == 1))


def load_aiff_audio(filepath):
    """
    读取 AIFF/AIFC 文件。

    AIFF 与 WAV 的主要差别之一是字节序不同：
    - WAV 常见为 little-endian；
    - AIFF 常见为 big-endian。
    因此读取后需要按 big-endian 解释采样字节。
    """
    with aifc.open(filepath, "rb") as af:
        fs = af.getframerate()
        channels = af.getnchannels()
        sample_width = af.getsampwidth()
        frames = af.readframes(af.getnframes())
    return fs, read_pcm_frames(frames, sample_width, channels, byteorder="big", unsigned_8bit=False)


def load_au_audio(filepath):
    """
    读取 AU/SND 文件。

    AU 容器可能保存线性 PCM，也可能保存 G.711 μ-law 等压缩编码。
    本项目为了让识别过程稳定可解释，只直接支持线性 PCM AU；
    如果 AU 文件是其他编码，程序会给出明确提示，而不是输出不可靠结果。
    """
    with sunau.open(filepath, "rb") as au:
        fs = au.getframerate()
        channels = au.getnchannels()
        sample_width = au.getsampwidth()
        comp_type = au.getcomptype()
        if comp_type != "NONE":
            raise ValueError(f"当前 AU/SND 文件使用 {comp_type} 编码，请使用线性 PCM AU 文件。")
        frames = au.readframes(au.getnframes())
    return fs, read_pcm_frames(frames, sample_width, channels, byteorder="big", unsigned_8bit=False)


def load_with_pydub(filepath):
    """
    读取 MP3、OGG、FLAC、M4A 等压缩音频格式。

    程序使用 imageio_ffmpeg 自带的 ffmpeg 解码为单声道 PCM 数据，
    避免手动配置系统环境变量，也避免 pydub 额外寻找 ffprobe。
    若环境没有这些组件，程序仍可正常识别 WAV、AIFF、AU。
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("读取该格式需要 ffmpeg。请执行：pip install imageio-ffmpeg") from exc

    # imageio_ffmpeg.get_ffmpeg_exe() 会返回 Python 包自带的 ffmpeg.exe 路径。
    # 这样即使系统环境变量 PATH 中没有 ffmpeg，本程序仍然可以解码 MP3 等格式。
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-v", "error",
        "-i", filepath,
        "-ac", "1",          # 输出单声道，减少后续算法处理复杂度。
        "-ar", str(FS),      # 重采样到 8000 Hz，与程序生成的 DTMF 测试音保持一致。
        "-f", "f32le",       # 输出 32 bit little-endian float PCM。
        "pipe:1",            # 不生成临时文件，直接从标准输出读取采样数据。
    ]

    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as exc:
        raise RuntimeError(f"音频解码失败：{exc}") from exc

    data = np.frombuffer(completed.stdout, dtype=np.float32).astype(np.float64)
    if data.size == 0:
        raise RuntimeError("音频解码失败：未读取到有效采样数据。")
    return FS, normalize_audio(data)


def load_audio_file(filepath):
    """
    根据文件扩展名选择合适的音频读取方式。

    原生支持：
    - .wav / .wave
    - .aif / .aiff / .aifc
    - .au / .snd

    可选支持：
    - .mp3 / .ogg / .flac / .m4a 等压缩格式，需要 pydub + ffmpeg。
    """
    # 先根据扩展名选择最合适的读取路径。
    # 最终每条路径都返回 (采样率, 单声道浮点数组)，后面的 recognize_audio 不需要关心原始格式。
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {".wav", ".wave"}:
        return load_wav_audio(filepath)
    if ext in {".aif", ".aiff", ".aifc"}:
        return load_aiff_audio(filepath)
    if ext in {".au", ".snd"}:
        return load_au_audio(filepath)
    return load_with_pydub(filepath)


def save_preview_wav(source_path, preview_path):
    """
    为 GUI 内部试听生成临时 WAV 文件。

    winsound 在 Windows 上只能稳定直接播放 WAV。
    为了避免 AIFF、AU、MP3 等格式弹出“打开方式”窗口，这里先复用 load_audio_file
    把任意支持格式解码成采样数组，再写成一个临时 WAV，最后仍用 winsound 播放。
    """
    fs, data = load_audio_file(source_path)
    data = normalize_audio(data)
    wavfile.write(preview_path, fs, (data * 32767).astype(np.int16))
    return preview_path


def goertzel_algorithm(samples, target_freq, fs):
    """
    Goertzel 算法：只计算指定频点的能量。

    对 DTMF 识别来说，我们只关心固定的几个标准频率。
    与完整 FFT 相比，Goertzel 算法更直接，计算量也更小。

    算法理解：
    - FFT 会把一个窗口内的所有频率成分都算出来；
    - Goertzel 更像是“只问某一个频率有没有明显存在”；
    - DTMF 只有少数候选频点，因此逐个计算这些频点能量即可。

    返回值不是实际物理功率，而是该目标频点在当前窗口中的相对能量。
    能量越大，说明当前窗口越可能包含该频率分量。
    """
    n = len(samples)

    # k 是目标频率在当前窗口频率网格中最接近的索引。
    # 由于窗口长度有限，目标频率不一定刚好落在整数频点上，因此这里使用四舍五入。
    k = int(0.5 + (n * target_freq) / fs)
    w = (2.0 * np.pi / n) * k
    cosine = np.cos(w)

    # s_prev 和 s_prev2 分别保存递推过程中的前一项和前两项。
    # Goertzel 的优势就是不用保存完整频谱，只需要递推这两个状态量。
    s_prev, s_prev2 = 0.0, 0.0

    for x in samples:
        s = x + 2.0 * cosine * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s

    # 递推结束后，用最后两个状态量计算目标频点能量。
    return s_prev2 ** 2 + s_prev ** 2 - 2.0 * cosine * s_prev * s_prev2


def recognize_audio(filepath):
    """
    识别音频文件中的 DTMF 拨号号码。

    处理链路：
    1. 读取音频文件并统一为单声道浮点信号；
    2. 按短时窗口扫描音频；
    3. 每个窗口分别检测低频组和高频组能量峰值；
    4. 将频率组合映射为数字；
    5. 合并连续窗口中的同一数字，得到最终号码。
    """
    # 读取音频。这里不关心输入文件原来是什么格式，
    # 因为 load_audio_file 已经把它统一转换为“采样率 + 一维音频数组”。
    fs, data = load_audio_file(filepath)
    data = normalize_audio(data)

    # 40 ms 窗口能覆盖多个 DTMF 周期，20 ms 步长用于平滑扫描。
    # 窗口太短：频率判断不稳定；窗口太长：按键边界不清晰。
    # 40 ms 对本项目生成的标准 DTMF 音频比较合适。
    window_size = int(fs * 0.04)
    step_size = int(fs * 0.02)

    recognized_number = ""
    last_detected = None

    # 使用滑动窗口遍历整段音频。
    # 每次取出一小段 window，判断这一小段是否包含一个 DTMF 按键音。
    for i in range(0, len(data) - window_size, step_size):
        window = data[i:i + window_size]

        # 分别计算低频组和高频组的候选频点能量。
        # DTMF 的一个有效按键必须同时有一个低频峰值和一个高频峰值。
        low_energies = {f: goertzel_algorithm(window, f, fs) for f in LOW_FREQS}
        high_energies = {f: goertzel_algorithm(window, f, fs) for f in HIGH_FREQS}

        # 找出当前窗口中最强的低频点和最强的高频点。
        best_low = max(low_energies, key=low_energies.get)
        best_high = max(high_energies, key=high_energies.get)

        # 归一化后，标准 DTMF 音在目标频点会有明显能量峰。
        # 固定阈值适合本课程生成的标准音频；如果处理嘈杂实录音，可改为自适应阈值。
        if low_energies[best_low] > 100.0 and high_energies[best_high] > 100.0:
            # 用“最佳低频 + 最佳高频”回到 DTMF_TABLE 中查找对应按键。
            current_digit = next(
                (char for char, freqs in DTMF_TABLE.items() if freqs == (best_low, best_high)),
                None,
            )
            if current_digit and current_digit != last_detected:
                # 一个按键音通常会持续多个窗口。
                # 如果每个窗口都追加一次，就会把一个 5 错误识别成 5555。
                # 因此只有当前按键与上一窗口不同，才追加到结果中。
                recognized_number += current_digit
                last_detected = current_digit
        else:
            # 能量低于阈值通常表示静音、间隔或非 DTMF 信号。
            # 把 last_detected 清空后，下一个相同数字才可以被识别为新的按键。
            last_detected = None

    return recognized_number


class DTMFApp:
    """DTMF 识别系统的 GUI 主类。"""

    def __init__(self, root):
        self.root = root
        self.root.title("拨号音分析系统 (DTMF Recognition)")
        self.root.geometry("750x760")
        self.root.configure(bg="#F4F5F7")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_file = os.path.join(self.base_dir, "generated_test.wav")
        self.audio_cache_dir = os.path.join(self.base_dir, "dtmf_cache")
        self.preview_file = os.path.join(self.base_dir, "preview_playback.wav")

        self.seq_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="未选择任何文件")
        self.result_var = tk.StringVar(value=" 等待识别... ")

        os.makedirs(self.audio_cache_dir, exist_ok=True)
        self.pre_generate_tones()
        self.seq_var.trace_add("write", self.update_display)
        self.setup_ui()

    def pre_generate_tones(self):
        """提前生成每个按键的试听音，点击拨号盘时可立即播放。"""
        for key, (f_l, f_h) in DTMF_TABLE.items():
            safe_key = {"*": "star", "#": "hash"}.get(key, key)
            path = os.path.join(self.audio_cache_dir, f"{safe_key}.wav")
            if not os.path.exists(path):
                t = np.linspace(0, 0.12, int(FS * 0.12), endpoint=False)
                sig = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
                wavfile.write(path, FS, (sig * 32767).astype(np.int16))

    def update_display(self, *args):
        """同步更新拨号盘上方的号码显示区域。"""
        if hasattr(self, "display_text_id"):
            self.canvas.itemconfig(self.display_text_id, text=self.seq_var.get())

    def setup_ui(self):
        """构建 GUI 页面，包括拨号盘、文件选择区、结果区和波形区。"""
        left_frame = tk.Frame(self.root, width=350, bg="#F4F5F7")
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        right_frame = tk.Frame(
            self.root,
            width=350,
            relief="flat",
            bg="white",
            highlightbackground="#E0E0E0",
            highlightthickness=1,
        )
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        tk.Label(left_frame, text="手机仿真拨号盘", font=("微软雅黑", 14, "bold"), bg="#F4F5F7").pack(pady=5)
        self.canvas = tk.Canvas(left_frame, width=310, height=560, bg="#F4F5F7", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_rectangle(15, 10, 295, 70, fill="white", outline="#E0E0E0", width=1)
        self.display_text_id = self.canvas.create_text(
            155, 40, text="", font=("Consolas", 26, "bold"), fill="#1976D2"
        )

        def create_btn(x, y, r, t1, t2, val):
            """在画布上绘制一个圆形拨号按钮，并绑定点击事件。"""
            tag = f"btn_{val}"
            self.canvas.create_oval(x - r + 1, y - r + 2, x + r + 1, y + r + 2, fill="#D6D8DC", outline="", tags=tag)
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="#E8EAF6", tags=tag)
            self.canvas.create_text(x, y - 8, text=t1, font=("微软雅黑", 22, "bold"), tags=tag)
            self.canvas.create_text(x, y + 14, text=t2, font=("微软雅黑", 8, "bold"), fill="gray", tags=tag)
            self.canvas.tag_bind(tag, "<Button-1>", lambda e, v=val: self.press_key(v))

        keys = [
            ("1", "", "1"), ("2", "ABC", "2"), ("3", "DEF", "3"),
            ("4", "GHI", "4"), ("5", "JKL", "5"), ("6", "MNO", "6"),
            ("7", "PQRS", "7"), ("8", "TUV", "8"), ("9", "WXYZ", "9"),
            ("*", "", "*"), ("0", "+", "0"), ("#", "", "#"),
        ]

        for i, (t1, t2, v) in enumerate(keys):
            row, col = i // 3, i % 3
            create_btn(65 + col * 90, 125 + row * 85, 36, t1, t2, v)

        call_x, call_y, call_r = 155, 485, 40
        tag_call = "call_btn"
        self.canvas.create_oval(
            call_x - call_r + 1,
            call_y - call_r + 2,
            call_x + call_r + 1,
            call_y + call_r + 2,
            fill="#a0c0a0",
            outline="",
            tags=tag_call,
        )
        self.canvas.create_oval(
            call_x - call_r,
            call_y - call_r,
            call_x + call_r,
            call_y + call_r,
            fill="#4CAF50",
            outline="",
            tags=tag_call,
        )
        self.canvas.create_text(call_x, call_y, text="CALL", fill="white", font=("微软雅黑", 12, "bold"), tags=tag_call)
        self.canvas.tag_bind(tag_call, "<Button-1>", lambda e: self.build_audio())

        clear_btn = self.canvas.create_text(65, 485, text="清空", font=("微软雅黑", 11, "bold"), fill="#757575")
        self.canvas.tag_bind(clear_btn, "<Button-1>", lambda e: self.seq_var.set(""))
        back_btn = self.canvas.create_text(245, 485, text="退格", font=("微软雅黑", 11, "bold"), fill="#757575")
        self.canvas.tag_bind(back_btn, "<Button-1>", lambda e: self.delete_key())

        header = tk.Frame(right_frame, bg="#3F51B5", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="信号解析终端", font=("微软雅黑", 12, "bold"), bg="#3F51B5", fg="white").pack()

        tk.Label(right_frame, text="待检测文件:", bg="white", font=("微软雅黑", 9, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
        tk.Label(right_frame, textvariable=self.file_var, fg="#616161", wraplength=300, bg="#F5F5F5", justify="left").pack(fill="x", padx=15, ipady=3)

        btn_box = tk.Frame(right_frame, bg="white")
        btn_box.pack(fill="x", padx=15, pady=10)
        tk.Button(btn_box, text="载入音频", command=self.load_file, relief="groove").pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(btn_box, text="试听", command=self.play_audio, relief="groove").pack(side="right", expand=True, fill="x", padx=2)

        tk.Button(
            right_frame,
            text="启动算法识别号码",
            font=("微软雅黑", 12, "bold"),
            bg="#FF9800",
            fg="white",
            relief="flat",
            command=self.run_recognition,
        ).pack(fill="x", padx=20, pady=(15, 5), ipady=6)

        tk.Label(right_frame, textvariable=self.result_var, font=("Consolas", 32, "bold"), fg="#D32F2F", bg="#FFEBEE").pack(pady=5, fill="x", padx=15, ipady=10)

        tk.Label(right_frame, text="实时时域波形 (Oscilloscope)", bg="white", font=("微软雅黑", 10, "bold")).pack(pady=(10, 0))
        self.fig, self.ax = plt.subplots(figsize=(4, 1.8), dpi=80)
        self.fig.patch.set_facecolor("#F5F5F5")
        self.ax.set_facecolor("#000000")
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.line, = self.ax.plot(np.zeros(200), color="#00FF00", linewidth=1.5)
        self.ax.set_ylim(-1.2, 1.2)
        self.fig.tight_layout(pad=0.5)

    def update_plot(self, signal):
        """更新示波器区域，只显示前 200 个采样点，避免整段波形过密。"""
        view_window = np.zeros(200)
        signal = np.asarray(signal, dtype=float)
        view_window[:min(200, len(signal))] = signal[:200]
        self.line.set_ydata(view_window)
        self.canvas_plot.draw_idle()

    def press_key(self, key):
        """处理拨号盘按键：追加号码、生成当前按键波形并播放提示音。"""
        self.seq_var.set(self.seq_var.get() + key)
        f_l, f_h = DTMF_TABLE[key]

        t = np.linspace(0, 0.15, int(FS * 0.15), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * f_l * t) + 0.5 * np.sin(2 * np.pi * f_h * t)
        self.update_plot(sig)

        safe_key = {"*": "star", "#": "hash"}.get(key, key)
        path = os.path.join(self.audio_cache_dir, f"{safe_key}.wav")
        if os.path.exists(path):
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def delete_key(self):
        """删除最后一个拨号字符，并清空示波器显示。"""
        s = self.seq_var.get()
        if s:
            self.seq_var.set(s[:-1])
        self.update_plot(np.zeros(200))

    def build_audio(self):
        """把当前拨号序列合成为 generated_test.wav，供后续识别和演示。"""
        seq = self.seq_var.get()
        if not seq:
            messagebox.showwarning("提示", "请先在左侧拨号盘输入号码。")
            return

        # 每个按键音持续 0.25 秒，按键之间保留 0.08 秒静音。
        # 静音段的作用是给识别算法一个“分隔”，尤其是连续输入相同数字时很重要。
        t_tone = np.linspace(0, 0.25, int(FS * 0.25), endpoint=False)
        pause = np.zeros(int(FS * 0.08))
        full = np.array([])

        for char in seq:
            f_l, f_h = DTMF_TABLE[char]
            # 按键音 = 低频正弦 + 高频正弦。
            # 两个分量的系数都设为 0.5，避免叠加后幅值过大。
            tone = 0.5 * np.sin(2 * np.pi * f_l * t_tone) + 0.5 * np.sin(2 * np.pi * f_h * t_tone)
            full = np.concatenate((full, tone, pause))

        # 保存为 16 bit PCM WAV，这是最常见、最容易被其他软件读取的音频格式。
        wavfile.write(self.target_file, FS, (full * 32767).astype(np.int16))
        self.file_var.set(self.target_file)
        messagebox.showinfo("成功", "拨号音已生成，请点击右侧“启动算法识别号码”。")

    def load_file(self):
        """从磁盘选择待识别音频文件。"""
        f = filedialog.askopenfilename(filetypes=SUPPORTED_FILETYPES, initialdir=self.base_dir)
        if f:
            self.file_var.set(f)

    def play_audio(self):
        """
        试听当前文件。

        旧版本对非 WAV 文件使用 os.startfile，会跳到系统默认播放器或弹出“打开方式”窗口。
        现在改为在程序内部完成试听：先把当前文件解码成临时 WAV，再用 winsound 播放。
        """
        path = self.file_var.get()
        if not os.path.exists(path):
            messagebox.showwarning("提示", "请先选择一个音频文件。")
            return

        try:
            # 先停止上一次异步播放，避免连续点击试听时多个声音叠在一起。
            winsound.PlaySound(None, winsound.SND_PURGE)

            ext = os.path.splitext(path)[1].lower()
            if ext in {".wav", ".wave"}:
                play_path = path
            else:
                play_path = save_preview_wav(path, self.preview_file)

            winsound.PlaySound(play_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            messagebox.showerror("试听失败", str(exc))

    def run_recognition(self):
        """运行识别算法，并把结果显示在 GUI 中。"""
        path = self.file_var.get()
        if not os.path.exists(path) or path == "未选择任何文件":
            messagebox.showwarning("提示", "请先载入或生成一个音频文件。")
            return

        try:
            self.result_var.set("识别中...")
            self.root.update()
            time.sleep(0.2)
            result = recognize_audio(path)
            self.result_var.set(result if result else "无结果")
        except Exception as exc:
            messagebox.showerror("识别失败", str(exc))
            self.result_var.set("识别失败")


if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()
