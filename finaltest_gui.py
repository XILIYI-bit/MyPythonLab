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
import queue
import os
import subprocess
import sunau
import threading
import time
import wave
import winsound
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter
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

# 实时监听参数说明。
# 这些参数只影响“麦克风实时识别”，不会改变文件识别和程序生成标准拨号音的逻辑。
#
# LIVE_BLOCK_SECONDS：声卡回调每次送入程序的音频块长度。
# 20 ms 的块足够短，界面波形刷新及时；又不会短到让 Tk 主线程频繁处理音频队列。
LIVE_BLOCK_SECONDS = 0.02
LIVE_BLOCK_SIZE = int(FS * LIVE_BLOCK_SECONDS)

# LIVE_WINDOW_SECONDS：每次做 DTMF 频率判决时使用的分析窗口。
# DTMF 最低频率是 697 Hz，50 ms 内已经包含三十多个周期，足够计算稳定频率；
# 同时窗口不宜过长，否则快速连点两个相同数字时，中间的短暂停顿会被窗口跨过去。
LIVE_WINDOW_SECONDS = 0.05
LIVE_WINDOW_SIZE = int(FS * LIVE_WINDOW_SECONDS)

# LIVE_HOP_SECONDS：滑动窗口步长。
# 10 ms 步长表示每 10 ms 重新判定一次，能在快速拨号时捕捉按键边界。
LIVE_HOP_SECONDS = 0.01
LIVE_HOP_SIZE = int(FS * LIVE_HOP_SECONDS)

# LIVE_MIN_RAW_RMS：原始麦克风输入的最低有效强度。
# 小于该值通常是背景底噪或安静间隔，直接作为“无按键”处理，减少人声和环境声误触发。
LIVE_MIN_RAW_RMS = 0.0035

# LIVE_CONFIRM_FRAMES：同一个数字连续出现多少个窗口后才确认输出。
# 设置为 2 可以兼顾快速连点；如果设成 1，人声瞬时谐波更容易误判。
LIVE_CONFIRM_FRAMES = 2

# LIVE_RELEASE_FRAMES / LIVE_RELEASE_BLOCKS：按键释放判定。
# 当检测窗口短暂失效，或输入块出现明显静音，就认为当前按键已经结束。
# 这对“55”这种同键快速连点很关键，否则第二个 5 会被当作第一个 5 的延续。
LIVE_RELEASE_FRAMES = 1
LIVE_RELEASE_BLOCKS = 1

# LIVE_FAST_CONFIRM_CONFIDENCE：强置信度快速确认阈值。
# 当低频组和高频组的主峰都远强于第二峰时，可以更快确认，降低快速拨号漏检。
LIVE_FAST_CONFIRM_CONFIDENCE = 18.0

# LIVE_MIN_EMIT_GAP_SECONDS：两次输出之间的最小间隔。
# 它防止一个长按键被连续多个窗口重复写入，同时保留快速连点的响应空间。
LIVE_MIN_EMIT_GAP_SECONDS = 0.035

# LIVE_ACTIVE_MIN_SECONDS：一个已确认按键至少保持的时间。
# 这样可以避免刚确认后立即被一个边缘窗口释放，导致同一个长按被拆成多个数字。
LIVE_ACTIVE_MIN_SECONDS = 0.045

# LIVE_RELATIVE_RELEASE_RATIO：相对能量释放比例。
# 录音环境不一定能回到绝对静音，所以用“当前块能量低于已确认按键峰值的一定比例”
# 来判断手指是否已经松开或手机是否已经停止发出当前拨号音。
LIVE_RELATIVE_RELEASE_RATIO = 0.38

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


def make_dtmf_bandpass(fs):
    """
    生成 DTMF 频段带通滤波器系数。

    DTMF 标准频率集中在 697 Hz 到 1633 Hz。
    这里保留约 650 Hz 到 1700 Hz 的频带，可以削弱人声低频、环境低频震动和高频噪声。

    答辩时可以说明：
    人声、敲击声、风噪的能量往往分布很宽，而 DTMF 只关心固定的八个频点；
    先做带通滤波，相当于把问题从“复杂环境声分析”缩小为“电话拨号频段检测”。
    """
    low = 650 / (fs / 2)
    high = 1700 / (fs / 2)
    return butter(4, [low, high], btype="bandpass")


def reduce_environment_noise(samples, fs=FS, noise_floor=0.02, bandpass=None, filter_state=None):
    """
    对实时环境音做轻量降噪。

    本项目不依赖额外 noisereduce 包，而是采用两步处理：
    1. 带通滤波，只保留 DTMF 所在频段；
    2. 噪声门限，将低于环境噪声水平的小幅值采样压低。

    这种方法对课堂演示场景足够直接，也能解释为信号与系统中的滤波和阈值判决。

    filter_state 是实时处理的关键：
    麦克风音频是一块一块送入程序的，如果每块都重新开始滤波，块边界会产生瞬态失真；
    保留 lfilter 的状态，相当于让滤波器连续工作，频谱会更稳定。
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return samples

    # b、a 是 IIR 带通滤波器的分子和分母系数。
    # 如果外部已经创建好滤波器，就复用它，避免实时循环里反复设计滤波器。
    b, a = bandpass if bandpass else make_dtmf_bandpass(fs)
    if filter_state is None:
        # 文件识别或一次性处理时不需要保存状态。
        filtered = lfilter(b, a, samples)
        next_state = None
    else:
        # 实时监听时传入上一块音频结束后的滤波状态，保证下一块从同一状态继续。
        filtered, next_state = lfilter(b, a, samples, zi=filter_state)
    # 注意这里不是把低于门限的采样点直接清零，而是压低到 35%。
    # 原因是 DTMF 本质是两条连续正弦波，硬清零会制造“削顶/断裂”，反而污染频谱。
    gate = max(noise_floor * 1.2, 0.002)
    filtered = np.where(np.abs(filtered) < gate, filtered * 0.35, filtered)
    # 如果外界声源离麦克风过近，滤波后幅度可能超过 1。这里仅做保护性缩放，
    # 保持后续绘图和阈值判断不被异常峰值拖垮。
    peak = np.max(np.abs(filtered)) if filtered.size else 0.0
    if peak > 1.0:
        filtered = filtered / peak
    if filter_state is None:
        return filtered
    return filtered, next_state


def detect_dtmf_digit(
    window,
    fs=FS,
    min_energy=2.0,
    dominance=2.8,
    min_pair_share=0.68,
    max_twist=16.0,
    min_spectral_purity=0.58,
    max_freq_error=32.0,
    min_sharpness=1.8,
    min_rms=0.004,
):
    """
    判断一个短时窗口中是否存在 DTMF 按键。

    返回值为 (digit, confidence)：
    - digit 为识别到的按键字符，未检测到则为 None；
    - confidence 是低频组和高频组峰值相对第二峰值的较小比值。

    dominance 用于避免把普通噪声误判为按键：真正 DTMF 信号在低频组和高频组各有一个明显峰值。

    本函数的核心思想可以用于答辩：
    不是“听起来像不像数字”，而是检查一个窗口是否同时满足 DTMF 的物理结构。
    一个有效按键必须有一个低频组频点和一个高频组频点，并且这两个频点要尖锐、稳定、
    能量占比足够高；人声虽然也可能含有接近频率，但通常是宽频和谐波混合，不会长期满足这些条件。
    """
    window = np.asarray(window, dtype=np.float64)
    if len(window) < int(fs * 0.04):
        return None, 0.0

    # 去直流分量。麦克风或声卡可能带有很小的 DC 偏移，
    # 不去掉会影响 RMS 和低频能量估计。
    window = window - np.mean(window)
    raw_rms = float(np.sqrt(np.mean(window ** 2))) if window.size else 0.0
    if raw_rms < min_rms:
        return None, 0.0

    # 汉宁窗用于减小窗口截断带来的频谱泄漏。
    # 答辩时可以说：实际窗口不一定刚好从正弦波零点开始结束，加窗能让目标频点更突出。
    tapered = window * np.hanning(len(window))

    # 分别计算低频组和高频组的候选频点能量。
    # DTMF 键盘每个按键都由“一低一高”两个频率组成，所以必须分组判断。
    low_energies = {f: goertzel_algorithm(tapered, f, fs) for f in LOW_FREQS}
    high_energies = {f: goertzel_algorithm(tapered, f, fs) for f in HIGH_FREQS}

    # 选出每组能量最大的频点，作为当前窗口的候选按键。
    best_low = max(low_energies, key=low_energies.get)
    best_high = max(high_energies, key=high_energies.get)
    low_values = sorted(low_energies.values(), reverse=True)
    high_values = sorted(high_energies.values(), reverse=True)
    low_peak = low_values[0]
    high_peak = high_values[0]

    # dominance / confidence：主峰相对第二峰的优势。
    # 如果人声在多个候选频点都有能量，主峰优势会下降，因此不会被当作稳定 DTMF。
    low_ratio = low_peak / max(low_values[1], 1e-9)
    high_ratio = high_peak / max(high_values[1], 1e-9)
    confidence = min(low_ratio, high_ratio)

    # pair_share：最佳低频 + 最佳高频在八个候选频点总能量中的占比。
    # 真正 DTMF 应该主要集中在两个频点；如果能量分散，说明更可能是人声或噪声。
    total_energy = sum(low_energies.values()) + sum(high_energies.values())
    pair_share = (low_peak + high_peak) / max(total_energy, 1e-9)

    # twist：低频和高频之间的强弱比。
    # 手机外放和麦克风会让高频变弱，所以阈值不能太死；但如果一边极强一边极弱，也不应通过。
    twist = max(low_peak, high_peak) / max(min(low_peak, high_peak), 1e-9)

    # 除了 Goertzel 的八个频点能量，还用 FFT 观察整个 DTMF 频带。
    # 这样可以判断目标频点是否真的是窄带峰值，而不是宽频噪声里碰巧最大的点。
    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    freqs = np.fft.rfftfreq(len(window), 1.0 / fs)
    band_mask = (freqs >= 600) & (freqs <= 1800)
    band_power = float(np.sum(spectrum[band_mask]))
    target_power = 0.0
    max_freq_error_seen = 0.0
    sharpness_values = []
    for target in (best_low, best_high):
        # 目标频率附近 32 Hz 的能量记为有效目标能量。
        # 这个范围允许手机扬声器和声卡采样带来的小偏移。
        target_mask = np.abs(freqs - target) <= 32
        target_power += float(np.sum(spectrum[target_mask]))
        if np.any(target_mask):
            local_freqs = freqs[target_mask]
            local_power = spectrum[target_mask]
            peak_freq = float(local_freqs[int(np.argmax(local_power))])
            max_freq_error_seen = max(max_freq_error_seen, abs(peak_freq - target))

        # sharpness 用目标频点能量对比左右 65 Hz 的“保护频点”能量。
        # 如果目标附近只是宽宽的一片能量，人声的可能性更高；真正 DTMF 会在标准频点形成尖峰。
        guard_energies = [
            goertzel_algorithm(tapered, guard_freq, fs)
            for guard_freq in (target - 65, target + 65)
            if 0 < guard_freq < fs / 2
        ]
        if guard_energies:
            target_energy = low_peak if target == best_low else high_peak
            sharpness_values.append(target_energy / max(max(guard_energies), 1e-9))
    spectral_purity = target_power / max(band_power, 1e-9)
    sharpness = min(sharpness_values) if sharpness_values else 0.0

    # 以下是逐层否决规则。每一层都对应一个答辩点：
    # 能量足够、组内主峰明确、双音集中、两音比例合理、频谱足够纯、频率偏差不大。
    if low_peak < min_energy or high_peak < min_energy:
        return None, confidence
    if confidence < dominance:
        return None, confidence
    if pair_share < min_pair_share:
        return None, confidence
    if twist > max_twist:
        return None, confidence
    if spectral_purity < min_spectral_purity:
        return None, confidence
    if sharpness < min_sharpness:
        return None, confidence
    if max_freq_error_seen > max_freq_error:
        return None, confidence

    digit = next(
        (char for char, freqs in DTMF_TABLE.items() if freqs == (best_low, best_high)),
        None,
    )
    return digit, confidence


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
    if n == 0:
        return 0.0

    # 直接使用目标频率本身，而不是四舍五入到 FFT 频率栅格。
    # 实时录音的窗口长度会变化；精确频率版本对 697、770 等非整数频点更稳定。
    w = 2.0 * np.pi * target_freq / fs
    cosine = np.cos(w)

    # s_prev 和 s_prev2 分别保存递推过程中的前一项和前两项。
    # Goertzel 的优势就是不用保存完整频谱，只需要递推这两个状态量。
    s_prev, s_prev2 = 0.0, 0.0

    for x in samples:
        s = x + 2.0 * cosine * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s

    # 递推结束后，用最后两个状态量计算目标频点能量。
    return max(s_prev2 ** 2 + s_prev ** 2 - 2.0 * cosine * s_prev * s_prev2, 0.0)


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
        self.root.geometry("820x780")
        self.root.configure(bg="#F4F5F7")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.target_file = os.path.join(self.base_dir, "generated_test.wav")
        self.audio_cache_dir = os.path.join(self.base_dir, "dtmf_cache")
        self.preview_file = os.path.join(self.base_dir, "preview_playback.wav")

        self.seq_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="未选择任何文件")
        self.result_var = tk.StringVar(value=" 等待识别... ")
        self.live_result_var = tk.StringVar(value="")
        self.live_status_var = tk.StringVar(value="未监听")
        self.live_noise_var = tk.StringVar(value="环境噪声: 未校准")

        self.active_page = None
        self.live_running = False
        self.live_stream = None
        self.live_queue = queue.Queue()
        self.live_buffer = np.zeros(0, dtype=np.float64)
        self.live_noise_floor = 0.002
        self.live_last_digit = None
        self.live_active_digit = None
        self.live_scan_pos = 0
        self.live_last_emit_pos = -10 ** 9
        self.live_release_pos = 0
        self.live_active_peak_rms = 0.0
        self.live_candidate_digit = None
        self.live_candidate_count = 0
        self.live_silence_blocks = 0
        self.live_input_silence_blocks = 0
        self.live_recent_raw_rms = 0.0
        self.live_input_fs = FS
        self.live_window_size = LIVE_WINDOW_SIZE
        self.live_hop_size = LIVE_HOP_SIZE
        self.live_min_emit_gap = int(FS * LIVE_MIN_EMIT_GAP_SECONDS)
        self.live_active_min_samples = int(FS * LIVE_ACTIVE_MIN_SECONDS)
        self.live_bandpass = None
        self.live_filter_state = None
        self.configure_live_processing(FS)
        self.live_lock = threading.Lock()

        os.makedirs(self.audio_cache_dir, exist_ok=True)
        self.pre_generate_tones()
        self.seq_var.trace_add("write", self.update_display)
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_live_processing(self, fs):
        """
        按照当前录音采样率配置实时检测窗口和连续滤波状态。

        有些声卡支持 8000 Hz，有些默认是 44100/48000 Hz。
        程序不再把每个小块强行重采样，而是按实际采样率重新计算窗口长度和滤波器；
        这样可以减少块边界重采样带来的抖动，实时波形和频率判决都会更稳定。
        """
        self.live_input_fs = int(fs)
        # 窗口、步长和最小输出间隔都用“秒”定义，再换算成当前采样率下的采样点数。
        # 这样无论声卡运行在 8 kHz 还是 48 kHz，算法的时间尺度都一致。
        self.live_window_size = max(int(self.live_input_fs * LIVE_WINDOW_SECONDS), int(self.live_input_fs * 0.04))
        self.live_hop_size = max(1, int(self.live_input_fs * LIVE_HOP_SECONDS))
        self.live_min_emit_gap = max(1, int(self.live_input_fs * LIVE_MIN_EMIT_GAP_SECONDS))
        self.live_active_min_samples = max(1, int(self.live_input_fs * LIVE_ACTIVE_MIN_SECONDS))

        # 实时滤波器只创建一次，并保存 zi 状态。zi 可以理解为 IIR 滤波器的“记忆”。
        self.live_bandpass = make_dtmf_bandpass(self.live_input_fs)
        b, a = self.live_bandpass
        self.live_filter_state = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)

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
        nav_frame = tk.Frame(self.root, bg="#E9ECF5", height=44)
        nav_frame.pack(side="top", fill="x")
        self.file_tab_btn = tk.Button(
            nav_frame,
            text="文件识别",
            command=lambda: self.switch_page("file"),
            relief="flat",
            font=("微软雅黑", 11, "bold"),
            padx=18,
        )
        self.file_tab_btn.pack(side="left", padx=(12, 4), pady=7)
        self.live_tab_btn = tk.Button(
            nav_frame,
            text="实时监听",
            command=lambda: self.switch_page("live"),
            relief="flat",
            font=("微软雅黑", 11, "bold"),
            padx=18,
        )
        self.live_tab_btn.pack(side="left", padx=4, pady=7)

        self.page_holder = tk.Frame(self.root, bg="#F4F5F7")
        self.page_holder.pack(side="top", fill="both", expand=True)
        self.file_page = tk.Frame(self.page_holder, bg="#F4F5F7")
        self.live_page = tk.Frame(self.page_holder, bg="#F4F5F7")

        left_frame = tk.Frame(self.file_page, width=350, bg="#F4F5F7")
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        right_frame = tk.Frame(
            self.file_page,
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
        self.canvas.tag_bind(clear_btn, "<Button-1>", lambda e: self.clear_sequence())
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
        self.setup_live_page()
        self.switch_page("file")

    def switch_page(self, page_name):
        """通过顶部按钮在文件识别页和实时监听页之间切换。"""
        if self.active_page == page_name:
            return
        for page in (self.file_page, self.live_page):
            page.pack_forget()
        if page_name == "file":
            self.file_page.pack(fill="both", expand=True)
            self.file_tab_btn.configure(bg="#3F51B5", fg="white")
            self.live_tab_btn.configure(bg="#D8DCE8", fg="#333333")
        else:
            self.live_page.pack(fill="both", expand=True)
            self.live_tab_btn.configure(bg="#3F51B5", fg="white")
            self.file_tab_btn.configure(bg="#D8DCE8", fg="#333333")
        self.active_page = page_name

    def setup_live_page(self):
        """构建实时监听页面，用于从麦克风环境音中识别 DTMF 拨号音。"""
        left_frame = tk.Frame(self.live_page, width=300, bg="#F4F5F7")
        left_frame.pack(side="left", fill="y", padx=14, pady=14)
        right_frame = tk.Frame(
            self.live_page,
            bg="white",
            highlightbackground="#E0E0E0",
            highlightthickness=1,
        )
        right_frame.pack(side="right", fill="both", expand=True, padx=14, pady=14)

        tk.Label(left_frame, text="实时环境监听", bg="#F4F5F7", font=("微软雅黑", 15, "bold")).pack(anchor="w", pady=(4, 12))
        tk.Label(left_frame, textvariable=self.live_status_var, bg="#E8F5E9", fg="#1B5E20", font=("微软雅黑", 11, "bold")).pack(fill="x", ipady=8)
        tk.Label(left_frame, textvariable=self.live_noise_var, bg="#F4F5F7", fg="#616161", font=("微软雅黑", 9)).pack(anchor="w", pady=(12, 6))

        tk.Button(
            left_frame,
            text="开始监听",
            command=self.start_live_listening,
            bg="#4CAF50",
            fg="white",
            relief="flat",
            font=("微软雅黑", 12, "bold"),
        ).pack(fill="x", pady=(18, 6), ipady=7)
        tk.Button(
            left_frame,
            text="停止监听",
            command=self.stop_live_listening,
            bg="#757575",
            fg="white",
            relief="flat",
            font=("微软雅黑", 12, "bold"),
        ).pack(fill="x", pady=6, ipady=7)
        tk.Button(
            left_frame,
            text="清空实时结果",
            command=self.clear_live_result,
            relief="groove",
            font=("微软雅黑", 11),
        ).pack(fill="x", pady=(6, 18), ipady=5)

        info = (
            "监听逻辑：麦克风采集 -> DTMF 频段带通滤波 -> 噪声门限降噪 -> "
            "Goertzel 频点能量检测。请把拨号音源靠近麦克风，环境越安静越稳定。"
        )
        tk.Label(left_frame, text=info, wraplength=260, justify="left", bg="#F4F5F7", fg="#555555").pack(anchor="w")

        header = tk.Frame(right_frame, bg="#3F51B5", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="实时识别结果", font=("微软雅黑", 12, "bold"), bg="#3F51B5", fg="white").pack()

        tk.Label(
            right_frame,
            textvariable=self.live_result_var,
            font=("Consolas", 36, "bold"),
            fg="#D32F2F",
            bg="#FFEBEE",
        ).pack(fill="x", padx=18, pady=(16, 10), ipady=14)

        tk.Label(right_frame, text="降噪后实时波形", bg="white", font=("微软雅黑", 10, "bold")).pack(pady=(8, 0))
        self.live_fig, self.live_ax = plt.subplots(figsize=(4.6, 2.2), dpi=85)
        self.live_fig.patch.set_facecolor("#F5F5F5")
        self.live_ax.set_facecolor("#000000")
        self.live_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.live_canvas_plot = FigureCanvasTkAgg(self.live_fig, master=right_frame)
        self.live_canvas_plot.get_tk_widget().pack(fill="both", expand=True, padx=18, pady=(6, 18))
        self.live_line, = self.live_ax.plot(np.zeros(400), color="#00E676", linewidth=1.3)
        self.live_ax.set_ylim(-1.2, 1.2)
        self.live_fig.tight_layout(pad=0.5)

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

    def clear_sequence(self):
        """清空拨号字符，同时清空示波器中的时域波形。"""
        self.seq_var.set("")
        self.update_plot(np.zeros(200))

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

    def clear_live_result(self):
        """清空实时监听页面上的识别结果和波形。"""
        self.live_result_var.set("")
        self.live_last_digit = None
        self.live_active_digit = None
        self.live_scan_pos = 0
        self.live_last_emit_pos = -10 ** 9
        self.live_release_pos = 0
        self.live_active_peak_rms = 0.0
        self.live_candidate_digit = None
        self.live_candidate_count = 0
        self.live_silence_blocks = 0
        self.live_input_silence_blocks = 0
        self.live_recent_raw_rms = 0.0
        self.live_buffer = np.zeros(0, dtype=np.float64)
        if self.live_bandpass is not None:
            b, a = self.live_bandpass
            self.live_filter_state = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)
        if hasattr(self, "live_line"):
            self.live_line.set_ydata(np.zeros(400))
            self.live_canvas_plot.draw_idle()

    def start_live_listening(self):
        """启动麦克风实时监听。"""
        if self.live_running:
            return

        try:
            import sounddevice as sd
        except ImportError:
            messagebox.showerror(
                "缺少实时录音库",
                "实时监听需要安装 sounddevice。\n请在命令行执行：py -3 -m pip install sounddevice",
            )
            return

        self.clear_live_result()
        self.live_queue = queue.Queue()
        self.live_noise_floor = 0.002
        self.live_running = True
        self.live_status_var.set("正在监听麦克风...")
        self.live_noise_var.set("环境噪声: 校准中")

        try:
            device_info = sd.query_devices(kind="input")
            default_fs = int(device_info.get("default_samplerate", FS))

            def audio_callback(indata, frames, callback_time, status):
                if status:
                    self.live_queue.put(("status", str(status)))
                mono = np.asarray(indata[:, 0], dtype=np.float64).copy()
                self.live_queue.put(("audio", mono))

            last_error = None
            for candidate_fs in dict.fromkeys([FS, default_fs]):
                try:
                    self.configure_live_processing(candidate_fs)
                    block_size = max(128, int(candidate_fs * LIVE_BLOCK_SECONDS))
                    self.live_stream = sd.InputStream(
                        samplerate=candidate_fs,
                        channels=1,
                        blocksize=block_size,
                        dtype="float32",
                        callback=audio_callback,
                    )
                    self.live_stream.start()
                    break
                except Exception as exc:
                    last_error = exc
                    self.live_stream = None
            else:
                raise last_error

            self.live_status_var.set(f"正在监听麦克风... {self.live_input_fs} Hz")
            self.root.after(30, self.process_live_audio)
        except Exception as exc:
            self.live_running = False
            self.live_status_var.set("监听启动失败")
            messagebox.showerror("监听失败", str(exc))

    def stop_live_listening(self):
        """停止麦克风监听并释放声卡资源。"""
        self.live_running = False
        stream = self.live_stream
        self.live_stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self.live_candidate_digit = None
        self.live_candidate_count = 0
        self.live_silence_blocks = 0
        self.live_input_silence_blocks = 0
        self.live_buffer = np.zeros(0, dtype=np.float64)
        self.live_active_digit = None
        self.live_scan_pos = 0
        self.live_last_emit_pos = -10 ** 9
        self.live_release_pos = 0
        self.live_active_peak_rms = 0.0
        if hasattr(self, "live_line"):
            self.live_line.set_ydata(np.zeros(400))
            self.live_canvas_plot.draw_idle()
        self.live_status_var.set("已停止监听")

    def on_close(self):
        """关闭窗口前释放实时监听资源。"""
        self.stop_live_listening()
        self.root.destroy()

    def update_live_plot(self, signal):
        """刷新实时监听页的降噪后波形。"""
        view_window = np.zeros(400)
        signal = np.asarray(signal, dtype=float)
        view_window[:min(400, len(signal))] = signal[:400]
        self.live_line.set_ydata(view_window)
        self.live_canvas_plot.draw_idle()

    def process_live_audio(self):
        """
        从麦克风队列取出音频块，完成降噪、DTMF 判决和界面更新。

        实时识别分成两层：
        - 块级处理：每 20 ms 估计输入强度、更新噪声底、做连续带通滤波；
        - 窗口级处理：在累计缓冲区中用 50 ms 滑动窗口做 DTMF 判决。

        这样设计的原因是，声卡回调块很短，适合实时刷新；但单块太短，不适合稳定判频。
        因此程序把多个块放进缓冲区，再用滑动窗口识别数字。
        """
        if not self.live_running:
            return

        got_audio = False
        latest_clean = None
        while True:
            try:
                item_type, payload = self.live_queue.get_nowait()
            except queue.Empty:
                break

            if item_type == "status":
                self.live_status_var.set(f"监听中: {payload}")
                continue

            raw = np.asarray(payload, dtype=np.float64)
            processing_fs = getattr(self, "live_input_fs", FS)
            raw_rms = float(np.sqrt(np.mean(raw ** 2))) if raw.size else 0.0
            self.live_recent_raw_rms = raw_rms

            # 在没有强按键音时逐步更新环境噪声估计，供噪声门限使用。
            # 这里使用缓慢更新，是为了避免把正在播放的拨号音误认为“新的噪声底”。
            adaptive_input_floor = max(LIVE_MIN_RAW_RMS, self.live_noise_floor * 2.2)
            if raw_rms < adaptive_input_floor:
                self.live_noise_floor = 0.96 * self.live_noise_floor + 0.04 * raw_rms
            self.live_noise_var.set(f"环境噪声: {self.live_noise_floor:.4f} / 输入强度: {raw_rms:.4f}")

            clean, self.live_filter_state = reduce_environment_noise(
                raw,
                processing_fs,
                self.live_noise_floor,
                self.live_bandpass,
                self.live_filter_state,
            )
            clean_rms = float(np.sqrt(np.mean(clean ** 2))) if clean.size else 0.0
            block_end_pos = len(self.live_buffer) + len(clean)

            # 块级静音判断：如果原始输入强度低于自适应门限，就认为这一块没有有效按键。
            # 直接将 clean 置零可以让后面的窗口更容易看到“按键之间的间隔”。
            if raw_rms < adaptive_input_floor:
                self.live_input_silence_blocks += 1
                clean = np.zeros_like(clean)
                clean_rms = 0.0
            else:
                self.live_input_silence_blocks = 0

            if self.live_active_digit is not None:
                # 记录当前已确认按键的能量峰值。
                # 快速连点同一个数字时，两次按键之间可能没有绝对静音，但会出现明显能量下跌。
                self.live_active_peak_rms = max(self.live_active_peak_rms, clean_rms)
                relative_floor = self.live_active_peak_rms * LIVE_RELATIVE_RELEASE_RATIO
                release_floor = max(adaptive_input_floor, relative_floor)
                active_age = block_end_pos - self.live_last_emit_pos
                # 只有当前按键已经保持了最短时间，才允许用能量谷释放。
                # 这能防止一个刚确认的长按被边缘窗口立即拆成多个相同数字。
                if active_age >= self.live_active_min_samples and clean_rms < release_floor:
                    self.live_input_silence_blocks = max(self.live_input_silence_blocks, LIVE_RELEASE_BLOCKS)

            if self.live_input_silence_blocks >= LIVE_RELEASE_BLOCKS:
                # 块级释放：当前按键结束，后面可以再次输出相同数字。
                # release_pos 记录释放发生的位置，用来避免还没扫描完的旧窗口反复触发同一个按键。
                self.live_active_digit = None
                self.live_active_peak_rms = 0.0
                self.live_candidate_digit = None
                self.live_candidate_count = 0
                self.live_release_pos = block_end_pos
            latest_clean = clean
            got_audio = True
            self.live_buffer = np.concatenate((self.live_buffer, clean))

            # 只保留最近 2 秒音频。实时识别不需要无限保存历史，否则长时间监听会越来越占内存。
            max_buffer = int(processing_fs * 2.0)
            if len(self.live_buffer) > max_buffer:
                trim = len(self.live_buffer) - max_buffer
                self.live_buffer = self.live_buffer[-max_buffer:]
                self.live_scan_pos = max(0, self.live_scan_pos - trim)
                self.live_last_emit_pos = max(-10 ** 9, self.live_last_emit_pos - trim)
                self.live_release_pos = max(0, self.live_release_pos - trim)

        if got_audio and len(self.live_buffer) > 0:
            self.update_live_plot(self.live_buffer[-400:])

        detected_in_pass = False
        processing_fs = getattr(self, "live_input_fs", FS)
        while self.live_scan_pos + self.live_window_size <= len(self.live_buffer):
            window_start = self.live_scan_pos
            window = self.live_buffer[window_start:window_start + self.live_window_size]
            window_rms = float(np.sqrt(np.mean(window ** 2))) if window.size else 0.0

            # 窗口强度太低时不进入频率判决。
            # 这样可以减少安静环境、键间空白和远处人声被误识别的概率。
            if window_rms < max(LIVE_MIN_RAW_RMS, self.live_noise_floor * 1.8):
                digit, confidence = None, 0.0
            else:
                # 频率判决由 detect_dtmf_digit 完成。这里传入的参数比文件识别更严格，
                # 因为麦克风实时环境中会出现人声、回声和桌面噪声。
                digit, confidence = detect_dtmf_digit(
                    window,
                    processing_fs,
                    min_energy=2.0,
                    dominance=2.8,
                    min_pair_share=0.68,
                    max_twist=16.0,
                    min_spectral_purity=0.58,
                    max_freq_error=32.0,
                    min_sharpness=1.8,
                    min_rms=max(LIVE_MIN_RAW_RMS, self.live_noise_floor * 1.8),
                )
            if digit:
                detected_in_pass = True
                self.live_silence_blocks = 0

                # 候选确认机制：
                # 单个窗口识别到数字不立刻输出；只有同一个数字连续出现，才认为它是真实按键。
                if digit == self.live_candidate_digit:
                    self.live_candidate_count += 1
                else:
                    self.live_candidate_digit = digit
                    self.live_candidate_count = 1

                # strong_hit 用于高置信度快速确认，stable_hit 用于普通稳定确认。
                # 两者都需要至少两个窗口，避免一帧毛刺直接变成数字。
                strong_hit = self.live_candidate_count >= 2 and confidence >= LIVE_FAST_CONFIRM_CONFIDENCE
                stable_hit = self.live_candidate_count >= LIVE_CONFIRM_FRAMES

                # active_digit 是“已经输出并且尚未释放”的按键。
                # 如果当前窗口还是同一个 active_digit，就不重复输出，防止长按 5 变成 55555。
                can_emit = digit != self.live_active_digit

                # 如果释放发生在缓冲区较后的位置，而当前扫描窗口还在释放点之前，
                # 说明这是旧按键的尾部窗口，不能用来触发新的同键输出。
                if self.live_active_digit is None and window_start < self.live_release_pos:
                    can_emit = False
                enough_gap = window_start - self.live_last_emit_pos >= self.live_min_emit_gap

                # 同时满足“可输出、距离上次输出足够远、候选足够稳定”后才写入界面。
                if can_emit and enough_gap and (strong_hit or stable_hit):
                    self.live_status_var.set(f"确认按键 {digit}，置信度 {confidence:.2f}")
                    self.live_result_var.set(self.live_result_var.get() + digit)
                    self.live_active_digit = digit
                    self.live_active_peak_rms = max(self.live_active_peak_rms, window_rms)
                    self.live_last_emit_pos = window_start
            else:
                # 窗口级释放：
                # 如果一个窗口已经无法通过 DTMF 判决，说明可能进入了键间间隔。
                # 对快速连点同一个数字，哪怕只有很短的间隔，也需要尽快释放 active_digit。
                self.live_silence_blocks += 1
                if self.live_silence_blocks >= LIVE_RELEASE_FRAMES:
                    if self.live_active_digit is not None:
                        self.live_active_digit = None
                        self.live_active_peak_rms = 0.0
                        self.live_release_pos = window_start + self.live_window_size
                    # 清空候选，下一段按键需要重新连续确认。
                    self.live_candidate_digit = None
                    self.live_candidate_count = 0
                if self.live_silence_blocks % 12 == 0:
                    self.live_status_var.set("正在监听麦克风...")
            self.live_scan_pos += self.live_hop_size

        if detected_in_pass:
            # 保留最近一小段上下文即可；扫描位置已经记录，后续会继续从未扫描处开始。
            pass

        self.root.after(30, self.process_live_audio)

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
