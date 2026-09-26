"""
算法 4: RNNoise (直接挂载 pyrnnoise 内部的 C 语言内核)
说明: 绕过 pyrnnoise 包内部 Graph 类的 Python Bug，使用 ctypes 直接调用其 C 动态库
"""
import os
import glob
import ctypes
import torch
import numpy as np
import scipy.signal as signal
import pyrnnoise

class RNNoiseDenoiser:
    def __init__(self, device="cpu"):
        self.name = "RNNoise (Official C)"
        self.type = "Official Pretrained Lightweight DL"
        self.device = torch.device("cpu")

        # 1. 自动定位已安装 pyrnnoise 包目录下的 C 动态库 (.dll / .so)
        pyrnnoise_dir = os.path.dirname(pyrnnoise.__file__)
        dll_files = []
        for ext in ["*.dll", "*.so", "*.dylib", "*.pyd"]:
            dll_files.extend(glob.glob(os.path.join(pyrnnoise_dir, "**", ext), recursive=True))

        if not dll_files:
            raise FileNotFoundError("未在 pyrnnoise 安装目录中找到 C 动态库文件。")

        # 匹配 C 动态库路径
        lib_path = dll_files[0]
        for f in dll_files:
            if "rnnoise" in os.path.basename(f).lower():
                lib_path = f
                break

        # 2. 通过 ctypes 直接载入 C 库底层接口
        self.lib = ctypes.CDLL(lib_path)

        # 绑定 C 语言函数签名
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]

        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]

        self.lib.rnnoise_process_frame.restype = ctypes.c_float
        self.lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]

        # 官方模型参数量与体积 (~85K 参数, ~206 KB)
        self.param_count = 85000
        self.model_size_kb = 206.3

    def get_model_size_kb(self):
        return self.model_size_kb

    def get_param_count(self):
        return self.param_count

    def process(self, y_tensor, sr=16000):
        if isinstance(y_tensor, torch.Tensor):
            y_np = y_tensor.detach().cpu().numpy()
        else:
            y_np = np.array(y_tensor)

        orig_len = len(y_np)

        # 1. 重采样: 16kHz -> 48kHz (RNNoise C 库原生固定要求 48kHz)
        if sr != 48000:
            y_48k = signal.resample_poly(y_np, up=48000, down=sr)
        else:
            y_48k = y_np.copy()

        # 2. 幅值缩放: [-1.0, 1.0] -> [-32768.0, 32767.0]
        y_48k_scaled = (y_48k * 32768.0).astype(np.float32)

        # 3. 480 点 (10ms) 分帧补零
        frame_size = 480
        pad_len = (frame_size - (len(y_48k_scaled) % frame_size)) % frame_size
        if pad_len > 0:
            y_48k_scaled = np.pad(y_48k_scaled, (0, pad_len), mode='constant')

        num_frames = len(y_48k_scaled) // frame_size
        out_48k_scaled = np.zeros_like(y_48k_scaled)

        # 4. 创建 C 语言 RNNoise 句柄并逐帧调用
        st = self.lib.rnnoise_create(None)
        in_buf = (ctypes.c_float * frame_size)()
        out_buf = (ctypes.c_float * frame_size)()

        try:
            for i in range(num_frames):
                frame_data = y_48k_scaled[i * frame_size : (i + 1) * frame_size]
                in_buf[:] = frame_data
                self.lib.rnnoise_process_frame(st, out_buf, in_buf)
                out_48k_scaled[i * frame_size : (i + 1) * frame_size] = np.array(out_buf)
        finally:
            self.lib.rnnoise_destroy(st)

        # 5. 幅值还原: [-32768.0, 32767.0] -> [-1.0, 1.0]
        out_48k = out_48k_scaled / 32768.0

        # 6. 重采样回 16kHz
        if sr != 48000:
            out_16k = signal.resample_poly(out_48k, up=sr, down=48000)
        else:
            out_16k = out_48k

        # 7. 消除相移延迟 (160 点延迟)，精准点对点对齐
        delay_samples = int(frame_size * (sr / 48000.0))
        out_aligned = out_16k[delay_samples : delay_samples + orig_len]

        if len(out_aligned) < orig_len:
            out_aligned = np.pad(out_aligned, (0, orig_len - len(out_aligned)))

        res_tensor = torch.from_numpy(out_aligned.astype(np.float32))
        if isinstance(y_tensor, torch.Tensor):
            res_tensor = res_tensor.to(y_tensor.device)

        return res_tensor