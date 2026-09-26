"""
算法 1: Boll 经典谱减法 (Spectral Subtraction, Boll 1979)
特点: 传统经典 DSP 算法，零参数，通过对静音帧估算噪声幅度并直接相减。
"""

import os
import time
import torch
import numpy as np


class BollSpectralSubtraction:
    def __init__(self, nperseg=512, noise_frames=10, device="cpu"):
        self.nperseg = nperseg
        self.noise_frames = noise_frames
        self.device = torch.device(device)
        self.name = "Boll_SS"
        self.type = "Traditional DSP"

    def get_model_size_kb(self):
        """传统 DSP 算法无神经网络参数，大小为 0 KB"""
        return 0.0

    def get_param_count(self):
        return 0

    def process(self, y_tensor, sr):
        """
        y_tensor: 1D PyTorch Tensor (单通道语音信号)
        sr: 采样率 (如 16000)
        """
        y_tensor = y_tensor.to(self.device)
        nperseg = self.nperseg
        window = torch.hann_window(nperseg, device=self.device)

        # 1. 短时傅里叶变换 (STFT)
        Zxx = torch.stft(
            y_tensor,
            n_fft=nperseg,
            hop_length=nperseg // 2,
            win_length=nperseg,
            window=window,
            return_complex=True
        )
        mag, phase = torch.abs(Zxx), torch.angle(Zxx)

        # 2. 估计噪声功率谱 (选取能量最低的前 k 帧)
        frame_energies = torch.sum(mag ** 2, dim=0)
        _, min_indices = torch.topk(frame_energies, k=min(self.noise_frames, mag.shape[1]), largest=False)
        noise_mag = torch.mean(mag[:, min_indices], dim=1, keepdim=True)

        # 3. 谱减操作: S_clean = max(|Y| - D_noise, 0)
        clean_mag = torch.clamp(mag - noise_mag, min=0.0)

        # 4. 逆短时傅里叶变换 (ISTFT)
        clean_stft = torch.polar(clean_mag, phase)
        denoised = torch.istft(
            clean_stft,
            n_fft=nperseg,
            hop_length=nperseg // 2,
            win_length=nperseg,
            window=window,
            length=y_tensor.shape[0]
        )
        return denoised