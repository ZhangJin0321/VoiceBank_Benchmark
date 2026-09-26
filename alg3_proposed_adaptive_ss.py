"""
算法 3: 改进的自适应过减谱减法 (Proposed Adaptive Spectral Subtraction)
特点: 基于后验信噪比动态调节过减因子 alpha 和谱下限 beta，结合 2D 频域平滑消除“音乐噪声”。
"""

import torch

class ProposedAdaptiveSS:
    def __init__(self, nperseg=512, noise_frames=10, device="cpu"):
        self.nperseg = nperseg
        self.noise_frames = noise_frames
        self.device = torch.device(device)
        self.name = "Proposed_Adaptive"
        self.type = "Traditional DSP (Improved)"

    def get_model_size_kb(self):
        return 0.0

    def get_param_count(self):
        return 0

    def process(self, y_tensor, sr):
        y_tensor = y_tensor.to(self.device)
        nperseg = self.nperseg
        window = torch.hann_window(nperseg, device=self.device)

        alpha_min, alpha_max = 0.8, 2.2
        beta_min, beta_max = 0.005, 0.02

        Zxx = torch.stft(
            y_tensor,
            n_fft=nperseg,
            hop_length=nperseg // 2,
            win_length=nperseg,
            window=window,
            return_complex=True
        )
        mag, phase = torch.abs(Zxx), torch.angle(Zxx)

        frame_energies = torch.sum(mag ** 2, dim=0)
        _, min_indices = torch.topk(frame_energies, k=min(self.noise_frames, mag.shape[1]), largest=False)
        noise_mag = torch.mean(mag[:, min_indices], dim=1, keepdim=True)

        # 计算后验 SNR
        post_snr = mag / (noise_mag + 1e-8)

        # 2D 滤波平滑 (消除独立噪声孤点)
        snr_4d = post_snr.unsqueeze(0).unsqueeze(0)
        snr_smoothed = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(snr_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_size=(3, 3), stride=1
        ).squeeze(0).squeeze(0)

        snr_db = 20.0 * torch.log10(torch.clamp(snr_smoothed, min=1e-3))
        norm_snr = torch.clamp((snr_db + 10.0) / 25.0, 0.0, 1.0)

        # 动态自适应映射
        alpha = alpha_max - (alpha_max - alpha_min) * norm_snr
        beta = beta_min + (beta_max - beta_min) * (1.0 - norm_snr)

        sub_mag = mag - alpha * noise_mag
        clean_mag = torch.maximum(sub_mag, beta * noise_mag)

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