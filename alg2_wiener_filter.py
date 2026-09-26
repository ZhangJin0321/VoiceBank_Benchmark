"""
算法 2: 维纳滤波 (Wiener Filtering)
特点: 基于最小均方误差 (MMSE) 准则计算频域增益，比 Boll 谱减法更平滑。
"""

import torch

class WienerFilter:
    def __init__(self, nperseg=512, noise_frames=10, floor=0.01, device="cpu"):
        self.nperseg = nperseg
        self.noise_frames = noise_frames
        self.floor = floor
        self.device = torch.device(device)
        self.name = "Wiener_Filter"
        self.type = "Traditional DSP"

    def get_model_size_kb(self):
        return 0.0

    def get_param_count(self):
        return 0

    def process(self, y_tensor, sr):
        y_tensor = y_tensor.to(self.device)
        nperseg = self.nperseg
        window = torch.hann_window(nperseg, device=self.device)

        Zxx = torch.stft(
            y_tensor,
            n_fft=nperseg,
            hop_length=nperseg // 2,
            win_length=nperseg,
            window=window,
            return_complex=True
        )
        mag, phase = torch.abs(Zxx), torch.angle(Zxx)

        # 估计噪声功率谱
        signal_pwr = mag ** 2
        frame_energies = torch.sum(signal_pwr, dim=0)
        _, min_indices = torch.topk(frame_energies, k=min(self.noise_frames, mag.shape[1]), largest=False)
        noise_pwr = torch.mean(signal_pwr[:, min_indices], dim=1, keepdim=True)

        # 计算维纳增益 G(f, t) = max(1 - P_noise / P_signal, floor)
        gain = torch.clamp(1.0 - (noise_pwr / (signal_pwr + 1e-8)), min=self.floor)
        clean_mag = mag * gain

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