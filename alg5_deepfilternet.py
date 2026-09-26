"""
算法 5: DeepFilterNet3 (前沿轻量级 Deep Filtering 模型 - 修复 2D 维度与 GPU 兼容)
"""
import torch

class DeepFilterNetDenoiser:
    def __init__(self, device=None):
        # 1. 设备感知：有 GPU 用 GPU，没有自动回退 CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.name = "DeepFilterNet3"
        self.type = "SOTA Lightweight Deep Learning"

        from df.enhance import init_df

        # 2. 初始化模型与 Rust 状态机
        self.model, self.df_state, _ = init_df()
        self.model = self.model.to(self.device)
        self.model.eval()

        # 3. 自动匹配底层 Rust 状态机生成的 CPU 特征张量至目标设备
        original_forward = self.model.forward
        target_device = self.device

        def auto_device_forward(*args, **kwargs):
            args = [a.to(target_device) if isinstance(a, torch.Tensor) else a for a in args]
            kwargs = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            return original_forward(*args, **kwargs)

        self.model.forward = auto_device_forward

        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.model_size_kb = (self.param_count * 4) / 1024.0

    def get_model_size_kb(self):
        return self.model_size_kb

    def get_param_count(self):
        return self.param_count

    def process(self, y_tensor, sr):
        from df.enhance import enhance

        orig_len = y_tensor.shape[0]
        y_cpu = y_tensor.detach().cpu()

        # 【核心修复】：强制将输入扩充为 2D 形状 [1, N]（df.analysis 必需）
        if y_cpu.ndim == 1:
            y_cpu = y_cpu.unsqueeze(0)  # [N] -> [1, N]
        elif y_cpu.ndim == 2 and y_cpu.shape[0] != 1:
            y_cpu = y_cpu.T  # [N, 1] -> [1, N]

        # 执行神经网络增强
        with torch.no_grad():
            enhanced = enhance(self.model, self.df_state, y_cpu)

        # 恢复 1D 数组、截断长度并推回全局指定设备 (GPU/CPU)
        enhanced = enhanced.squeeze(0)[:orig_len].to(self.device)
        return enhanced