import os
import time
import glob
import random
import psutil
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import torch

# 导入 4 个保留的算法模块（已移除 RNNoise）
from alg1_boll_ss import BollSpectralSubtraction
from alg2_wiener_filter import WienerFilter
from alg3_proposed_adaptive_ss import ProposedAdaptiveSS
from alg5_deepfilternet import DeepFilterNetDenoiser

from plot_paper_supplementary import plot_pareto_frontier
from plot_paper_supplementary import plot_snr_robustness_boxplot

# ==================== 0. 环境与画图全局设置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_audio_gpu(filepath):
    """读取音频并转换为标准 normalized float32 Tensor"""
    sr, data = wavfile.read(filepath)
    data = torch.from_numpy(data.astype(np.float32)).to(DEVICE)
    if data.ndim > 1:
        data = torch.mean(data, dim=1)
    max_val = torch.max(torch.abs(data))
    if max_val > 0:
        data = data / max_val
    return sr, data


def save_audio_from_gpu(filepath, sr, tensor_data):
    """保存 Tensor 为 int16 WAV 音频文件（不计入算法纯耗时）"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tensor_data = tensor_data.detach()
    tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=1.0, neginf=-1.0)
    max_val = torch.max(torch.abs(tensor_data))
    if max_val > 1.0:
        tensor_data = tensor_data / max_val
    signal_np = (tensor_data * 32767.0).clamp(-32768, 32767).cpu().numpy().astype(np.int16)
    wavfile.write(filepath, sr, signal_np)


def calculate_snr_gpu(clean, processed):
    """计算信噪比 SNR (dB)"""
    min_len = min(len(clean), len(processed))
    clean, processed = clean[:min_len], processed[:min_len]
    noise = clean - processed
    clean_pwr = torch.sum(clean ** 2)
    noise_pwr = torch.sum(noise ** 2) + 1e-10
    return float((10 * torch.log10(clean_pwr / noise_pwr)).cpu().item())


# ==================== 1. 单样本独立画图与导出函数 ====================
def plot_individual_sample_plots(sample_data_list, alg_names, output_dir="./outpng"):
    """
    针对选定的测试样本，按样本单独生成并导出时域对比图与语谱对比图
    文件命名前缀自动挂载样本文件名（如 p226_001_waveform.png, p226_001_spectrogram.png）
    """
    os.makedirs(output_dir, exist_ok=True)
    column_titles = ["Clean (GT)", "Noisy"] + alg_names
    num_cols = len(column_titles)

    for sample in sample_data_list:
        base_name = sample['file_name']
        sample_stem = os.path.splitext(base_name)[0]
        sr = sample['sr']
        signals = [sample['clean'], sample['noisy']] + [sample['denoised'][alg] for alg in alg_names]

        # ---------------- 1. 单样本时域波形对比图 ----------------
        fig_time, axes_time = plt.subplots(1, num_cols, figsize=(20, 3.2), dpi=300)
        fig_time.suptitle(f"样本 [{base_name}] 时域波形 (Time-Domain) 横向对比", fontsize=13, fontweight='bold', y=1.02)

        for c_idx, sig in enumerate(signals):
            ax = axes_time[c_idx]
            sig_np = sig.cpu().numpy()
            time_axis = np.linspace(0, len(sig_np) / sr, len(sig_np))
            color = '#1f77b4' if c_idx >= 2 else ('green' if c_idx == 0 else 'red')
            ax.plot(time_axis, sig_np, color=color, linewidth=0.8)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_title(column_titles[c_idx], fontsize=10, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel("幅度 (Amplitude)", fontsize=9)
            ax.set_xlabel("时间 (s)", fontsize=8)

        plt.tight_layout()
        time_plot_path = os.path.join(output_dir, f"{sample_stem}_waveform.png")
        plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------- 2. 单样本频域语谱对比图 ----------------
        fig_spec, axes_spec = plt.subplots(1, num_cols, figsize=(20, 3.2), dpi=300)
        fig_spec.suptitle(f"样本 [{base_name}] 频域语谱图 (STFT Spectrogram) 横向对比", fontsize=13, fontweight='bold', y=1.02)

        for c_idx, sig in enumerate(signals):
            ax = axes_spec[c_idx]
            sig_np = sig.cpu().numpy()
            ax.specgram(sig_np, NFFT=512, Fs=sr, noverlap=256, cmap='magma')
            ax.set_title(column_titles[c_idx], fontsize=10, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel("频率 (Hz)", fontsize=9)
            else:
                ax.set_yticks([])
            ax.set_xlabel("时间 (s)", fontsize=8)

        plt.tight_layout()
        spec_plot_path = os.path.join(output_dir, f"{sample_stem}_spectrogram.png")
        plt.savefig(spec_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"🖼️ 已独立导出样本 [{sample_stem}] 图像 -> {time_plot_path} | {spec_plot_path}")


# ==================== 2. 主控评测流程 ====================
def run_benchmark():
    clean_dir = "./dataset/clean_testset_wav"
    noisy_dir = "./dataset/noisy_testset_wav"
    out_wav_dir = "./outwav"
    out_png_dir = "./outpng"

    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = glob.glob(os.path.join(clean_dir, "*.wav"))
    clean_dict = {os.path.basename(f): f for f in clean_files}

    if not noisy_files:
        print(f"❌ 未找到数据集文件，请检查文件夹路径: {noisy_dir}")
        return

    # 1. 实例化 4 种保留算法（不包含 RNNoise）
    alg_instances = [
        BollSpectralSubtraction(device=DEVICE),
        WienerFilter(device=DEVICE),
        ProposedAdaptiveSS(device=DEVICE),
        DeepFilterNetDenoiser(device=DEVICE)
    ]

    alg_names = [alg.name for alg in alg_instances]
    process = psutil.Process(os.getpid())

    # 随机选定 5 个测试音频进行画图展示
    random.seed(42)
    selected_sample_files = set(random.sample(noisy_files, min(5, len(noisy_files))))
    sample_data_list = []

    stats = {
        alg.name: {
            "out_snr": [], "snr_imp": [], "time_ms": [],
            "rtf": [], "mem_mb": [], "model_size_kb": alg.get_model_size_kb(),
            "param_count": alg.get_param_count()
        }
        for alg in alg_instances
    }

    print("=" * 105)
    print(f"🖥️  计算设备: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU Mode'})")
    print(f"📁 启动 VoiceBank 完整评估，涵盖 4 种核心算法，共测试 {len(noisy_files)} 条音频")
    print("=" * 105)

    for idx, noisy_path in enumerate(noisy_files, 1):
        base_name = os.path.basename(noisy_path)
        clean_path = clean_dict.get(base_name)
        if not clean_path:
            continue

        # 先读取 Clean 音频（不加入算法耗时统计）
        _, clean_gpu = load_audio_gpu(clean_path)

        current_sample_record = {"file_name": base_name, "sr": None, "clean": clean_gpu, "noisy": None,
                                 "denoised": {}} if noisy_path in selected_sample_files else None

        for alg in alg_instances:
            mem_before = process.memory_info().rss / (1024 * 1024)

            # =================== 【纯算力耗时精准测量区】 ===================
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # 1. 读取带噪音频文件
            sr, noisy_gpu = load_audio_gpu(noisy_path)

            # 2. 降噪算法处理
            denoised_gpu = alg.process(noisy_gpu, sr)

            # 3. 还原/同步时域信号
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # ==============================================================

            mem_after = process.memory_info().rss / (1024 * 1024)

            # 后续评估指标计算与文件保存（完全不占用上述计时）
            in_snr = calculate_snr_gpu(clean_gpu, noisy_gpu)
            out_snr = calculate_snr_gpu(clean_gpu, denoised_gpu)
            snr_imp = out_snr - in_snr
            audio_duration_sec = len(noisy_gpu) / float(sr)
            rtf = (elapsed_ms / 1000.0) / audio_duration_sec

            # 保存处理后的 WAV 音频
            out_wav_path = os.path.join(out_wav_dir, alg.name, base_name)
            save_audio_from_gpu(out_wav_path, sr, denoised_gpu)

            # 记录统计数据
            stats[alg.name]["out_snr"].append(out_snr)
            stats[alg.name]["snr_imp"].append(snr_imp)
            stats[alg.name]["time_ms"].append(elapsed_ms)
            stats[alg.name]["rtf"].append(rtf)
            stats[alg.name]["mem_mb"].append(max(0.01, mem_after - mem_before))

            if current_sample_record is not None:
                current_sample_record["sr"] = sr
                current_sample_record["noisy"] = noisy_gpu
                current_sample_record["denoised"][alg.name] = denoised_gpu

        if current_sample_record is not None:
            sample_data_list.append(current_sample_record)

        if idx % 100 == 0 or idx == len(noisy_files):
            print(f"评估进度: [{idx}/{len(noisy_files)}] 条音频完成...")

    # ==================== 3. 打印全量学术对比表格 ====================
    print("\n" + "=" * 105)
    print(f"【VoiceBank 全量算法性能学术汇总表 - 样本数: {len(noisy_files)} 条】")
    print("=" * 105)
    print(
        f"{'算法名称 (Algorithm)':<20} | {'SNR 提升(dB)':<12} | {'平均耗时(ms)':<10} | {'RTF 实时因子':<12} | {'内存(MB)':<10} | {'模型体积':<12}")
    print("=" * 105)
    for alg in alg_instances:
        name = alg.name
        avg_snr = np.mean(stats[name]["snr_imp"])
        avg_time = np.mean(stats[name]["time_ms"])
        avg_rtf = np.mean(stats[name]["rtf"])
        avg_mem = np.mean(stats[name]["mem_mb"])
        size_str = f"{stats[name]['model_size_kb']:.1f} KB" if stats[name][
                                                                   'model_size_kb'] < 1024 else f"{stats[name]['model_size_kb'] / 1024:.2f} MB"

        print(
            f"{name:<20} | {avg_snr:<+12.2f} | {avg_time:<10.2f} | {avg_rtf:<12.4f} | {avg_mem:<10.2f} | {size_str:<12}")
    print("=" * 105)

    # ==================== 4. 单样本独立图像渲染与导出 ====================
    print("\n🎨 正在为选定的 5 个样本单独绘制导出高分辨率对比图...")
    plot_individual_sample_plots(sample_data_list, alg_names, output_dir=out_png_dir)
    print(f"\n🎉 所有对比实验与可视化渲染全部顺利完成！")

    # 1. 构造帕累托绘图所需字典
    summary_for_pareto = {}
    for alg in alg_instances:
        summary_for_pareto[alg.name] = {
            "snr_imp": np.mean(stats[alg.name]["snr_imp"]),
            "time_ms": np.mean(stats[alg.name]["time_ms"]),
            "mem_mb": np.mean(stats[alg.name]["mem_mb"]),
            "is_proposed": (alg.name == "Proposed_Adaptive")
        }

    plot_pareto_frontier(summary_for_pareto, output_dir=out_png_dir)

    # 2. 构造箱线图所需的逐样本记录
    raw_records = []
    for alg_name, item in stats.items():
        if "in_snr" in item and len(item["in_snr"]) > 0:
            for in_snr, snr_imp in zip(item["in_snr"], item["snr_imp"]):
                raw_records.append({
                    "in_snr": in_snr,
                    "alg_name": alg_name,
                    "snr_imp": snr_imp
                })
        else:
            print(f"⚠️ 提示: {alg_name} 的 stats 中未记录 'in_snr'，跳过箱线图绘制。")

    # 仅当有数据时才调用箱线图绘制
    if raw_records:
        plot_snr_robustness_boxplot(raw_records, output_dir=out_png_dir)


if __name__ == "__main__":
    run_benchmark()