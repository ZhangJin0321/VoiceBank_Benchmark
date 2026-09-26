import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 0. 绘图全局风格设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="SimHei")


# ==================== 1. 帕累托前沿图 (Pareto Frontier) ====================
def plot_pareto_frontier(summary_stats, output_dir="./outpng"):
    """
    绘制 SNR 提升 vs 处理延时 (Pareto Frontier) 散点/前沿图
    summary_stats: 字典结构，包含各算法的平均指标
      {
         'AlgName': {'snr_imp': float, 'time_ms': float, 'mem_mb': float, 'is_proposed': bool}, ...
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 配色与标记定义
    color_map = {
        "Boll_SS": "#2ca02c",
        "Wiener_Filter": "#ff7f0e",
        "Proposed_Adaptive": "#d62728",  # 突出红色
        "DeepFilterNet3": "#1f77b4"
    }

    x_times = []
    y_snrs = []
    names = []

    for name, stat in summary_stats.items():
        x = stat['time_ms']
        y = stat['snr_imp']
        mem = stat['mem_mb']
        color = color_map.get(name, "#7f7f7f")

        x_times.append(x)
        y_snrs.append(y)
        names.append(name)

        # 气泡大小映射内存开销 (基准点大小 150，随内存线性放大)
        bubble_size = 180 + mem * 15

        # 绘制散点
        is_proposed = stat.get('is_proposed', False)
        marker = '★' if is_proposed else 'o'

        if is_proposed:
            ax.scatter(x, y, s=bubble_size * 1.5, color=color, alpha=0.95,
                       edgecolors='black', linewidth=1.8, label=f"{name} (本文)", zorder=5)
        else:
            ax.scatter(x, y, s=bubble_size, color=color, alpha=0.8,
                       edgecolors='gray', linewidth=1.0, label=name, zorder=3)

        # 标注文字偏移处理
        offset_x = 1.15 if x < 10 else 0.85
        offset_y = y + 0.15 if name != "DeepFilterNet3" else y - 0.35
        ax.annotate(
            f"{name}\n({x:.2f}ms, {y:+.2f}dB)",
            (x, y),
            xytext=(x * offset_x, offset_y),
            fontsize=9.5,
            fontweight='bold' if is_proposed else 'normal',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='gray', lw=0.8)
        )

    # 绘制对数 X 轴（应对从 4ms 到 120ms 的跨度）
    ax.set_xscale('log')
    ax.set_xlabel("平均算法处理延时 Processing Latency (ms) [对数坐标轴]", fontsize=11, fontweight='bold')
    ax.set_ylabel("信噪比提升 SNR Improvement (dB)", fontsize=11, fontweight='bold')
    ax.set_title("算法 SNR 提升与处理延时权衡关系 (Pareto Trade-off)", fontsize=13, fontweight='bold', pad=12)

    # 添加气泡含义说明
    ax.text(0.03, 0.05, "* 气泡节点大小代表运行内存 (RAM) 占用开销",
            transform=ax.transAxes, fontsize=9, color='#555555',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8))

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "pareto_frontier_snr_vs_latency.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 帕累托前沿图已成功导出至: {save_path}")


# ==================== 2. 按输入 SNR 分组鲁棒性箱线图 ====================
def plot_snr_robustness_boxplot(raw_sample_records, output_dir="./outpng"):
    """
    按初始信噪比 (Input SNR) 分组，绘制各算法 SNR 提升的箱线图
    raw_sample_records: 包含每条样本详细指标的列表
      [
        {'in_snr': float, 'alg_name': str, 'snr_imp': float}, ...
      ]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 对输入 SNR 进行区间离散化 (0dB, 5dB, 10dB, 15dB 离散桶)
    def bin_snr(in_snr):
        if in_snr < 2.5:
            return "0 dB"
        elif in_snr < 7.5:
            return "5 dB"
        elif in_snr < 12.5:
            return "10 dB"
        else:
            return "15 dB"

    # 格式转换
    data_for_df = []
    for record in raw_sample_records:
        data_for_df.append({
            "Input_SNR_Group": bin_snr(record['in_snr']),
            "Algorithm": record['alg_name'],
            "SNR_Imp": record['snr_imp']
        })

    import pandas as pd
    df = pd.DataFrame(data_for_df)

    # 2. 绘图设置
    plt.figure(figsize=(10, 5.5), dpi=300)
    palette = {
        "Boll_SS": "#2ca02c",
        "Wiener_Filter": "#ff7f0e",
        "Proposed_Adaptive": "#d62728",
        "DeepFilterNet3": "#1f77b4"
    }

    ax = sns.boxplot(
        data=df,
        x="Input_SNR_Group",
        y="SNR_Imp",
        hue="Algorithm",
        order=["0 dB", "5 dB", "10 dB", "15 dB"],
        palette=palette,
        width=0.6,
        fliersize=2,
        linewidth=1.0
    )

    ax.set_title("不同输入信噪比 (Input SNR) 下算法 SNR 提升分布与鲁棒性对比", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("初始带噪语音信噪比 (Input SNR Level)", fontsize=11, fontweight='bold')
    ax.set_ylabel("信噪比提升增益 SNR Improvement (dB)", fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="降噪算法", title_fontsize='10', fontsize=9.5, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "snr_robustness_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 鲁棒性分组箱线图已成功导出至: {save_path}")


# ==================== 3. 示例测试与集成演示 ====================
if __name__ == "__main__":
    # 模拟与测试数据（实际运行时可由 run_master_benchmark 生成的数据结构传入）
    mock_summary = {
        "Boll_SS": {"snr_imp": 0.42, "time_ms": 3.85, "mem_mb": 0.05, "is_proposed": False},
        "Wiener_Filter": {"snr_imp": 0.58, "time_ms": 4.12, "mem_mb": 0.06, "is_proposed": False},
        "Proposed_Adaptive": {"snr_imp": 0.73, "time_ms": 4.04, "mem_mb": 0.04, "is_proposed": True},
        "DeepFilterNet3": {"snr_imp": 3.15, "time_ms": 118.50, "mem_mb": 18.20, "is_proposed": False},
    }

    # 运行帕累托前沿绘图
    plot_pareto_frontier(mock_summary)

    # 模拟 824 条样本的箱线图数据
    np.random.seed(42)
    mock_raw_records = []
    snr_levels = [0, 5, 10, 15]
    algs = [
        ("Boll_SS", 0.42),
        ("Wiener_Filter", 0.58),
        ("Proposed_Adaptive", 0.73),
        ("DeepFilterNet3", 3.15)
    ]

    for _ in range(200):  # 模拟样本
        snr_base = float(np.random.choice(snr_levels)) + np.random.normal(0, 0.5)
        for alg_name, base_imp in algs:
            # 引入小范围高斯噪声模拟单样本波动
            imp = base_imp + np.random.normal(0, 0.35)
            mock_raw_records.append({
                "in_snr": snr_base,
                "alg_name": alg_name,
                "snr_imp": imp
            })

    # 运行箱线图绘图
    plot_snr_robustness_boxplot(mock_raw_records)