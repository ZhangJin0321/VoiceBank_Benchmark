# VoiceBank Benchmark: 语音降噪算法多维度性能评估系统

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

本项目是一个专注于语音降噪（Speech Enhancement/Denoising）算法的综合性测试平台。基于国际标准的 **VoiceBank-DEMAND** 数据集，提供了从传统信号处理（DSP）算法到现代深度学习（Deep Learning）模型的全方位评估。

除了评估传统质量指标（SNR 提升）外，系统还针对**工程端侧部署需求**，精准记录算法的处理延时（Latency）、实时因子（RTF）、内存开销（RAM）和模型参数量，并自动生成科研级别的可视化图表。

---

## 🚀 核心功能

- **多算法横向对比**：内置 5 种典型语音降噪算法：
  - `Boll_SS`：经典 Boll 谱减法 (DSP)
  - `Wiener_Filter`：维纳滤波算法 (DSP)
  - `Proposed_Adaptive`：改进型自适应过减谱减法 (DSP)
  - `RNNoise`：基于循环神经网络的轻量级降噪算法 (Hybrid/RNN)
  - `DeepFilterNet3`：基于复数域深度滤波的高质量降噪网络 (DL)
- **多维度工程指标评估**：
  - **降噪质量**：信噪比增益 ($\Delta\text{SNR}$)
  - **工程算力**：平均处理延时 ($T_{\text{proc}}$)、实时因子 ($\text{RTF}$)
  - **资源占用**：内存峰值增量 ($\Delta\text{RAM}$)、模型体积与参数量
- **可视化产物自动生成**：
  - 自动绘制 **SNR vs. 延时帕累托前沿图 (Pareto Frontier)**
  - 自动绘制 **按输入 SNR 分组的抗噪鲁棒性箱线图 (Grouped Boxplot)**
  - 导出典型样本的时域波形图与 STFT 语谱图对比

---

## 📁 项目结构

```text
VoiceBank_Benchmark/
├── dataset/                  # VoiceBank-DEMAND 测试集 (Clean & Noisy 音频)
├── outpng/                   # 自动生成的科研对比图表 (.png)
├── outwav/                   # 算法处理后的降噪音频文件 (.wav)
├── alg1_boll_ss.py           # Boll 谱减法实现
├── alg2_wiener_filter.py     # 维纳滤波算法实现
├── alg3_proposed_adaptive_ss.py # 自适应谱减法实现
├── alg4_rnnoise.py           # RNNoise 算法接口
├── alg5_deepfilternet.py     # DeepFilterNet3 模型接口
├── download_dll.py           # 依赖项/动态链接库自动下载脚本
├── plot_paper_supplementary.py # 论文补充图表绘制脚本
├── run_master_benchmark.py   # 测试主程序入口
├── requirements.txt          # Python 依赖项列表
└── README.md                 # 项目说明文档