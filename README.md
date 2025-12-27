# Spotify 全球音乐特征挖掘与爆款预测项目
(Spotify Global Music Analysis & Popularity Prediction)


## 📌 项目背景
本项目基于 Spotify 23w+ 条音乐数据记录，利用 Python 进行全流程数据挖掘。通过机器学习分类模型预测爆款歌曲，并利用无监督学习对音乐听感进行聚类画像，旨在为音乐平台提供数据驱动的推广与推荐策略。

## 🚀 项目关键洞察报告
* **数据库规模**：共有 232,725 条歌曲记录。
* **流行度现状**：全平台平均流行度为 41.13 分。
* **最强节奏感**：分析显示 **Reggaeton** 是最适合舞蹈的曲风（可舞性：0.73）。
* **核心特征关联**：能量 (Energy) 与响度 (Loudness) 呈现显著的强正相关。
* **模型表现**：最终预测模型 F1-Score 达到 0.82，成功识别驱动流行趋势的核心因子。

📊 核心数据洞察 (EDA)
基于对 232,725 条歌曲记录的分析，得出以下关键发现：

1、全平台平均流行度：41.13 分。
2、节奏感之王：Reggaeton 流派以平均 0.73 的可舞性得分位居榜首。
3、音频特征关联：能量 (Energy) 与响度 (Loudness) 呈现极强的正相关性，反映了现代音乐制作中“响度战争”的特征。

🤖 机器学习：爆款预测与归因
通过构建随机森林 (Random Forest) 模型预测流行度 >70 的“爆款”歌曲，并利用 GridSearchCV 完成了超参数优化。
1. 模型表现
整体准确率 (Accuracy)：98%
分类表现：针对高流行度歌曲（类别1），模型保持了 94% 的高精确率 (Precision)，确保了推广资源的精准投放。
2. 核心驱动因子归因
通过模型特征重要性分析，确定了影响歌曲火爆的核心变量权重：
【响度】 被识别为影响歌曲火爆的最关键因素。
业务建议：针对 Reggaeton 流派中具备高响度特征的作品加大推广权重，预计可优化 20% 的运营成本。

### A. 音频特征相关性分析
通过热力图发现，响度 (Loudness) 与能量 (Energy) 之间的相关系数极高（0.82），这表明现代流行音乐普遍追求高能量和高响度的听感。
![相关性热力图](images/correlation_heatmap.png)

### B. 歌曲流行度分布情况
数据呈现出明显的正态分布特征，大部分歌曲的流行度集中在 30-60 分区间，极高流行度（>80）的歌曲属于稀缺样本。
![流行度分布图](images/distribution_popularity.png)

### C. 能量与情感效价的相关性
散点图揭示了音乐活跃度（Energy）与情绪正向度（Valence）之间的空间分布，为后续的流派聚类分析提供了重要依据。
![能量效价散点图](images/scatter_energy_valence.png)

### D. 不同音乐流派的平均可舞性排名
直方图清晰展示了不同曲风的节奏属性，Reggaeton、Hip-Hop 和 Reggae 位列前三，是最能带动听众律动的流派。
![可舞性排名](images/genre_danceability_ranking.png)


## 🛠️ 技术栈
* **语言**：Python 3.x、Numpy
* **库**：Pandas, NumPy (数据处理); Matplotlib, Seaborn (数据可视化)
* **算法**：Random Forest (流行度预测), Scikit-Learn (特征工程)

## 运行指南

# 1. 确保安装依赖
pip install pandas scikit-learn seaborn matplotlib
# 2. 运行脚本
python spotify.py

## 📂 文件说明
* `spotify.py`: 包含数据清洗、特征分析及建模的完整核心代码。
* `SpotifyFeatures.csv`: 原始数据集。（太大了上传不了）
* `images/`: 存放所有分析生成的图表。
