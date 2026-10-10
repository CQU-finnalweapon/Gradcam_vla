# Gradcam_vla

VLA 模型的三相机 Grad-CAM、输入像素 Jacobian 与物理 state 梯度可视化工具。支持离线单帧分析及整条 episode 逐帧计算，不控制机器人，不更新模型参数。

当前主要 pipeline 使用 **Spirit Qwen3-VL**：固定采样噪声，完整执行去噪推理，选取 `action_step=10`（从零计数）对每个动作分量分别求导。每帧保存带符号 FP32 原始 Jacobian，再独立生成视频或 state×action 热力图。输入 Jacobian 与中间特征层 Grad-CAM 是两种不同的分析方式。

## 功能与文件

| 入口 | 功能 |
| --- | --- |
| `episode_jacobian.py` | 整条 episode 三相机像素和物理 state Jacobian，多 GPU 按帧分工、输入分块、断点恢复 |
| `visualize_episode_jacobian.py` | 相机视频、单帧 state 矩阵、整段 RMS 矩阵、指定 state 随时间变化 |
| `validate_episode_jacobian.py` | 完整帧序列、矩阵形状、有限值与固定噪声检查 |
| `run_spirit_action_jacobian.py` | 单帧全动作 chunk 像素梯度模长 |
| `run_spirit_action_jacobian_multigpu.py` | 单帧按动作输出维度多卡并行 |
| `run_spirit_qwen.py` | Spirit Qwen3-VL 视觉特征层 Grad-CAM |
| `run.py` / `gradcam.py` | PI0.5 / SigLIP Grad-CAM |
| `compare_checkpoints.py` | checkpoint 可视化对比 |

三路相机：`cam_high`、`left_wrist_high`、`right_wrist_high`；支持文档列出的腕部相机别名。

## 环境与模型

推荐为 Spirit 单独创建 Python 环境。实测环境使用 PyTorch 2.9.1、torchvision 0.24.1、Transformers 4.57.1；直接依赖见 `requirements-spirit.txt`。

```bash
python -m pip install -r requirements-spirit.txt
```

还需要自行准备：

- 与 checkpoint 配套的 mozbrain / Spirit 模型源码及其依赖，由 `--repo` 指定，不能任意替换成另一版本的模型结构。
- 合并后的 HF safetensors checkpoint、配置、归一化统计量和 tokenizer，由 `--checkpoint` 指定。
- 匹配的本地 Qwen backbone 配置和图像处理器资源，由 `--assets` 指定。
- 相机与物理 state 定义兼容的 LeRobot v2 episode 数据集，由 `--dataset` 指定。

本仓库不包含上述模型源码、权重和数据。LeRobot v3 共享视频布局或其他 VLA 架构需要额外适配。部分旧脚本保留原实验机的默认路径，换机器时请设置对应路径；episode 入口支持下面列出的路径参数。PI0.5 使用配套仓库的 patched Transformers，与 Spirit 环境分开，见 [PI0.5 说明](PI05_GradCAM_README.md)。

## 计算整条 episode

在本仓库根目录运行，将示例路径替换为本机实际路径：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python episode_jacobian.py \
  --repo /path/to/checkpoint-matched/mozbrain \
  --checkpoint /path/to/merged-checkpoint \
  --assets /path/to/qwen-backbone-assets \
  --dataset /path/to/lerobot-v2-dataset \
  --episode 0 --action-step 10 --seed 0 --gpus 0,1,2 \
  --image-tile-size 128x160 --vjp-chunk 1 \
  --output results/episode0_action10
```

默认计算所有帧；`--frames 150` 可先运行单帧样例。同目录同配置重跑会跳过有效结果。每张 GPU 都需要完整模型副本；图块重建完整图像再推理，缩小图块不会按面积同比降低主干激活显存。可用 `--activation-offload` 将保存的激活移到 CPU。

每帧 NPZ 包含：

- `jac_image_<camera>`：`[action_dim, RGB, H, W]`，对模型有效 RGB `[0,1]` 像素求导。
- `jac_state`：`[state_dim, action_dim]`，对归一化前的打包物理 state 求导。
- 输入 state、预测 action chunk、原图、维度标签及模型/帧/噪声元数据。

输出动作沿用 checkpoint 反归一化后的表示，可能是 delta action，并非自动转换成世界坐标绝对姿态。当前样例模型屏蔽 state 第 2、9 维，因此其导数为零。

## 可视化

```bash
# 两排相机视频：上排原图、下排梯度模长；每帧独立归一化
python visualize_episode_jacobian.py \
  --input results/episode0_action10 --mode camera --normalization relative \
  --output results/episode0_action10/camera_relative.mp4

# 指定帧的 state × action 偏导绝对值
python visualize_episode_jacobian.py \
  --input results/episode0_action10 --mode state-frame --frames 150 \
  --output results/episode0_action10/state150.png

# 全 episode 的 state × action RMS
python visualize_episode_jacobian.py \
  --input results/episode0_action10 --mode state-rms \
  --output results/episode0_action10/state_rms.png

# 指定输入 state 维度随时间变化
python visualize_episode_jacobian.py \
  --input results/episode0_action10 --mode state-time --state-index 6 \
  --output results/episode0_action10/state6_time.png
```

相机热图为选定动作分量和 RGB 通道上的 Jacobian L2 模长。默认 `relative` 使用每帧三路相机合并像素的 99.7 百分位上限，同帧共享色标；跨帧颜色不能比较绝对梯度大小。`--normalization global` 使用全段统一色标。用 `--cameras`、`--actions`、`--states` 选择相机或矩阵维度。可视化不加载模型、不占用 GPU。

## 验证与资源参考

```bash
python -m pytest test_gradcam.py test_action_jacobian.py test_episode_jacobian.py -q
python validate_episode_jacobian.py --input results/episode0_action10
```

已完成的 511 帧、30 FPS episode 示例：3 张 A100 80GB 计算约 59 分钟；单帧 NPZ 约 88 MB，合计约 45 GB。实际开销随模型、图像大小、动作维度、存储与硬件而变化。权重、NPZ、视频、图片、日志和缓存均通过 `.gitignore` 排除。

更多说明：[Episode pipeline](Episode_Jacobian使用说明.md)、[多卡 Jacobian](多卡Jacobian使用说明.md)、[全动作像素梯度](全动作像素梯度说明.md)、[Spirit Grad-CAM](Spirit_Qwen_GradCAM说明.md)。详细实验文档中的绝对路径是原运行环境示例。
