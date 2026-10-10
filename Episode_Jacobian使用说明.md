# Episode 逐帧原始 Jacobian pipeline

## 计算定义

每帧使用三路相机、该帧物理 state 和任务文本，固定同一组高斯噪声，完成原生 10 步去噪后选取 `action_step=10`（从零计数，第 11 个动作）。对这个时间步的每一个动作维度分别求导。

- 相机：`J_image[d,c,h,w] = ∂a[10,d]/∂I[c,h,w]`，保留正负号，不预先求模长。
- state：`J_state[i,d] = ∂a[10,d]/∂physical_state[i]`，行是物理 state，列是动作。
- 输出动作在 checkpoint 统计量反归一化后的坐标下；保留 checkpoint 的 delta action 表示，没有额外变成全局绝对姿态。
- 输入 state 为 checkpoint 规定顺序的打包物理 state，位于归一化之前。它不是对视频数据里的四元数等底层存储字段直接求导。
- 图片是模型有效输入 RGB `[0,1]`，在 Qwen 归一化和时间复制之前。主相机当前为 `352×480`，腕部为 `256×320`；原始视频上展示的图是插值结果。
- 模型权重冻结、eval；BF16 autocast 前向、一阶自动微分、FP32 保存。没有更新参数，不使用 GT action loss。
- 当前 checkpoint 推理会屏蔽 state 维度 2、9；其导数为零，保留原生行为。

## 指定不同数据集 / episode

```bash
cd /mnt/vepfs01/output/klay.zhou
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venvs/mozbrain-spirit-gradcam/bin/python -u gradcam_vla/episode_jacobian.py \
  --repo /mnt/vepfs02/output/anthony.cao/mozbrain_spirit_new_topic \
  --checkpoint /mnt/vepfs01/output/klay.zhou/gradcam_vla/checkpoints/pourcorn_step20000 \
  --assets /mnt/vepfs01/output/klay.zhou/gradcam_vla/checkpoints/Qwen3-VL-4B-Instruct-assets \
  --dataset /mnt/vepfs02/output/anthony.cao/datasets/22303_pourcorn_udas/dataset_0_57_ae6e0e92b5fa7267 \
  --episode 0 --action-step 10 --seed 0 --gpus 2,5,7 \
  --image-tile-size 128x160 --vjp-chunk 1 --tiles-per-backward 0 \
  --output gradcam_vla/results/pourcorn_step20000_episode0_action10
```

替换 `--dataset` 和 `--episode` 可以计算其他兼容数据集中的 episode。默认处理该 episode 的**全部帧**，不跳帧。

适配范围：当前 Spirit Qwen3-VL co-training 架构、LeRobot v2 风格的 `meta/info.json`、`meta/tasks.jsonl`、单 episode parquet 和三路视频。相机通过现有别名匹配；state 使用 dataset `observation.state`（须已是 checkpoint 所要求的物理坐标/顺序），或使用指定模型仓库的 ObservationStandardizer 从原始机器人字段打包。其他机器人、state 表示、LeRobot v3 共享视频布局或不同 VLA 架构并非自动兼容。

脚本从 `--repo` 导入模型类和预处理。`--checkpoint` 需要合并后的 HF safetensors 权重及 config/tokenizer；原始 FSDP checkpoint 需要先合并。`--assets` 是对应 Qwen backbone 的本地 config 和图像处理器配置。

## 多卡、分块与恢复

多卡按视频帧分工，每张 GPU 只加载一次模型，连续处理其帧段。相比按 20 个输出再分卡，这样避免每帧重复加载和跨卡合并。

- `--frames 150`：只算样例帧。
- `--frames 148,149,150,151`：只算指定测试片段。
- 不传 `--frames`：整条 episode。
- 相同输出目录再次运行：校验身份与已有 NPZ，跳过有效帧，补算丢失/损坏的帧。
- `--gpus 2,5,7`：指定实际 CUDA ID；每卡需要完整模型副本。
- `--vjp-chunk 1`：一次对一个动作输出求导。增加可提高批量，但可能增加显存/变慢。
- `--image-tile-size 128x160`：有效模型像素上的高×宽；仍重建三路完整画面，不裁图推理。
- `--tiles-per-backward 0`：一次请求所有图块和 state 的梯度；非零时将输入叶子分组求导，会重复上游反向，通常更慢。
- `--activation-offload`：把前向保存的张量暂存 CPU，降低 GPU 激活显存，但增加传输和反向时间。
- `--compress`：NPZ 无损压缩；默认不压缩，优先顺序写盘性能。

图块大小主要控制输入叶子及返回梯度缓冲区；因为仍然进行完整画面的前向计算，缩小图块不会按面积同比降低模型权重或主干激活显存。显存不足时还需要结合 `--vjp-chunk 1`、`--tiles-per-backward` 和 `--activation-offload` 调整。不同分块保留完整图像上下文。

一个当前样例的 NPZ 约 88 MB（包括原图、模型输入等），511 帧约 45 GB。工作进程不会将全 episode Jacobian 常驻内存，每帧算完即落盘。写盘使用临时文件加原子替换，避免将半写文件当作完成结果。

运行身份包含 checkpoint/模型代码/图像配置、数据集与 parquet、episode、动作时间步、随机种子等。改变这些设置应使用新输出目录。更换 GPU、图块大小、批量、压缩方式可以续算，但浮点重复计算不保证逐位相同。

## 输出格式

```text
OUTPUT/
  manifest.json         # 数据、模型、帧数、FPS、运行身份
  status.json           # 已完成帧和 episode_complete
  jobs/<id>/worker*.log # 本次各卡日志
  frames/
    frame_000000.npz
    frame_000001.npz
    ...
```

每个 NPZ：

| Key | 形状 / 类型 | 意义 |
|---|---|---|
| `jac_image_cam_high` | `[D,3,H,W]` FP32 | 主相机带符号偏导 |
| `jac_image_left_wrist_high` | `[D,3,H,W]` FP32 | 左腕带符号偏导 |
| `jac_image_right_wrist_high` | `[D,3,H,W]` FP32 | 右腕带符号偏导 |
| `jac_state` | `[S,D]` FP32 | 输入物理 state × 动作列 |
| `physical_state` | `[S]` | 求导位置的真实 state |
| `selected_action` | `[D]` | 被求导的动作时间步 |
| `predicted_actions` | `[1,T,D]` | 完整动作 chunk，供核查 |
| `image_*` | uint8 RGB | 有效模型图像 |
| `original_*` | uint8 RGB | 原视频帧，后续可视化无需再读视频 |
| `state_labels` / `action_labels` | 字符串数组 | 各维度名称 |
| `metadata` | JSON 字符串 | 帧号、时间戳、task、固定 noise hash、坐标定义等 |

`D`、`S` 来自 checkpoint；当前均为 20。`jac_state` 与相机 Jacobian 采用不同轴序以方便矩阵显示，元数据明确记录。

读取：

```python
with np.load('frames/frame_000150.npz', allow_pickle=False) as z:
    J_state = z['jac_state']  # [physical_state, action]
    J_cam = z['jac_image_cam_high']  # [action, RGB, H, W]
    heat = np.sqrt((J_cam.astype(np.float64) ** 2).sum(axis=(0, 1)))
```

## 可视化

脚本：`visualize_episode_jacobian.py`。不加载模型，不占 GPU。

### 相机视频 / 单帧图

```bash
.venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/visualize_episode_jacobian.py \
  --input gradcam_vla/results/pourcorn_step20000_episode0_action10 \
  --mode camera --cameras all --normalization relative \
  --output gradcam_vla/results/pourcorn_step20000_episode0_action10/camera_episode_relative.mp4
```

- `--cameras cam_high`、`--cameras left_wrist_high,right_wrist_high` 可选相机。
- `--actions 9` 只看动作第 9 维；`--actions 0,1,2` 聚合多个动作列；默认全动作列。
- 像素热图 `sqrt(sum_selected_action,sum_RGB J²)`。
- 默认 `--normalization relative`：每帧独立计算所选相机全部像素热图的 99.7 百分位上限，同一帧三路相机共享该上限；跨帧颜色不代表相同的绝对梯度大小。
- 已恢复旧版两排布局和显示公式：第一排原图、第二排热图，线性颜色 `z=clip(H/upper,0,1)`，透明度 `alpha=0.58*z`。固定透明度、平方根色阶和第三排已回滚。
- `--normalization global` 恢复整条视频统一色标：从每帧各路热图确定性抽样得到全局 99.7 百分位；此模式下可用 `--vmax` 固定上限。
- 相机输出附带 `.diagnostics.json`，记录每帧各相机的梯度模长中位数、99/99.7 百分位、最大值、非零比例和色标上限。
- `--frames 150 --output .../frame150.png` 输出单帧对比图。
- MP4 按源 FPS 编码，要求连续帧，避免把跳帧视频伪装为原时间轴。

### 指定帧 state × action

```bash
.venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/visualize_episode_jacobian.py \
  --input gradcam_vla/results/pourcorn_step20000_episode0_action10 \
  --mode state-frame --frames 150 --output /tmp/state_frame150.png
```

每个格子为 `abs(J_state[i,d])`。用 `--states 0,1,6` 选输入行，用 `--actions 0,1,2` 选输出列。单位为所选输出动作单位 / 所选物理 state 单位，各维度存在量纲差异。

### 整段 state × action RMS 汇总

```bash
.venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/visualize_episode_jacobian.py \
  --input gradcam_vla/results/pourcorn_step20000_episode0_action10 \
  --mode state-rms --output /tmp/state_episode_rms.png
```

`H[i,d] = sqrt(mean_frame J_state[frame,i,d]²)`，不会因不同帧正负相反而抵消。

### 指定 state 随视频时间变化

```bash
.venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/visualize_episode_jacobian.py \
  --input gradcam_vla/results/pourcorn_step20000_episode0_action10 \
  --mode state-time --state-index 6 --output /tmp/state6_over_time.png
```

行是视频帧/时间，列是动作维度，格子为该 state 的偏导绝对值。所有矩阵同时保存 `.npy` 数值和 `.json` 可视化配置。

默认可视化要求完整 episode；只有明确指定 `--frames` 或 `--allow-partial` 才使用片段，输出元数据会记录是否完整。

## 完整结果检查

```bash
PYTHONNOUSERSITE=1 .venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/validate_episode_jacobian.py \
  --input gradcam_vla/results/pourcorn_step20000_episode0_action10
```

逐帧流式读取原始矩阵，检查完整帧序列、输入与输出维度、FP32 有限值、固定噪声一致性、被屏蔽 state 的零导数，以及选定动作与完整 chunk 的对应关系；输出 `validation.json`。该步骤不使用 GPU。

## 当前完整 episode 样例

目录：`results/pourcorn_step20000_episode0_action10`。使用 pourcorn step20000、episode 0、固定 seed 0、action step 10，共 511 帧、30 FPS。

- 三卡（2、5、7）本次计算任务实际耗时 3543.16 秒，约 59.1 分钟，包含模型加载与写盘；已存在的第 150 帧被跳过。
- 511 个原始 NPZ 总计 45,039,358,314 字节，约 45.04 GB。
- `frame150_comparison.png` / `frame150_state.png`：指定帧相机与 state 示例。
- `camera_episode_relative.mp4`：当前版逐帧归一化、两排三相机原图与热图对比。
- `camera_episode.mp4`：历史 global 色标视频，保留供对照。
- `state_episode_rms.png`：全 511 帧 state × action RMS。
- `state6_over_time.png`：输入 state 第 6 维（左夹爪）的帧 × action 热图。
- `validation.json`：全量原始矩阵检查结果。

本次 511 帧全部通过 FP32 形状/有限值检查，noise SHA 全部一致，原生与可微去噪输出最大绝对差为 0。视频已逐帧解码验证：511 帧、30 FPS、1440×740、17.033 秒；RMS 矩阵为 20×20，单 state 时间矩阵为 511×20。详见 `visualization_validation.json`。

算法测试覆盖有符号解析导数、物理归一化、state 屏蔽、图块/VJP 分组一致性、原子写盘及恢复、RMS 防止符号抵消，以及不同数据集/episode 下三相机视频对齐。固定噪声不意味着 BF16 计算在不同设备或分组下逐位相同。

### 当前显示方式：旧版布局 + 逐帧 relative

旧 `camera_episode.mp4` 使用线性色阶加随梯度变化的透明度。全局上限为 0.0127502，第 500 帧右腕相机梯度模长的 99 百分位仅 0.000186148，旧版对应透明度约 0.00847，因此容易看起来没有热图。

当前 `camera_episode_relative.mp4` 恢复旧版线性色阶与随梯度变化的透明度，只将归一化范围改为每帧。设第 t 帧三路相机热图为 H，u_t 为该帧合并像素的 99.7 百分位，则显示 z=clip(H/u_t,0,1)，alpha=0.58*z。每帧 u_t 记录在视频同名 JSON 的 `frame_color_upper` 中。

上一轮三排增强版 `camera_episode_visibility.mp4` 作为历史结果保留，其显示选项已从当前脚本回滚。所有版本使用同一组原始 NPZ，没有重新计算或改变梯度。

```bash
PYTHONNOUSERSITE=1 .venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/visualize_episode_jacobian.py \
  --input gradcam_vla/results/pourcorn_step20000_episode0_action10 \
  --mode camera --normalization relative \
  --output gradcam_vla/results/pourcorn_step20000_episode0_action10/camera_episode_relative.mp4
```
