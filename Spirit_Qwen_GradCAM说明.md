# Spirit Qwen3-VL 三相机 Grad-CAM

运行脚本：`run_spirit_qwen.py`。此适配使用 `/mnt/vepfs02/output/anthony.cao/mozbrain_spirit_new_topic` 下的模型与机器人预处理代码，配置、tokenizer、权重来自用户指定训练任务的 `checkpoints/020000`；未修改该仓库。

当前样本为数据集 `dataset_0_57_ae6e0e92b5fa7267` 的 episode 0、frame 150，任务为 `Pour corns into the bowl`。

解释目标为最终预测动作 `action[10, 9]`，索引从 0 开始，即动作 chunk 中第 11 个动作的右臂末端位置 Z 分量。它经过 checkpoint 自带统计量反归一化。使用固定 seed 0，完整运行 10 步 Euler 采样。

## 可视化层与方法

Hook 层是 `model.paligemma_with_expert.qwen.model.visual.blocks.23` 的输出，即 Qwen3-VL 视觉编码器第 24 个、也是最后一个 Transformer block，位于视觉 token merger 之前。

对每路相机分别计算通道权重 `alpha[k] = mean_spatial(d action[10,9] / d A[:,k])`，得到 `signed_cam = sum_k(alpha[k] * A[:,k])`。标准 Grad-CAM 使用 `ReLU(signed_cam)`；第三行使用 `ReLU(-signed_cam)` 作为负方向诊断图。

三路相机的原始 patch 网格分别为 22×30、16×20、16×20。Qwen 将 token 按 2×2 空间分组排列，脚本先恢复正常行列顺序，再上采样、映射到相机原图。最终图片的平滑细节来自插值，原始空间分辨率仍是上述 patch 网格。

## comparison.png 的九张图

| | 左列 | 中列 | 右列 |
|---|---|---|---|
| 第一行 | 主相机原图 | 左腕相机原图 | 右腕相机原图 |
| 第二行 | 主相机正向 Grad-CAM | 左腕相机正向 Grad-CAM | 右腕相机正向 Grad-CAM |
| 第三行 | 主相机负向诊断图 | 左腕相机负向诊断图 | 右腕相机负向诊断图 |

各热力图独立缩放颜色，不能靠不同相机的颜色亮度比较其绝对贡献。亮区是所选动作标量在所选层上的梯度归因，不是物体检测结果或注意力概率。第三行不是“模型不关注”的区域；它表示该签名 CAM 的负值部分。

Qwen3-VL 还有从较早视觉层进入语言模型的 DeepStack 支路。本结果只解释最后视觉 block 的特征，不能覆盖所有视觉支路。也不能与此前 PI0.5/SigLIP 的图直接做数值比较。

## 校验与来源

- 严格核对全部权重键和形状，载入 checkpoint 归一化统计量。
- 与同仓库原生采样器比较完整动作张量，结果写入 `metadata.json`。
- 可微采样保留原生代码的实际数值行为，包括 `noise` 与 `x_t` 原地更新导致的补齐维度处理逻辑。
- 仅绕过动作反归一化函数的 `no_grad` 装饰器，计算公式不变。
- 输出保留原图、模型输入、原始有符号 CAM、预测动作、代码来源 SHA256。
- Qwen 4B backbone 配置从该训练日志恢复；图像处理器配置复用本地 Qwen3-VL 同系列资源，图像均值、标准差、rescale 和 merge 参数与训练日志一致，详情记录于 `checkpoint_provenance.json`。

## 重跑

```bash
cd /mnt/vepfs01/output/klay.zhou
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venvs/mozbrain-spirit-gradcam/bin/python gradcam_vla/run_spirit_qwen.py
```

共享文件系统读取约 21GB 权重可能需要数分钟。脚本先将 mmap 张量复制到普通 CPU 内存，再载入 GPU。可用 `--episode`、`--frame`、`--action-step`、`--action-dim`、`--seed` 和 `--output` 调整样本与解释目标。
