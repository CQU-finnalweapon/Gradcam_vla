# 全动作 Jacobian 多卡并行和图像分块

入口：`run_spirit_action_jacobian_multigpu.py`。沿用指定 Spirit/Qwen3-VL 仓库、20000 step checkpoint 和 pourcorn 数据集。每张 GPU 加载同一个冻结模型，使用相同图像、state、文本和固定噪声，只对分配到的动作输出求导。两张卡分别计算 `[0,600)`、`[600,1200)`，四张卡各计算 300 项。动作顺序为时间优先，即 `index = timestep * 20 + dimension`。

最终在有效输入像素网格上求 `sqrt(sum_gpu sum_action sum_rgb gradient²)`。先合并平方和，再开方，再插值；不合并各卡已经渲染的热图。默认完整覆盖 1200 项输出，没有抽样近似。

## 推荐命令

在 `/mnt/vepfs01/output/klay.zhou` 下运行，并按实际空闲情况指定 GPU：

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venvs/mozbrain-spirit-gradcam/bin/python -u \
  gradcam_vla/run_spirit_action_jacobian_multigpu.py \
  --gpus 3,6 \
  --vjp-chunk 1 \
  --image-tile-size 128x160 \
  --tiles-per-backward 0 \
  --output gradcam_vla/results/my_multigpu_run
```

`--gpus` 是传给各子进程 `CUDA_VISIBLE_DEVICES` 的设备 ID 或 UUID，不是父进程可见列表中的相对编号。模型副本占用每张卡的显存：多卡分配动作输出不会将模型权重显存除以卡数。

## 可调选项

| 参数 | 含义 | 显存与速度 |
|---|---|---|
| `--gpus 3,6` | 选择并行 GPU，可增加到更多空闲卡 | 分摊动作输出的反向工作，各卡仍保留完整前向图 |
| `--vjp-chunk 1` | 一次求导的动作输出数量 | 1 最保守；2/4/8 增加批量临时梯度和显存，是否更快需实测 |
| `--image-tile-size full` | 每路整图作为一个输入叶子 | 默认；没有裁图推理 |
| `--image-tile-size 128x160` | 高×宽，单位是模型实际 RGB 输入像素 | 每个区域变成独立求导叶子，再拼回原位置；边缘不足一块自动保留 |
| `--tiles-per-backward 0` | 0 表示全部图块一起求导 | 通常最快，不重复上游反向 |
| `--tiles-per-backward 4` | 每次只请求 4 块的梯度 | 减少返回图像梯度缓冲，但重复上游反向，可能显著变慢 |
| `--activation-offload` | 前向保存的 autograd 张量暂存在 CPU | 实际降低 GPU 激活显存，但消耗 CPU RAM/PCIe 带宽，多轮反向可能很慢 |
| `--reference DIR` | 和已有单卡完整结果校验 | 校验样本、动作预测和全部有效输入/原图展示热图 |
| `--episode/--frame/--seed` | 调整数据帧及固定采样噪声 | 所有 GPU 使用相同值 |

这里图像切块是真正的 autograd 输入分组。所有区域仍以原顺序拼成三路完整图像再送进 Qwen，未更改图片分辨率、位置信息或视觉上下文。`128x160` 时三路共 17 个区域。只改 tile size 而保持 `tiles-per-backward=0`，不会明显降低全模型激活内存。

Transformer 会将所有视觉 token 相互耦合。只请求部分区域的导数，依然要经过完整的视觉/语言/动作计算图。因此图像越小块不意味着总显存按比例降低；它也不能把一张显存不够放模型的 GPU 变得可用。显存调节顺序建议：先将 VJP chunk 降到 1；仍不足时使用 offload；仅在返回梯度缓冲成为问题时限制 tiles per backward。

当前多卡计算仍为反归一化动作的 Jacobian，BF16 autocast 前向，FP32 局部平方累加、CPU FP64 卡间累加，不改变之前指标的动作单位或语义。

## 结果和诊断

- `comparison.png` 和三个单相机图：完整结果。
- `action_jacobian_norm.npz`：合并后的浮点数据，包含有效输入和原图展示坐标。
- `predicted_actions.npy`：完整预测动作。
- `multigpu_metadata.json`：每张卡动作范围、图块数、反向耗时、峰值 allocated/reserved 显存、总耗时和参考误差。
- `partials/<本次唯一编号>/workerN.log`：每个进程的独立日志。
- 同目录 `workerN.npz/json`：原始平方和、动作和输入校验信息。每次使用唯一目录，旧结果不会混入新的归并。

任一 worker 失败，会终止本次其余 worker 并保留日志；不把部分输出当成完整热图。合并前检查动作范围连续无重复、输入/配置一致、各卡预测动作一致。对于新的样本，先运行单卡或用小规模验证确认预算，再提高 VJP chunk。

## 本次实测

样本仍为 episode 0 / frame 150，GPU 3 和 6 各负责 600 项输出，`vjp-chunk=1`、`image-tile-size=128x160`、`tiles-per-backward=0`。

- 每卡反向耗时 258.1 / 271.0 秒；旧单卡日志中的反向区间约 507 秒，反向部分约加速 1.9 倍。共享机器的其他负载和非同时运行的基线会影响这个比较。
- 最慢 worker 从加载到计算完成 427.1 秒，约 7.1 分钟；旧单卡完整样例约 631.7 秒。此次首次合并严格阈值检查失败后，直接利用保存的部分结果重做归并，无需重新反向。
- 每卡峰值 allocated 显存约 45.07 GiB，reserved 约 46.19 GiB。`nvidia-smi` 含其他进程和 CUDA 运行时的数值不能直接当成本脚本的张量显存。
- 动作预测与旧单卡最大差异为 0。
- 有效输入热图相对 L2 误差为 0.057%～0.077%，最大像素绝对误差 7.14e-5。结果并非逐位相同；原来 `rtol=2e-4, atol=1e-7` 的严格逐像素检查未通过，已如实保存在元数据中。
- 同一计算图重复反向的单动作梯度相对波动约 1.4%～1.8%，切块/整图两次反向差异约 1.2%～1.3%；直接检查拼接运算的梯度路由误差为 0。因此不能把所有 BF16 原始梯度差异归因于多卡合并或图块拼接。
- 默认参考结果门限明确配置为 `--reference-relative-l2 0.002` 和 `--reference-max-abs 0.0001`，不是逐像素严格等同断言。两者均写入结果，可按需求收紧。
- 小规模实际模型测试运行了 `vjp-chunk=2, tiles-per-backward=8`，8 个输出反向 7.73 秒，峰值 allocated 45.17 GiB。这组配置相对默认并未节省显存或加速，说明细分图像不应作为主要加速手段。
- `--activation-offload` 已在真实模型单项动作测试中跑通，峰值 allocated 20.75 GiB、reserved 20.94 GiB，单项反向 2.82 秒，原生/可微动作误差 0。尚未将该选项用于完整 1200 项输出计时，因此不要把这个小测试的总时间当作完整计算时间。

完整样例：`results/pourcorn_step20000_jacobian_2gpu_tile128x160/comparison.png`。

若只需重新合并已完成的计算，可使用 `--merge-only results/.../partials/<任务编号>`，同时指定对应 `--gpus`、`--output` 和可选 `--reference`。`multigpu_metadata.json` 的 `merge_only=true` 时，`seconds` 是此次合并时间，不能当作完整模型计算耗时；应看 `max_worker_seconds` 和各 worker 的 `backward_seconds`。
