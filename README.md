# Minimax-H3-P5K-Bench

MiniMax H3(33B 全模态视频生成模型)在 **8× NVIDIA RTX PRO 5000 72GB(Blackwell)** 上的本地部署、推理加速与基准测试报告。

## 硬件环境

| 项 | 规格 |
|---|---|
| GPU | 8 × NVIDIA RTX PRO 5000 72GB Blackwell(sm_120 / CC 12.0,350W),共 576GB 显存 |
| 互连 | PCIe(无 NVLink);NUMA0: GPU0-3↔CPU 0-127,NUMA1: GPU4-7↔CPU 128-255 |
| CPU / 内存 / 磁盘 | 256 核 / 743GB / 1TB |
| 驱动 / CUDA | 580.126.20 / CUDA 13.0 |
| 平台 | 腾讯云 CVM(ap-beijing) |

## TL;DR(2026-09-10 实测,768P / 10s 片 / 50 步 / seed=1101)

| 方案 | E2E(s) | 峰值显存/卡 | 8卡吞吐(视频秒/h) | **元/视频秒** | vs 官方 ¥0.57/s |
|---|---|---|---|---|---|
| INT8 单卡(kitchen_int8+offload) | 2233.0 | 44GB* | 128(8实例,理论) | ¥0.217* | 2.6× 便宜* |
| BF16 4卡 TP2+U2 | 728.4 | 64.9GB | 98(2实例) | ¥0.283 | 2.0× 便宜 |
| **FP8 4卡 TP2+U2(推荐)** | **646.3** | **52.9GB** | **111(2实例,双实例并发已验证)** | **¥0.250** | **2.3× 便宜** |
| FP8 + batching×4 | 646.3(串行) | 59.9GB | 111(无增益) | ¥0.250 | sglang 0.5.19 对 H3 串行执行,批处理不生效 |
| BF16+batching×4 | OOM | 72.3GB | 0 | — | 排队请求编码/VAE 显存重叠 OOM |
| 8×INT8@0.5MP | 704.6 | 61.5GB | 408(理论) | ¥0.068 | 8.4× 便宜(理论) |

> *理论/分析值;FP8+batching 结果见 [benchmarks.md](docs/benchmarks.md)。8 卡月成本按 20000 元(2500 元/卡×8)。
> **质量结论:INT8 vs BF16 768P 同条件出片肉眼无可辨差异**(对照拼图见 results/quality/)。
> FastH3 蒸馏版被 sglang 0.5.19 阻塞(缺 video_sparse_attn_h3 后端),权重已就绪待框架更新。

## 仓库结构

```
├── docs/
│   ├── architecture.md   # 架构设计(H3 模型结构 + 部署拓扑)
│   ├── deployment.md     # 部署文档(环境/镜像/各方案命令/踩坑)
│   ├── user-guide.md     # 使用手册(API/WebUI/参数表/FAQ)
│   └── benchmarks.md     # 测试性能指标(方法论 + 全量数据)
├── results/              # 原始数据(JSONL/CSV)与图表
├── scripts/              # 基准测试与权重下载脚本
└── assets/               # 基准素材(t2va prompt、fl2va 参考图)与样片截图
```

## 当前进度

- [x] 环境准备:Docker 29.7.2 + nvidia-container-toolkit 1.20.0,容器内 8 卡可见
- [x] SGLang H3 镜像(sglang 0.5.19 + diffusion + comfy-kitchen,cu130;关键坑:基础镜像 diffusers 0.37 无 H3,须升 0.40)
- [x] 权重下载(COS 内网 1179MB/s,FL2VA 144GB / 2 分钟)
- [x] 测试1:INT8 单卡全档位(0.3/0.5/0.7MP/768P + fl2va 图生视频)
- [x] 测试2:BF16 4 卡 TP2+U2 全档位 + fl2va
- [x] 测试3:候选 A(BF16×2)/ B(FP8×2)/ E(batching,OOM 复现)/ D(理论分析);C(FastH3)被框架版本阻塞
- [x] 质量评估:INT8 vs BF16 四档位对照拼图 + PSNR/SSIM(含口径说明)
- [x] WebUI(Gradio + Nginx Basic Auth admin/tencentcloud)
- [x] FP8+batching 候选F 实测(见 benchmarks.md §5/§6)

## License

- 本仓库文档与脚本:MIT
- MiniMax-H3 模型权重遵循 [MiniMax 社区许可](https://huggingface.co/MiniMaxAI/MiniMax-H3)(排除美/欧/英/韩地区),权重不入本仓库
