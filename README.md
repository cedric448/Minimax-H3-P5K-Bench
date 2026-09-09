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

## TL;DR(结论速览,随测试进度更新)

| 方案 | 配置 | 分辨率 | E2E 延迟 | 峰值显存/卡 | 吞吐(视频秒/小时) | 元/视频秒 | vs 官方 API |
|---|---|---|---|---|---|---|---|
| 测试1:INT8 单卡 | kitchen_int8 / ConvRot INT8 | 待测 | 待测 | 待测 | 待测 | 待测 | — |
| 测试2:BF16 4卡 | TP2+Ulysses2 | 待测 | 待测 | 待测 | 待测 | 待测 | — |
| 测试3:8卡最优 | 待定 | 待测 | 待测 | 待测 | 待测 | 待测 | — |

> 对标:MiniMax 官方 API 768P 定价 $0.08/s ≈ ¥0.57/s。本地成本按 8 卡 × 2500 元/卡/月 = 20000 元/月核算:**元/视频秒 = 20000 / (吞吐 T[视频秒/小时] × 720)**。

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
- [x] SGLang H3 镜像(sglang[diffusion] + comfy-kitchen,cu130)
- [ ] 权重下载(COS 内网,FL2VA ~144GB)
- [ ] 测试1:INT8 单卡(0.3/0.5/0.7MP/768P)
- [ ] 测试2:BF16 4 卡 TP2+U2(0.3/0.5/0.7MP/768P)
- [ ] 测试3:8 卡极致吞吐 + 成本排名
- [ ] WebUI(Gradio + Nginx Basic Auth)
- [ ] 质量评估(INT8 vs BF16、与官方 API 对比)

## License

- 本仓库文档与脚本:MIT
- MiniMax-H3 模型权重遵循 [MiniMax 社区许可](https://huggingface.co/MiniMaxAI/MiniMax-H3)(排除美/欧/英/韩地区),权重不入本仓库
