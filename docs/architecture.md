# 架构设计

## 1. MiniMax H3 模型架构

MiniMax H3(2026-07-31 发布,2026-08-03 开源 H3-Base)是全模态视频生成系统,由三个模块串联:

```
用户输入(文/图/视频/音频)
   │
   ▼
H3-Context-IR(多模态指令理解 → 结构化提示词)   ← 托管 API,未开源
   │
   ▼
H3-Base(核心生成,768P + 原生立体声)            ← 开源,本项目本地部署部分
   │
   ▼
H3-Regenerate-2K(768P → 2K 再生)               ← 托管 API,未开源
```

**开源边界:本地部署最高 768P;2K 需本地 768P 产物调用官方 Regeneration API($0.05/s)。**

### H3-Base 组件

| 组件 | 说明 |
|---|---|
| H3-Omni-Transformer | 33B 参数 dense 单流 Transformer;其中 ~13B 位于 AdaLN 调制分支,**推理时可预计算缓存,无需常驻显存**;3D MM-RoPE(t,h,w);无模态专属结构(模态参数仅存在于输入/输出层与 AdaLN) |
| H3-Encoder | 完整 Qwen3-VL-32B 权重,取第 50 层隐状态输入 Transformer |
| H3-VisualVAE | 时间因果视频自编码器,空间 16× / 时间 4× 压缩,24 通道隐空间(f16t4d24);patch 1×2×2 后有效空间下采样 32×;ViT 解码器 |
| H3-AudioVAE | 32kHz 立体声音频 VAE |

### 生成规格

- 时长 4-15s,24fps(5s→124 帧、10s→243 帧、15s→362 帧)
- 输出 H.264 视频 + AAC 32kHz 立体声(音画一体生成)
- 宽高须 32 像素对齐
- 检查点:FL2VA(t2va 文生视频 / fl2va 首尾帧图生视频)与 Ref2VA(参考图/视频/音频生成)两套独立权重,请求内互斥

## 2. 本项目部署拓扑

### 硬件与 NUMA 约束

8 × RTX PRO 5000 72GB(sm_120)无 NVLink,PCIe 互连,双 NUMA 节点:

```
NUMA 0(CPU 0-127)                NUMA 1(CPU 128-255)
┌─────────────────────┐          ┌─────────────────────┐
│  GPU0   GPU1        │          │  GPU4   GPU5        │
│  GPU2   GPU3        │          │  GPU6   GPU7        │
└─────────────────────┘          └─────────────────────┘
   节点内 PCIe(NODE)                跨节点 SYS(慢,避免跨节点组实例)
```

**规则:任何多卡实例的 GPU 必须落在同一 NUMA 节点内,并以 numactl 绑定对应节点。**

### 测试方案拓扑

| 方案 | 拓扑 | 说明 |
|---|---|---|
| 测试1 INT8 单卡 | 1×GPU(容器) | 精度/显存验证;768P 需 layerwise offload |
| 测试2 BF16 4卡 | TP2 + Ulysses2 × 4 GPU(单 NUMA 节点) | 官方 4×H100 最快拓扑;Ulysses4 在 72GB 卡必 OOM(94GB/卡) |
| 测试3 候选 | 2×(TP2+U2)每 NUMA 一实例 / FP8 / FastH3 蒸馏 / 多实例 INT8 | 吞吐与成本排名,详见 benchmarks.md |

### 服务与网络架构

```
用户/压测机
   │
   ├─ :8080 Nginx(Basic Auth:admin)─── 反代 :7860
   │                                      ┌────────────────────┐
   │                                      │ Gradio WebUI 容器   │
   │                                      └────────┬───────────┘
   │                                               │ localhost
   ├─ :30010 SGLang 实例1(FL2VA, GPU0-3, NUMA0)◀┘
   ├─ :30011 SGLang 实例2(FL2VA, GPU4-7, NUMA1)
   └─ bench.py(直连 SGLang,绕开代理,采样 nvidia-smi)
```

- SGLang 端口仅监听本机/内网,不对公网暴露(无鉴权)
- WebUI 仅经 Nginx Basic Auth 对外
- 基准脚本与 WebUI 的 HTTP 客户端均 `trust_env=False`,避免残留 http_proxy 劫持 localhost

## 3. 分辨率档位映射

本地 SGLang 以 `target.short_edge` 指定分辨率;MP 档沿用 ComfyUI Megapixels 惯例(16:9):

| 档位 | 尺寸(px) | short_edge | 像素量 | 定位 |
|---|---|---|---|---|
| 0.3MP | 736×416 | 416 | 0.31MP | 抽卡/预览档 |
| 0.5MP | 960×544 | 544 | 0.52MP | 质量/吞吐折中 |
| 0.7MP | 1152×640 | 640 | 0.74MP | 接近官方画质 |
| 768P | 1344×768 | 768 | 1.03MP | 官方画质配方 |

> 官方 API 仅有 480P/768P/2K 档;本地任意 short_edge 可用,但仅 768 是官方画质配方。
