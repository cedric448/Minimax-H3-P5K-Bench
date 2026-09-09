# 部署文档

> 环境:腾讯云 CVM ap-beijing,8× RTX PRO 5000 72GB(Blackwell sm_120),驱动 580.126.20,CUDA 13.0,256 核 / 743GB 内存 / 1TB 磁盘。

## 1. 环境准备(已完成步骤实录)

### 1.1 Docker + GPU 容器运行时

```bash
# TencentOS 4 的 EPOL 源自带 docker-ce(内网直装,免代理)
yum install -y docker-ce docker-ce-cli containerd docker-buildx-plugin

# nvidia-container-toolkit 不在 EPOL,加官方 repo(需代理)后安装
startvpn   # 本机 v2ray 代理函数,http://127.0.0.1:1087
curl -sL -o /etc/yum.repos.d/nvidia-container-toolkit.repo \
  https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
yum install -y nvidia-container-toolkit   # 1.20.0

# 配置 nvidia runtime + 腾讯云 Docker Hub 内网镜像加速
nvidia-ctk runtime configure --runtime=docker
# /etc/docker/daemon.json 增加 "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
systemctl enable --now docker

# 验证:容器内 8 卡可见
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

### 1.2 SGLang H3 镜像

```bash
docker pull lmsysorg/sglang:latest     # 走腾讯云镜像加速,约 52GB
docker run -d --name sglang-build --gpus all lmsysorg/sglang:latest sleep infinity
docker exec sglang-build bash -c "
  pip install -i https://mirrors.cloud.tencent.com/pypi/simple --pre 'sglang[diffusion]' comfy-kitchen
  pip install -U -i https://mirrors.cloud.tencent.com/pypi/simple diffusers==0.40.0
"
docker commit sglang-build sglang-h3:sm120-v2 && docker rm -f sglang-build
```

> ⚠️ **关键坑**:基础镜像自带 diffusers 0.37.0 早于 H3 开源(2026-08),不含 `MiniMaxH3ModularPipeline`,必须升级到 ≥0.38(已验证 0.40.0)。sglang 不 pin diffusers 版本,升级安全。

版本快照(已验证 import OK):
`sglang 0.5.19 / torch 2.13.0+cu130 / diffusers 0.40.0 / transformers 5.12.1 / comfy-kitchen 0.2.33`

### 1.3 权重(FL2VA,~144GB)

```bash
# COS 同地域(ap-beijing)内网直连免流量费;断点续传 + 并发下载
export TENCENTCLOUD_SECRET_ID=...  TENCENTCLOUD_SECRET_KEY=...
python3 scripts/download_cos.py --variant FL2VA --workers 8
```

- 桶:`cedricbwang-public-1258272081`(私有)/`MiniMax-H3/`,FL2VA 完整(126 文件口径),Ref2VA 上传未完成
- 权重布局:`/root/h3/weights/MiniMax-H3/{model_index.json, modular_model_index.json, FL2VA/...}`
- 根目录 `model_index.json` 与 `modular_model_index.json` 需从 HuggingFace 补齐(COS 副本缺失):
  `curl -L -o <dest> https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/model_index.json`(需代理)

## 2. 各方案部署命令

### 测试1:INT8 单卡(容器)

```bash
# 路线A:在线量化(需 comfy-kitchen;sm_120 兼容性若失败走路线B)
docker run -d --name h3-int8 --gpus '"device=0"' --network host --shm-size 64g \
  -v /root/h3/weights:/weights sglang-h3:sm120-v2 \
  bash -c "sglang serve --model-path /weights/MiniMax-H3 --model-variant fl2va \
    --num-gpus 1 --quantization kitchen_int8 --attention-backend fa \
    --performance-mode memory --layerwise-offload-components dit,text_encoder \
    --dit-layerwise-resident-layers 20 --enable-torch-compile false --port 30010"

# 路线B:预量化 ConvRot INT8 文件(ModelScope Comfy-Org/MiniMax-H3 下载)
#   --component-weights-paths.transformer <int8_convrot.safetensors>(自描述,不可再叠 --quantization)
#   编码器可配 NVFP4-AWQ(14.6GB,CC≥10)
# 0.3-0.7MP 可先试 --performance-mode speed 全驻留(~52GB 权重),OOM 再降级 memory+offload
```

### 测试2:BF16 4 卡 TP2+Ulysses2(容器)

```bash
docker run -d --name h3-bf16-tp2u2 --gpus '"device=0,1,2,3"' --network host --shm-size 64g \
  -v /root/h3/weights:/weights sglang-h3:sm120-v2 \
  numactl --cpunodebind=0 --membind=0 \
  sglang serve --model-path /weights/MiniMax-H3 --model-variant fl2va \
    --num-gpus 4 --tp-size 2 --ulysses-degree 2 \
    --encoder-parallel auto --performance-mode speed --host 0.0.0.0 --port 30010
```

> 勿用 Ulysses4:官方 4×H200 实测峰值 94.3GB/卡,72GB 卡必 OOM。
> 备选:`--tp-size 4 --ulysses-degree 1`(49.8GB/卡)或 `--use-fsdp-inference true --ulysses-degree 4`(57GB/卡)。

### 测试3:8 卡吞吐候选

| 候选 | 命令差异(其余同测试2) |
|---|---|
| A 基线 | 起两个实例:GPU0-3(NUMA0)+ GPU4-7(NUMA1),端口 30010/30011 |
| B FP8 | 追加 `--quantization fp8 --quantization-ignored-layers blocks.0.attn token_refiner` |
| C FastH3 | `--model-path FastVideo/FastH3-4-step-Preview-v1-VSA-DataFree`(固定 5 步,T2VA only) |
| D 高并行 | 4×2 卡或 8×1 卡 INT8(复用测试1 配方) |
| E 开关 | `--minimax-h3-adaln-cache-*` / `--enable-breakable-cuda-graph true --warmup-resolutions 1344x768` / `--encoder-parallel dp --batching-max-size >1` / `--enable-torch-compile true`(会变数值,仅吞吐) |

## 3. WebUI + Nginx

```bash
# WebUI 容器(--network host 直连本机 SGLang)
docker build -t h3-webui /root/h3/webui
docker run -d --name h3-webui --network host \
  -e SGLANG_ENDPOINT=http://127.0.0.1:30010 \
  -v /root/h3/outputs:/data/outputs h3-webui

# Nginx 鉴权网关
yum install -y nginx
htpasswd -bc /etc/nginx/.htpasswd admin tencentcloud
cp /root/h3/webui/nginx_h3webui.conf /etc/nginx/conf.d/ && systemctl reload nginx
# 访问 http://<host>:8080(仅限内网/受信 IP;公网需 HTTPS+白名单)
```

## 4. 踩坑记录

| # | 坑 | 规避 |
|---|---|---|
| 1 | 基础镜像 diffusers 0.37.0 无 MiniMaxH3 | 升级 diffusers≥0.38(本文 1.2) |
| 2 | Ulysses4 在 72GB 卡 OOM | 用 TP2+U2 / TP4+U1 |
| 3 | 分辨率非 32 对齐 → patchify 崩溃 | 档位映射表固定 32 对齐尺寸 |
| 4 | 残留 http_proxy 劫持 localhost | 脚本 `trust_env=False`;curl 加 `--noproxy '*'` |
| 5 | VAE 放进 offload 列表 → 每 decode tile 重流 ~9GB | offload 仅 dit,text_encoder |
| 6 | 同 seed 重复请求命中缓存 | 基准复核换 seed |
| 7 | 首条视频含 JIT 编译/加载 | warmup 1 次再计时 |
| 8 | `quality:high`(Cache-DiT)仅支持 4×H200 精确负载 | 不作依赖 |
| 9 | torch.compile 改变数值 | 质量基准一律 false |
| 10 | COS 走公网下载收流量费 | 同地域用内网(默认域名自动解析内网) |
