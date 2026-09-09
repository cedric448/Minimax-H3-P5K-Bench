# 使用手册

## 1. API 调用(SGLang 本地服务)

本地 SGLang 暴露 OpenAI 风格异步任务接口,与官方 `/v2/video_generation`(`resolution` 语法)**不兼容**。

### 流程

```
POST /v1/videos            提交生成任务 → 返回 id
GET  /v1/videos/{id}       轮询状态(pending/running/completed/failed)
GET  /v1/videos/{id}/content   下载 MP4(含原生音频)
```

### 文生视频(t2va)

```bash
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "<结构化 prompt,见 assets/ballet_t2va.txt>",
    "seconds": 10,
    "task": "t2va",
    "conditions": [],
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0},
    "quality": "lossless",
    "num_outputs_per_prompt": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3,
    "seed": 1101
  }'
```

### 图生视频(fl2va,首帧)

```bash
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "<prompt>",
    "seconds": 10,
    "task": "fl2va",
    "conditions": [{
      "type": "image",
      "uri": "file:///root/h3/de930b24-b17b-4973-9c14-d62a1f9618c7.jpg",
      "role": "keyframe",
      "frame_index": 0
    }],
    "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 10.0},
    "num_inference_steps": 50, "flow_shift": 12, "audio_flow_shift": 3, "seed": 1101
  }'
```

### 轮询与下载

```bash
while true; do
  status=$(curl -sS http://127.0.0.1:30010/v1/videos/${ID} | jq -r .status)
  [ "$status" = completed ] && break
  [ "$status" = failed ] && exit 1
  sleep 2
done
curl -sSL http://127.0.0.1:30010/v1/videos/${ID}/content -o out.mp4
```

## 2. 参数表

| 参数 | 取值/默认 | 说明 |
|---|---|---|
| task | `t2va` / `fl2va` | 文生视频 / 首尾帧图生视频(Ref2VA 变体另有 `ref2va`,本部署未含) |
| target.short_edge | 416/544/640/768 | 对应 0.3/0.5/0.7MP/768P 档(16:9),宽高自动 32 对齐 |
| target.aspect_ratio | 16:9 等 / auto | t2va 必须显式;i2va 用 auto 随输入图 |
| seconds / duration_seconds | 4-15 | 帧数自动对齐(5s→124f,10s→243f) |
| num_inference_steps | 默认 50 | FastH3 蒸馏版固定 5 |
| flow_shift / audio_flow_shift | 12 / 3 | 官方默认 |
| seed | 整数 | 固定 seed 可复现;注意同 seed 重跑可能命中结果缓存 |
| quality | lossless(默认)/ high | high(Cache-DiT)仅支持 4×H200 精确负载 |
| conditions | image 条目 | fl2va:frame_index 0(首帧)/ -1(尾帧),可两者 |

## 3. WebUI 使用

- 访问 `http://<内网IP>:8080`,Basic Auth:用户 `admin`,密码 `tencentcloud`
- 选择任务(文生/图生)、分辨率档、时长、seed、steps → 生成 → 页内播放(含音频)并可下载
- 图生视频需上传首帧图(参考图 `assets/de930b24-b17b-4973-9c14-d62a1f9618c7.jpg`)

## 4. 基准脚本

```bash
# t2va 全矩阵(4 档 × warmup1 + 正式3)
python3 scripts/bench.py --endpoint http://127.0.0.1:30010 --tag <方案名> --task t2va
# 吞吐模式(并发 worker 连续提交)
python3 scripts/bench.py --endpoint http://127.0.0.1:30010 --tag <方案名> --task t2va \
  --tiers 768P --mode throughput --concurrency 4 --runs 8
# 结果:results/<tag>_<task>.jsonl(E2E/queue/gen/download 分解 + 峰值显存/功耗)
```

## 5. FAQ

**Q: 为什么没有 2K?** H3-Regenerate-2K 未开源,本地封顶 768P。需 2K:本地 768P 产物调官方 Regeneration API($0.05/s)。

**Q: 提示词怎么写?** 推荐官方结构化三段式(`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`,样例见 assets/);官方 Context-IR API 可自动生成该结构(输入 $0.90/M tok、输出 $3.60/M tok)。

**Q: 0.3/0.5/0.7MP 是官方档位吗?** 不是。官方 API 仅 480P/768P/2K;MP 档为本地推理的像素量档位(ComfyUI 惯例),仅 768P 有官方画质配方。

**Q: 中文语音不稳?** 社区已知问题,建议音频参考或多次生成;基准中中文语音 case 单列。

**Q: INT8 会更快吗?** 不一定。量化节省显存/权重带宽;提速主要来自 cu130(硬件反量化)、更低步数、更低分辨率。单卡 INT8 定位是容量与精度验证。
