#!/usr/bin/env python3
"""MiniMax-H3 WebUI(单文件 Gradio 应用,规范见 /root/h3/p.md §7)

功能:prompt 输入、分辨率档(0.3/0.5/0.7MP/768P)、时长、seed、steps,
提交本地 SGLang(/v1/videos)→ 轮询 → 下载 → 在线播放(含原生音频)。

部署:容器内运行,默认监听 127.0.0.1:7860,由宿主机 Nginx 反代 + Basic Auth 对外。
环境变量:
  SGLANG_ENDPOINT  SGLang 服务地址(默认 http://127.0.0.1:30010)
  WEBUI_OUT_DIR    输出目录(默认 /root/h3/outputs/webui)
"""
import os
import shutil
import time
import uuid

import gradio as gr
import requests

SGLANG_ENDPOINT = os.environ.get("SGLANG_ENDPOINT", "http://127.0.0.1:30010").rstrip("/")
OUT_DIR = os.environ.get("WEBUI_OUT_DIR", "/root/h3/outputs/webui")
PROMPT_FILE = os.environ.get("WEBUI_PROMPT_FILE", "/root/h3/scripts/prompts/ballet_t2va.txt")

TIERS = {"0.3MP (736×416)": 416, "0.5MP (960×544)": 544,
         "0.7MP (1152×640)": 640, "768P (1344×768)": 768}
REF_IMAGE = "/root/h3/de930b24-b17b-4973-9c14-d62a1f9618c7.jpg"

os.makedirs(OUT_DIR, exist_ok=True)

# 绕开 http_proxy(残留代理会劫持 localhost 请求)
SESSION = requests.Session()
SESSION.trust_env = False


def _default_prompt():
    try:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def generate(prompt, tier_label, duration, seed, steps, task, ref_image):
    if not prompt or not prompt.strip():
        raise gr.Error("prompt 不能为空")
    short_edge = TIERS[tier_label]
    duration = float(duration)
    steps = int(steps)
    seed = int(seed)

    if task.startswith("图生视频"):
        # 上传文件在 WebUI 容器 /tmp 下,SGLang 容器不可见 → 拷到共享挂载目录
        src = getattr(ref_image, "name", None) or ref_image
        if not src or not os.path.isfile(src):
            raise gr.Error("图生视频需要上传首帧图")
        uploads = os.path.join(OUT_DIR, "uploads")
        os.makedirs(uploads, exist_ok=True)
        shared = os.path.join(uploads, f"ref_{uuid.uuid4().hex[:8]}" + os.path.splitext(src)[1])
        shutil.copy(src, shared)
        payload = {
            "model": "MiniMaxAI/MiniMax-H3",
            "prompt": prompt,
            "seconds": int(duration),
            "task": "fl2va",
            "conditions": [{"type": "image", "uri": "file://" + shared,
                            "role": "keyframe", "frame_index": 0}],
            "target": {"short_edge": short_edge, "aspect_ratio": "auto",
                       "duration_seconds": duration},
            "quality": "lossless", "num_outputs_per_prompt": 1,
            "num_inference_steps": steps, "flow_shift": 12.0,
            "audio_flow_shift": 3.0, "seed": seed,
        }
    else:
        payload = {
            "model": "MiniMaxAI/MiniMax-H3",
            "prompt": prompt,
            "seconds": int(duration),
            "task": "t2va",
            "conditions": [],
            "target": {"short_edge": short_edge, "aspect_ratio": "16:9",
                       "duration_seconds": duration},
            "quality": "lossless", "num_outputs_per_prompt": 1,
            "num_inference_steps": steps, "flow_shift": 12.0,
            "audio_flow_shift": 3.0, "seed": seed,
        }

    t0 = time.time()
    try:
        r = SESSION.post(f"{SGLANG_ENDPOINT}/v1/videos", json=payload, timeout=120)
        if r.status_code >= 400:
            raise gr.Error(f"提交失败 {r.status_code}: {r.text[:300]}")
        vid = r.json().get("id") or r.json().get("task_id")
        while True:
            st = SESSION.get(f"{SGLANG_ENDPOINT}/v1/videos/{vid}", timeout=60).json()
            status = st.get("status", "")
            if status in ("completed", "succeeded"):
                break
            if status in ("failed", "cancelled"):
                raise gr.Error(f"生成失败: {str(st)[:300]}")
            time.sleep(2)
        out_path = os.path.join(OUT_DIR, f"{task[:4]}_{tier_label.split()[0]}_{uuid.uuid4().hex[:8]}.mp4")
        with SESSION.get(f"{SGLANG_ENDPOINT}/v1/videos/{vid}/content",
                         timeout=600, stream=True) as r:
            if r.status_code >= 400:
                raise gr.Error(f"下载失败 {r.status_code}")
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except requests.ConnectionError:
        raise gr.Error(f"无法连接 SGLang 服务({SGLANG_ENDPOINT}),请确认服务已启动")
    elapsed = time.time() - t0
    return out_path, f"完成:耗时 {elapsed:.1f}s,文件 {os.path.basename(out_path)}"


def build_ui():
    with gr.Blocks(title="MiniMax-H3 WebUI") as demo:
        gr.Markdown(f"# MiniMax-H3 视频生成\n后端:`{SGLANG_ENDPOINT}`(SGLang /v1/videos)")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=10, value=_default_prompt())
                task = gr.Radio(["文生视频 (t2va)", "图生视频 (fl2va 首帧)"],
                                value="文生视频 (t2va)", label="任务")
                ref_image = gr.File(label="首帧图(图生视频时必传)", file_types=["image"])
                with gr.Row():
                    tier = gr.Dropdown(list(TIERS.keys()), value="768P (1344×768)", label="分辨率")
                    duration = gr.Slider(4, 15, 10, step=1, label="时长(秒)")
                with gr.Row():
                    seed = gr.Number(1101, label="seed", precision=0)
                    steps = gr.Number(50, label="steps", precision=0)
                btn = gr.Button("生成", variant="primary")
            with gr.Column(scale=1):
                video = gr.Video(label="结果(含音频)")
                status = gr.Markdown("就绪")
        btn.click(generate, [prompt, tier, duration, seed, steps, task, ref_image],
                  [video, status])
    return demo


if __name__ == "__main__":
    # allowed_paths:允许返回共享输出目录中的视频文件(否则 Gradio 报 InvalidPathError)
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True,
                      allowed_paths=[OUT_DIR])
