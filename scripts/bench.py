#!/usr/bin/env python3
"""MiniMax-H3 本地 SGLang 基准测试脚本(规范见 /root/h3/p.md §5)

用法示例:
  # t2va 全矩阵(4 档分辨率 × warmup + 3 次):
  python3 bench.py --endpoint http://127.0.0.1:30010 --tag test2_tp2u2 --task t2va
  # 单档:
  python3 bench.py --endpoint http://127.0.0.1:30010 --tag test1_int8 --task t2va --tiers 768P
  # fl2va 图生视频验证(768P × 1 次):
  python3 bench.py --endpoint http://127.0.0.1:30010 --tag test2_tp2u2 --task fl2va --runs 1 --warmup 0
  # 吞吐模式(并发 4 worker × 每个跑 2 条,测稳态吞吐):
  python3 bench.py --endpoint http://127.0.0.1:30010 --tag test3_A --task t2va --tiers 768P --mode throughput --concurrency 4 --runs 8

注意:脚本内置绕开 http_proxy(session.trust_env=False),本机服务直连。
"""
import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import requests

# ---- 基准常量(与 p.md §5 对齐) ----
TIERS = {"0.3MP": 416, "0.5MP": 544, "0.7MP": 640, "768P": 768}
DEFAULT_SEED = 1101
DEFAULT_STEPS = 50
FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0
DEFAULT_DURATION = 10.0
BASE = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE, "prompts", "ballet_t2va.txt")
REF_IMAGE = "/root/h3/de930b24-b17b-4973-9c14-d62a1f9618c7.jpg"
RESULTS_DIR = "/root/h3/results"
OUTPUTS_DIR = "/root/h3/outputs"

POLL_INTERVAL = 2.0


def load_prompt():
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read().strip()


class VramSampler:
    """后台线程按间隔采样 nvidia-smi 显存/功耗。"""

    def __init__(self, interval=1.0):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.interval = interval

    def _probe(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        ts = time.time()
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3:
                idx, mem, pwr = parts
                try:
                    self.samples.append((ts, int(idx), int(mem), float(pwr)))
                except ValueError:
                    pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._probe()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def peak(self):
        """返回 {gpu_idx: peak_mem_mb} 与 {gpu_idx: peak_power_w}"""
        mem, pwr = {}, {}
        for _, idx, m, p in self.samples:
            mem[idx] = max(mem.get(idx, 0), m)
            pwr[idx] = max(pwr.get(idx, 0), p)
        return mem, pwr


class SglangClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint.rstrip("/")
        self.s = requests.Session()
        self.s.trust_env = False  # 绕开代理,本机/内网直连

    def submit(self, payload):
        r = self.s.post(f"{self.endpoint}/v1/videos", json=payload, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"submit failed {r.status_code}: {r.text[:500]}")
        data = r.json()
        return data.get("id") or data.get("task_id")

    def poll(self, vid):
        r = self.s.get(f"{self.endpoint}/v1/videos/{vid}", timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"poll failed {r.status_code}: {r.text[:500]}")
        return r.json()

    def download(self, vid, path):
        r = self.s.get(f"{self.endpoint}/v1/videos/{vid}/content", timeout=600, stream=True)
        if r.status_code >= 400:
            raise RuntimeError(f"download failed {r.status_code}: {r.text[:500]}")
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    def wait(self, vid, t_start):
        """轮询到完成;返回 (完成时间, 状态, 首个running时间或None, 轮询次数)"""
        first_running = None
        n = 0
        while True:
            n += 1
            st = self.poll(vid)
            status = st.get("status", "")
            if status in ("completed", "succeeded"):
                return time.time(), status, first_running, n
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"task {vid} {status}: {json.dumps(st)[:500]}")
            if status in ("running", "in_progress", "processing") and first_running is None:
                first_running = time.time()
            time.sleep(POLL_INTERVAL)


def build_payload(task, short_edge, duration, seed, steps, prompt, ref_image, ratio=None):
    if task == "t2va":
        return {
            "model": "MiniMaxAI/MiniMax-H3",
            "prompt": prompt,
            "seconds": int(duration),
            "task": "t2va",
            "conditions": [],
            "target": {"short_edge": short_edge, "aspect_ratio": ratio or "16:9",
                       "duration_seconds": duration},
            "quality": "lossless",
            "num_outputs_per_prompt": 1,
            "num_inference_steps": steps,
            "flow_shift": FLOW_SHIFT,
            "audio_flow_shift": AUDIO_FLOW_SHIFT,
            "seed": seed,
        }
    elif task == "fl2va":
        return {
            "model": "MiniMaxAI/MiniMax-H3",
            "prompt": prompt,
            "seconds": int(duration),
            "task": "fl2va",
            "conditions": [{
                "type": "image",
                "uri": "file://" + ref_image,
                "role": "keyframe",
                "frame_index": 0,
            }],
            "target": {"short_edge": short_edge, "aspect_ratio": ratio or "auto",
                       "duration_seconds": duration},
            "quality": "lossless",
            "num_outputs_per_prompt": 1,
            "num_inference_steps": steps,
            "flow_shift": FLOW_SHIFT,
            "audio_flow_shift": AUDIO_FLOW_SHIFT,
            "seed": seed,
        }
    raise ValueError(f"unknown task {task}")


def run_single(client, task, tier, short_edge, duration, seed, steps, prompt,
               ref_image, tag, run_idx, warmup):
    payload = build_payload(task, short_edge, duration, seed, steps, prompt, ref_image)
    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    try:
        vid = client.submit(payload)
        t_submit = time.time()
        t_done, status, t_running, n_polls = client.wait(vid, t0)
        out = os.path.join(OUTPUTS_DIR, f"{tag}_{task}_{tier}_run{run_idx}_{vid}.mp4")
        client.download(vid, out)
        t_dl = time.time()
        size = os.path.getsize(out)
    finally:
        sampler.stop()
    peak_mem, peak_pwr = sampler.peak()
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag, "task": task, "tier": tier, "short_edge": short_edge,
        "duration_s": duration, "seed": seed, "steps": steps,
        "warmup": warmup,
        "video_id": str(vid),
        "e2e_s": round(t_dl - t0, 2),
        "submit_s": round(t_submit - t0, 2),
        "queue_s": round((t_running - t_submit) if t_running else 0.0, 2),
        "gen_s": round((t_done - t_running) if t_running else (t_done - t_submit), 2),
        "download_s": round(t_dl - t_done, 2),
        "polls": n_polls,
        "output": out, "bytes": size,
        "peak_vram_mb_per_gpu": peak_mem,
        "peak_power_w_per_gpu": peak_pwr,
    }


def run_throughput(client, task, tier, short_edge, duration, seed, steps, prompt,
                   ref_image, tag, concurrency, total_runs):
    """并发 worker 连续提交,测稳态吞吐(视频秒/小时)。"""
    done_records = []
    lock = threading.Lock()
    counter = {"i": 0}

    def worker(wid):
        client_w = SglangClient(client.endpoint)
        while True:
            with lock:
                if counter["i"] >= total_runs:
                    return
                counter["i"] += 1
                idx = counter["i"]
            rec = run_single(client_w, task, tier, short_edge, duration, seed, steps,
                             prompt, ref_image, f"{tag}_w{wid}", idx, warmup=False)
            with lock:
                done_records.append(rec)

    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    sampler.stop()
    peak_mem, peak_pwr = sampler.peak()
    total_video_s = duration * len(done_records)
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag, "task": task, "tier": tier, "short_edge": short_edge,
        "duration_s": duration, "seed": seed, "steps": steps,
        "mode": "throughput", "concurrency": concurrency,
        "completed": len(done_records),
        "wall_s": round(wall, 2),
        "video_seconds_total": total_video_s,
        "throughput_video_s_per_hour": round(total_video_s / wall * 3600, 1),
        "mean_e2e_s": round(sum(r["e2e_s"] for r in done_records) / len(done_records), 2) if done_records else None,
        "peak_vram_mb_per_gpu": peak_mem,
        "peak_power_w_per_gpu": peak_pwr,
        "runs": done_records,
    }


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:30010")
    ap.add_argument("--tag", required=True, help="方案标识,如 test1_int8 / test2_tp2u2")
    ap.add_argument("--task", choices=["t2va", "fl2va"], default="t2va")
    ap.add_argument("--tiers", default=None,
                    help="逗号分隔:0.3MP,0.5MP,0.7MP,768P;t2va 默认全部,fl2va 默认 768P")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--runs", type=int, default=3, help="每档正式计时次数")
    ap.add_argument("--warmup", type=int, default=1, help="每档 warmup 次数(不计入统计)")
    ap.add_argument("--mode", choices=["latency", "throughput"], default="latency")
    ap.add_argument("--concurrency", type=int, default=4, help="吞吐模式并发 worker 数")
    ap.add_argument("--ref-image", default=REF_IMAGE)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    prompt = load_prompt()
    if args.tiers:
        tiers = [t.strip() for t in args.tiers.split(",")]
    else:
        tiers = list(TIERS.keys()) if args.task == "t2va" else ["768P"]
    for t in tiers:
        assert t in TIERS, f"未知档位 {t},可选 {list(TIERS)}"

    client = SglangClient(args.endpoint)
    jsonl = os.path.join(RESULTS_DIR, f"{args.tag}_{args.task}.jsonl")

    for tier in tiers:
        short_edge = TIERS[tier]
        if args.mode == "throughput":
            rec = run_throughput(client, args.task, tier, short_edge, args.duration,
                                 args.seed, args.steps, prompt, args.ref_image,
                                 args.tag, args.concurrency, args.runs)
            append_jsonl(jsonl, rec)
            print(f"[{tier}] 吞吐: {rec['throughput_video_s_per_hour']} 视频秒/小时 "
                  f"(并发{args.concurrency} × {rec['completed']}条, wall {rec['wall_s']}s, "
                  f"峰值显存/卡 {max(rec['peak_vram_mb_per_gpu'].values()) if rec['peak_vram_mb_per_gpu'] else 0}MB)")
            continue
        # latency 模式:warmup + runs
        records = []
        for i in range(args.warmup):
            rec = run_single(client, args.task, tier, short_edge, args.duration,
                             args.seed, args.steps, prompt, args.ref_image,
                             args.tag, i, warmup=True)
            append_jsonl(jsonl, rec)
            print(f"[{tier}] warmup#{i}: e2e={rec['e2e_s']}s (不计入)")
        for i in range(args.runs):
            rec = run_single(client, args.task, tier, short_edge, args.duration,
                             args.seed, args.steps, prompt, args.ref_image,
                             args.tag, i, warmup=False)
            append_jsonl(jsonl, rec)
            records.append(rec)
            print(f"[{tier}] run#{i}: e2e={rec['e2e_s']}s "
                  f"(queue={rec['queue_s']}s gen={rec['gen_s']}s dl={rec['download_s']}s "
                  f"peakVRAM={max(rec['peak_vram_mb_per_gpu'].values()) if rec['peak_vram_mb_per_gpu'] else 0}MB)")
        if records:
            mean = sum(r["e2e_s"] for r in records) / len(records)
            mn = min(r["e2e_s"] for r in records)
            mx = max(r["e2e_s"] for r in records)
            peak = max(max(r["peak_vram_mb_per_gpu"].values()) for r in records if r["peak_vram_mb_per_gpu"])
            print(f"== [{tier}] {args.task} n={len(records)}: e2e mean={mean:.1f}s "
                  f"min={mn:.1f}s max={mx:.1f}s 峰值显存/卡={peak}MB → "
                  f"单请求折算吞吐 {args.duration/mean*3600:.0f} 视频秒/小时")


if __name__ == "__main__":
    main()
