#!/usr/bin/env python3
"""从 COS 下载 MiniMax-H3 FL2VA 权重(用户恢复下载任务后执行)

桶: cedricbwang-public-1258272081(ap-beijing,与本机同地域走内网免流量费)
前缀: MiniMax-H3/FL2VA/  → 保存到 /root/h3/weights/MiniMax-H3/FL2VA/
特性: 并发下载、断点续传(按文件大小校验跳过已完成文件)、完成后输出核对清单

用法:
  python3 download_cos.py                # 默认只下 FL2VA(基准测试只需它)
  python3 download_cos.py --variant Ref2VA   # 等 COS 传完后可下 Ref2VA
  python3 download_cos.py --workers 16
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from qcloud_cos import CosConfig, CosS3Client

BUCKET = "cedricbwang-public-1258272081"
REGION = "ap-beijing"
PREFIX = "MiniMax-H3/"
DEST_ROOT = "/root/h3/weights/MiniMax-H3"

# AK/SK 从环境变量读取(勿硬编码;值见 /root/h3/p.md)
SID = os.environ.get("TENCENTCLOUD_SECRET_ID")
SKEY = os.environ.get("TENCENTCLOUD_SECRET_KEY")


def get_client():
    if not SID or not SKEY:
        sys.exit("请先 export TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY(值见 /root/h3/p.md)")
    # 同地域 CVM 默认域名自动解析内网(已验证 169.254.0.49),免流量费
    return CosS3Client(CosConfig(Region=REGION, SecretId=SID, SecretKey=SKEY))


def list_all(client, prefix):
    objs, marker = [], ""
    while True:
        resp = client.list_objects(Bucket=BUCKET, Prefix=prefix, Marker=marker, MaxKeys=1000)
        objs += resp.get("Contents", [])
        if resp.get("IsTruncated") == "true":
            marker = resp["NextMarker"]
        else:
            return objs


def download_one(client, key, size, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return ("skip", key, size)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    client.download_file(Bucket=BUCKET, Key=key, DestFilePath=tmp,
                         MAXThread=5, PartSize=20)
    os.replace(tmp, dest)
    if os.path.getsize(dest) != size:
        return ("size_mismatch", key, size)
    return ("ok", key, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="FL2VA", choices=["FL2VA", "Ref2VA"],
                    help="默认 FL2VA(基准测试只需);Ref2VA 在 COS 传完后可下")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = get_client()
    prefix = f"{PREFIX}{args.variant}/"
    objs = list_all(client, prefix)
    total = sum(int(o["Size"]) for o in objs)
    print(f"{prefix}: {len(objs)} 个文件, 共 {total/1e9:.1f} GB")
    if not objs:
        sys.exit("COS 上该变体还没有文件(Ref2VA 尚未上传完成)")

    dest_root = DEST_ROOT  # key 已含变体子目录(FL2VA/xxx),不再拼接变体名
    t0 = time.time()
    stats = {"ok": 0, "skip": 0, "size_mismatch": 0}
    done_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, client, o["Key"], int(o["Size"]),
                          os.path.join(dest_root, o["Key"][len(PREFIX):])): o for o in objs}
        for fut in as_completed(futs):
            status, key, size = fut.result()
            stats[status] += 1
            done_bytes += size
            mb = size / 1e6
            print(f"[{status}] {key} ({mb:.1f}MB)  进度 {done_bytes/1e9:.1f}/{total/1e9:.1f}GB", flush=True)

    dt = time.time() - t0
    print(f"\n完成: 新下 {stats['ok']} / 跳过(已存在) {stats['skip']} / 大小异常 {stats['size_mismatch']}, "
          f"耗时 {dt/60:.1f} 分钟, 均速 {done_bytes/1e6/max(dt,1):.0f} MB/s")
    # 核对清单
    manifest = os.path.join("/root/h3/results", f"cos_download_{args.variant}_manifest.txt")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w") as f:
        for o in sorted(objs, key=lambda x: x["Key"]):
            f.write(f"{o['Size']}\t{o['Key']}\n")
    print(f"清单已写入 {manifest}")


if __name__ == "__main__":
    main()
