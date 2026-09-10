FROM python:3.11-slim

WORKDIR /app

# 腾讯云 pypi 镜像(容器构建走内网/直连,不走代理)
RUN pip install --no-cache-dir gradio requests -i https://mirrors.cloud.tencent.com/pypi/simple

COPY app.py /app/app.py
# 基准 prompt 文件供默认值读取
COPY prompts/ballet_t2va.txt /app/prompts/ballet_t2va.txt

ENV SGLANG_ENDPOINT=http://127.0.0.1:30010 \
    WEBUI_OUT_DIR=/outputs/webui \
    WEBUI_PROMPT_FILE=/app/prompts/ballet_t2va.txt \
    no_proxy=127.0.0.1,localhost

# 运行时须与 SGLang 容器使用相同挂载点(-v /root/h3/outputs:/outputs),
# 否则上传的首帧图对 SGLang 容器不可见(跨容器路径一致性)

# --network host 模式运行,直接访问宿主机 SGLang 与输出目录
EXPOSE 7860
CMD ["python", "app.py"]
