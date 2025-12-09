# 多个proxy需要启动多个http server

## 非radix启动命令

```bash
VLLM_USE_V1=1 python -m lm_service.apis.vllm.epd_api_server \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --proxy-addr /tmp/proxy \
    --port 5580 \
    --host 0.0.0.0 \
    --encode-addr-list  /tmp/encoder_0 \
    --pd-addr-list /tmp/prefill_decode_0 \
    --allowed-local-media-path /workspace/l00807937/EPD_Timecount_v0.11.0/image/
```

## radix启动命令

```bash
VLLM_USE_V1=1 python -m lm_service.apis.vllm.epd_api_server \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --port 5580 \
    --host 0.0.0.0 \
    --metastore-client-config '{
        "metastore_client": "RedisMetastoreClient",
        "metastore_address": "redis://127.0.0.1:6379/0"
        }' \
    --allowed-local-media-path /workspace/l00807937/EPD_Timecount_v0.11.0/image/
```

## 发送请求

```bash
curl -X POST  http://127.0.0.1:5580/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "/workspace/models/Qwen2.5-VL-7B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///workspace/l00807937/EPD_Timecount_v0.11.0/image/work.jpg"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```
