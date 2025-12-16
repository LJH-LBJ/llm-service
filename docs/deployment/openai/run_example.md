# API接口文档

本服务接口遵循 RESTful 设计，提供标准 HTTP 状态码返回。  
数据交互格式为 JSON（除 `/metrics` 及静态文档资源）。

---

## 目录

- [启动命令](#启动命令)
- [参数说明](#参数说明)
- [公开文档与元数据](#公开文档与元数据)
- [核心API](#核心api)
- [健康检查](#健康检查)
- [监控与其它接口](#监控与其它接口)
- [未实现接口](#未实现接口)

---

## 启动命令

```bash

python -m lm_service.entrypoints.openai.epd_api_server \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --proxy-addr /tmp/proxy \
    --port 5580 \
    --host 0.0.0.0 \
    --encode-addr-list  /tmp/encoder_0 \
    --pd-addr-list /tmp/prefill_decode_0 \
    --allowed-local-media-path /workspace/l00807937/EPD_Timecount_v0.11.0/image/

```

支持通过连接Redis作为metastore方式注册和发现

```bash

python -m lm_service.entrypoints.openai.epd_api_server \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --port 5580 \
    --host 0.0.0.0 \
    --metastore-client-config '{
        "metastore_client": "RedisMetastoreClient",
        "metastore_address": "redis://127.0.0.1:6379/0"
        }' \
    --allowed-local-media-path /workspace/l00807937/EPD_Timecount_v0.11.0/image/

```

## 参数说明

本接口为 OpenAI API 兼容启动方式，以下为各参数详细说明：

| 参数                          | 说明                                       | 示例/默认值                                                   |
|-------------------------------|--------------------------------------------|--------------------------------------------------------------|
| `--model`                     | 指定模型文件或目录                         | `/workspace/models/Qwen2.5-VL-7B-Instruct`                   |
| `--proxy-addr`                | Proxy 进程通信地址                         | `/tmp/proxy`                                                 |
| `--port`                      | HTTP 服务监听端口                          | `5580`                                                       |
| `--host`                      | HTTP 服务监听主机（IP）                     | `0.0.0.0`（全网可访问）                                       |
| `--encode-addr-list`          | 编码(encoder)服务地址列表                  | `/tmp/encoder_0`                                             |
| `--pd-addr-list`              | 预填充/解码(pd)服务地址列表                | `/tmp/prefill_decode_0`                                      |
| `--allowed-local-media-path`  | 本地资源访问白名单目录                     | `/workspace/`          |
| `--metastore-client-config`   | 元数据存储配置；格式为 JSON 字符串，指定存储后端与连接参数 | `'{"metastore_client":"RedisMetastoreClient","metastore_address":"redis://127.0.0.1:6379/0"}'` |

> 注意：多个 proxy 需要启动多个 http server

---

## 公开文档与元数据

| 路径                  | 方法   | 功能                  |
| --------------------- | ------ | --------------------- |
| /openapi.json         | GET    | OpenAPI元数据（API描述信息） |
| /docs                 | GET    | Swagger UI可视化API文档 |
| /docs/oauth2-redirect | GET    | Swagger OAuth2重定向页面 |
| /redoc                | GET    | ReDoc可视化API文档    |

`curl http://host:port/openapi.json`

`curl http://host:port/docs`

`curl http://host:port/docs/oauth2-redirect`

`curl http://host:port/redoc`

---

## 核心API

### 1. 聊天接口

- **路径**：`/v1/chat/completions`
- **方法**：POST
- **描述**：基于对话历史生成LLM回复
- **请求体举例**：

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

- **响应体举例**：

    ```json

    {
      "id": "chatcmpl-38c6c4cd17794416afbdd014db2f8408",
      "object": "chat.completion",
      "created": 1765438622,
      "model": "/workspace/models/Qwen2.5-VL-3B-Instruct",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The text in the image translates to \"My work never ends.\""
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 216,
        "completion_tokens": 26,
        "total_tokens": 242
      }
    }

    ```

---

### 2. 传统补全接口

- **路径**：`/v1/completions`
- **方法**：POST
- **描述**：普通文本补全（非对话）。
- **请求体举例**：

    ```json

    curl -X POST http://127.0.0.1:5580/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "/workspace/models/Qwen2.5-VL-7B-Instruct",
        "prompt": "Once upon a time, there was a wise assistant who",
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": false
      }'

    ```

- **响应体举例**：

    ```json

    {
      "id": "cmpl-598be898c76349519d1c0deb4dea1111",
      "object": "text_completion",
      "created": 1765423874,
      "model": "/workspace/models/Qwen2.5-VL-7B-Instruct",
      "choices": [
        {
          "index": 0,
          "text": " could answer any question. At the end of each day, the Helper would review its answers to ensure that it was up to date. This is what I call the **Helper**.\n```\ntype Helper struct {\n    askQuestions []string\n    answerQuestions []string\n}\n```\nHere's an example of creating and using a `Helper` object:\n\n```\nhelper := Helper{\n    askQuestions: []string{\"What is the meaning of life?\", \"Is there life on other planets?\"},\n    answerQuestions: []string{\"The meaning of life is subjective.\", \"We have not yet found any evidence of life elsewhere.\"},\n}\n```"
        }
      ],
      "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 128,
        "total_tokens": 139
      }
    }

    ```

---

## 健康检查

- **路径**：`/check_health`
- **方法**：GET
- **描述**：服务健康状态自检。
- **成功响应**：

    ```json

    {"status": "ok"}

    ```

---

## 监控与其它接口

- **路径**：`/metrics`
- **方法**：GET
- **描述**：返回模型或服务实例各项运行指标。

    **响应示例：**

    ```text

    Server Type: PD_INSTANCE
      Address: ipc:///tmp/prefill_decode_0
        ec_role: PD_INSTANCE
        addr: ipc:///tmp/prefill_decode_0
        Avg e2e time requests: 6212.255 ms
        Avg queue time requests: 0.217 ms
        Avg prefill time requests: 653.453 ms
        Avg mean time per output token requests: 76.416 ms
        Avg time to first token: 4302.684 ms
        Avg proxy ttft: 10151.510 ms
        Avg proxy to instance requests time: 1.393 ms
    Server Type: E_INSTANCE
      Address: ipc:///tmp/encoder_0
        ec_role: E_INSTANCE
        addr: ipc:///tmp/encoder_0
        Avg e2e time requests: 3926.159 ms
        Avg queue time requests: 0.219 ms
        Avg prefill time requests: 3.850 ms
        Avg mean time per output token requests: 0.000 ms
        Avg time to first token: 0.000 ms
        Avg proxy ttft: 0.000 ms
        Avg proxy to instance requests time: 4.444 ms

    ```

---

## 未实现接口

- **路径**：`/abort`

- **方法**：POST

- **状态**：*未实现*

- **描述**：接口定义但尚未提供具体实现，调用将返回默认或为空。

- **路径**：`/start_profile`

- **方法**：POST

- **状态**：*未实现*

- **描述**：接口定义但尚未提供具体实现，调用将返回默认或为空。

- **路径**：`/stop_profile`

- **方法**：POST

- **状态**：*未实现*

- **描述**：接口定义但尚未提供具体实现，调用将返回默认或为空。

---
