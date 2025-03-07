# TODO

## 任务1
- 使用agi服务直接替换openai，要研究兼容性，参见聊天的主要入口；争议：没人会在这个平台直接使用openai，可以改动尽量避免
- 适配语音输入
- 适配图片和文件输入
- 适配语音和图片输出
- web检索
- 知识库检索
- 推理

## 聊天适配
- 通用流程
- /api/chat/completions
- 统一调用chat.generate_chat_completion == chat_completion_handler 处理请求
- chat.generate_chat_completion：会处理openai 和 ollama请求 ；请求以openai 的api格式为准
- 以上分别调用openai.generate_chat_completion 或 ollama.generate_chat_completion 完成实际请求
- agi实际openai接口，完成相关工作

## 语音输入处理
- 调研语音输入的端口

## 图片和文件输入处理
- 调研文件上传端口和输入

## 知识库检索处理
- 调试参数
- 相关入口
- process_chat_payload
- chat_completion_files_handler
- generate_queries
- 流程： 获取model信息，在处理payload的时候，检测到模型关联了知识库，则收集知识库信息，线请求关联知识，然后拼接请求，然后实现聊天
- 数据格式：
- model.get("info", {}).get("meta", {}).get("knowledge", False)
- knowledge 格式 {
    "name": item.get("name"),
    "type": "collection",
    "collection_names": item.get("collection_names"),
    "legacy": True,
}
- form_data["files"]
- form_data["metadata"]["files"]

## web检索处理
- 调试参数
- process_chat_payload
- chat_completion_files_handler
- 数据格式：
- form_data["features"]["web_search"]
- form_data["type"] = "web_search"
- openai 请求参数："metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": "query_generation",
            "task_body": form_data,
            "chat_id": form_data.get("chat_id", None),
        }

## 推理处理
- 调试参数
