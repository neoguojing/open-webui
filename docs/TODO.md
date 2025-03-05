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

## web检索处理
- 调试参数

## 推理处理
- 调试参数
