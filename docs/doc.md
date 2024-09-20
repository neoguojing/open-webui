# 执行流程

## ollama流程
- ChatCompletionMiddleware
- - chat_completion_filter_functions_handler ： 数据库检索函数包，并执行相关函数
- - chat_completion_tools_handler： 检索和组装 tool的system prompt，通过llm组织函数调用格式，调用工具函数
- - - generate_chat_completions： 使用llm，通过prompt让llm返回json格式的函数调用请求
- - - - generate_ollama_chat_completion: 组装请求，调用ollama
- - chat_completion_files_handler：通过查询知识库返回上下文
- - - get_rag_context：
- - - - query_collection_with_hybrid_search 或 query_collection： 执行知识库查询
