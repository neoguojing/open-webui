# TODO

## 任务1
- 使用agi服务直接替换openai，要研究兼容性，参见聊天的主要入口；争议：没人会在这个平台直接使用openai，可以改动尽量避免
- 适配语音输入和输出: 需调试
- 适配图片: DONE
- 文件输入
- 适配图片输出: DONE
- web检索: 引用需要调整
- 知识库检索： test
- 推理

## 聊天适配
- 通用流程
- /api/chat/completions
- 请求
```
{
	"stream": true,
	"model": "agi",
	"messages": [{
		"role": "user",
		"content": "hello"
	}, {
		"role": "assistant",
		"content": "你好，有什么可以帮助你的吗？如果你有关于科学文章、股票信息或者图像生成的问题，随时告诉我。"
	}, {
		"role": "user",
		"content": "你的名字"
	}],
	"params": {},
	"features": {
		"image_generation": false,
		"code_interpreter": false,
		"web_search": false
	},
	"variables": {
		"{{USER_NAME}}": "neo",
		"{{USER_LOCATION}}": "Unknown",
		"{{CURRENT_DATETIME}}": "2025-03-09 19:24:42",
		"{{CURRENT_DATE}}": "2025-03-09",
		"{{CURRENT_TIME}}": "19:24:42",
		"{{CURRENT_WEEKDAY}}": "Sunday",
		"{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
		"{{USER_LANGUAGE}}": "zh-CN"
	},
	"model_item": {
		"id": "agi",
		"object": "model",
		"created": 1677654321,
		"owned_by": "openai",
		"name": "agi",
		"openai": {
			"id": "agi",
			"object": "model",
			"created": 1677654321,
			"owned_by": "neo"
		},
		"urlIdx": 0,
		"actions": []
	},
	"session_id": "DTMcwB0v5KkvcVkIAAAB",
	"chat_id": "6d57d18f-be8a-4430-85ae-5a882bfb8852",
	"id": "41f5fdfd-d0aa-4d24-8925-fb61bdede70a"
}
返回
{"status":true,"task_id":"e2e89373-022b-4344-8dbc-e045633e0332"}
```
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
- knowledge 格式 
- {
    "name": item.get("name"),
    "type": "collection",
    "collection_names": item.get("collection_names"),
    "legacy": True,
}
{
	"id": item.get("collection_name"),
	"name": item.get("name"),
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
- 流程：
- - 利用大模型自动补全问题
- - 分析历史数据： web
- - 符合检索结果 list
```
{
		'description': '俄乌战争是21世纪以来影响最为深远的国际冲突之一。文章从历史背景、战争的直接影响、国际社会的反应及全球影响等方面展开论述，深刻剖析了俄乌战争在政治、经济、安全、能源、粮食等领域的连锁效应。',
		'embedding_config': '{"engine": "ollama", "model": "bge-m3:latest"}',
		'language': 'zh-CN',
		'source': 'https://news.qq.com/rain/a/20250108A01LEN00',
		'start_index': 3,
		'title': '俄乌战争的根源、现状与影响：一场改变世界格局的冲突_腾讯新闻',
		'score': 0.6120820045471191
	}
```
- - 结合上下文，使用system格式的消息请求llm
## 推理处理
- 调试参数

## openweb ui
## 消息事件处理类
- chatCompletionEventHandler

## 聊天消息处理
- chatEventHandler
- chatCompletedHandler
- 此处讲后端消息返回，并转换为MessageType
- TODO: 依据不同的对象填充不同类型的消息，特别是图片和音频消息
- TODO： python后端是否对agi的消息做了额外的处理
### 端消息类型 需要适配
```
interface MessageType {
		id: string;
		model: string;
		content: string;
		files?: { type: string; url: string }[];
		timestamp: number;
		role: string;
		statusHistory?: {
			done: boolean;
			action: string;
			description: string;
			urls?: string[];
			query?: string;
		}[];
		status?: {
			done: boolean;
			action: string;
			description: string;
			urls?: string[];
			query?: string;
		};
		done: boolean;
		error?: boolean | { content: string };
		sources?: string[];
		code_executions?: {
			uuid: string;
			name: string;
			code: string;
			language?: string;
			result?: {
				error?: string;
				output?: string;
				files?: { name: string; url: string }[];
			};
		}[];
		info?: {
			openai?: boolean;
			prompt_tokens?: number;
			completion_tokens?: number;
			total_tokens?: number;
			eval_count?: number;
			eval_duration?: number;
			prompt_eval_count?: number;
			prompt_eval_duration?: number;
			total_duration?: number;
			load_duration?: number;
			usage?: unknown;
		};
		annotation?: { type: string; rating: number };
	}
```