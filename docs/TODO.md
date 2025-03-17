# TODO

## 任务1
- 创建agi分支，而不是借用openai的分支
- 使用agi服务直接替换openai，要研究兼容性，参见聊天的主要入口；争议：没人会在这个平台直接使用openai，可以改动尽量避免
- 适配语音输入和输出: 需调试
- 适配图片: DONE
- 文件输入
- 适配图片输出: DONE
- web检索: 引用需要调整 Done
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
	"files": [
		{
			"type": "file",
			"file": {
			"id": "370664f8-f179-473d-9a0c-b3f07fc0349f",
			"user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
			"hash": "3e11c6cbdad114f4807614b25e6e94a83378905f091130c277c68178e0e327df",
			"filename": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
			"data": {
				"content": "电子发票(普通发票) 发票号码:\n开票日期:\n购\n买\n方\n信\n息统一社会信用代码 /纳税人识别号:销\n售\n方\n信\n息统一社会信用代码 /纳税人识别号:名称: 名称:\n项目名称 规格型号 单 位数 量单 价金 额税率/征收率 税 额\n合计\n价税合计(大写) (小写)\n备\n注\n开票人:24312000000280517071\n2024年09月12日\n上海商汤科技开发有限公司\n91310115MA1HB3LY4M上海椒客多餐饮服务有限公司\n91310105MACBK27P2A\n¥353.47 ¥3.53\n叁佰伍拾柒圆整 ¥357.00\n李小育\n李小育*餐饮服务 *餐饮服务 1% 353.47 3.53 353.47 1"
			},
			"meta": {
				"name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
				"content_type": "application/pdf",
				"size": 83703,
				"data": {},
				"collection_name": "file-370664f8-f179-473d-9a0c-b3f07fc0349f"
			},
			"created_at": 1741852282,
			"updated_at": 1741852282
			},
			"id": "370664f8-f179-473d-9a0c-b3f07fc0349f",
			"url": "http://10.8.10.82:8090/api/v1/files/370664f8-f179-473d-9a0c-b3f07fc0349f",
			"name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
			"collection_name": "file-370664f8-f179-473d-9a0c-b3f07fc0349f",
			"status": "uploaded",
			"size": 83703,
			"error": "",
			"itemId": "07534a1a-a86a-4213-a451-25fc6064524c"
		}
	],
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
		"info": {
			"id": "rag",
			"user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
			"base_model_id": "gemma3:27b",
			"name": "rag",
			"params": {},
			"meta": {
				"profile_image_url": "/static/favicon.png",
				"description": null,
				"capabilities": {
					"vision": true,
					"citations": true
				},
				"suggestion_prompts": null,
				"tags": [],
				"knowledge": [{
					"id": "3242bbd4-4a09-47d0-a704-dcbd5d665774",
					"user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
					"name": "test",
					"description": "个人文档",
					"data": {
						"file_ids": ["eb495463-9977-45d6-abd5-50a6365cacac"]
					},
					"meta": null,
					"access_control": null,
					"created_at": 1741763827,
					"updated_at": 1741775216,
					"user": {
						"id": "50e5febb-e4d7-4caa-9965-751160245ab6",
						"name": "neo",
						"email": "guojing_neo@163.com",
						"role": "admin"
					},
					"files": [{
						"id": "eb495463-9977-45d6-abd5-50a6365cacac",
						"meta": {
							"name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
							"content_type": "application/pdf",
							"size": 83703,
							"data": {},
							"collection_name": "3242bbd4-4a09-47d0-a704-dcbd5d665774"
						},
						"created_at": 1741775216,
						"updated_at": 1741775216
					}],
					"type": "collection"
				}]
			},
			"access_control": {
				"read": {
					"group_ids": [],
					"user_ids": []
				},
				"write": {
					"group_ids": [],
					"user_ids": []
				}
			},
			"is_active": true,
			"updated_at": 1741851344,
			"created_at": 1741851344
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
- api/v1/files/ file.py ,文件保存到了存储里
- 调用process_file： 将文件存储到知识库里
- - 返回
```
{
    "id": "017e1bf1-5f91-4d95-8789-fae28cc5e593",
    "user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
    "hash": "3e11c6cbdad114f4807614b25e6e94a83378905f091130c277c68178e0e327df",
    "filename": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
    "data": {
        "content": "电子发票(普通发票) 发票号码:\n开票日期:\n购\n买\n方\n信\n息统一社会信用代码 /纳税人识别号:销\n售\n方\n信\n息统一社会信用代码 /纳税人识别号:名称: 名称:\n项目名称 规格型号 单 位数 量单 价金 额税率/征收率 税 额\n合计\n价税合计(大写) (小写)\n备\n注\n开票人:24312000000280517071\n2024年09月12日\n上海商汤科技开发有限公司\n91310115MA1HB3LY4M上海椒客多餐饮服务有限公司\n91310105MACBK27P2A\n¥353.47 ¥3.53\n叁佰伍拾柒圆整 ¥357.00\n李小育\n李小育*餐饮服务 *餐饮服务 1% 353.47 3.53 353.47 1"
    },
    "meta": {
        "name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
        "content_type": "application/pdf",
        "size": 83703,
        "data": {},
        "collection_name": "file-017e1bf1-5f91-4d95-8789-fae28cc5e593"
    },
    "created_at": 1741759604,
    "updated_at": 1741759604
}
```
- 基于文件的聊天请求格式
```
{
	"stream": true,
	"model": "qwq:latest",
	"messages": [ {
		"role": "user",
		"content": "改文档说了啥？"
	}],
	"params": {},
	"files": [{
		"type": "file",
		"file": {
			"id": "017e1bf1-5f91-4d95-8789-fae28cc5e593",
			"user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
			"hash": "3e11c6cbdad114f4807614b25e6e94a83378905f091130c277c68178e0e327df",
			"filename": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
			"data": {
				"content": "电子发票(普通发票) 发票号码:\n开票日期:\n购\n买\n方\n信\n息统一社会信用代码 /纳税人识别号:销\n售\n方\n信\n息统一社会信用代码 /纳税人识别号:名称: 名称:\n项目名称 规格型号 单 位数 量单 价金 额税率/征收率 税 额\n合计\n价税合计(大写) (小写)\n备\n注\n开票人:24312000000280517071\n2024年09月12日\n上海商汤科技开发有限公司\n91310115MA1HB3LY4M上海椒客多餐饮服务有限公司\n91310105MACBK27P2A\n¥353.47 ¥3.53\n叁佰伍拾柒圆整 ¥357.00\n李小育\n李小育*餐饮服务 *餐饮服务 1% 353.47 3.53 353.47 1"
			},
			"meta": {
				"name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
				"content_type": "application/pdf",
				"size": 83703,
				"data": {},
				"collection_name": "file-017e1bf1-5f91-4d95-8789-fae28cc5e593"
			},
			"created_at": 1741759604,
			"updated_at": 1741759604
		},
		"id": "017e1bf1-5f91-4d95-8789-fae28cc5e593",
		"url": "http://10.8.10.82:8090/api/v1/files/017e1bf1-5f91-4d95-8789-fae28cc5e593",
		"name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
		"collection_name": "file-017e1bf1-5f91-4d95-8789-fae28cc5e593",
		"status": "uploaded",
		"size": 83703,
		"error": "",
		"itemId": "02da21dd-e1b5-4fbb-9c86-8c15a220b32b"
	}],
	"features": {
		"image_generation": false,
		"code_interpreter": false,
		"web_search": false
	},
	"variables": {
		"{{USER_NAME}}": "neo",
		"{{USER_LOCATION}}": "Unknown",
		"{{CURRENT_DATETIME}}": "2025-03-12 14:14:15",
		"{{CURRENT_DATE}}": "2025-03-12",
		"{{CURRENT_TIME}}": "14:14:15",
		"{{CURRENT_WEEKDAY}}": "Wednesday",
		"{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
		"{{USER_LANGUAGE}}": "zh-CN"
	},
	"model_item": {
		"id": "qwq:latest",
		"name": "qwq:latest",
		"object": "model",
		"created": 1741759127,
		"owned_by": "ollama",
		"ollama": {
			"name": "qwq:latest",
			"model": "qwq:latest",
			"modified_at": "2025-03-12T13:20:21.664248788+08:00",
			"size": 19851349390,
			"digest": "cc1091b0e276012ba4c1662ea103be2c87a1543d2ee435eb5715b37b9b680d27",
			"details": {
				"parent_model": "",
				"format": "gguf",
				"family": "qwen2",
				"families": ["qwen2"],
				"parameter_size": "32.8B",
				"quantization_level": "Q4_K_M"
			},
			"urls": [0]
		},
		"actions": []
	},
	"session_id": "-V2VOBs8OYpzx4CbAAAB",
	"chat_id": "9232c3bb-c7e0-44ac-98cd-b13c54bd5d02",
	"id": "d7b1ae20-6e8c-45d6-b9ff-f56ed506cc8b"
}
```

## 知识库
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

- 引用类型：
- {
    "sources": [
        {
            "source": {
                "id": "3242bbd4-4a09-47d0-a704-dcbd5d665774",
                "user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
                "name": "test",
                "description": "个人文档",
                "meta": null,
                "access_control": null,
                "created_at": 1741763827,
                "updated_at": 1741775216,
                "user": {
                    "id": "50e5febb-e4d7-4caa-9965-751160245ab6",
                },
                "files": [
                    {
                        "id": "eb495463-9977-45d6-abd5-50a6365cacac",
                        "meta": {
                            "name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
                            "content_type": "application/pdf",
                            "size": 83703,
                            "data": {},
                            "collection_name": "3242bbd4-4a09-47d0-a704-dcbd5d665774"
                        },
                        "created_at": 1741775216,
                        "updated_at": 1741775216
                    }
                ],
                "type": "collection"
            },
            "document": [""            ],
            "metadata": [
                {
                    "author": "China Tax",
                    "created_by": "50e5febb-e4d7-4caa-9965-751160245ab6",
                    "creationdate": "D:20240912212111",
                    "creator": "Suwell",
                    "embedding_config": "{\"engine\": \"\", \"model\": \"sentence-transformers/all-MiniLM-L6-v2\"}",
                    "file_id": "eb495463-9977-45d6-abd5-50a6365cacac",
                    "hash": "3e11c6cbdad114f4807614b25e6e94a83378905f091130c277c68178e0e327df",
                    "moddate": "D:20240912212111",
                    "name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
                    "ofd2pdflib": "ofd2pdflib/2.2.24.0111.1407",
                    "page": 0,
                    "page_label": "1",
                    "producer": "Suwell OFD convertor",
                    "source": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
                    "start_index": 0,
                    "total_pages": 1
                }
            ],
            "distances": [
                0.6697336820511759
            ]
        }
    ]
}
### 知识库管理 knowledge.py
- create 创建知识库
- / 列举知识库
- add 添加文件到知识库
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