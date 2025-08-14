---
date: 2025-08-11
title: langchain实践
description: 
mermaid: true
mathjax: true
category: [智能体]
tags: [langchain, 智能体, 大模型]
ogImage: https://astro-yi.obs.cn-east-3.myhuaweicloud.com/avatar.png
---

# 预备内容

- **LangChain 官网**：https://www.langchain.com/
- **Python 文档入口**：https://python.langchain.com/docs/get_started/introduction

# hello world 例子
## 环境设置
创建环境
`conda create --name langchain_env python=3.11`

激活环境
`conda activate langchain_env`

安装包
`conda install langchain -c conda-forge`
`pip install -qU "langchain[openai]"`

**注册使用 LangSmith**
LangSmith 用于追踪、调试和评估 LLM 应用    
- 官网：https://smith.langchain.com/ ，可使用谷歌登录
- 获取 API 密钥和项目名称：Settings-API Keys

**环境变量**
.env 文件
```python
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
OPENROUTER_API_KEY=""
```
*OpenRouter官网： https://openrouter.ai/

## 主代码
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载所有环境变量
load_dotenv()

# LangSmith的配置，确保开启
os.environ["LANGSMITH_TRACING"] = "true"

# 初始化 ChatOpenAI 模型
model = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY") # 使用 os.getenv() 更安全
)

from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# 调用模型并获取返回结果
response = model.invoke(messages)

# 打印回复的文本内容
print(response.content)
```

## 提示模版类
```python
# 导入提示模板类
from langchain_core.prompts import ChatPromptTemplate

# 定义系统和用户的模板
system_template = "将以下内容从英语翻译成{language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 使用模板生成提示词
prompt = prompt_template.invoke({"language": "意大利语", "text": "Hello, world!"})

# 将格式化后的提示词传递给模型(流式输出)
for chunk in model.stream(prompt):
    print(chunk.content, end="", flush=True)
```

# RAG 应用

**教程链接**：https://python.langchain.com/docs/tutorials/rag  
将自己的文档（PDF、文本文件等）变成一个可供 LLM 检索的知识库

## 概述（索引+检索生成）
![LangChain-RAG1](/pic/langchain实践/LangChain-RAG1.png)  
1. 加载：通过文档加载器加载数据
2. 拆分：将大的 `Documents` 分解成更小的块，对于索引数据和将数据传入模型都很有用，因为大块数据更难搜索，而且无法适应模型有限的上下文窗口
3. 存储：需要一个地方来存储和索引分割的内容，以便以后可以对其进行搜索，通常通过使用向量存储和嵌入模型来完成
![LangChain-RAG2](/pic/langchain实践/LangChain-RAG2.png)  
4. 检索：给定用户输入，使用检索器从存储中检索相关的片段
5. 生成：ChatModel / LLM 使用包含问题和检索数据的提示产生一个答案  

一旦索引了数据，将使用 **LangGraph** 作为编排框架来实施检索和生成步骤

## 环境设置


    
- **你将学到什么？**
    
    - 如何加载一个文档。
        
    - 如何将文档分割成小块（chunking）。
        
    - 如何使用嵌入模型（embedding model）将文本转成向量。
        
    - 如何将向量存入向量数据库（vector store）。
        
    - 如何用一个检索器（retriever）来查找相关文档。
        
    - 如何将检索到的文档传递给 LLM，生成最终答案。
        

#### 案例 2：构建一个能使用工具的 Agent

Agent 是 LangChain 中最强大的概念之一，它让 LLM 能够自主决策并使用外部工具。这个案例会让你理解 Agent 的工作原理。

- **教程链接**：[https://python.langchain.com/docs/modules/agents/how_to/](https://www.google.com/search?q=https://python.langchain.com/docs/modules/agents/how_to/)
    
- **你将学到什么？**
    
    - 如何定义一个工具（Tool），比如一个搜索工具或计算器。
        
    - 如何将工具提供给 Agent。
        
    - 如何观察 Agent 的“思考”过程（**Thought**）和它选择执行的“行动”（**Action**）。
        
    - 如何让 Agent 根据任务需求，自主选择并使用合适的工具。
        

### 3. 如何实践这些案例

你可以将这些案例的代码复制到你本地的 **Jupyter Notebook** 或 **Python 文件**中。

1. **安装必要的库**：通常你需要安装 `langchain` 和一些其他库，比如 `openai`、`tiktoken` 等。
    
2. **设置 API 密钥**：大多数案例都需要一个 OpenAI 的 API 密钥。你需要创建一个 `.env` 文件，将你的密钥保存在里面，然后加载。
    
3. **逐行运行代码**：不要一次性复制代码，尝试理解每一行代码的作用。修改一些参数，看看结果有什么变化，这是最好的学习方法。
    

### 总结

- **第一步：** 从官方文档的 **Tutorials** 部分开始。
    
- **第二步：** 专注于 **RAG** 和 **Agent** 这两个核心案例。
    
- **第三步：** 动手运行代码，并尝试修改它。这是将理论知识转化为实践技能的关键。