# LectureAgent Package
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import re
import nltk
from typing import List, Tuple

import bs4
import getpass
import os
from dotenv import load_dotenv




load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true" #langgarph追踪

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",   # 通义千问 Embedding 模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


model = init_chat_model(
    model="qwen-plus",
    model_provider="together",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    timeout=10,
    max_tokens=1000
)
@dataclass
class ResponseFormat:
    """Response schema for the agent."""

    papertitle1: str
    url1: str
    papertitle2: str
    url2: str
    papertitle3: str
    url3: str
    papertitle4: str | None=None
    url4: str | None=None
    papertitle5: str | None=None
    url5: str | None=None
    




# -----------------------------
# 3. 工具：arXiv 查询
# -----------------------------
@tool()
def axriv_search(query: str):
    """
    在 arXiv 上搜索相关论文，返回前 5 篇论文的标题和链接。
    """
    print("arxiv search called with query:", query)
    return query
@tool()
def google_search(query: str):
    """
    使用 Google 搜索相关论文，返回前 5 篇论文的标题和链接。
    """
    print("google search called with query:", query)

    return query
# -----------------------------
# 4. 创建 Agent
# -----------------------------
tools = [axriv_search, google_search]

agent = create_agent(
    model=model,       # 已经初始化好的 ChatModel
    tools=tools,

)


# -----------------------------
# 5. 对外接口
# -----------------------------
def ask(query: str) -> str:
    """
    用户输入 → Agent → 自动调用提取关键词 + arXiv 搜索 → 输出论文标题+链接
    """
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},

    )


# -----------------------------
# 6. 主程序
# -----------------------------
if __name__ == "__main__":
    user_q = "RAG cache 类似的论文有哪些？"
    answer = ask(user_q)
    print("=== 搜索结果 ===")
    print(answer)
