# LectureAgent Package
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json
import os
import re
import shutil
import tempfile
import nltk
import requests
from xml.etree import ElementTree as ET
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from rank_bm25 import BM25Okapi
import bs4
import getpass


load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # langgraph追踪

if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0,
    timeout=10,
    max_tokens=1000,
)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_MAX_RESULTS = 10
SERPER_SCHOLAR_TYPE = "scholar"


@dataclass
class SearchResult:
    articles: List[Dict[str, str]]
    temp_dir: Path
    json_path: Path


def create_temp_results_dir() -> Path:
    """创建存放搜索结果的临时文件夹."""
    return Path(tempfile.mkdtemp(prefix="lectureagent_"))


def write_results_to_json(articles: List[Dict[str, str]], temp_dir: Path) -> Path:
    """将文章列表写入临时目录中的 JSON 文件."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    json_path = temp_dir / "results.json"
    payload = [{"title": item["title"], "url": item["url"]} for item in articles]
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return json_path


def delete_temp_results_dir(temp_dir: Path) -> None:
    """删除指定的临时目录."""
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def _init_scholar_client(max_results: int) -> GoogleSerperAPIWrapper:
    """初始化 Serper Google Scholar 客户端."""
    if not os.getenv("SERPER_API_KEY"):
        raise EnvironmentError(
            "SERPER_API_KEY 未配置，无法进行 Google Scholar 搜索。"
        )
    return GoogleSerperAPIWrapper(type=SERPER_SCHOLAR_TYPE, k=max_results)


def _build_arxiv_query(user_input: str) -> str:
    """将用户输入转换为 arXiv 兼容的检索语句."""
    cleaned = re.sub(r"\s+", " ", user_input).strip()
    if not cleaned:
        raise ValueError("检索内容不能为空")
    # 基于关键词在题目、摘要和分类中检索
    tokens = re.split(r"[，,.;；]", cleaned)
    clauses = []
    for token in filter(None, (t.strip() for t in tokens)):
        # arXiv 查询：ti 标题，abs 摘要，cat 分类
        clause = f"(ti:{token} OR abs:{token} OR cat:{token})"
        clauses.append(clause)
    return " AND ".join(clauses) if clauses else f"(ti:{cleaned} OR abs:{cleaned})"


def search_arxiv_articles(
    domain_text: str, max_results: int = DEFAULT_MAX_RESULTS
) -> SearchResult:
    """
    根据领域描述在 arXiv 检索相关文章.

    Args:
        domain_text: 用户输入的领域或文章内容描述.
        max_results: 返回的文章数量，默认 10.

    Returns:
        包含标题与链接的字典列表.
    """
    query = _build_arxiv_query(domain_text)
    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "LectureAgent/0.1 (contact: placeholder@example.com)",
    }

    response = requests.get(
        ARXIV_API_URL,
        params=params,
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    articles: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NAMESPACE):
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NAMESPACE).strip()
        link = entry.findtext("atom:id", default="", namespaces=ATOM_NAMESPACE).strip()
        if title and link:
            articles.append({"title": title, "url": link})

    articles = articles[:max_results]
    temp_dir = create_temp_results_dir()
    json_path = write_results_to_json(articles, temp_dir)
    return SearchResult(articles=articles, temp_dir=temp_dir, json_path=json_path)


def search_google_scholar_articles(
    query: str, max_results: int = DEFAULT_MAX_RESULTS
) -> SearchResult:
    """
    使用 Google Scholar 获取最相关的论文标题和链接.

    Args:
        query: 用户输入的检索关键词或领域描述.
        max_results: 返回条数，默认 10.

    Returns:
        SearchResult: 包含文章列表与结果文件路径.
    """
    client = _init_scholar_client(max_results)
    response = client.results(query)
    organic = response.get("organic", []) if isinstance(response, dict) else []

    articles: List[Dict[str, str]] = []
    for item in organic:
        title = (item.get("title") or "").strip()
        url = (item.get("link") or "").strip()
        if title and url:
            articles.append({"title": title, "url": url})

    articles = articles[:max_results]
    temp_dir = create_temp_results_dir()
    json_path = write_results_to_json(articles, temp_dir)
    return SearchResult(articles=articles, temp_dir=temp_dir, json_path=json_path)


@tool("google_scholar_search")
def google_scholar_search_tool(query: str) -> str:
    """
    LangChain 工具：通过 Google Scholar 搜索研究文章.

    Args:
        query: 用户输入的检索关键词.

    Returns:
        str: 结果摘要（含 JSON 文件路径），供代理回复用户。
    """
    result = search_google_scholar_articles(query)
    lines = [
        f"{idx+1}. {item['title']} | {item['url']}"
        for idx, item in enumerate(result.articles)
    ]
    summary = "\n".join(lines) if lines else "未找到相关结果。"
    return f"{summary}\n\nJSON 文件：{result.json_path}"