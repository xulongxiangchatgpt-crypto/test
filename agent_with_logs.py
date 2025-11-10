#!/usr/bin/env python3
"""
增强版Agent - 添加详细的Monitor RAG日志记录
"""

import requests
from typing import Dict, List, Any
from openai import OpenAI
import re
import json
from termcolor import colored
import os
import sys
import time
import multiprocessing
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
# from hipporag import HippoRAG  # 注释掉，使用FAISS版本
from copy import deepcopy
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MonitorRAG')

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from llm_agent.utils import LLMAgent
from configs.common_config import CommonConfig
from llm_agent.tools.tool_manager import ToolManager
from mcp_sandbox.MCP.rag_tool_emb_llm_judge import search_local_documents


class BaseAgent:
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config: LLMAgent = LLMAgent(llm_config)
        self.rag_monitor = OpenAI(
            api_key=CommonConfig["OPENAI_CONFIG"]["authorization"],
            base_url=CommonConfig["OPENAI_CONFIG"]["url"]
        )
        self.rag_querier = OpenAI(
            api_key=CommonConfig["OPENAI_CONFIG"]["authorization"],
            base_url=CommonConfig["OPENAI_CONFIG"]["url"]
        )
        self.rag_injector = OpenAI(
            api_key=CommonConfig["OPENAI_CONFIG"]["authorization"],
            base_url=CommonConfig["OPENAI_CONFIG"]["url"]
        )
        
        # RAG参数
        self.rag_chunk = 200  # 监控窗口大小
        self.rag_overlapping = 50  # 重叠窗口大小
        self.max_rag = 3  # 最大RAG中断次数
        
        logger.info(f"BaseAgent initialized with RAG parameters: chunk={self.rag_chunk}, overlapping={self.rag_overlapping}, max_rag={self.max_rag}")

    def check_rag(self, text: str):
        """
        检查是否需要RAG检索
        """
        logger.info(f"🔍 [RAG Monitor] 开始检查文本是否需要RAG检索")
        logger.info(f"📝 [RAG Monitor] 输入文本长度: {len(text)} 字符")
        logger.info(f"📄 [RAG Monitor] 输入文本预览: {text[:100]}...")
        
        prompt_1 = f"""
Analyze the following text and determine if responding to it accurately requires retrieving information from an external source.
If you find any doubt or uncertainty about a concept or term in the text, consider it necessary to rag. You should tend to use rag because it helps with reasoning.

If retrieval is required, answer: yes
If no retrieval is required, answer: no

Text:
{text}

Judgment:
"""
        try:
            logger.info(f"🤖 [RAG Monitor] 调用GPT-4.1-mini进行RAG判断")
            start_time = time.time()
            
            response = self.rag_monitor.chat.completions.create(
                model = 'gpt-4.1-mini',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_1}
                ],
                max_tokens=150,
                temperature=0
            )
            
            end_time = time.time()
            result = response.choices[0].message.content.strip()
            
            logger.info(f"⏱️ [RAG Monitor] API调用耗时: {end_time - start_time:.2f}秒")
            logger.info(f"🎯 [RAG Monitor] 判断结果: {result}")
            
        except Exception as e:
            logger.error(f"❌ [RAG Monitor] API调用失败: {e}")
            return None

        if result.lower() != 'yes':
            logger.info(f"✅ [RAG Monitor] 判断结果: 不需要RAG检索")
            return None

        logger.info(f"🔄 [RAG Monitor] 需要RAG检索，开始生成查询")
        
        prompt_2 = f"""
Your task is to generate a single, concise, and effective search query for retrieving the information required by the text below.

## Instructions
1. Return **only the search query** itself.
2. Do not include any explanations, punctuation, quotation marks, or other text.
3. The query should be direct and contain only the most essential keywords.

## Text
{text}

Search Query:
"""
        try:
            logger.info(f"🔍 [RAG Querier] 调用GPT-4.1-mini生成搜索查询")
            start_time = time.time()
            
            response = self.rag_querier.chat.completions.create(
                model = 'gpt-4.1-mini',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_2}
                ],
                max_tokens=150,
                temperature=0
            )
            
            end_time = time.time()
            result = response.choices[0].message.content.strip()
            
            logger.info(f"⏱️ [RAG Querier] API调用耗时: {end_time - start_time:.2f}秒")
            logger.info(f"🎯 [RAG Querier] 生成的搜索查询: '{result}'")
            
            return result
        except Exception as e:
            logger.error(f"❌ [RAG Querier] API调用失败: {e}")
            return None

    def rag_search(self, query: str) -> str:
        """
        执行RAG搜索
        """
        logger.info(f"🔍 [RAG Search] 开始搜索，查询: '{query}'")
        start_time = time.time()
        
        try:
            docs = search_local_documents(query)
            end_time = time.time()
            
            logger.info(f"⏱️ [RAG Search] 搜索耗时: {end_time - start_time:.2f}秒")
            logger.info(f"📊 [RAG Search] 返回文档数量: {len(docs)}")
            
            if self.llm_config.is_debug:
                logger.info(f"📄 [RAG Search] 搜索结果预览:")
                for i, doc in enumerate(docs[:3]):  # 只显示前3个结果
                    logger.info(f"  [{i+1}] {doc[:100]}...")
            
            result = json.dumps(docs, indent=4)
            logger.info(f"✅ [RAG Search] 搜索完成，结果长度: {len(result)} 字符")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [RAG Search] 搜索失败: {e}")
            return json.dumps([], indent=4)

    def add_rag(self, text: str, rag_result: str) -> str:
        """
        将RAG结果集成到文本中
        """
        logger.info(f"🔄 [RAG Injector] 开始集成RAG结果")
        logger.info(f"📝 [RAG Injector] 原始文本长度: {len(text)} 字符")
        logger.info(f"📄 [RAG Injector] RAG结果长度: {len(rag_result)} 字符")
        
        prompt = f"""
You are an intelligent assistant that seamlessly integrates retrieved information into ongoing text generation. Your task is to naturally incorporate the retrieved information into the given text as if it were part of the original thought process.

## Instructions:
1. Integrate the retrieved information naturally into the text flow
2. Make it sound like the author just "thought of" or "remembered" this information
3. Use phrases like "I recall that...", "This reminds me that...", "I should also consider...", etc.
4. Maintain the original writing style and tone
5. Do not use explicit citations or references
6. Keep the integration smooth and conversational

## Original Text:
{text}

## Retrieved Information:
{rag_result}

## Enhanced Text:
"""
        try:
            logger.info(f"🤖 [RAG Injector] 调用GPT-4o进行内容集成")
            start_time = time.time()
            
            response = self.rag_injector.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            end_time = time.time()
            result = response.choices[0].message.content.strip()
            
            logger.info(f"⏱️ [RAG Injector] API调用耗时: {end_time - start_time:.2f}秒")
            logger.info(f"📏 [RAG Injector] 集成后文本长度: {len(result)} 字符")
            logger.info(f"✅ [RAG Injector] 内容集成完成")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [RAG Injector] 内容集成失败: {e}")
            return text

    def call_api(self, user_prompt: str, enable_rag: bool = True):
        """
        调用API生成内容，支持Monitor-based RAG
        """
        logger.info(f"🚀 [API Call] 开始API调用")
        logger.info(f"📝 [API Call] 用户提示长度: {len(user_prompt)} 字符")
        logger.info(f"🔧 [API Call] RAG功能: {'启用' if enable_rag else '禁用'}")
        
        if not enable_rag:
            logger.info(f"⚡ [API Call] RAG已禁用，直接生成内容")
            try:
                response = self.llm_config.call_model(user_prompt)
                logger.info(f"✅ [API Call] 内容生成完成")
                return {"content": response, "type": "completed"}
            except Exception as e:
                logger.error(f"❌ [API Call] 内容生成失败: {e}")
                return {"content": "", "type": "error", "error": str(e)}

        # Monitor-based RAG逻辑
        logger.info(f"🔍 [Monitor RAG] 开始Monitor-based RAG流程")
        
        rag_count = 0
        current_prompt = user_prompt
        rag_check_text = ""
        
        while rag_count < self.max_rag:
            logger.info(f"🔄 [Monitor RAG] 第 {rag_count + 1} 次生成循环")
            
            try:
                # 流式生成
                logger.info(f"⚡ [Monitor RAG] 开始流式生成")
                response = self.llm_config.call_model_stream(current_prompt)
                
                generated_text = ""
                rag_check_text = ""
                
                for chunk in response:
                    if 'content' in chunk and chunk['content']:
                        generated_text += chunk['content']
                        rag_check_text += chunk['content']
                        
                        # 检查是否需要RAG
                        if len(rag_check_text) >= self.rag_chunk:
                            logger.info(f"🎯 [Monitor RAG] 达到RAG检查阈值 ({self.rag_chunk} 字符)")
                            
                            # 检查是否需要RAG
                            rag_query = self.check_rag(rag_check_text)
                            
                            if rag_query:
                                logger.info(f"🔄 [Monitor RAG] 触发RAG检索，查询: '{rag_query}'")
                                
                                # 执行RAG搜索
                                rag_result = self.rag_search(rag_query)
                                
                                # 集成RAG结果
                                enhanced_text = self.add_rag(rag_check_text, rag_result)
                                
                                # 更新提示
                                current_prompt = user_prompt + "\n\n" + enhanced_text
                                rag_count += 1
                                
                                logger.info(f"🔄 [Monitor RAG] RAG集成完成，开始新的生成循环")
                                logger.info(f"📊 [Monitor RAG] 当前RAG次数: {rag_count}/{self.max_rag}")
                                
                                # 重置检查文本，保留重叠部分
                                rag_check_text = rag_check_text[-self.rag_overlapping:]
                                break
                            else:
                                logger.info(f"✅ [Monitor RAG] 无需RAG检索，继续生成")
                                # 重置检查文本，保留重叠部分
                                rag_check_text = rag_check_text[-self.rag_overlapping:]
                
                # 如果正常完成生成
                if not rag_query:
                    logger.info(f"✅ [Monitor RAG] 生成完成，无RAG中断")
                    return {"content": generated_text, "type": "completed"}
                else:
                    logger.info(f"🔄 [Monitor RAG] 因RAG中断，继续生成")
                    
            except Exception as e:
                logger.error(f"❌ [Monitor RAG] 生成过程中出错: {e}")
                return {"content": "", "type": "error", "error": str(e)}
        
        logger.warning(f"⚠️ [Monitor RAG] 达到最大RAG次数限制 ({self.max_rag})")
        return {"content": generated_text, "type": "max_rag_reached"}


class Eigen1Agent:
    def __init__(self, debug: bool = False, log_dir: str = "logs"):
        """
        初始化Eigen1Agent，使用o1-mini模型
        """
        logger.info(f"🚀 [Eigen1Agent] 初始化开始")
        logger.info(f"🔧 [Eigen1Agent] Debug模式: {debug}")
        logger.info(f"📁 [Eigen1Agent] 日志目录: {log_dir}")
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置
        common_config = CommonConfig()
        
        # 使用o1-mini模型配置
        llm_config = {
            'model': 'o1-mini',
            'base_url': common_config["OPENAI_CONFIG"]["url"],
            'api_key': common_config["OPENAI_CONFIG"]["authorization"],
            'generation_config': {
                'max_tokens': 4000,
                'temperature': 0.7,
            },
            'stop_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
            'tool_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
            'is_debug': debug
        }
        
        self.deepseek_api_url = common_config['DEEPSEEK_CONFIG']['url']
        self.deepseek_api_key = common_config['DEEPSEEK_CONFIG']['authorization']
        self.sandbox_url = common_config['SANDBOX']['tool_link']
        
        self.chat_obj = LLMAgent(self.deepseek_api_url, self.deepseek_api_key, self.sandbox_url)
        
        logger.info(f"✅ [Eigen1Agent] 初始化完成")
        logger.info(f"🤖 [Eigen1Agent] 使用模型: o1-mini")
        logger.info(f"🔗 [Eigen1Agent] 工具服务器: {self.sandbox_url}")

    def _forward_solver(self, query: str):
        """
        Solver步骤
        """
        logger.info(f"🧠 [Solver] 开始Solver步骤")
        logger.info(f"📝 [Solver] 查询: {query}")
        
        user_prompt = f"""
请按照以下步骤分析问题：{query}

1. 首先进行本地文档搜索，查找相关信息
2. 评估搜索结果的质量和相关性
3. 进行网络搜索验证和补充信息
4. 综合所有信息给出最终答案

请开始执行：
"""
        
        try:
            result = self.chat_obj.call_model(user_prompt, assistant_prefix="我将按照工作流程来解决这个问题。")
            logger.info(f"✅ [Solver] Solver步骤完成")
            logger.info(f"📏 [Solver] 结果长度: {len(result)} 字符")
            return result
        except Exception as e:
            logger.error(f"❌ [Solver] Solver步骤失败: {e}")
            return str(e)

    def forward(self, query: str, question_id: str = "default"):
        """
        完整的Eigen1工作流程
        """
        logger.info(f"🎯 [Eigen1Agent] 开始处理查询")
        logger.info(f"🆔 [Eigen1Agent] 问题ID: {question_id}")
        logger.info(f"📝 [Eigen1Agent] 查询: {query}")
        
        # Step 1: Solver
        solver_result = self._forward_solver(query)
        
        logger.info(f"🎉 [Eigen1Agent] 处理完成")
        logger.info(f"📊 [Eigen1Agent] 最终结果长度: {len(solver_result)} 字符")
        
        return solver_result


if __name__ == "__main__":
    # 测试代码
    logger.info("🧪 开始测试增强版Agent")
    
    # 创建配置
    common_config = CommonConfig()
    llm_config = {
        'model': 'o1-mini',
        'base_url': common_config["OPENAI_CONFIG"]["url"],
        'api_key': common_config["OPENAI_CONFIG"]["authorization"],
        'generation_config': {
            'max_tokens': 4000,
            'temperature': 0.7,
        },
        'stop_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        'tool_condition': r'<code[^>]*>((?:(?!<code).)*?)</code>',
        'is_debug': True
    }
    
    # 创建BaseAgent实例
    agent = BaseAgent(llm_config)
    
    # 测试查询
    test_query = "分析一下京东和淘宝的区别"
    logger.info(f"🔍 测试查询: {test_query}")
    
    # 执行查询
    result = agent.call_api(test_query, enable_rag=True)
    
    logger.info(f"✅ 测试完成，结果类型: {result.get('type', 'unknown')}")
