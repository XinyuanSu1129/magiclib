"""
magiclib / generator

Attention:
1. statements should be abbreviated
2. Pay attention to the usage traffic of API keys
3. The character is added each time, and the price is calculated last
"""


# 导入顺序不同有可能导致程序异常
from . import general  # 当前未用到，预计参与分析时用到

from openai import OpenAI
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import abstractmethod
from typing import Union, Optional, List, Dict, Callable, Tuple

current_time_zone_location = 'Shanghai'
DeepSeek_api = 'sk-cc2167b962444015a28d989478add7eb'

class AI:

    # 公有参数初始化
    def __init__(self, client: Optional[OpenAI] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, max_tokens: int = 2048, temperature: float = 0.7,
                 top_p: float = 1.0, n: int = 1, stream: bool = False, stop: Union[str, list, None] = None,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.


        :param client: (OpenAI) OpenAI 实例化对象
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 或 'deepseek-reasoner'
        :param messages: (list) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param n: (int) 生成多少个独立回复选项 (消耗 n 倍 token)，如 n=3 会返回 3 种不同回答
        :param stream: (bool) 是否启用流输出 (逐字返回)
        :param stop: (str / list) 停止生成的标记，遇到这些字符串时终止输出
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作
        """

        # 参数初始化
        self.client = client
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        # 回答
        self.response = None

        # 初始化字符统计计数器
        self.total_input_tokens = 0  # 用户输入总 tokens
        self.total_output_tokens = 0  # AI 模型输出总 tokens

    # 与 AI 模型对话
    @abstractmethod
    def chat(self):
        pass


class DeepSeek(AI):

    # DeepSeek 特有参数
    def __init__(self, client: Optional[OpenAI] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None,  reasoning_steps: bool = False, precision: str = 'medium',
                 cache_enabled: bool = True):
        """
        DeepSeek 特有参数初始化
        Initialization of DeepSeek's unique parameters.

        :param client: (OpenAI) OpenAI 实例化对象
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 或 'deepseek-reasoner'
        :param messages: (list) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息
        :param reasoning_steps: (bool) 是否显示推理步骤 (仅限 reasoner 模型)
        :param precision: (str) 计算精度，'low'、'medium' 与 'high'
        :param cache_enabled: (bool) 是不启用响应缓存
        """

        super().__init__(model=model, messages=messages, client=client)

        # 检查 OpenAI 实例化
        if client is None:
            self.client = OpenAI(
                api_key=DeepSeek_api,  # API Key
                base_url='https://api.deepseek.com/v1',  # DeepSeek
            )

        # 检查 model 实例化
        if model is None:
            self.model = 'deepseek-chat'

        # 检查 messages 实例化
        if messages is None:
            self.messages = [  # message 是一个 list，包含多个消息对象，最后一个消息对象为当前发的内容
                        {"role": "system",   # role 是消息发送者的身份，有 "system", "user" 与 "assistant"
                         "content": "You are a helpful AI assistant who answers users' questions."}  # 消息文本内容
                        ]

        # 参数初始化
        self.reasoning_steps = reasoning_steps
        self.precision = precision
        self.cache_enabled = cache_enabled

        # 初始化字符统计计数器
        self.total_input_deepseek_chat_hit_tokens = 0  # 用户输入命中 tokens deepseek-chat
        self.total_input_deepseek_chat_miss_tokens = 0  # 用户输入未命中 tokens deepseek-chat
        self.total_input_deepseek_chat_tokens = 0  # 用户输入总 tokens deepseek-chat
        self.total_output_deepseek_chat_tokens = 0  # AI 模型输出总 tokens deepseek-reasoner

        self.total_input_deepseek_reasoner_hit_tokens = 0  # 用户输入命中 tokens deepseek-reasoner
        self.total_input_deepseek_reasoner_miss_tokens = 0  # 用户输入未命中 tokens deepseek-reasoner
        self.total_input_deepseek_reasoner_tokens = 0  # 用户输入总 tokens deepseek-reasoner
        self.total_output_deepseek_reasoner_tokens = 0  # AI 模型输出总 tokens deepseek-reasoner

        # 模型与价格
        self.input_price_of_millions_of_hit_tokens_of_deepseek_chat = 0.5  # 命中 deepseek-chat
        self.input_price_of_millions_of_miss_tokens_of_deepseek_chat = 2  # 未命中 deepseek-chat
        self.output_price_of_millions_of_tokens_of_deepseek_chat = 8  # 输出 deepseek-chat

        self.input_price_of_millions_of_hit_tokens_of_deepseek_reasoner = 1  # 命中 deepseek-reasoner
        self.input_price_of_millions_of_miss_tokens_of_deepseek_reasoner = 4  # 未命中 deepseek-reasoner
        self.output_price_of_millions_of_tokens_of_deepseek_reasoner = 16  # 输出 deepseek-reasoner

        self.in_discounts = False  # 是否在折扣范围内
        self.current_price = 0.0
        self.input_price_of_millions_of_hit_tokens_of_deepseek_chat_discounts = 0.25  # 命中 deepseek-chat
        self.input_price_of_millions_of_miss_tokens_of_deepseek_chat_discounts = 1  # 未命中 deepseek-chat
        self.output_price_of_millions_of_tokens_of_deepseek_chat_discounts = 4  # 输出 deepseek-chat

        self.input_price_of_millions_of_hit_tokens_of_deepseek_reasoner_discounts = 0.25  # 命中 deepseek-reasoner
        self.input_price_of_millions_of_miss_tokens_of_deepseek_reasoner_discounts = 1  # 未命中 deepseek-reasoner
        self.output_price_of_millions_of_tokens_of_deepseek_reasoner_discounts = 4  # 输出 deepseek-reasoner

    # 与 DeepSeek 聊天
    def chat(self, model: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.7, top_p: float = 1.0,
             presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> None:
        """
        与 DeepSeek 的指定模型聊天，在优惠时段 (北京时间 00:30-08:30) 时改用 deepseek-reasoner 模型
        Chat with the designated model of DeepSeek and switch to the deepseek-reasoner model during
        the promotional period (00:30-08:30 Beijing time).

        注意：
        只有在空的一行输入换行符 '\n' 或空按“回车”才会将内容输入给 AI 模型，否则只是换到下一行并等待继续输入

        Note
        Only when a newline character '\n' is entered on an empty line or an empty "Enter" is pressed will the content
        be input into the AI model; otherwise, it will simply move to the next line and wait for further input.

        :param model: (str) 指定使用的模型，如 'deepseek-chat' 或 'deepseek-reasoner'
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作

        :return: None
        """

        # 检查赋值
        if model is None:
            model = self.model

        # 获取当前北京时间
        beijing_time = datetime.now(ZoneInfo(f"Asia/{current_time_zone_location}"))
        current_hour_min = (beijing_time.hour, beijing_time.minute)

        # 定义时间区间 [00:30, 08:30]
        start_time = (0, 30)  # 00:30
        end_time = (8, 30)  # 08:30

        # 判断是否在优惠时间 (Beijing 00:30-08:30) 内
        if start_time <= current_hour_min < end_time and model == 'deepseek-chat':
            model = 'deepseek-reasoner'
            model_verison = 'DeepSeek-R1-0528'
            self.in_discounts = True
            hour, minute = datetime.now().hour, datetime.now().minute
            print(f'The current time is \033[34m{hour}:{minute}\033[0m, and the model has been converted to '
                  f'\033[31m{model} ({model_verison})\033[0m.')
        else:
            model_verison = ''
            if model == 'deepseek-chat':
                model_verison = 'DeepSeek-V3-0324'
            elif model == 'deepseek-reasoner':
                model_verison = 'DeepSeek-R1-0528'

            print(f"Let's start chatting! The current model is \033[31m{model} ({model_verison})\033[0m.")

        # 对话循环
        while True:

            # 获取用户输入
            user_input_list = []
            print('\033[1m\033[92mUser\033[0m: ', end='')
            while True:
                line = input('')
                if line == '':
                    break
                user_input_list.append(line)

            user_input = "\n".join(user_input_list)

            # 退出条件
            if user_input.lower() in ['退出', 'exit', 'quit']:
                total_input_hit_tokens = (self.total_input_deepseek_chat_hit_tokens +
                                          self.total_input_deepseek_reasoner_hit_tokens)
                total_input_miss_tokens = (self.total_input_deepseek_chat_miss_tokens +
                                           self.total_input_deepseek_reasoner_miss_tokens)
                print(f'\nIn this conversation, the input contains \033[92m{self.total_input_tokens}\033[0m characters,'
                      f' where \033[3m{total_input_hit_tokens}\033[0m hit '
                      f'\033[3m{total_input_miss_tokens}\033[0m misses, '
                      f'and the output has \033[95m{self.total_output_tokens}\033[0m characters.')
                print(f'The conversation is over. Goodbye ^_< !\n')
                break

            # 添加用户消息到对话历史
            self.messages.append({"role": "user", "content": user_input})

            # 调用 DeepSeek API
            self.response = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stream=self.stream,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,

                # reasoning_steps=self.reasoning_steps if self.model == 'deepseek-reasoner' else False,
                # precision=self.precision,
                # cache_enabled=self.cache_enabled
            )

            # 获取 AI 回复
            ai_reply = self.response.choices[0].message.content

            prompt_tokens = self.response.usage.prompt_tokens  # 输入总 tokens
            prompt_hit_tokens = self.response.usage.prompt_cache_hit_tokens  # 输入命中 tokens
            prompt_miss_tokens = self.response.usage.prompt_cache_miss_tokens  # 输入未命中 tokens
            completion_tokens = self.response.usage.completion_tokens  # 输出 tokens

            # prompt_tokens 计算
            if self.response.model == 'deepseek-chat':
                self.total_input_deepseek_chat_tokens += prompt_tokens
                self.total_input_deepseek_chat_hit_tokens += prompt_hit_tokens
                self.total_input_deepseek_chat_miss_tokens += prompt_miss_tokens
            elif self.response.model == 'deepseek-reasoner':
                self.total_input_deepseek_reasoner_tokens += prompt_tokens
                self.total_input_deepseek_reasoner_hit_tokens += prompt_hit_tokens
                self.total_input_deepseek_reasoner_miss_tokens += prompt_miss_tokens
            self.total_input_tokens += prompt_tokens  # 总长度添加

            # completion_tokens 计算
            if self.response.model == 'deepseek-chat':
                self.total_output_deepseek_chat_tokens += completion_tokens
            elif self.response.model == 'deepseek-reasoner':
                self.total_output_deepseek_reasoner_tokens += completion_tokens
            self.total_output_tokens += completion_tokens  # 总长度添加

            # 打印并保存 AI 回复
            print(f"\033[1m\033[95mDeepSeek\033[0m:\033[35;2m {ai_reply}\033[0m\n")
            self.messages.append(dict({"role": "assistant", "content": ai_reply}))

        return None

    # 计算使用的费用
    def calculate_cost(self) -> None:
        """
        计算本次使用的费用
        Calculate the cost for this use.

        :return: None
        """

        # 初始化费用参数
        input_price = 0.0
        output_price = 0.0

        if not self.in_discounts:
            # 输入价格
            input_price += (self.total_input_deepseek_chat_hit_tokens * 1 / 1000000 *  # 命中 deepseek-chat 模型
                            self.input_price_of_millions_of_hit_tokens_of_deepseek_chat)
            input_price += (self.total_input_deepseek_chat_miss_tokens * 1 / 1000000 *  # 未命中 deepseek-chat 模型
                            self.input_price_of_millions_of_miss_tokens_of_deepseek_chat)

            input_price += (self.total_input_deepseek_reasoner_hit_tokens * 1 / 1000000 *  # 命中 deepseek-reasoner 模型
                            self.input_price_of_millions_of_hit_tokens_of_deepseek_reasoner)
            input_price += (self.total_input_deepseek_reasoner_miss_tokens * 1 / 1000000 *  # 未命中 deepseek-reasoner
                            self.input_price_of_millions_of_miss_tokens_of_deepseek_reasoner)
            self.current_price += input_price

            # 输出价格
            output_price += (self.total_output_deepseek_chat_tokens * 1 / 1000000 *  # deepseek-chat 模型
                             self.output_price_of_millions_of_tokens_of_deepseek_chat)
            output_price += (self.total_output_deepseek_reasoner_tokens * 1 / 1000000 *  # deepseek-reasoner 模型
                             self.output_price_of_millions_of_tokens_of_deepseek_reasoner)
            self.current_price += output_price

        else:  # 在折扣范围内
            # 输入价格
            input_price += (self.total_input_deepseek_chat_hit_tokens * 1 / 1000000 *  # 命中 deepseek-chat 模型
                            self.input_price_of_millions_of_hit_tokens_of_deepseek_chat_discounts)
            input_price += (self.total_input_deepseek_chat_miss_tokens * 1 / 1000000 *  # 未命中 deepseek-chat 模型
                            self.input_price_of_millions_of_miss_tokens_of_deepseek_chat_discounts)

            input_price += (self.total_input_deepseek_reasoner_hit_tokens * 1 / 1000000 *  # 命中 deepseek-reasoner 模型
                            self.input_price_of_millions_of_hit_tokens_of_deepseek_reasoner_discounts)
            input_price += (self.total_input_deepseek_reasoner_miss_tokens * 1 / 1000000 *  # 未命中 deepseek-reasoner
                            self.input_price_of_millions_of_miss_tokens_of_deepseek_reasoner_discounts)
            self.current_price += input_price

            # 输出价格
            output_price += (self.total_output_deepseek_chat_tokens * 1 / 1000000 *  # deepseek-chat 模型
                            self.output_price_of_millions_of_tokens_of_deepseek_chat_discounts)
            output_price += (self.total_output_deepseek_reasoner_tokens * 1 / 1000000 *  # deepseek-reasoner 模型
                            self.output_price_of_millions_of_tokens_of_deepseek_reasoner_discounts)
            self.current_price += output_price

        print(f'The quotation for this conversation is \033[31;2m¥{self.current_price:,.4f}\033[0m.\n')

    # 对以往对话进行总结
    def summarize_conversation(self, len_lim: int = 1000) -> None:
        """
        总结对话历史，控制上下文长度，当历史对话总字符超过 len_lim 时，生成简洁总结替代历史记录

        :param len_lim: (int) 总结对话时的长度阈值，未达限制时不进行总结，默认为 1000
        :return: None
        """

        # 计算当前所有消息的总字符数
        total_chars = sum(len(msg["content"]) for msg in self.messages)

        # 如果总字符数小于等于 len_lim，无需总结
        if total_chars <= len_lim:
            print(f"\033[90m[No summary needed - Dialogue history within {len_lim} characters]\033[0m")
            return

        print(f"\033[90m[Dialogue history exceeds {len_lim} characters, generating summary...]\033[0m")

        # 保存当前对话历史（总结前）
        original_messages = self.messages.copy()

        try:
            # 构建总结请求 - 使用更明确的指令
            summary_prompt = [
                {
                    "role": "system",
                    "content": f"You are a professional dialogue summarization assistant. "
                               f"Summarize the following conversation concisely in its original language, "
                               f"preserving key information and decisions. "
                               f"Summary must be under {len_lim} characters. "
                               f"Output ONLY the summary content without any extra text."
                },
                {
                    "role": "user",
                    "content": "Summarize this conversation:\n" + "\n".join(
                        f"{msg['role']}: {msg['content']}" for msg in self.messages
                    )
                }
            ]

            # 调用API生成总结
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt,
                max_tokens=len_lim,  # 限制总结长度
                temperature=0.3  # 更确定的输出
            )

            # 获取总结内容
            summary = response.choices[0].message.content

            # 检查总结长度
            if len(summary) > len_lim:
                summary = summary[:len_lim - 3] + "..."  # 确保不超过 len_lim 字符

            # 更新对话历史：保留总结 + 最新两次对话
            # 关键更新：在system消息中明确说明包含总结和最新对话
            self.messages = [
                {
                    "role": "system",
                    "content": f"<SUMMARY>{summary}</SUMMARY>\n"
                               f"<CONTEXT>Previous conversation summarized. Latest two messages retained"
                               f" for context.</CONTEXT>"
                },
                *original_messages[-2:]  # 保留最新的用户输入和AI回复
            ]

            # 显示总结结果
            print(f"\033[92mSummary generated:\033[0m {summary}")
            print(f"\033[90m[History reduced from {total_chars} to {len(summary)} chars + latest 2 messages]\033[0m")

        except Exception as e:
            print(f"\033[91mSummary failed: {str(e)}\033[0m")
            # 恢复原始对话历史
            self.messages = original_messages

        return None

    # 清空聊天缓存内容
    def reset_conversation(self, prompt: Optional[str] = None, preserve_system: bool = False) -> None:
        """
        清空 self.messages 中的内容
        Clear the content in self.messages.

        :param prompt: (str) 可选的系统提示内容
        :param preserve_system: (bool) 是否保留原有的系统消息，默认为 False
        :return: None
        """

        # 保留系统消息的逻辑
        if preserve_system:
            # 筛选出所有系统消息
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            self.messages = system_messages
        else:
            self.messages = []

        # 添加新系统提示
        if prompt is not None:
            self.messages.append({
                "role": "system",
                "content": prompt
            })

        # 状态反馈
        print("\033[35;2mself.messages\033[0m only retains \033[31msystem\033[0m information" if preserve_system
              else "The content in \033[35;2mself.messages\033[0m has been \033[31mreset\033[0m")

    # 利用 DeepSeek 模型修改手稿
    def revise_manuscript(self, manuscript: Optional[str] = None, advice: Optional[str] = None,
                          to_chat: bool = True) -> str or list:
        """
        让 DeepSeek deepseek-reasoner 模型协助修改手稿，仅支持单次修改，后续问题需要转移至 chat() 中
        Let the DeepSeek deepseek-reasoner model assist in modifying the manuscript.
        It only supports single modification. Subsequent issues need to be transferred to chat()

        :param manuscript: (str) 需要修改的稿件内容，该内容为必填项
        :param advice: (str) 意见，往往是审稿人给出的，为选填项
        :param to_chat: (bool) 在生成修改后的文本后是否继续对话，默认为 True

        :return revised_text: (str) 修改后的文本
        """

        model_revise_manuscript = 'deepseek-reasoner'  # 应用 deepseek-reasoner 模型

        if advice is not None:  # 有修改意见
            stipulation = '''# Role
            You are a professional manuscript editing assistant with strong language skills and extensive experience in 
            text optimization. Please refine the language and improve the structure of the manuscript I provide, making 
            it clearer, more natural, and aligned with academic writing standards. Additionally, incorporate the 
            revision requests I provide and make appropriate adjustments or enhancements to the content accordingly.

            # Task
            I will provide the following items in sequence:
            1.	The original manuscript content;
            2.	Specific revision requests to be addressed.

            # Output Requirements
            1.	Basic Improvements:
            •	Correct all grammatical and spelling errors
            •	Enhance sentence fluency and readability
            •	Ensure the accuracy and consistency of technical terms
            •	Modify sentence structure to avoid redundancy
            2.	Targeted Revisions:
            •	Strictly follow the revision requests I provide
            •	Focus on and revise the specific areas mentioned in the comments
            •	Preserve the original meaning and tone of the text
            3.	Revision Marking System:
            Please use the following tags to indicate your edits:
            •	[Added]: for newly added content
            •	[Deleted]: for removed content
            •	[Modified]: for rewritten or revised content

            # Final Summary
            1.	Degree of revision (e.g., major revision 80%, minor revision 15%)
            2.	Summary of key changes made
            3.	A brief evaluation of the revised manuscript'''
            self.messages = [{"role": "system", "content": stipulation}]
            content = ("# Original Manuscript\n"
                       + manuscript.strip() + "\n\n"
                       + "# Revision Advice\n"
                       + advice.strip())
            messages_manuscript = {"role": "user", "content": content}
            self.messages.append(messages_manuscript)

        else:  # 无修改意见
            stipulation = '''# Role
            You are a professional manuscript editing assistant with strong language skills and extensive experience 
            in text optimization. Please refine the language and improve the structure of the manuscript I provide, 
            making it clearer, more natural, and aligned with academic writing standards. Additionally, incorporate 
            the revision requests I provide and make appropriate adjustments or enhancements to the content accordingly.

            # Task
            I will provide my manuscript. Please help me complete the revision.

            # Output Requirements
            1.	Basic Improvements:
            •	Correct all grammatical and spelling errors
            •	Enhance sentence fluency and readability
            •	Ensure the accuracy and consistency of technical terms
            •	Modify sentence structure to avoid redundancy
            2.	Targeted Revisions:
            •	Strictly follow the revision requests I provide
            •	Focus on and revise the specific areas mentioned in the comments
            •	Preserve the original meaning and tone of the text
            3.	Revision Marking System:
            Please use the following tags to indicate your edits:
            •	[Added]: for newly added content
            •	[Deleted]: for removed content
            •	[Modified]: for rewritten or revised content

            # Final Summary
            1.	Degree of revision (e.g., major revision 80%, minor revision 15%)
            2.	Summary of key changes made
            3.	A brief evaluation of the revised manuscript'''
            self.messages = [{"role": "system", "content": stipulation}]
            content = ("# Original Manuscript\n"
                       + manuscript.strip())
            messages_manuscript = {"role": "user", "content": content}
            self.messages.append(messages_manuscript)

        # 调用 DeepSeek API
        self.response = self.client.chat.completions.create(
            model=model_revise_manuscript,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            stream=self.stream,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        # 获取 AI 回复
        choices = self.response.choices
        if len(choices) == 1:  # 如果只生成一条消息
            ai_reply = choices[0].message.content.strip()
        else:  # 如果生成多条消息
            ai_reply = "\n\n".join(
                f"[Response {i + 1}]:\n{choice.message.content.strip()}"
                for i, choice in enumerate(choices)
            )

        prompt_tokens = self.response.usage.prompt_tokens  # 输入总 tokens
        prompt_hit_tokens = self.response.usage.prompt_cache_hit_tokens  # 输入命中 tokens
        prompt_miss_tokens = self.response.usage.prompt_cache_miss_tokens  # 输入未命中 tokens
        completion_tokens = self.response.usage.completion_tokens  # 输出 tokens

        # prompt_tokens 计算
        if self.response.model == 'deepseek-chat':
            self.total_input_deepseek_chat_tokens += prompt_tokens
            self.total_input_deepseek_chat_hit_tokens += prompt_hit_tokens
            self.total_input_deepseek_chat_miss_tokens += prompt_miss_tokens
        elif self.response.model == 'deepseek-reasoner':
            self.total_input_deepseek_reasoner_tokens += prompt_tokens
            self.total_input_deepseek_reasoner_hit_tokens += prompt_hit_tokens
            self.total_input_deepseek_reasoner_miss_tokens += prompt_miss_tokens
        self.total_input_tokens += prompt_tokens  # 总长度添加

        # completion_tokens 计算
        if self.response.model == 'deepseek-chat':
            self.total_output_deepseek_chat_tokens += completion_tokens
        elif self.response.model == 'deepseek-reasoner':
            self.total_output_deepseek_reasoner_tokens += completion_tokens
        self.total_output_tokens += completion_tokens  # 总长度添加

        # 打印并保存 AI 回复
        print(f"\033[1m\033[95mDeepSeek\033[0m:\033[35;2m {ai_reply}\033[0m\n")
        self.messages.append(dict({"role": "assistant", "content": ai_reply}))

        if to_chat:
            self.chat()

        return ai_reply
