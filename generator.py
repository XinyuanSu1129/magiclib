"""
magiclib / generator

Attention:
1.  statements should be abbreviated
2.  Pay attention to the usage traffic of API keys
3.  The character is added each time, and the price is calculated last
"""


# 导入顺序不同有可能导致程序异常
from . import general  # 当前未用到，预计参与分析时用到

import inspect
from openai import OpenAI
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import abstractmethod
from typing import Union, Optional, List, Dict, Callable, Tuple

# 所用时区
current_time_zone_location = 'Shanghai'

# DeepSeek
DeepSeek_api_key = 'sk-cc2167b962444015a28d989478add7eb'  # MiaomiaoSu from 2025-08-07 ¥20
DeepSeek_base_url = 'https://api.deepseek.com/v1'

# Avaliable large AI models 1
other_api_key = 'sk-flk6RxbjuWqApkKaU0DUJDP3FsG6QBI2hjHkwRRyU6briHqZ'  # DeepSeek-R1-671B to 2025-09-09
other_base_url = 'https://lmhub.fatui.xyz/v1'
avaliable_model = ['deepseek-ai/DeepSeek-R1',  # DeepSeek
                   'gpt-oss-120b',  # ChatGPT
                   'gemini-2.5-pro',  # Gemini
                   'moonshot-v1-128k',  # Moonshot
                   'glm-4-0520',  # 智谱
                   'Qwen/QVQ-72B-Preview',  # 通义千问
                   'abab6.5s',  # MiniMax
                   'baidu/ERNIE-4.5-300B-A47B',  # 文心一言
                   'SparkDesk-4.0Ultra',  # 讯飞星火
                   'hunyuan-large-longcontext',  # 腾讯混元
                   'command-r-plus',  # Cohere
                   'yi-large',  # 零一万物
                   'mistral-large-pixtral-2411',  # Mistral AI
                   'meta-llama/llama-4-maverick-17b-128e-instruct',  # Llama
                   'doubao-1.5-thinking-pro',  # 豆包
                   ]

""" AI 大模型总类 """
class AI:
    """
    AI 大模型公有参数部分

    This section contains the public parameters and methods of the AI large model class.
    The methods should include chat().
    """

    # 公有参数初始化
    def __init__(self, instance_id: Optional[str] = None, client: Optional[OpenAI] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, max_tokens: int = 2048, temperature: float = 0.7,
                 top_p: float = 1.0, n: int = 1, stream: bool = False, stop: Union[str, list, None] = None,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
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
        self.instance_id = instance_id
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

        # 输出信息
        self.response = None

        # 初始化字符统计计数器
        self.total_input_tokens = 0  # 用户输入总 tokens
        self.total_output_tokens = 0  # AI 模型输出总 tokens

    # 确保可以从实例变量中找到 instance_id
    def __repr__(self):
        return f"{self.instance_id}"

    # 与 AI 模型对话
    @abstractmethod
    def chat(self):
        pass


""" 真人模型 """
class Human:
    """
    此为真人交互时占用类

    It can be interaction among multiple users, achieving interaction between users and between users
    and large AI models.
    """

    # 初始化
    def __init__(self, instance_id: Optional[str] = None, model: Optional[str] = None):
        """
        Human 类参数初始化，主要参数需要与类 AI 相同
        Initialize the parameters of the Human class. The main parameters need to be the same as those of the class AI.

        :param instance_id: (str) 实例 id
        :param model: (str) 无模型，固定为 None
        """

        self.instance_id = instance_id
        self.model = model

        # 输入信息
        self.messages = None

        # 输出信息
        self.response = None

    # 确保可以从实例变量中找到 instance_id
    def __repr__(self):
        return f"{self.instance_id}"

    # 用户输入内容
    def chat(self, messages: Optional[List[dict]] = None) -> str:
        """
        用户收到信息，返回信息，仅一次，不会循环
        The user receives the message and returns it only once, without any loops.

        :param messages: (List[dict]) 用户收到的信息，用户收到信息中 'system' 将突出显示，'user' 将为主要内容

        :return human_reply: (str) 用户返回的信息，返回的信息将保存在 self.messages 中的最后一条并以 'assistant' 的身份保存
        """

        # 检查 messages 是否输入
        if messages is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                            f"messages cannot be None.")

        if len(messages) == 0:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"messages cannot be an empty list.")

        self.messages = messages.copy()  # 拷贝一份，避免修改外部列表

        # 打印所有消息内容 (role 和 content），根据角色加颜色
        print("\n\033[3mAll messages\033[0m:")
        for i, msg in enumerate(self.messages, 1):
            role = msg['role']
            content = msg['content']
            if role == 'user':
                print(f"{i}. \033[1m\033[95mUser\033[0m:\033[35;2m {content}\033[0m")
            elif role == 'system':
                print(f"{i}. \033[1m\033[91mSystem\033[0m:\033[31m {content}\033[0m")
            elif role == 'assistant':
                print(f"{i}. \033[1m\033[92mAssistant\033[0m: \033[32m {content}\033[0m")
            else:
                # 其他角色正常打印，无色彩
                print(f"{i}. {role}: {content}")

        # 从 messages 中找到 system 内容
        system_content = ""
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
                break

        # 提示输入
        human_reply = input("\n\033[1m\033[92mAssistant\033[0m: ")

        # 将用户输入以 assistant 角色追加到 self.messages
        self.messages.append({'role': 'assistant', 'content': human_reply})

        # 保存回复内容
        self.response = human_reply

        return human_reply


""" DeepSeek 大模型 """
class DeepSeek(AI):
    """
    DeepSeek

    Use the DeepSeek model for chatting and analysis.
    """

    # DeepSeek 特有参数
    def __init__(self,

                 # 公有参数初始化 (12)
                 instance_id: Optional[str] = None, client: Optional[OpenAI] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, max_tokens: int = 2048, temperature: float = 0.7,
                 top_p: float = 1.0, n: int = 1, stream: bool = False, stop: Union[str, list, None] = None,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0,

                 # DeepSeek 特有参数初始化 (3)
                 reasoning_steps: bool = False, precision: str = 'medium', cache_enabled: bool = True):
        """
        DeepSeek 特有参数初始化
        Initialization of DeepSeek's unique parameters.

        # 公有参数 (12)
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
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

        # DeepSeek 特有参数 (3)
        :param reasoning_steps: (bool) 是否显示推理步骤 (仅限 reasoner 模型)
        :param precision: (str) 计算精度，'low'、'medium' 与 'high'
        :param cache_enabled: (bool) 是不启用响应缓存
        """

        super().__init__(instance_id=instance_id, client=client, model=model, messages=messages,
                         max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop,
                         presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)

        self.api_key = other_api_key    # DeepSeek-R1-671B to 2025-09-09
        self.base_url = other_base_url

        # 检查 OpenAI 实例化
        if client is None:
            self.client = OpenAI(
                api_key=self.api_key,  # API Key
                base_url=self.base_url,  # DeepSeek
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

        # 特有参数初始化 (3)
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
    def chat(self, model: Optional[str] = None, system_content: Optional[str] = None, max_tokens: int = 500,
             temperature: float = 0.7, top_p: float = 1.0, presence_penalty: float = 0.0,
             frequency_penalty: float = 0.0) -> str:
        """
        与 DeepSeek 的指定模型聊天，在优惠时段 (北京时间 00:30-08:30) 时改用 deepseek-reasoner 模型
        Chat with the designated model of DeepSeek and switch to the deepseek-reasoner model during
        the promotional period (00:30-08:30 Beijing time).

        注意：
        1.  想要退出需要输入：'退出', 'exit' 或 'quit'
        2.  只有在空的一行输入换行符 '\n' 或空按“回车”才会将内容输入给 AI 模型，否则只是换到下一行并等待继续输入

        Note:
        1.  To exit, you need to enter: '退出', 'exit' or 'quit'.
        2.  Only when a newline character '\n' is entered on an empty line or an empty "Enter" is pressed will
            the content be input into the AI model; otherwise, it will simply move to the next line and wait
            for further input.

        :param model: (str) 指定使用的模型，如 'deepseek-chat' 或 'deepseek-reasoner'
        :param system_content: (str) 'role': 'system' 中的 content 的内容，被赋值时会消除前面的所有对话记录。
                               如果未赋值则运用初始信息，默认为初始信息
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作

        :return ai_reply: (str) DeepSeek AI 返回的消息
        """

        # 检查赋值
        if model is None:
            model = self.model
        if system_content is not None:
            self.messages = [{"role": "system",
                              "content": system_content}]

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
        ai_reply = ''
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

        return ai_reply

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


""" 其它推理大模型 """
class OtherAI(AI):
    """
    其它小渠道大推理模型

    This part of the model is not necessarily reliable, but it is low in price and can well demonstrate its
    advantages in frequently used scenarios.
    """

    # 公有参数初始化
    def __init__(self, instance_id: Optional[str] = None, client: Optional[OpenAI] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, max_tokens: int = 2048, temperature: float = 0.7,
                 top_p: float = 1.0, n: int = 1, stream: bool = False, stop: Union[str, list, None] = None,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
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

        super().__init__(instance_id=instance_id, client=client, model=model, messages=messages,
                         max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop,
                         presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)

        self.api_key = other_api_key  # DeepSeek-R1-671B to 2025-09-09
        self.base_url = other_base_url

        # 检查 OpenAI 实例化
        if client is None:
            self.client = OpenAI(
                api_key=self.api_key,  # API Key
                base_url=self.base_url,  # DeepSeek
            )

        # 检查 model 实例化
        if model is None:
            self.model = 'deepseek-ai/DeepSeek-R1'

        # 检查 messages 实例化
        if messages is None:
            self.messages = [  # message 是一个 list，包含多个消息对象，最后一个消息对象为当前发的内容
                {"role": "system",  # role 是消息发送者的身份，有 "system", "user" 与 "assistant"
                 "content": "You are a helpful AI assistant who answers users' questions."}  # 消息文本内容
            ]

    #  与 AI 大模型聊天
    def chat(self, model: Optional[str] = None, system_content: Optional[str] = None, max_tokens: int = 500,
             temperature: float = 0.7, top_p: float = 1.0, presence_penalty: float = 0.0,
             frequency_penalty: float = 0.0) -> str:
        """
        与 AI 大模型聊天，该部分语言模型将不会统计输入、输出 tokens 与费用
        When chatting with the AI large model, this part of the language model will not count the input,
        output tokens and fees.

        注意：
        1.  想要退出需要输入：'退出', 'exit' 或 'quit'
        2.  只有在空的一行输入换行符 '\n' 或空按“回车”才会将内容输入给 AI 模型，否则只是换到下一行并等待继续输入

        Note:
        1.  To exit, you need to enter: '退出', 'exit' or 'quit'.
        2.  Only when a newline character '\n' is entered on an empty line or an empty "Enter" is pressed will
            the content be input into the AI model; otherwise, it will simply move to the next line and wait
            for further input.

        :param model: (str) 指定使用的模型，如 'deepseek-chat' 或 'deepseek-reasoner'
        :param system_content: (str) 'role': 'system' 中的 content 的内容，被赋值时会消除前面的所有对话记录。
                               如果未赋值则运用初始信息，默认为初始信息
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作

        :return ai_reply: (str) AI 返回的消息
        """

        # 检查赋值
        if model is None:
            model = self.model
        if system_content is not None:
            self.messages = [{"role": "system",
                              "content": system_content}]

        print(f"Let's start chatting! The current model is \033[31m{model}\033[0m.")

        # 对话循环
        ai_reply = ''
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
                print(f'The conversation is over. Goodbye ^_< !\n')
                break

            # 添加用户消息到对话历史
            self.messages.append({"role": "user", "content": user_input})

            # 调用 API
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
            )

            # 获取 AI 回复
            ai_reply = self.response.choices[0].message.content

            # 打印并保存 AI 回复
            print(f"\033[1m\033[95mAI\033[0m:\033[35;2m {ai_reply}\033[0m\n")
            self.messages.append(dict({"role": "assistant", "content": ai_reply}))

        return ai_reply


""" AI 大模型的应用 """
class Assist(DeepSeek):
    """
    应用各种 AI 大模型完成生产力活动

    Apply various AI large models such as ChatGPT, DeepSeek, Claude, Gemini, Grok, etc., to complete productivity
    activities such as drawing, writing articles, revising articles, and analyzing data.
    """

    # 初始化，应当包含所有 AI 大模型的参数
    def __init__(self,

                 # 公有参数初始化 (12)
                 instance_id: Optional[str] = None, client: Optional[OpenAI] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, max_tokens: int = 2048, temperature: float = 0.7,
                 top_p: float = 1.0, n: int = 1, stream: bool = False, stop: Union[str, list, None] = None,
                 presence_penalty: float = 0.0, frequency_penalty: float = 0.0,

                 # DeepSeek 特有参数初始化 (3)
                 reasoning_steps: bool = False, precision: str = 'medium', cache_enabled: bool = True):
        """
        DeepSeek 特有参数初始化
        Initialization of DeepSeek's unique parameters.

        # 公有参数 (12)
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
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

        # DeepSeek 特有参数 (3)
        :param reasoning_steps: (bool) 是否显示推理步骤 (仅限 reasoner 模型)
        :param precision: (str) 计算精度，'low'、'medium' 与 'high'
        :param cache_enabled: (bool) 是不启用响应缓存
        """

        super().__init__(instance_id=instance_id, client=client, model=model, messages=messages, max_tokens=max_tokens,
                         temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop,
                         presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                         reasoning_steps=reasoning_steps, precision=precision, cache_enabled=cache_enabled)

    # 利用 DeepSeek 模型修改手稿
    def revise_manuscript(self, model: Optional[str] = None, manuscript: Optional[str] = None,
                          advice: Optional[str] = None, to_chat: bool = True) -> str:
        """
        让 DeepSeek 的 AI 大模型协助修改手稿，仅支持单次修改，后续问题需要转移至 chat() 中
        Let the DeepSeek deepseek-reasoner model assist in modifying the manuscript.
        It only supports single modification. Subsequent issues need to be transferred to chat()

        :param model: (str) 修改文章所用模型，默认为 'deepseek-reasoner'
        :param manuscript: (str) 需要修改的稿件内容，该内容为必填项
        :param advice: (str) 意见，往往是审稿人给出的，为选填项
        :param to_chat: (bool) 在生成修改后的文本后是否继续对话，默认为 True

        :return revised_text: (str) 修改后的文本
        """

        if model is None:
            model_revise_manuscript = 'deepseek-reasoner'  # 应用 deepseek-reasoner 模型
        else:
            model_revise_manuscript = model

        if advice is not None:  # 有修改意见
            stipulation = '''# Role
                You are a professional manuscript editing assistant with strong language skills and extensive 
                experience in text optimization. Please refine the language and improve the structure of the 
                manuscript I provide, making it clearer, more natural, and aligned with academic writing standards. 
                Additionally, incorporate the revision requests I provide and make appropriate adjustments or 
                enhancements to the content accordingly.

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
                You are a professional manuscript editing assistant with strong language skills and extensive 
                experience in text optimization. Please refine the language and improve the structure of the 
                manuscript I provide, making it clearer, more natural, and aligned with academic writing standards. 
                Additionally, incorporate the revision requests I provide and make appropriate adjustments or 
                enhancements to the content accordingly.

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


""" AI 大模型互动区 """
class Muse:
    """
    各种 AI 大模型与人类休闲交互区

    Apply various large-scale AI models such as ChatGPT, DeepSeek, Claude, Gemini, and Grok to interact casually
    with humans and enhance inspiration.
    """

    # 初始化，应当包含所有 AI 大模型的参数
    def __init__(self, man_number: int = 0, ai_number: int = 0):
        """
        DeepSeek 特有参数初始化
        Initialization of DeepSeek's unique parameters.

        # 玩家数量配置 (2)
        :param man_number: (int) 真人玩家的数量，默认为 None，表示根据需要分配
        :param ai_number: (int) AI 玩家的数量，默认为 None，表示根据需要分配
        """

        # 玩家配置
        self.player = None

        # 真人玩家参数
        self.man_number = man_number

        # AI 玩家参数
        self.ai_number = ai_number

    # 配置环境，几位真人，几个 AI
    def setup_environment(self, man_number: Optional[int] = None, ai_number: Optional[int] = None,
                          default_ai: str = 'deepseek', show_result: bool = False, **kwargs) -> dict:
        """
        环境配置，有几位真人玩家与几个 AI，多出的 AI 用 DeepSeek 补全
        The environmental configuration includes several real players and several ais.
        The extra ais are completed with DeepSeek.

        :param man_number: (int) 真人玩家的数量，默认为 None，表示根据需要分配
        :param ai_number: (int) AI 玩家的数量，分配 AI 之和的总数需要小于等于 AI 玩家的总数。不足的用 DeepSeek AI 补全。
                                默认为 None，表示根据需要分配
        :param default_ai: (str) 默认的 AI 模型为哪个，即多出的 AI，默认为 DeepSeek，可用的 AI 模型如下：
                           ['deepseek', 'chatgpt', 'gemini', 'moonshot', 'zhipu', 'tongyiqianwen', 'minimax',
                           'wenxinyiyan', 'xunfeixinghuo', 'tengxunhunyuan', 'cohere', 'lingyiwanwu', 'mistral',
                           'llama', 'doubao']
        :param show_result: (bool) 是否打印分配结果，默认为 False

        :return instance: (dict) 实例化的全部玩家，key 值与 instance_id 一样，value 为类 Human 或 AI 对象

        --- **kwargs ---

        # 可用 AI 大模型 (15)
        - deepseek_ai_number: (int) DeepSeek 的 AI 大模型的个数
        - chatgpt_ai_number: (int) ChatGPT 的 AI 大模型的个数
        - gemini_ai_number: (int) Gemini 的 AI 大模型的个数
        - moonshot_ai_number: (int) Moonshot 的 AI 大模型的个数
        - zhipu_ai_number: (int) ZHIPU 的 AI 大模型的个数
        - tongyiqianwen_ai_number: (int) TONGYIQIANWEN 的 AI 大模型的个数
        - minimax_ai_number: (int) MiniMax 的 AI 大模型的个数
        - wenxinyiyan_ai_number: (int) WENXINYIYAN 的 AI 大模型的个数
        - xunfeixinghuo_ai_number: (int) XUNFEIXINGHUO 的 AI 大模型的个数
        - tengxunhunyuan_ai_number: (int) TENGXUNHUNYUAN 的 AI 大模型的个数
        - cohere_ai_number: (int) Cohere 的 AI 大模型的个数
        - lingyiwanwu_ai_number: (int) LINGYIWANWU 的 AI 大模型的个数
        - mistral_ai_number: (int) Mistral 的 AI 大模型的个数
        - llama_ai_number: (int) Llama 的 AI 大模型的个数
        - doubao_ai_number: (int) DOUBAO 的 AI 大模型的个数
        """

        # 玩家数量分配
        if man_number is None:
            man_number = self.man_number
        if ai_number is None:
            ai_number = self.ai_number

        # 检查输入是否均大于 0
        if man_number < 0 or ai_number < 0:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"all parameters must be greater than 0.")

        # 模型种类与完整模型名的映射
        ai_model_map = {
            'deepseek': 'deepseek-ai/DeepSeek-R1',  # DeepSeek
            'chatgpt': 'gpt-oss-120b',  # ChatGPT
            'gemini': 'gemini-2.5-pro',  # Gemini
            'moonshot': 'moonshot-v1-128k',  # Moonshot
            'zhipu': 'glm-4-0520',  # 智谱
            'tongyiqianwen': 'Qwen/QVQ-72B-Preview',  # 通义千问
            'minimax': 'abab6.5s',  # MiniMax
            'wenxinyiyan': 'baidu/ERNIE-4.5-300B-A47B',  # 文心一言
            'xunfeixinghuo': 'SparkDesk-4.0Ultra',  # 讯飞星火
            'tengxunhunyuan': 'hunyuan-large-longcontext',  # 腾讯混元
            'cohere': 'command-r-plus',  # Cohere
            'lingyiwanwu': 'yi-large',  # 零一万物
            'mistral': 'mistral-large-pixtral-2411',  # Mistral AI
            'llama': 'meta-llama/llama-4-maverick-17b-128e-instruct',  # Llama
            'doubao': 'doubao-1.5-thinking-pro',  # 豆包
        }

        instances = {}

        # 实例化 Human 类
        for i in range(man_number):
            human_key = f"human_{i + 1}"
            instances[human_key] = Human(instance_id=human_key, model=None)

        # 校验 default_ai
        if default_ai not in ai_model_map:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"default_ai must be one of the following: {list(ai_model_map.keys())}")

        # 获取各类型数量并校验
        ai_counts = {}
        for ai_type in ai_model_map.keys():
            count = kwargs.get(f"{ai_type}_ai_number", 0) or 0
            if count < 0:
                raise ValueError(f"{ai_type}_ai_number cannot be less than 0.")
            ai_counts[ai_type] = count

        # 校验 ai_number
        total_specified = sum(ai_counts.values())
        if ai_number < total_specified:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"ai_number ({ai_number}) must be greater than or equal to the sum of the quantities "
                             f"of each type ({total_specified}).")

        # 分配多余的到 default_ai
        if ai_number > total_specified:
            ai_counts[default_ai] += ai_number - total_specified

        # 实例化 OtherAI
        for ai_type, model_name in ai_model_map.items():
            for i in range(ai_counts[ai_type]):
                instance_id = f"{ai_type}_{i + 1}"
                instances[instance_id] = OtherAI(instance_id=instance_id, model=model_name)

        # 可选调试输出
        if show_result:
            print("AI allocation result:")
            for ai_type, count in ai_counts.items():
                print(f"{ai_type}: {count}")
            print(f"\nTotal: {man_number + sum(ai_counts.values())}  Human: {man_number}  AI: {ai_number}")

        self.player = instances

        return instances


""" 应用 Duck Typing 来调用实例中的 API 与 URL """


def set_api_config(ai_instance: object, api_url_pair: str):
    """
    可以改动已实例化的 AI 模型中的 API 配置

    :param ai_instance: (object) 已实例化的 AI 模型对象
    :param api_url_pair: (str) API 与 URL 配对的名称
    """

    # 获取当前函数名（用于错误信息）
    method_name = inspect.currentframe().f_code.co_name

    # 预定义的API配置 - 确保这些变量在作用域内
    try:
        api_configs = {
            # DeepSeek 官方
            'ds': {"api_key": DeepSeek_api_key, "base_url": DeepSeek_base_url},
            # 渠道 AI
            '1': {"api_key": other_api_key, "base_url": other_base_url},
        }
    except NameError as e:
        raise NameError(f"\033[95mIn {method_name}\033[0m, Configuration variable not found: {str(e)}") from None

    # 1.  验证配置是否存在
    if api_url_pair not in api_configs:
        available_configs = ", ".join(api_configs.keys())
        # 使用实例的类名而不是当前类的类名
        class_name = ai_instance.__class__.__name__
        raise ValueError(f"\033[95mIn {method_name} for {class_name}\033[0m, "
                         f"The configuration '{api_url_pair}' does not exist. "
                         f"Available configurations: {available_configs}")

    config = api_configs[api_url_pair]

    # 2.  鸭子类型检查 - 确保实例有必要的属性
    required_attrs = {'api_key', 'base_url'}
    missing_attrs = [attr for attr in required_attrs if not hasattr(ai_instance, attr)]

    if missing_attrs:
        # 使用实例的类名
        class_name = ai_instance.__class__.__name__
        raise AttributeError(f"\033[95mIn {method_name} for {class_name}\033[0m, "
                             f"The instance is missing necessary attributes: "
                             f"{', '.join(missing_attrs)}. "
                             f"Required attributes: {', '.join(required_attrs)}")

    # 3.  应用配置
    ai_instance.api_key = config["api_key"]
    ai_instance.base_url = config["base_url"]
