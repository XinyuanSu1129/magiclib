"""
magiclib / generator

Attention:
1. statements should be abbreviated
2. Pay attention to the usage traffic of API keys
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

        self.reasoning_steps = reasoning_steps
        self.precision = precision
        self.cache_enabled = cache_enabled

    # 与 DeepSeek 聊天
    def chat(self) -> None:
        """
        与 DeepSeek 的 deepseek-chat 模型聊天，在优惠时段 (北京时间 00:30-08:30) 时改用 deepseek-reasoner 模型
        chat with DeepSeek's deepseek-chat model and switch to the deepseek-reasoner model during the promotional
        period (00:30-08:30 Beijing time)

        :return: None
        """

        # 获取当前北京时间
        beijing_time = datetime.now(ZoneInfo(f"Asia/{current_time_zone_location}"))
        current_hour_min = (beijing_time.hour, beijing_time.minute)

        # 定义时间区间 [00:30, 08:30]
        start_time = (0, 30)  # 00:30
        end_time = (8, 30)  # 08:30

        # 判断是否在优惠时间 (Beijing 00:30-08:30) 内
        if start_time <= current_hour_min < end_time:
            self.model = 'deepseek-reasoner'
            hour, minute = datetime.now().hour, datetime.now().minute
            print(f'The current time is \033[34m{hour}:{minute}\033[0m, and the model has been converted to '
                  f'\033[31m{self.model}\033[0m.')

        while True:
            # 获取用户输入
            user_input = input("\033[1m\033[92mUser\033[0m: ")

            # 退出条件
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("\nThe conversation is over. Goodbye!")
                break

            # 添加用户消息到对话历史
            self.messages.append({"role": "user", "content": user_input})

            # 调用 DeepSeek API
            self.response = self.client.chat.completions.create(
                model=self.model,
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

            # 打印并保存 AI 回复
            print(f"\n\033[1m\033[95mDeepSeek\033[0m:\033[35;2m {ai_reply}\033[0m\n")
            self.messages.append(dict({"role": "assistant", "content": ai_reply}))

        return None
