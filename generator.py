"""
magiclib / generator

Attention:
1.  statements should be abbreviated
2.  Pay attention to the usage traffic of API keys
3.  The character is added each time, and the price is calculated last
"""


# 导入顺序不同有可能导致程序异常
from . import general, grapher

import os
import time
import json
import base64
import inspect
import requests
from PIL import Image
from io import BytesIO
import simpleaudio as sa
from pandas import DataFrame
from datetime import datetime
from zoneinfo import ZoneInfo
from abc import abstractmethod
import matplotlib.pyplot as plt
from pydub import AudioSegment
from typing import Union, Optional, List, Dict, Callable, Tuple

# 所用时区
current_time_zone_location = 'Shanghai'

# DeepSeek
DeepSeek_api_key = 'sk-cc2167b962444015a28d989478add7eb'  # MiaomiaoSu from 2025-08-07 ¥20
DeepSeek_base_url = 'https://api.deepseek.com/v1'

# Avaliable large AI models
other_api_key = 'sk-flk6RxbjuWqApkKaU0DUJDP3FsG6QBI2hjHkwRRyU6briHqZ'  # DeepSeek-R1-671B to 2025-09-09
other_base_url = 'https://lmhub.fatui.xyz/v1'
avaliable_model = [
    # Avaliable & Free
    'deepseek-ai/DeepSeek-R1'  # DeepSeek
    'deepseek-ai/DeepSeek-V3'  # DeepSeek
    'gemini-2.5-flash',  # Gemini
    'gemini-2.0-flash',  # Gemini
    'gpt-oss-120b',  # ChatGPT
    'command-r-plus'  # Cohere
    'mistral-large-latest'  # Mistral AI
    'mistral-large-pixtral-2411'  # Mistral AI
    'doubao-1.5-thinking-pro',  # 豆包
    'doubao-1.5-ui-tars',  # 豆包
    'glm-4-0520',  # 智谱清言
    'internvl3-latest',  # 浦源书生
    'internlm3-latest',  # 浦源书生
    'hunyuan-large-vision',  # 腾讯混元
    'hunyuan-large-longcontext',  # 腾讯混元
    'SparkDesk-4.0Ultra',  # 讯飞星火
    'Qwen/Qwen3-235B-A22B-Instruct-2507',  # 通义千问
    'Qwen/Qwen3-235B-A22B-Thinking-2507',  # 通义千问
    'yi-large',  # 零一万物
    'yi-large-fc',  # 零一万物
    'yi-large-preview',  # 零一万物

    # Generate image
    'black-forest-labs/FLUX.1-pro',  # Black Forest Lab  目前无法使用
    'black-forest-labs/FLUX.1-dev',  # Black Forest Lab
    'black-forest-labs/FLUX.1-schnell',  # Black Forest Lab
    'LoRA/black-forest-labs/FLUX.1-dev',  # Black Forest Lab
    'stabilityai/stable-diffusion-3-5-large',  # StabilityAI
    'stabilityai/stable-diffusion-xl-base-1.0',  # StabilityAI
    'Kwai-Kolors/Kolors'  # Kolors
    
    # Generate voice  目前不可用
    'fishaudio/fish-speech-1.4',  # FishAudio
    'fishaudio/fish-speech-1.5',  # FishAudio
    'FunAudioLLM/CosyVoice2-0.5B',  # CosyVoice

    # Unavaliable
    'gemini-2.5-pro',  # Gemini
    'meta-llama/llama3-70b-8192'  # Meta
    'MiniMaxAI/MiniMax-M1-80k'  # MiniMax
    'abab6.5s',  # MiniMax
    'distil-whisper-large-v3-en',  # ChatGPT
    'stabilityai/stable-diffusion-3-5-large',  # StabilityAI
    'BAAI/bge-reranker-v2-m3',  # 智源研究院
    'moonshotai/Kimi-Dev-72B',  # 月之暗面
    'moonshot-v1-128k',  # 月之暗面
    'zyAI/netzy-latest',  # 正一 AI
    'baidu/ERNIE-4.5-300B-A47B',  # 文心一言
    'stepfun-ai/step3',  # 阶跃星辰

    # Charging
    'claude-sonnet-4',  # Claude

    # Unkonw or other set
    'LoRA/black-forest-labs/FLUX.1-dev'  # Black Forest Lab
    'fishaudio/fish-speech-1.5'  # FishAudio
    'meta-llama/llama-4-maverick-17b-128e-instruct',  # Llama
]


""" AI 工具包 """
class Tools:
    """
    AI 大模型可以调用此部分的工具

    The AI invokes the tools of this part when the model is permitted. This part may accept class attributes
    and proceed step by step, as well as static methods.
    """

    # 无需初始化，被调用时赋值
    def __init__(self):

        # 实例化类 grapher.Plotter
        self.graphter_instanced = grapher.Plotter()

        # generate_image()
        self.image_data_list = []  # 图片的 list，base64格式
        self.image_list = []  # save_image()
        self.save_path = None  # save_image()
        self.image_seed = None
        self.response_created = None
        self.response_total_tokens = 0
        self.response_input_tokens = 0
        self.response_output_tokens = 0
        self.response_text_tokens = 0
        self.response_image_tokens = 0

        # generate_voice()
        self.voice = None

    # 读取 TXT 文件
    def read_txt(self, txt_path: str) -> str:
        """
        让 AI 大模型读取 TXT 中的数据，返回为数据的 data_dic。读取时的数据为两例并用空格分割
        Let the AI large model read the data in TXT and return the data_dic of the data.
        The data at the time of reading consists of two examples separated by Spaces.

        :param txt_path: (str) TXT 文件路径，可以是文件路径，也可以是目录，若被赋值则对该路径文件 / 目录进行处理

        :return status: (str) 返回信息，让 AI 大模型明白数据已经读取
        """

        self.graphter_instanced.read_txt(txt_path=txt_path)
        status = 'The data in the TXT file has been read.'

        return status

    # 读取 Excel 文件
    def read_excel(self, excel_path: str) -> str:
        """
        让 AI 大模型读取 Excel 中的数据，返回为数据的 data_dic。读取时无行索引且第一列为表头
        Let the AI large model read the data in Excel and return the data_dic of the data. When reading,
        there is no row index and the first column is the header of the table.
        The data at the time of reading consists of two examples separated by Spaces.

        :param excel_path: (str) Excel 文件路径，可以是文件路径，也可以是目录，若被赋值则对该路径文件 / 目录进行处理

        :return status: (str) 返回信息，让 AI 大模型明白数据已经读取
        """

        self.graphter_instanced.read_excel(excel_path=excel_path)
        status = 'The data in the Excel file has been read.'

        return status

    # 读取 JSON 文件
    def read_json(self, json_path: str) -> str:
        """
        让 AI 大模型读取 JSON 中的数据，该数据需要为 magiclib 保存的 JSON 格式
        Let the AI large model read the data in JSON, which needs to be in the JSON format saved for magiclib.

        :param json_path: (str) Excel 文件路径，可以是文件路径，也可以是目录，若被赋值则对该路径文件 / 目录进行处理

        :return status: (str) 返回信息，让 AI 大模型明白数据已经读取
        """

        self.graphter_instanced.read_json(json_path=json_path)
        status = 'The data in the JSON file has been read.'

        return status

    # 绘制线形图
    def plot_line(self):
        """
        根据读取到的数据绘制线形图
        Draw a line graph based on the read data.

        :return status: (str) 返回信息，让 AI 大模型明白线形图已绘制
        """

        self.graphter_instanced.plot_line()
        status = 'The line graph has been drawn.'

        return status

    # 绘制散点形图
    def plot_scatter(self):
        """
        根据读取到的数据绘制散点图
        Draw a scatter graph based on the read data.

        :return status: (str) 返回信息，让 AI 大模型明白散点图已绘制
        """

        self.graphter_instanced.plot_scatter()
        status = 'The scatter plot has been drawn.'

        return status

    # 生成图片  用到 AI 大模型
    def generate_image(self, prompt: str, save_path: Optional[str] = None, model: str = 'black-forest-labs/FLUX.1-dev',
                       size: str = '1024x1024', n: int = 1, seed: int = None) -> str:
        """
        根据用户要求生成图片
        Generate images according to user requirements.

        :param prompt: (str) 生成图片的英文提示词，为必需输入项。注意：在输入前必需将 prompt 转换成英文！
        :param save_path: (str) 保存的目录路径，若输入则按照路径保存
        :param model: (str) 生成图片的模型，默认为 Black Forest Lab 的 black-forest-labs/FLUX.1-dev
        :param size: (str) 图片的大小，最大 '2048x2048'，默认为 '1024x1024'
        :param n: (int) 生成图片的数量，默认为 1。注意：目前只能为 1
        :param seed: (int) 随机种子，默认为无种子

        :return status: (str) 图片绘制成功与否的信息
        """

        # =============== 注意更新 ===============
        generate_image_api_key = other_api_key
        generate_image_base_url = other_base_url
        generate_image_model = model
        # =============== 注意更新 ===============

        # 检查输入
        self.image_seed = seed
        self.save_path = save_path

        # 构建 URL
        base_url = generate_image_base_url
        endpoint = '/images/generations'
        generate_image_url = base_url + endpoint

        # 构建 headers
        headers = {
            "Authorization": f"Bearer {generate_image_api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体
        request_body = {
            "model": generate_image_model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": "b64_json",
            "seed": seed
        }

        # 发送请求
        response = requests.post(url=generate_image_url, headers=headers, json=request_body)
        response_dic = response.json()  # 转化为 JSON格式

        # 判断请求是否成功
        if response.status_code != 200:
            message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
            status = f"\033[31mRequest failed!\033[0m status_code: {response.status_code} ({message})"
            return status

        # 创建实例属性
        self.response_created = response_dic.get('created', None)

        # 总 tokens
        self.response_total_tokens = response_dic.get('usage', {}).get('total_tokens', 0)
        self.response_input_tokens = response_dic.get('usage', {}).get('input_tokens', 0)
        self.response_output_tokens = response_dic.get('usage', {}).get('output_tokens', 0)

        # 细节 tokens
        input_tokens_details = response_dic.get('usage', {}).get('input_tokens_details', {})
        self.response_text_tokens = input_tokens_details.get('text_tokens', 0)
        self.response_image_tokens = input_tokens_details.get('image_tokens', 0)

        self.image_data_list = response_dic.get('data', [])
        result_list = []
        if self.image_data_list:
            for idx, img_item in enumerate(self.image_data_list):

                # 处理 b64_json 的格式
                if "b64_json" in img_item and img_item["b64_json"]:
                    b64_img = img_item["b64_json"]
                    # 去掉前缀
                    if b64_img.startswith("data:image"):
                        b64_img_clean = b64_img.split(",")[1]
                    else:
                        b64_img_clean = b64_img
                    result_list.append(b64_img_clean)

                # 处理 url 的格式
                elif "url" in img_item and img_item["url"]:
                    result_list.append(img_item["url"])

                # 处理其它格式
                else:
                    result_list.append("No valid image data found")

        self.image_list = result_list

        status = 'No pictures were generated.'
        for idx, b64_data in enumerate(result_list):  # 遍历多个 Base64 图片

            # 显示图片
            image_bytes = base64.b64decode(b64_data)
            image = Image.open(BytesIO(image_bytes))

            # 创建没有边距的 figure
            fig = plt.figure(figsize=(image.width / 100, image.height / 100), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])  # 直接占满
            ax.set_axis_off()
            fig.add_axes(ax)

            ax.imshow(image)
            plt.show()

            # 保存图片
            if save_path is not None:  # 如果 save_path 的值不为 None，则保存
                status = f'The image has been generated and saved in the {save_path} directory.'
                file_name = "Generate_image.png" if idx == 0 else f"Generate_image_{idx}.png"
                full_file_path = os.path.join(save_path, file_name)

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = f"Generate_image_{idx}_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)

                    while os.path.exists(full_file_path):  # 不断递增，直到文件名不重复
                        count += 1
                        file_name = f"Generate_image_{idx}_{count}.png"
                        full_file_path = os.path.join(save_path, file_name)

                # 保存 Base64 图片
                with open(full_file_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))

            else:  # 如果 save_path 的值为 None
                status = 'The image has been successfully generated but not saved.'

        return status

    # 保存图片
    def save_image(self, save_path: Optional[str] = None) -> str:
        """
        将上一张用 generate_image() 生成的图片保存下来，需要在前文或此处提供保存路径
        TTo save the previous image generated by generate_image(), you need to provide the save path in
        the previous text or here.

        :param save_path: (str) 保存的目录路径，若输入则按照路径保存，没有则找之前是否输入过路径

        :return status: (str) 图片保存成功与否的信息
        """

        # 检查参数
        if not self.image_data_list:   # 是否有图片需要被保存
            status = 'No pictures need to be saved.'
            return status

        if save_path is None:  # 保存路径是否被提供
            if self.save_path is None:
                status = 'No save path is provided.'
                return status
            else:
                save_path = self.save_path

        for idx, b64_data in enumerate(self.image_list):  # 遍历多个 Base64 图片
            file_name = "Generate_image.png" if idx == 0 else f"Generate_image_{idx}.png"
            full_file_path = os.path.join(save_path, file_name)

            if os.path.exists(full_file_path):  # 查看该文件名是否存在
                count = 1
                file_name = f"Generate_image_{idx}_{count}.png"
                full_file_path = os.path.join(save_path, file_name)

                while os.path.exists(full_file_path):  # 不断递增，直到文件名不重复
                    count += 1
                    file_name = f"Generate_image_{idx}_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)

            # 保存 Base64 图片
            with open(full_file_path, "wb") as f:
                f.write(base64.b64decode(b64_data))

        status = f'The image has been saved in the {save_path} directory.'

        return status

    # 生成声音 用到 AI 大模型
    def generate_voice(self, text: str, save_path: Optional[str] = None, model: str = 'fishaudio/fish-speech-1.5',
                       voice: str = 'alloy', temperature: float = 1, speech_rate: float = 1, pitch: float = 1) -> str:
        """
        将输入的文本转化为语音
        Convert the input text into speech.

        :param text: (str) 需要转化为语音的文本
        :param save_path: (str) 保存的目录路径，若输入则按照路径保存
        :param model: (str) 所用 AI 大模型，默认为 fishaudio/fish-speech-1.5
        :param voice: (str) 语音风格，可选 'alloy', 'echo', 'shimmer'，默认为 'alloy'
        :param temperature: (float) 随机性，0-1，越低随机性越低，默认为 0.7
        :param speech_rate: (float) 语速，默认 1.0
        :param pitch: (float) 语调，默认 1.0

        :return status: (str) 声音生成的信息
        """

        # =============== 注意更新 ===============
        generate_voice_api_key = other_api_key
        generate_voice_base_url = other_base_url
        generate_voice_model = model
        # =============== 注意更新 ===============

        # 检查输入
        self.save_path = save_path

        # 构建 URL
        base_url = generate_voice_base_url
        endpoint = '/audio/speech'
        generate_image_url = base_url + endpoint

        # 构建 headers
        headers = {
            "Authorization": f"Bearer {generate_voice_api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体
        request_body = {
            "model": generate_voice_model,
            "voice": voice,
            "input": text,
            "format": "mp3",
            "temperature": temperature,
            "speech_rate": speech_rate,
            "pitch": pitch
        }

        # 发送请求
        response = requests.post(url=generate_image_url, headers=headers, json=request_body)

        # 判断请求是否成功
        if response.status_code != 200:
            message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
            status = f"\033[31mRequest failed!\033[0m status_code: {response.status_code} ({message})"
            return status

        # 假设 response.content 是从 /v1/audio/speech 得到的 MP3 二进制
        audio_bytes = response.content
        self.voice = audio_bytes

        # 用 pydub 读取 MP3
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")

        # 播放
        play_obj = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        play_obj.wait_done()  # 等待播放结束

        # 保存音频
        if save_path is not None:  # 如果 save_path 的值不为 None，则保存
            status = f'The voice has been played and saved in {save_path}. Continue the conversation'
            file_name = "Generate_voice.png" if idx == 0 else f"Generate_voice_{idx}.png"
            full_file_path = os.path.join(save_path, file_name)

            if os.path.exists(full_file_path):  # 查看该文件名是否存在
                count = 1
                file_name = f"Generate_voice_{idx}_{count}.png"
                full_file_path = os.path.join(save_path, file_name)

                while os.path.exists(full_file_path):  # 不断递增，直到文件名不重复
                    count += 1
                    file_name = f"Generate_voice_{idx}_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)

            # 保存 Base64 图片
            with open(full_file_path, "wb") as f:
                f.write(response.content)

        else:  # 如果 save_path 的值为 None
            status = "The voice has finished playing. Continue the conversation."

        return status


""" AI 大模型总类 """
class AI:
    """
    AI 大模型公有参数部分

    This section contains the public parameters and methods of the AI large model class.
    The methods should include continue_chat().
    """

    # 创建 Tools 实例
    tools_instance = Tools()
    # 可用工具及描述
    toolkit = [
        {
            "type": "function",
            "function": {
                "name": "read_txt",
                "description": "Read a TXT file and save a dictionary of data, with the file name as the key and "
                               "DataFrame as the value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "txt_path": {
                            "type": "string",
                            "description": "Path to the TXT file or directory."
                        }
                    },
                    "required": ["txt_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_excel",
                "description": "Read an Excel file and save a dictionary of data, with the file name as the key and "
                               "DataFrame as the value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "excel_path": {
                            "type": "string",
                            "description": "Path to the Excel file or directory."
                        }
                    },
                    "required": ["excel_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_json",
                "description": "Read a JSON file in magiclib format and save a dictionary of data, with "
                               "the file name as the key and DataFrame as the value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "json_path": {
                            "type": "string",
                            "description": "Path to the JSON file or directory."
                        }
                    },
                    "required": ["json_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "plot_line",
                "description": "Draw a line plot based on the data previously read. No parameters required.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "plot_scatter",
                "description": "Draw a scatter plot based on the data previously read. No parameters required.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate images based on the prompts provided by the user. You can appropriately "
                               "supplement the details of the pictures, but the content needs to be converted into "
                               "English. You can choose to save the generated image to the local directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The English prompt for generating images. Must be provided and in English."
                        },
                        "save_path": {
                            "type": ["string", "null"],
                            "description": "Directory path to save the generated images. If not provided, "
                                           "images will not be saved."
                        },
                        "model": {
                            "type": "string",
                            "description": "The image generation model to use. Default is "
                                           "'black-forest-labs/FLUX.1-dev'."
                        },
                        "size": {
                            "type": "string",
                            "description": "Image size, maximum '2048x2048'. Default is '1024x1024'."
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of images to generate. Currently only 1 is supported.",
                            "default": 1
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Random seed for reproducible results. If set to a fixed integer, the same "
                                           "prompt will produce the same image across multiple runs (given the same "
                                           "model and parameters). If omitted, a random seed is used by default."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_image",
                "description": "To save the previous image generated by generate_image(), you need to provide the "
                               "save path in the previous text or here.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "save_path": {
                            "type": "string",
                            "description": "Directory path to save images. Optional. If not provided, the previously "
                                           "set path will be used."
                        }
                    },
                    "required": []
                }
            }
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "generate_voice",
        #         "description": "Convert text to speech and you can choose to save it.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "text": {
        #                     "type": "string",
        #                     "description": "The text content to be converted into speech."
        #                 },
        #                 "save_path": {
        #                     "type": "string",
        #                     "description": "Optional directory path to save the generated voice file. "
        #                                    "If not provided, the voice will only be played."
        #                 },
        #                 "model": {
        #                     "type": "string",
        #                     "description": "The AI voice model to use, default is 'fishaudio/fish-speech-1.5'."
        #                 },
        #                 "voice": {
        #                     "type": "string",
        #                     "description": "Voice style for speech synthesis, options include 'alloy', 'echo', "
        #                                    "'shimmer'. Default is 'alloy'."
        #                 },
        #                 "temperature": {
        #                     "type": "number",
        #                     "description": "Randomness of voice generation, 0-1. Lower values produce more "
        #                                    "deterministic results. Default is 1."
        #                 },
        #                 "speech_rate": {
        #                     "type": "number",
        #                     "description": "Speech rate, default is 1.0."
        #                 },
        #                 "pitch": {
        #                     "type": "number",
        #                     "description": "Pitch of the generated voice, default is 1.0."
        #                 }
        #             },
        #             "required": ["text"]
        #         }
        #     }
        # },
    ]
    # 工具与对应方法
    tool_methods = {
        "read_txt": tools_instance.read_txt,
        "read_excel": tools_instance.read_excel,
        "read_json": tools_instance.read_json,
        "plot_line": tools_instance.plot_line,
        "plot_scatter": tools_instance.plot_scatter,
        "generate_image": tools_instance.generate_image,
        "save_image": tools_instance.save_image,
        # "generate_voice": tools_instance.generate_voice,
    }

    # 常见 HTTP 状态码说明
    status_code_messages = {
        200: "OK - Request succeeded",  # 请求成功
        201: "Created - Resource successfully created",  # 已成功创建资源
        202: "Accepted - Request accepted but not yet processed",  # 已接受请求，但尚未处理完成
        204: "No Content - Successful request but no content returned",  # 请求成功，但无返回内容
        301: "Moved Permanently - Resource permanently redirected",  # 资源永久重定向
        302: "Found - Resource temporarily redirected",  # 资源临时重定向
        304: "Not Modified - Cached version is still valid",  # 缓存未更新
        400: "Bad Request - Invalid request parameters or format",  # 请求参数错误或格式不正确
        401: "Unauthorized - Authentication failed or missing",  # 未授权或身份验证失败
        403: "Forbidden - Access to the resource is denied",  # 无权限访问资源
        404: "Not Found - Resource not found",  # 资源不存在
        408: "Request Timeout - The request took too long",  # 请求超时
        429: "Too Many Requests - Rate limit exceeded",  # 请求次数过多，被限流
        500: "Internal Server Error - Server encountered an error",  # 服务器内部错误
        502: "Bad Gateway - Invalid response from upstream server",  # 网关错误
        503: "Service Unavailable - Server temporarily unavailable",  # 服务器暂时不可用（过载或维护中）
        504: "Gateway Timeout - Upstream server took too long to respond"  # 网关超时
    }

    # 公有参数初始化
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, ai_keyword: Optional[str] = None,
                 instance_id: Optional[str] = None, information: Optional[str] = None,
                 max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 1.0, n: int = 1, stream: bool = False,
                 stop: Union[str, list, None] = None, presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                 seed: Optional[int] = None, tools: Optional[list] = None, tool_choice: str = "auto"
                 ):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        # 必要参数 (4)
        :param api_key: (str) 输入的 API KEY，即 API 密钥
        :param base_url: (str) 输入的 base URL
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 与 'deepseek-reasoner'
        :param messages: (list) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息

        # 自定义参数 (3)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型

        # 附加参数 (11)
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param n: (int) 生成多少个独立回复选项 (消耗 n 倍 token)，如 n=3 会返回 3 种不同回答
        :param stream: (bool) 是否启用流输出 (逐字返回)
        :param stop: (str / list) 停止生成的标记，遇到这些字符串时终止输出
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作
        :param seed: (int) 随机种子，默认为 None，表示完全随机
        :param tools: (list) 工具包，默认为无工具，[]，为防止可变实参，因而为 None
        :param tool_choice: (str) 工具选取方式，"auto" 为自动选取，"none" 为决不会选取"，
                            {"type": "function", "function": {"name": "xxx"}} 为强制调用指定工具，并且只能调用它。
                            "required"(部分文档称为 {"type": "function", "function": "required"} 的形式)，
                            模型必须调用某个工具，但可以自己选择哪一个

        # 数据结构
        1.  一般文本对话时的 messages 结构:
        messages = [  # message 是一个 list，包含多个消息对象，最后一个消息对象为当前发的内容
                    {"role": "system",   # role 是消息发送者的身份，有 "system", "user", "assistant", ("tool")
                     "content": "You are a helpful AI assistant who answers users' questions."}  # 消息文本内容
                    ]
        2.  收到带有 tool 请求的 messages:
        messages = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_001",
                        "function": {
                            "name": "my_function",
                            "arguments": "{\"x\":3, \"y\":5}"
                            }
                        }]
                    }
        3.  返回带有 tool 请求的 messages:
        {
          "role": "tool",
          "tool_call_id": "call_001",
          "content": "{\"result\": 8}"
        }
        """

        # 必要参数 (4)
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = other_api_key

        if base_url is not None:
            self.base_url = base_url
        else:
            self.base_url = other_base_url

        if model is not None:
            self.model = model
        else:
            self.model = 'deepseek-ai/DeepSeek-R1'

        if messages is not None:
            self.messages = messages
        else:
            self.messages = [{"role": "system", "content": "You are a helpful assistant. You will kindly answer "
                                                           "users' messages and use tools at the appropriate time."}]

        # 自定义参数 (3)
        self.ai_keyword = ai_keyword
        self.instance_id = instance_id
        self.information = information

        # 附加参数 (11)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n  # 一般模型不用该参数
        self.stream = stream  # 一般模型不用该参数
        self.stop = stop  # 一般模型不用该参数
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        if tools is None:  # 为防止可变实参，因而为 None
            self.tools = AI.toolkit
        self.tool_choice = tool_choice

        # 输入参数 (3)
        self.target_url = None
        self.headers = None
        self.request_body_dict = None

        # 输出参数 (11)
        self.response = None
        self.response_id = None
        self.response_model = None
        self.response_object = None
        self.response_created = None
        self.response_content = None
        self.response_finish_reason = None
        self.response_index = None
        self.response_prompt_tokens = 0
        self.response_completion_tokens = 0
        self.response_total_tokens = 0

        # 颜色设置
        self.system_role_color = '\033[91m'  # 亮红色
        self.system_content_color = '\033[31m'  # 红色
        self.system_remark_color = '\033[31;2m'  # 暗红色
        self.user_role_color = '\033[92m'  # 亮绿色
        self.user_content_color = '\033[32m'  # 绿色
        self.user_remark_color = '\033[32;2m'  # 暗绿色
        self.assistant_role_color = '\033[95m'  # 亮粉色
        self.assistant_content_color = '\033[35;2m'  # 粉色
        self.assistant_remark_color = '\033[35;2m'  # 暗粉色
        self.tool_role_color = '\033[94m',  # 亮蓝色
        self.tool_content_color = '\033[34m'  # 蓝色
        self.tool_remark_color = '\033[34;2m',  # 暗蓝色

        self.bold = '\033[1m'  # 加粗
        self.system_remind = '\033[90m'  # 亮黑色
        self.end_style = '\033[0m'  # 还原

        # 费用计算
        self.total_cost = 0  # 总价格
        self.input_price_per_million_tokens = 4  # 输入价格 (百万 tokens) 默认为 DeepSeek-R1 模型的费用，均按照未命中处理
        self.output_price_per_million_tokens = 16  # 输出价格 (百万 tokens)

        # 其他参数
        self.stream_begin_output = True  # stream 开始返回信息

    # 确保可以从实例变量中找到 instance_id
    def __repr__(self):
        return f"{self.instance_id}"

    # 与 AI 大模型聊天
    def chat(self, messages: Optional[List[dict]] = None, print_response: bool = True) -> List[dict]:
        """
        与 AI 大模型聊天，单次交互，传入完整 messages，返回 AI 回复并保存
        Chat with the AI large model, with only one interaction, return one AI response and save it.

        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param print_response: (bool) 是否打印返回的 content，默认为 True

        :return ai_reply: (list) AI 返回的消息列表 messages
        """

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages

        # 构建 URL
        endpoint = "/chat/completions"
        target_url = self.base_url + endpoint

        # 构建 headers
        headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.headers = headers

        # 构建请求体
        request_body_dict = {
            # 重要参数
            "model": self.model,
            "messages": self.messages,
            # 文本参数
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            # 种子参数
            "seed": self.seed,
            # 工具参数
            "tools": self.tools,
            "tool_choice": self.tool_choice
        }

        # 发送请求
        response = requests.post(url=target_url, headers=headers, json=request_body_dict)

        # 判断请求是否成功
        if response.status_code != 200:
            message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
            print(f"\033[31mRequest failed!\033[0m status_code: {response.status_code} ({message})")

        # 解析返回参数
        response_dict = response.json()
        self.response_id = response_dict.get("id")  # 本次请求的唯一标识符
        self.response_model = response_dict.get("model")  # 使用的模型名称
        self.response_object = response_dict.get("object")  # 返回对象类型，一般是 "chat.completion"
        self.response_created = response_dict.get("created")  # 创建时间，Unix 时间戳

        choices = response_dict.get("choices", [])  # 返回的回答列表，通常只有一个
        if choices:
            first_choice = choices[0]
            self.response_content = first_choice.get("message", {}).get("content")  # AI 生成的回答文本
            self.response_finish_reason = first_choice.get("finish_reason")  # 结束原因，如 "stop"
            self.response_index = first_choice.get("index")  # 选项索引

        usage = response_dict.get("usage", {})  # 令牌使用情况统计
        self.response_prompt_tokens += usage.get("prompt_tokens", 0)  # 提示词消耗的 tokens 数
        self.response_completion_tokens += usage.get("completion_tokens", 0)  # 生成内容消耗的 tokens 数
        self.response_total_tokens += usage.get("total_tokens", 0)  # 总共消耗的 tokens 数

        # 打印回复
        if print_response:
            print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                  f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

        # 保存 AI 回复为 assistant 角色追加到 self.messages
        self.messages.append({"role": "assistant", "content": self.response_content})

        reply_messages = self.messages

        return reply_messages

    #  与 AI 大模型持续聊天
    def continue_chat(self, system_content: Optional[str] = None, messages: Optional[List[dict]] = None,
                      end_token: str = '', stream: bool = True) -> List[dict]:
        """
        与 AI 大模型连续聊天，支持流式，可使用类 Tools 中的工具
        Continuous chatting with AI large models, supporting streaming, can use the Tools in the class Tools.

        注意：
        1.  想要退出需要输入：'退出', 'exit' 或 'quit'
        2.  end_token 默认情况下，只有在空的一行输入换行符 '\n' 或空按“回车”才会将内容输入给 AI 模型，否则只是换到下一行并等待继续输入，
            此情况下最下面的换行符 \n 不会保留
        3.  允许使用类 Tools 中的方法。顺序为：user-assistant-tool-assistant

        Note:
        1.  To exit, you need to enter: '退出', 'exit' or 'quit'.
        2.  By default, the content of end_token will only be input to the AI model when a newline character
            '\n' is entered on an empty line or when an empty "Enter" is pressed; otherwise, it will simply move to
            the next line and wait for further input. In this case, the bottom newline character \n will
            not be retained.
        3.  Methods in the class Tools are allowed to be used. The sequence is: user-assistant-tool-assistant

        :param system_content: (str) 'role': 'system' 中的 content 的内容，被赋值时会消除前面的所有对话记录。
                               如果未赋值则运用初始信息，默认为初始信息
        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param end_token: (str) 输入结束 token，在检测到该 token 并后紧跟 '\n' 时结束输入过程并输入，默认为换行符 '' 代表换行符，
                                此时在检测到一个空行后紧跟一个换行符代表输入结束。此参数不允许包含换行符
        :param stream: (bool) 是否启用流输出 (逐字返回)，默认为 True

        :return ai_reply: (list) AI 返回的消息列表 messages
        """

        # 检查 end_token 是否包含换行符 '\n'
        if '\n' in end_token or '\r' in end_token:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"end_token cannot contain line breaks.")

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages
        # 检查 system_content 赋值
        if system_content is not None:
            self.messages = [{"role": "system",
                              "content": system_content}]

        # 构建 URL
        endpoint = "/chat/completions"
        target_url = self.base_url + endpoint

        # 构建 headers
        headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.headers = headers

        # 构建请求体
        request_body_dict = {
            # 重要参数
            "model": self.model,
            "messages": self.messages,
            # 文本参数
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            # 流式输出
            "stream": stream,
            # 种子参数
            "seed": self.seed,
            # 工具参数
            "tools": self.tools,
            "tool_choice": self.tool_choice
        }

        print(f"Let's start chatting! The current model is \033[31m{self.model}\033[0m.")

        # 对话循环
        reply_messages = []
        while True:

            # 获取用户输入
            user_input_list = []
            prompt = f"{self.bold}{self.user_role_color}User{self.end_style}: "  # 绿色加粗 User:

            if end_token == '':
                # 空 token：空行换行直接结束
                while True:
                    line = input(prompt)
                    if line == '':  # 空行直接结束
                        break
                    user_input_list.append(line)
                    prompt = f"{self.bold}{self.user_role_color}----> {self.end_style}"
            else:
                # 非空 token: 以 token 结尾换行才结束
                while True:
                    line = input(prompt)
                    if line.endswith(end_token):
                        content_line = line[:-len(end_token)].rstrip()
                        # 如果这一行只有 token（去掉后为空），保留一个空行
                        if content_line == '':
                            user_input_list.append('')
                        else:
                            user_input_list.append(content_line)
                        break
                    user_input_list.append(line)
                    prompt = f"{self.bold}{self.user_role_color}----> {self.end_style}"

            user_input = "\n".join(user_input_list)

            # 退出条件
            if user_input.lower() in ['退出', 'exit', 'quit']:

                if stream:  # '流式'
                    print(f'The conversation is over. Goodbye ^_< !\n')
                else:  # 非'流式'
                    print(
                        f'\nIn this conversation, the input contains {self.user_role_color}'
                        f'{self.response_prompt_tokens}{self.end_style} '
                        f'characters, and the output has {self.assistant_role_color}{self.response_completion_tokens}'
                        f'{self.end_style} characters.')
                    print(f'The conversation is over. Goodbye ^_< !\n')
                break

            # 添加用户消息到对话历史
            self.messages.append({"role": "user", "content": user_input})

            # 更新请求体
            request_body_dict["messages"] = self.messages

            # '流式'
            if stream:

                # 流式请求 使用 requests 以流式方式 POST 请求接口
                with requests.post(url=target_url, headers=headers, json=request_body_dict, stream=True) as response:
                    # 判断请求是否成功
                    if response.status_code != 200:
                        message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
                        print(f"\033[31mRequest failed!\033[0m status_code: {response.status_code} ({message})")

                    ai_reply = ""
                    tool_calls = {}  # 用于累积工具调用信息

                    # iter_lines() 会按行（以换行符分隔）逐行读取服务器返回的流数据
                    for line in response.iter_lines():
                        if line:  # 过滤掉空行（有些 SSE 数据可能包含心跳或空行）
                            decoded_line = line.decode("utf-8")  # 将字节数据解码成字符串
                            # OpenAI 风格的 SSE(Server-Sent Events) 数据以 "data: " 开头
                            if decoded_line.startswith("data: "):
                                # 去掉开头的 "data: " 前缀，获取纯 JSON 数据部分
                                json_data = decoded_line[len("data: "):]

                                # 如果是 "[DONE]" 表示流式输出已经结束
                                if json_data.strip() == "[DONE]":
                                    break  # 跳出循环

                                try:
                                    # 将 JSON 字符串解析为 Python 字典
                                    content_dict = json.loads(json_data)

                                    # 只在第一次解析时获取元信息
                                    if self.stream_begin_output:
                                        self.response_id = content_dict.get("id")
                                        self.response_model = content_dict.get("model")
                                        self.response_object = content_dict.get("object")
                                        self.response_created = content_dict.get("created")

                                        print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                                              f"{self.assistant_content_color}", end="", flush=True)

                                        self.stream_begin_output = False  # 获取本次信息后关闭

                                    # 处理 choices
                                    choices = content_dict.get("choices", [])
                                    if choices:
                                        first_choice = choices[0]
                                        delta = first_choice.get("delta", {})

                                        # 追加文本
                                        if "content" in delta:
                                            text_piece = delta["content"]
                                            if isinstance(text_piece, str):
                                                ai_reply += text_piece
                                                print(text_piece, end="", flush=True)

                                        # 处理工具调用（累积参数）
                                        if "tool_calls" in delta:
                                            for call in delta["tool_calls"]:
                                                index = call["index"]

                                                # 初始化工具调用记录
                                                if index not in tool_calls:
                                                    tool_calls[index] = {
                                                        "id": "",
                                                        "function": {"name": "", "arguments": ""}
                                                    }

                                                # 更新工具调用ID
                                                if call.get("id"):
                                                    tool_calls[index]["id"] = call["id"]

                                                # 更新函数名称
                                                if "function" in call and call["function"].get("name"):
                                                    tool_calls[index]["function"]["name"] = call["function"]["name"]

                                                # 累积参数
                                                if "function" in call and call["function"].get("arguments"):
                                                    tool_calls[index]["function"]["arguments"] += call["function"][
                                                        "arguments"]

                                except json.JSONDecodeError:
                                    # 如果解析 JSON 出错（可能是心跳包或非 JSON 格式内容），则跳过
                                    continue

                    print(f"{self.end_style}\n")  # 结束颜色
                    self.stream_begin_output = True  # 下一次打印时变成首次输出

                    # 添加AI回复到消息历史
                    self.messages.append({"role": "assistant", "content": ai_reply})

                    # 处理累积的工具调用 (在流结束后)
                    if tool_calls:
                        # 按索引排序工具调用
                        sorted_tool_calls = [tool_calls[idx] for idx in sorted(tool_calls.keys())]

                        for call in sorted_tool_calls:
                            func_name = call["function"]["name"]
                            args_str = call["function"]["arguments"]

                            try:
                                # 尝试解析参数
                                args = json.loads(args_str) if args_str.strip() else {}

                                if func_name in AI.tool_methods:
                                    # 尝试调用工具
                                    result = AI.tool_methods[func_name](**args)
                                else:
                                    result = f"Unknown tool: {func_name}"
                            except json.JSONDecodeError:
                                result = f"Invalid arguments format for tool: {func_name}"
                            except Exception as e:
                                result = f"The tool cannot be executed {func_name}: {str(e)}"

                            # 发送工具结果回模型
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(obj={"result": result}, ensure_ascii=False)
                            })

                            # 获取调用工具后的AI回答
                            self.chat()

                        reply_messages = self.messages

            # 非 '流式'
            else:
                # 发送请求
                response = requests.post(url=target_url, headers=headers, json=request_body_dict)

                # 判断请求是否成功
                if response.status_code != 200:
                    message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
                    print(f"\033[31mRequest failed!\033[0m status_code: {response.status_code} ({message})")

                # 解析返回参数
                response_dict = response.json()
                self.response_id = response_dict.get("id")  # 本次请求的唯一标识符
                self.response_model = response_dict.get("model")  # 使用的模型名称
                self.response_object = response_dict.get("object")  # 返回对象类型，一般是 "chat.completion"
                self.response_created = response_dict.get("created")  # 创建时间，Unix 时间戳

                choices = response_dict.get("choices", [])
                if choices:
                    first_choice = choices[0]
                    message = first_choice.get("message", {})

                    # 保存 AI 消息内容
                    self.response_content = message.get("content")
                    self.response_finish_reason = first_choice.get("finish_reason")
                    self.response_index = first_choice.get("index")

                    # 保存 AI 回复为 assistant 角色追加到 self.messages
                    self.messages.append({"role": "assistant", "content": self.response_content})

                    # 打印回复
                    print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                          f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

                    #  处理 tool_calls
                    if "tool_calls" in message:

                        for call in message["tool_calls"]:
                            func_name = call["function"]["name"]
                            args = json.loads(call["function"]["arguments"])

                            # 动态调用对应函数
                            if func_name in AI.tool_methods:  # 注意用你之前定义的工具字典
                                result = AI.tool_methods[func_name](**args)
                            else:
                                result = f"Unknown tool: {func_name}"

                            # 把结果发回给模型
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(obj={"result": result}, ensure_ascii=False)
                            })

                            # 获取调用工具后的 AI 回答
                            self.chat()

                usage = response_dict.get("usage", {})  # 令牌使用情况统计
                self.response_prompt_tokens += usage.get("prompt_tokens", 0)  # 提示词消耗的 tokens 数
                self.response_completion_tokens += usage.get("completion_tokens", 0)  # 生成内容消耗的 tokens 数
                self.response_total_tokens += usage.get("total_tokens", 0)  # 总共消耗的 tokens 数

                reply_messages = self.messages

        return reply_messages

    # 计算使用的费用
    def calculate_cost(self) -> None:
        """
        计算本次使用的费用
        Calculate the cost for this use.

        :return: None
        """

        # 防止 None 出错
        prompt_tokens = self.response_prompt_tokens or 0
        completion_tokens = self.response_completion_tokens or 0

        # 费用计算
        input_cost = (prompt_tokens / 1_000_000) * self.input_price_per_million_tokens
        output_cost = (completion_tokens / 1_000_000) * self.output_price_per_million_tokens
        total_cost = input_cost + output_cost

        self.total_cost = total_cost  # 保存到类的属性中

        print(
            f"{self.system_role_color}{self.model}{self.end_style} | {self.user_role_color}Input{self.end_style}: "
            f"{self.user_content_color}{prompt_tokens}{self.end_style} tokens "
            f"({self.user_remark_color}¥{input_cost:.4f}{self.end_style}), "
            f"{self.assistant_role_color}Output{self.end_style}: {self.assistant_content_color}{completion_tokens}"
            f"{self.end_style} tokens ({self.assistant_remark_color}¥{output_cost:.4f}{self.end_style}) | "
            f"Total: {self.system_remark_color}¥{total_cost:,.4f}{self.end_style}\n")

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
            print(f"{self.system_remind}[No summary needed - Dialogue history within {len_lim} "
                  f"characters]{self.end_style}")
            return

        print(f"{self.system_remind}[Dialogue history exceeds {len_lim} characters, "
              f"generating summary...]{self.end_style}")

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

            # 构建 URL
            endpoint = "/chat/completions"
            target_url = self.base_url + endpoint

            # 构建 headers
            headers = {
                # "User-Agent": "<…>",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            self.headers = headers

            # 构建请求体
            request_body_dict = {
                "model": self.model,
                "messages": summary_prompt,
                "max_tokens": len_lim,
                "temperature": 0.3,
            }

            # 发送请求
            response_dict = requests.post(url=target_url, headers=headers, json=request_body_dict).json()
            summary = response_dict.get("choices", [])[0].get("message", {}).get("content")  # 返回的回答列表，通常只有一个

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
            print(f"{self.system_role_color}Summary generated:{self.end_style} {summary}")
            print(
                f"{self.system_role_color}[History reduced from {total_chars} to {len(summary)} chars + "
                f"latest 2 messages]{self.end_style}")

        except Exception as e:
            print(f"\033[31mSummary failed: {str(e)}\033[0m")
            # 恢复原始对话历史
            self.messages = original_messages

        return None

    # 清空聊天缓存内容
    def reset_conversation(self, system_prompt: Optional[str] = None, preserve_system: bool = False) -> None:
        """
        清空 self.messages 中的内容
        Clear the content in self.messages.

        :param system_prompt: (str) 可选的系统提示内容
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
        if system_prompt is not None:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 状态反馈
        print(f"{self.assistant_content_color}mself.messages{self.end_style} only retains "
              f"{self.system_role_color}system{self.end_style} information" if preserve_system
              else f"The content in {self.assistant_content_color}self.messages{self.end_style} has been "
                   f"{self.system_content_color}reset{self.end_style}")

        return None


""" 真人模型 """
class Human:
    """
    此为真人交互时占用类

    It can be interaction among multiple users, achieving interaction between users and between users
    and large AI models.
    """

    # 初始化
    def __init__(self, ai_keyword: Optional[str] = None, instance_id: Optional[str] = None,
                 information: Optional[str] = None,):
        """
        Human 类参数初始化，主要参数需要与类 AI 相同
        Initialize the parameters of the Human class. The main parameters need to be the same as those of the class AI.

        # 自定义参数 (3)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型
        """

        # 自定义参数 (3)
        self.ai_keyword = ai_keyword
        self.instance_id = instance_id
        self.information = information

        # 输入信息
        self.messages = None

        # 输出信息
        self.response = None

        # 颜色设置
        self.system_role_color = '\033[91m'  # 亮红色
        self.system_content_color = '\033[31m'  # 红色
        self.system_remark_color = '\033[31;2m'  # 暗红色
        self.user_role_color = '\033[92m'  # 亮绿色
        self.user_content_color = '\033[32m'  # 绿色
        self.user_remark_color = '\033[32;2m'  # 暗绿色
        self.assistant_role_color = '\033[95m'  # 亮粉色
        self.assistant_content_color = '\033[35;2m'  # 粉色
        self.assistant_remark_color = '\033[35;2m'  # 暗粉色
        self.tool_role_color = '\033[94m'  # 亮蓝色
        self.tool_content_color = '\033[34m'  # 蓝色
        self.tool_remark_color = '\033[34;2m',  # 暗蓝色

        self.bold = '\033[1m'  # 加粗
        self.system_remind = '\033[90m'  # 亮黑色
        self.end_style = '\033[0m'  # 还原

    # 确保可以从实例变量中找到 instance_id
    def __repr__(self):
        return f"{self.instance_id}"

    # 以真人的身份与 AI 大模型对话
    def chat(self, messages: Optional[List[dict]] = None, input_role_user: bool = True,
             end_token: str = '') -> List[dict]:
        """
        用户收到信息，返回信息，仅一次，不会循环
        The user receives the message and returns it only once, without any loops.

        :param messages: (List[dict]) 用户收到的信息，用户收到信息中 'system' 将突出显示，'user' 将为主要内容
        :param input_role_user: (bool) 用户输入在 messages 中记录为 'user' (True) 还是 'assistant' (False)，默认为 True
        :param end_token: (str) 此参数不允许包含换行符。end_token 默认情况下，只有在空的一行输入换行符 '\n' 或空按“回车”才会将
                                内容输入，否则只是换到下一行并等待继续输入，此情况下最下面的换行符 \n 不会保留

        :return reply_messages: (List[dict]) 用户回复后的消息列表，包含新追加的消息
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

        # 检查 end_token 是否包含换行符 '\n'
        if '\n' in end_token or '\r' in end_token:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"end_token cannot contain line breaks.")

        self.messages = messages.copy()  # 拷贝一份，避免修改外部列表

        # 打印所有消息内容 (role 和 content），根据角色加颜色
        print("\n\033[3mAll messages\033[0m:")
        for i, msg in enumerate(self.messages, 1):
            role = msg['role']
            content = msg['content']
            if role == 'user':
                print(f"{i}. {self.bold}{self.user_role_color}User{self.end_style}: "
                      f"{self.user_content_color}{content}{self.end_style}")
            elif role == 'assistant':
                print(f"{i}. {self.bold}{self.assistant_role_color}Assistant{self.end_style}: "
                      f"{self.assistant_content_color}{content}{self.end_style}")
            elif role == 'system':
                print(f"{i}. {self.bold}{self.system_role_color}System{self.end_style}: "
                      f"{self.system_content_color}{content}{self.end_style}")
            elif role == 'tool':
                result_str = json.loads(content)["result"]
                print(f"{i}. {self.bold}{self.tool_role_color}Tool{self.end_style}: "
                      f"{self.tool_content_color}{result_str}{self.end_style}")
            else:
                # 其他角色正常打印，无色彩
                print(f"{i}. {role}: {content}")

        # 根据 input_role_user 决定输入提示角色和颜色
        if input_role_user:
            input_role_name = "User"
            input_color_start = "\033[1m\033[92m"  # 粉色加粗
            input_color_end = "\033[0m"
        else:
            input_role_name = "Assistant"
            input_color_start = "\033[1m\033[95m"  # 绿色加粗
            input_color_end = "\033[0m"

        input_prompt = f"\n{input_color_start}{input_role_name}{input_color_end}: "

        # 获取用户输入
        if end_token == '':
            # 空 token：出现一个空行就结束
            lines = []
            first_line = input(input_prompt)

            if first_line != '':
                lines.append(first_line)
                # 计算续行提示符长度
                main_prompt_text = f"{input_role_name}: "
                main_prompt_len = len(main_prompt_text)
                continuation_prompt_text = '-' * (main_prompt_len - 3) + '-> '
                continuation_prompt = f"{input_color_start}{continuation_prompt_text}{input_color_end}"

                while True:
                    line = input(continuation_prompt)
                    if line == '':
                        break
                    lines.append(line)

            human_reply = "\n".join(lines)

        else:
            # 非空 token：以 token 结尾换行才结束
            lines = []
            first_line = input(input_prompt)

            if first_line.endswith(end_token):
                # 保留去掉 end_token 后的内容
                content_line = first_line[:-len(end_token)].rstrip()
                lines.append(content_line)
            else:
                lines.append(first_line)
                # 计算续行提示符长度
                main_prompt_text = f"{input_role_name}: "
                main_prompt_len = len(main_prompt_text)
                continuation_prompt_text = '-' * (main_prompt_len - 3) + '-> '
                continuation_prompt = f"{input_color_start}{continuation_prompt_text}{input_color_end}"

                while True:
                    line = input(continuation_prompt)
                    if line.endswith(end_token):
                        content_line = line[:-len(end_token)].rstrip()
                        lines.append(content_line)  # 这里保留去掉 token 的内容
                        break
                    lines.append(line)

            human_reply = "\n".join(lines)

        # 决定追加消息的角色
        role_to_append = 'user' if input_role_user else 'assistant'

        # 将用户输入追加到 self.messages
        self.messages.append({'role': role_to_append, 'content': human_reply})

        # 保存回复内容
        self.response = human_reply

        reply_messages = self.messages

        return reply_messages


""" DeepSeek 大模型 """
class DeepSeek(AI):
    """
    DeepSeek

    Use the DeepSeek model for chatting and analysis.
    """

    pass


""" Gemini 大模型 """
class Gemini:
    pass


""" 即梦 AI 图 & 视频模型 """
class Jimeng:
    pass


""" AI 大模型的应用 """
class Assist(AI):
    """
    应用各种 AI 大模型完成生产力活动

    Apply various AI large models such as ChatGPT, DeepSeek, Claude, Gemini, Grok, etc., to complete productivity
    activities such as drawing, writing articles, revising articles, and analyzing data.
    """

    # 初始化，应当包含所有 AI 大模型的参数
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None,
                 messages: Optional[List[dict]] = None, ai_keyword: Optional[str] = None,
                 instance_id: Optional[str] = None, information: Optional[str] = None,
                 max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 1.0, n: int = 1, stream: bool = False,
                 stop: Union[str, list, None] = None, presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        # 必要参数 (4)
        :param api_key: (str) 输入的 API KEY，即 API 密钥
        :param base_url: (str) 输入的 base URL
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 与 'deepseek-reasoner'
        :param messages: (list) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息

        # 自定义参数 (3)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型

        # 附加参数 (8)
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param n: (int) 生成多少个独立回复选项 (消耗 n 倍 token)，如 n=3 会返回 3 种不同回答
        :param stream: (bool) 是否启用流输出 (逐字返回)
        :param stop: (str / list) 停止生成的标记，遇到这些字符串时终止输出
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作
        """

        super().__init__(
            # 必要参数 (4)
            api_key=api_key, base_url=base_url, model=model, messages=messages,

            # 自定义参数 (3)
            ai_keyword=ai_keyword, instance_id=instance_id, information=information,

            # 附加参数 (8)
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty
        )

    # 利用 DeepSeek 模型修改手稿
    def revise_manuscript(self, model: Optional[str] = None, manuscript: Optional[str] = None,
                          advice: Optional[str] = None, to_chat: bool = True) -> str:
        """
        让 DeepSeek 的 AI 大模型协助修改手稿，仅支持单次修改，后续问题需要转移至 continue_chat() 中
        Let the DeepSeek deepseek-reasoner model assist in modifying the manuscript.
        It only supports single modification. Subsequent issues need to be transferred to continue_chat()

        :param model: (str) 修改文章所用模型，默认为 'deepseek-reasoner'
        :param manuscript: (str) 需要修改的稿件内容，该内容为必填项
        :param advice: (str) 意见，往往是审稿人给出的，为选填项
        :param to_chat: (bool) 在生成修改后的文本后是否继续对话，默认为 True

        :return revised_text: (str) 修改后的文本
        """

        if model is None:
            model_revise_manuscript = self.model
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

        # 构建 URL
        endpoint = "/chat/completions"
        target_url = self.base_url + endpoint

        # 构建 headers
        headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.headers = headers

        # 构建请求体
        request_body_dict = {
            # 重要参数
            "model": model_revise_manuscript,
            "messages": self.messages,
            # 文本参数
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            # 种子参数
            "seed": self.seed,
            # 工具参数
            "tools": self.tools,
            "tool_choice": self.tool_choice
        }

        # 发送请求
        response = requests.post(url=target_url, headers=headers, json=request_body_dict)

        # 判断请求是否成功
        if response.status_code != 200:
            message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
            print(f"\033[31mRequest failed!\033[0m status_code: {response.status_code} ({message})")

        # 解析返回参数
        response_dict = response.json()
        self.response_id = response_dict.get("id")  # 本次请求的唯一标识符
        self.response_model = response_dict.get("model")  # 使用的模型名称
        self.response_object = response_dict.get("object")  # 返回对象类型，一般是 "chat.completion"
        self.response_created = response_dict.get("created")  # 创建时间，Unix 时间戳

        choices = response_dict.get("choices", [])  # 返回的回答列表，通常只有一个
        response_content = ''
        if choices:
            first_choice = choices[0]
            self.response_content = response_content = first_choice.get("message", {}).get("content")  # AI 生成的回答文本
            self.response_finish_reason = first_choice.get("finish_reason")  # 结束原因，如 "stop"
            self.response_index = first_choice.get("index")  # 选项索引

        usage = response_dict.get("usage", {})  # 令牌使用情况统计
        self.response_prompt_tokens = usage.get("prompt_tokens", 0)  # 提示词消耗的 tokens 数
        self.response_completion_tokens = usage.get("completion_tokens", 0)  # 生成内容消耗的 tokens 数
        self.response_total_tokens = usage.get("total_tokens", 0)  # 总共消耗的 tokens 数

        # 打印回复
        print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
              f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

        # 保存 AI 回复为 assistant 角色追加到 self.messages
        self.messages.append({"role": "assistant", "content": self.response_content})

        if to_chat:
            self.continue_chat()

        return response_content


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
        self.player_configuration = None

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
            instances[human_key] = Human(instance_id=human_key)

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
                instances[instance_id] = AI(instance_id=instance_id, model=model_name)

        # 可选调试输出
        if show_result:
            print("AI allocation result:")
            for ai_type, count in ai_counts.items():
                print(f"{ai_type}: {count}")
            print(f"\nTotal: {man_number + sum(ai_counts.values())}  Human: {man_number}  AI: {ai_number}")

        self.player_configuration = instances

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
            'ds': {"api_key": DeepSeek_api_key, "base_url": DeepSeek_base_url, "model": "deepseek-reasoner"},
            # 渠道 AI
            '1': {"api_key": other_api_key, "base_url": other_base_url, "model": "gemini-2.5-pro"},
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
    ai_instance.model = config["model"]
