"""
magiclib / generator

------------------------------------------------------------------------------------------------------------------------
magiclib / generator is a versatile Python library for data processing, analysis, and visualization. It provides
modules for reading and managing data from TXT, Excel, JSON, and PDF files, supports plotting functions such as line
and scatter plots, and offers machine learning capabilities including embedding, classification, and iterative batch
processing via the generator module. Additionally, magiclib includes practical utility tools for saving, loading, and
listing historical conversations, searching articles by ISSN, generating and saving images, and handling document
content for downstream analysis. Its integrated toolkit makes it a flexible platform for systematic data analysis and
workflow automation.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""


from . import general, grapher, author

import re
import os
import sys
import time
import json
import hmac
import base64
import hashlib
import inspect
import requests
import platform
import mimetypes
import subprocess
import numpy as np
import urllib.parse
from PIL import Image
from io import BytesIO
from google import genai
import simpleaudio as sa
from pandas import DataFrame
from zoneinfo import ZoneInfo
from abc import abstractmethod
from google.genai import types
from pydub import AudioSegment
from google.genai import errors
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from datetime import datetime, timezone
from requests.exceptions import HTTPError
from typing import Union, Optional, List, Dict, Callable, Tuple
from google.generativeai.types import FunctionDeclaration, Tool


# 所用时区
current_time_zone_location = 'Shanghai'

# Avaliable large AI models
api_key_1 = 'sk-flk6RxbjuWqApkKaU0DUJDP3FsG6QBI2hjHkwRRyU6briHqZ'  # expiration date 2025-11-16
api_key_2 = 'sk-6BHzr6iwXuhojiGAmeJumgejirrhRtizEiE4ItxSBwh3W1yO'  # 40$
base_url_1 = 'https://lmhub.fatui.xyz/v1'
success_requests_per_minute = 20  # 在此范围内属于正常现象
avaliable_model = [
    # Avaliable & Free
    'deepseek-ai/DeepSeek-R1'  # DeepSeek
    'deepseek-ai/DeepSeek-V3'  # DeepSeek
    'gemini-2.5-pro',  # Gemini
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
model_note = """
此部分模型：
1. 部分模型无法使用 tools & tool_choice 参数，需要将 tools = [] 即可

1. For some models, the tools & tool_choice parameter cannot be used. Just set tools to []
"""

# DeepSeek
DeepSeek_api_key = 'sk-cc2167b962444015a28d989478add7eb'  # MiaomiaoSu from 2025-08-07 ¥20
DeepSeek_base_url = 'https://api.deepseek.com/v1'
DeepSeek_avaliable_model = ['deepseek-chat', 'deepseek-reasoner', 'deepseek-coder']

# Gemini
Gemini_api_key_1 = 'AIzaSyBub6Dw-9hd9sjpiyCeOxLkd39ZGP-LJog'
Gemini_base_url = 'https://generativelanguage.googleapis.com'

# Jimeng
Jimeng_api_key = 'JIMENG-API-KEY'  # 带有 Jimeng 密钥无法上传至 github，使用时注意修改
Jimeng_api_secret = 'JIMENG-API-SECRET'

# sef.messages 保存的目录路径
messages_save_path = '/Users/sumiaomiao/Downloads/messages_save'  # 注意修改


""" AI 工具包 """
class Tools:
    """
    AI 大模型可以调用此部分的工具

    The AI invokes the tools of this part when the model is permitted. This part may accept class attributes
    and proceed step by step, as well as static methods.

    注意：
    1.  工具中打印的提示信息 (非正式) 均为灰色
    2.  在成功完成调用后需要给用户打印信息
    3.  返回内容有参数时，需要调整语序将参数内容放在最后
    4.  不要 raise 错误，用 return status 的形式向 AI 返回错误

    Note:
    1.  The prompt messages printed in the tool (informal) are all gray.
    2.  After the call is successfully completed, information needs to be printed for the user.
    3.  When the returned content contains parameters, the word order needs to be adjusted to
        place the parameter content at the end.
    4.  Do not raise the error. return the error to the AI in the form of return status.
    """

    # 无需初始化，被调用时赋值
    def __init__(self):

        # 实例化类 grapher.Plotter
        self.plotter_instanced = grapher.Plotter()

        # 实例化类 author.PDF
        self.pdf_instanced = author.PDF()

        # 实例化类 author.ArticleFetcher
        self.articlefetcher_instanced = author.ArticleFetcher()

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

        # text_to_speech()
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

        self.plotter_instanced.read_txt(txt_path=txt_path)
        status = 'The data in the TXT file has been read.'
        print(f'\033[90m[{status}]\033[0m\n')

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

        self.plotter_instanced.read_excel(excel_path=excel_path)
        status = 'The data in the Excel file has been read.'
        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 读取 JSON 文件
    def read_json(self, json_path: str) -> str:
        """
        让 AI 大模型读取 JSON 中的数据，该数据需要为 magiclib 保存的 JSON 格式
        Let the AI large model read the data in JSON, which needs to be in the JSON format saved for magiclib.

        :param json_path: (str) Excel 文件路径，可以是文件路径，也可以是目录，若被赋值则对该路径文件 / 目录进行处理

        :return status: (str) 返回信息，让 AI 大模型明白数据已经读取
        """

        self.plotter_instanced.read_json(json_path=json_path)
        status = 'The data in the JSON file has been read.'
        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 绘制线形图
    def plot_line(self) -> str:
        """
        根据读取到的数据绘制线形图
        Draw a line graph based on the read data.

        :return status: (str) 返回信息，让 AI 大模型明白线形图已绘制
        """

        self.plotter_instanced.plot_line()
        status = 'The line graph has been drawn.'
        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 绘制散点形图
    def plot_scatter(self) -> str:
        """
        根据读取到的数据绘制散点图
        Draw a scatter graph based on the read data.

        :return status: (str) 返回信息，让 AI 大模型明白散点图已绘制
        """

        self.plotter_instanced.plot_scatter()
        status = 'The scatter plot has been drawn.'
        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 搜索文章
    def seek_doi(self, issn_list: list = None, number: int = 500, query: Optional[str] = None,
                 from_pub_date: Optional[str] = None, until_pub_date: Optional[str] = None):
        """
        获取目标期刊下的文章的 DOI，可以加过滤器
        To obtain the DOI of the articles in the target journal, a filter can be added.

        :param issn_list: (list) 需要搜索的期刊
        :param number: (int) 每个期刊搜索文章的数量
        :param query: (str) 相关内容，将在全局搜索
        :param query: (str) 相关内容，将在全局搜索
        :param from_pub_date: (str) 搜索的其实日期 "2020-01-01"
        :param until_pub_date: (str) 搜索的结束日期 "2022-12-31"

        :return status: (str) 返回信息，让 AI 大模型明白文章已检索
        """

        doi_list = self.articlefetcher_instanced.seek_doi(issn_list=issn_list,
                                                          number=number,
                                                          query=query,
                                                          show_result=True,
                                                          from_pub_date=from_pub_date,
                                                          until_pub_date=until_pub_date,
                                                          )
        print(f'\033[90m[The article has been retrieved.]\033[0m\n')
        status = f'These articles have been displayed to the user, with DOI:\n{doi_list}'

        return status

    # 读取 PDF 文件
    def read_pdf(self, pdf_path: str) -> str:
        """
        读取 PDF 文件，输入的可以是文件路径，也可以是目录路径，但只会读取有效的 PDF 文件
        When reading PDF files, the input can be either the file path or the directory path,
        but only valid PDF files will be read.

        :param pdf_path: (list / str)  要读取的 PDF 文件路径列表

        :return status: (str) 返回信息，让 AI 大模型明白 PDF 文件已读取
        """

        pdf_text_dict = self.pdf_instanced.read_pdf(pdf_path=pdf_path, min_word_count=0, recursive=False)
        print(f'\033[90m[The specified PDF has been successfully read.]\033[0m\n')
        status = (f'The specified PDF has been successfully read, with content (The key is the file name of the PDF, '
                  f'and the value is the content of the PDF file):\n{pdf_text_dict}')

        return status

    # 打印 PDF 文件
    def print_pdf(self, pdf_path: str) -> str:
        """
        打印 PDF 文件，输入的可以是文件路径，也可以是目录路径，但只会打印有效的 PDF 文件
        When printing PDF files, the input can be either the file path or the directory path,
        but only valid PDF files will be printed.

        :param pdf_path: (list / str)  要打印的 PDF 文件路径列表

        :return status: (str) 返回信息，让 AI 大模型明白已将内容发送给打印机
        """

        self.pdf_instanced.print_pdf(file_list=pdf_path, show_result=False)
        status = 'The specified PDF has been successfully printed.'
        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 生成图片  用到 AI 大模型
    def generate_image(self, prompt: str, save_path: Optional[str] = None,
                       model: str = 'stabilityai/stable-diffusion-3-5-large',
                       size: str = '1024x1024', n: int = 1, seed: int = None) -> str:
        """
        根据用户要求生成图片
        Generate images according to user requirements.

        :param prompt: (str) 生成图片的英文提示词，为必需输入项。注意：在输入前必需将 prompt 转换成英文！
        :param save_path: (str) 保存的目录路径，若输入则按照路径保存
        :param model: (str) 生成图片的模型，默认为 Black Forest Lab 的 stabilityai/stable-diffusion-3-5-large
        :param size: (str) 图片的大小，最大 '2048x2048'，默认为 '1024x1024'
        :param n: (int) 生成图片的数量，默认为 1。注意：目前只能为 1
        :param seed: (int) 随机种子，默认为无种子

        :return status: (str) 图片绘制成功与否的信息
        """

        # =============== 注意更新 ===============
        generate_image_api_key = api_key_1
        generate_image_base_url = base_url_1
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
            status = f"[{model}] Request failed! status_code: {response.status_code} ({message})"
            print(f'\033[90m[{status}]\033[0m\n')

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

        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 保存图片
    def save_image(self, save_path: str) -> str:
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

        if not os.path.isdir(save_path):  # 检查是否为目录路径
            status = 'The path of save_path needs to be a directory path.'
            return status

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
        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # 文本转音频 用到 AI 大模型
    def text_to_speech(self, text: str, save_path: Optional[str] = None, model: str = 'fishaudio/fish-speech-1.5',
                       voice: Optional[str] = None, temperature: float = 1, speech_rate: float = 1, pitch: float = 1)\
            -> str:
        """
        将输入的文本转化为语音
        Convert the input text into speech.

        :param text: (str) 需要转化为语音的文本
        :param save_path: (str) 保存的目录路径，若输入则按照路径保存
        :param model: (str) 所用 AI 大模型，默认为 fishaudio/fish-speech-1.5
        :param voice: (str) 语音风格，可选 'alloy', 'echo', 'shimmer'
        :param temperature: (float) 随机性，0-1，越低随机性越低，默认为 0.7
        :param speech_rate: (float) 语速，默认 1.0
        :param pitch: (float) 语调，默认 1.0

        :return status: (str) 声音生成的信息
        """

        # =============== 注意更新 ===============
        generate_voice_api_key = api_key_1
        generate_voice_base_url = base_url_1
        generate_voice_model = model
        # =============== 注意更新 ===============

        # 检查输入
        if save_path is None:  # 保存路径是否被提供
            if self.save_path is None:
                status = 'No save path is provided.'
                return status
            else:
                save_path = self.save_path

        if save_path is not None and not os.path.isdir(save_path):  # 检查是否为目录路径
            status = 'The path of save_path needs to be a directory path.'
            return status

        # 构建 URL
        base_url = generate_voice_base_url
        endpoint = '/audio/speech'
        text_to_speech_url = base_url + endpoint

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
        response = requests.post(url=text_to_speech_url, headers=headers, json=request_body)

        # 判断请求是否成功
        if response.status_code != 200:
            message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
            status = f"[{model}] Request failed! status_code: {response.status_code} ({message})"
            print(f'\033[90m[{status}]\033[0m\n')

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
        if save_path is not None:  # 如果 save_path 不为 None，则保存

            file_name = "Generate_voice.mp3"
            full_file_path = os.path.join(save_path, file_name)

            count = 1
            while os.path.exists(full_file_path):  # 文件已存在则自动加 _1, _2 ...
                file_name = f"Generate_voice_{count}.mp3"
                full_file_path = os.path.join(save_path, file_name)
                count += 1

            # 保存 Base64 图片
            with open(full_file_path, "wb") as f:
                f.write(response.content)

            status = f'The voice has been played and saved in {save_path}. Continue the conversation'

        else:  # 如果 save_path 的值为 None
            status = "The voice has finished playing. Continue the conversation."

        print(f'\033[90m[{status}]\033[0m\n')

        return status
    
    # 音频转文本
    def speech_to_text(self):
        pass

    # Embedding
    def embedding(self, embedding_text: Union[list, dict], save_path: str = None, model: str = 'text-embedding-004',
                  save_title_to_input_text: bool = False) -> str:
        """
        将文本内容添加 embedding，以便后续分类
        Add embeddings to the text content for subsequent classification.

        :param embedding_text: (list / dict) 需要 embedding 的文本内容，或文章内容，dict: {filename: extracted text}
        :param save_path: (str) 保存的目录路径，按照路径保存
        :param model: (str) 所用 AI 大模型，默认为 text-embedding-004
        :param save_title_to_input_text: (bool) 是否用 embedding_text 的 key 为保存内容，默认为 False

        :return status: (str) embedding 的结果
        """

        # =============== 注意更新 ===============
        generate_voice_api_key = api_key_1
        generate_voice_base_url = base_url_1
        embedding_model = model
        # =============== 注意更新 ===============

        # 检查输入
        if save_path is None:  # 保存路径是否被提供
            if self.save_path is None:
                status = 'No save path is provided.'
                return status
            else:
                save_path = self.save_path

        if save_path is None or not os.path.isdir(save_path):  # 检查是否为目录路径
            status = 'The save_path needs to be entered and must be the directory path.'
            return status

        # 检查赋值
        if not isinstance(embedding_text, dict) and save_title_to_input_text:
            status = 'save_title_to_input_text is True only when embedding_text is a dictionary.'
            return status

        embedding_text_list = []
        if isinstance(embedding_text, list):  # list
            embedding_text_list = embedding_text
        elif isinstance(embedding_text, dict):  # dict
            titles_list = list(embedding_text.keys())
            embedding_text_list = [embedding_text[title] for title in titles_list]  # 显式顺序

        # 构建 URL
        base_url = generate_voice_base_url
        endpoint = '/embeddings'
        embedding_url = base_url + endpoint

        # 构建 headers
        headers = {
            "Authorization": f"Bearer {generate_voice_api_key}",
            "Content-Type": "application/json"
        }

        # 构建请求体
        request_body = {
            "model": embedding_model,
            "input": embedding_text_list,  # 文本或文本 list
        }

        # 发送请求
        response = requests.post(url=embedding_url, headers=headers, json=request_body)

        # 判断请求是否成功
        if response.status_code != 200:
            message = AI.status_code_messages.get(response.status_code, "Unknown Error")  # 未知错误
            status = f"[{model}] Request failed! status_code: {response.status_code} ({message})"
            print(f'\033[90m[{status}]\033[0m\n')

            return status

        # 解析 JSON 响应
        res_json = response.json()

        # 提取 embedding， 如果 input 是 list，data 会对应每条文本
        embedding_list = [item['embedding'] for item in res_json['data']]

        # 转成 numpy 数组 (可选)
        embeddings = np.array(embedding_list)

        # save_title_to_input_text 为 True 且 embedding_text 为 dict
        if save_title_to_input_text:
            input_to_save = list(embedding_text.keys())  # 保存标题
        else:
            input_to_save = embedding_text_list  # 保存正文内容

        # 保存 embedding
        if save_path is not None:  # 如果 save_path 不为 None，则保存
            os.makedirs(save_path, exist_ok=True)  # 确保目录存在

            file_name = "Embedding_result.npz"
            full_file_path = os.path.join(save_path, file_name)

            count = 1
            while os.path.exists(full_file_path):  # 文件已存在则自动加 _1, _2 ...
                file_name = f"Embedding_result_{count}.npz"
                full_file_path = os.path.join(save_path, file_name)
                count += 1

            np.savez(full_file_path, embeddings=embeddings, input_text=input_to_save)

            status = f"Embedding has been successfully computed and saved to {full_file_path}."

        else:  # 如果 save_path 的值为 None
            status = "Embedding has been successfully computed. Continue the conversation."

        print(f'\033[90m[{status}]\033[0m\n')

        return status

    # Rerank
    def rerank(self):
        pass


""" AI 大模型总类 """
class AI:
    """
    AI 大模型公有参数部分

    This section contains the public parameters and methods of the AI large model class.
    The methods should include continue_chat().

    # 数据结构
    1.  一般文本对话时的 messages 结构:
    messages =  [  # message 是一个 list，包含多个消息对象，最后一个消息对象为当前发的内容
                {"role": "system",   # role 是消息发送者的身份，有 "system", "user", "assistant", ("tool")
                 "content": "You are a helpful AI assistant who answers users' questions."},  # 消息文本内容
                 "reasoning_content": "(AI 的思考内容)"  # 返回时不需要带入，部分 AI 无此功能，
                                                       # 常见参数: reasoning_content, reasoning, thoughts, thinking
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

    4.  流式结构:
    data:{
        "id": "... ...",
        "object": "chat.completion.chunk",
        "created": 1755421355,
        "model": "deepseek-ai/DeepSeek-V3",
        "choices":[{
            "index": 0,
            "delta": {
                "content": "(part content)",
                "reasoning_content": null
            },
            "finish_reason": stop  (非最后一条为 null)
        }],
        "system_fingerprint": "",
        "usage": {
            "prompt_tokens":1267,
            "completion_tokens":63,
            "total_tokens":1330
        }
    }

    5.  带有 tool 的流式结构:
    data:{
        "id": "... ...",
        "object": "chat.completion.chunk",
        "created": 1755421355,
        "model": "deepseek-ai/DeepSeek-V3",
        "choices":[{
            "index": 0,
            "delta": {
                "content": "(part content)",
                "reasoning_content": "tool_calls",  (非最后一条为 null)
                "tool_calls": [{
                    "index": 0,
                    "id": null,
                    "type": null,
                    "function": {
                        "arguments": "{\""}
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }],
        "system_fingerprint": "",
        "usage": {
            "prompt_tokens":1267,
            "completion_tokens":63,
            "total_tokens":1330
        }
    }
    """

    # 可用工具及描述
    toolkit = [
        {
            "type": "function",
            "function": {
                "name": "save_messages_to_txt",
                "description": f"Save the conversation history to a formatted TXT file. "
                               f"The file will be saved in the {messages_save_path} directory."
                               f"When saving a conversation, please summarize the current conversation "
                               f"into 2 to 4 words (preferably in English, with underscores between words), "
                               f"and name it accordingly. "
                               f"When saving files, you need to include the file extension '.txt'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Name of the TXT file to save the conversation history."
                        }
                    },
                    "required": ["file_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "load_messages_from_txt",
                "description": f"Load conversation history from a formatted TXT file and append it to the current "
                               f"conversation. The file must be located in the {messages_save_path} directory."
                               f"When loading files, you need to include the file extension '.txt'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "Name of the TXT file to load (without path)"
                        }
                    },
                    "required": ["file_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_historical_conversations",
                "description": f"List all saved historical conversation files in the {messages_save_path} directory, "
                               f"sorted by modification time with the latest first. Each file's save time "
                               f"(extracted from its content) is displayed alongside the filename.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
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
                "name": "seek_doi",
                "description": "Search for articles based on the ISSN of the journal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issn_list": {
                            "type": ["list", "null"],
                            "description": 'The ISSN list of a journal, if there is only one journal such as '
                                           '["1095-9238"], by default ["1095-9238"] represents JAS journals.',
                            "default": ["1095-9238"],
                        },
                        "number": {
                            "type": "integer",
                            "description": "The number of search articles.",
                            "default": 100
                        },
                        "query": {
                            "type": ["string", "null"],
                            "description": "The content that requires conditional search is needed "
                                           "when filtering articles.",
                            "default": "null"
                        },
                        "from_pub_date": {
                            "type": ["string", "null"],
                            "description": 'The actual date of the search, such as "2020-01-01".',
                            "default": "null"
                        },
                        "until_pub_date": {
                            "type": ["string", "null"],
                            "description": 'The end date of the search, such as "2022-12-31".',
                            "default": "null"
                        },
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_pdf",
                "description": "Reading PDF files can be either the file path or the directory path. In most cases, "
                               "it is the PDF of academic articles. This method is called when users need to know "
                               "the content of a PDF article or summarize the key points of the article.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "The path of the PDF file or the directory path."
                        }
                    },
                    "required": ["pdf_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "print_pdf",
                "description": "Print the PDF based on the PDF file path or directory path entered by the user. "
                               "Print when the user needs it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "The path of the PDF file or the directory path."
                        }
                    },
                    "required": ["pdf_path"]
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
                                           "'stabilityai/stable-diffusion-3-5-large'."
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
        #         "name": "text_to_speech",
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
        {
            "type": "function",
            "function": {
                "name": "embedding",
                "description": "The embedding model is used to add vectors to the content to facilitate "
                               "subsequent classification and discrimination functions. "
                               "The dictionary obtained from read_pdf can be directly subjected to embedding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "embedding_text": {
                            "type": ["string", "object"],
                            "description": "Text content or article content that requires embedding."
                        },
                        "save_path": {
                            "type": "string",
                            "description": "Save the path."
                        },
                        "model": {
                            "type": "string",
                            "description": "The embedding model to use. Default is 'text-embedding-004'."
                        },
                        "save_title_to_input_text": {
                            "type": "boolean",
                            "description": "Whether to use the key of embedding_text to save the content, "
                                           "the default is False."
                        },
                    },
                    "required": ["embedding_text", "save_path"]
                }
            }
        },

    ]

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

    # 常见 AI 思考参数
    ai_thinking_parameters = ["reasoning_content", "reasoning", "thoughts", "thinking", "raw_cot"]

    # 参数初始化
    def __init__(self,

                 # 必要参数 (4)
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, messages: Optional[List[dict]] = None,

                 # 自定义参数 (4)
                 ai_keyword: Optional[str] = None, instance_id: Optional[str] = None,
                 information: Optional[str] = None, show_reasoning: bool = False,

                 # 附加参数 (11)
                 max_tokens: int = 128000, temperature: float = 0.7, top_p: float = 1.0, n: int = 1,
                 stream: bool = False, stop: Union[str, list, None] = None, presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0, seed: Optional[int] = None, tools: Optional[list] = None,
                 tool_methods: Optional[dict] = None, tool_choice: str = "auto"
                 ):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        # 必要参数 (4)
        :param api_key: (str) 输入的 API KEY，即 API 密钥
        :param base_url: (str) 输入的 base URL
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 与 'deepseek-reasoner'
        :param messages: (list) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息

        # 自定义参数 (4)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型
        :param show_reasoning: (bool) 是否打印推理过程，如果有推理的话。默认为 False

        # 附加参数 (12)
        :param max_tokens: (int) 生成的最大 token 数 (输入 + 输出)
        :param temperature: (float) 控制输出的随机性 (0.0-2.0)，数值越低越确定，越高越有创意
        :param top_p: (float) 核采样概率 (0.0-1.0)，仅保留概率累计在前 top_p 的词汇，与 temperature 二选一
        :param n: (int) 生成多少个独立回复选项 (消耗 n 倍 token)，如 n=3 会返回 3 种不同回答
        :param stream: (bool) 是否启用流输出 (逐字返回)
        :param stop: (str / list) 停止生成的标记，遇到这些字符串时终止输出
        :param presence_penalty: (float)  避免重复主题 (-2.0-2.0)，正值降低重复提及同一概念的概率，适合长文本生成
        :param frequency_penalty: (float) 避免重复词汇 (-2.0-2.0)，正值降低重复用词概率，适合技术文档写作
        :param seed: (int) 随机种子，默认为 None，表示完全随机
        :param tools: (list) 工具信息条，用于描述工具
        :param tool_methods: (dict) 工具包，放有具体工具
        :param tool_choice: (str) 工具选取方式，"auto" 为自动选取，"none" 为决不会选取"，
                            {"type": "function", "function": {"name": "xxx"}} 为强制调用指定工具，并且只能调用它。
                            "required"(部分文档称为 {"type": "function", "function": "required"} 的形式)，
                            模型必须调用某个工具，但可以自己选择哪一个
        """

        # 必要参数 (4)
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = api_key_1

        if base_url is not None:
            self.base_url = base_url
        else:
            self.base_url = base_url_1

        if model is not None:
            self.model = model
        else:
            self.model = 'deepseek-ai/DeepSeek-R1'

        if messages is not None:
            self.messages = messages
        else:
            self.messages = [{"role": "system", "content": "You are a helpful assistant. You will kindly answer "
                                                           "users' messages and use tools at the appropriate time."}]

        # 自定义参数 (4)
        self.ai_keyword = ai_keyword
        self.instance_id = instance_id
        self.information = information
        self.show_reasoning = show_reasoning

        # 附加参数 (11)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n  # 一般模型不用该参数
        self.stream = stream
        self.stop = stop  # 一般模型不用该参数
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed
        if tools is not None:
            self.tools = tools
        else:
            self.tools = AI.toolkit
        if tool_methods is not None:
            self.tool_methods = tool_methods
        else:
            # 创建 Tools 实例
            self.tools_instance = Tools()
            self.tool_methods = {
                "save_messages_to_txt": self.save_messages_to_txt,
                "load_messages_from_txt": self.load_messages_from_txt,
                "list_historical_conversations": self.list_historical_conversations,

                "read_txt": self.tools_instance.read_txt,
                "read_excel": self.tools_instance.read_excel,
                "read_json": self.tools_instance.read_json,
                "plot_line": self.tools_instance.plot_line,
                "plot_scatter": self.tools_instance.plot_scatter,
                "seek_doi": self.tools_instance.seek_doi,
                "read_pdf": self.tools_instance.read_pdf,
                "print_pdf": self.tools_instance.print_pdf,
                "generate_image": self.tools_instance.generate_image,
                "save_image": self.tools_instance.save_image,
                # "text_to_speech": tools_instance.text_to_speech,
                # "speech_to_text": self.tools_instance.speech_to_text,
                "embedding": self.tools_instance.embedding,
                # "rerank": self.tools_instance.rerank,
            }
        self.tool_choice = tool_choice

        # 输入参数 (3)
        self.target_url = None
        self.headers = None
        self.request_body_dict = None

        # 输出参数 (13)
        self.response = None  # 不包含流式
        self.response_status = 0  # 初始化输出状态
        self.response_id = None
        self.response_model = None
        self.response_object = None
        self.response_created = None
        self.response_reasoning = None
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
        self.tool_role_color = '\033[94m'  # 亮蓝色
        self.tool_content_color = '\033[34m'  # 蓝色
        self.tool_remark_color = '\033[34;2m'  # 暗蓝色

        self.bold = '\033[1m'  # 加粗
        self.system_remind = '\033[90m'  # 亮黑色
        self.end_style = '\033[0m'  # 还原

        # 费用计算
        self.total_cost = 0  # 总价格
        self.input_price_per_million_tokens = 4  # 输入价格 (百万 tokens) 默认为 DeepSeek-R1 模型的费用，均按照未命中处理
        self.output_price_per_million_tokens = 16  # 输出价格 (百万 tokens)

        # 其他参数
        self.start_time = None  # 记录对话开始时间
        self.stream_begin_output = True  # stream 开始返回信息
        self.reasoning_output = True  # AI 思考开始返回信息

        # 敏感内容
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_ONLY_HIGH"},
        ]

    # 确保可以从实例变量中找到 instance_id
    def __repr__(self):
        return f"{self.instance_id}"

    # 与 AI 大模型聊天
    def chat(self, messages: Optional[List[dict]] = None, show_response: bool = True, raise_error: bool = False,
             return_all_messages: bool = True) -> List[dict] or str:
        """
        与 AI 大模型聊天，单次交互，传入完整 messages，返回 AI 回复并保存。允许调用工具，调用返回时以 system 返回给 AI 工具调用结果，
        且不会自动二次调用 AI 处理工具调用后的结果
        Chat with the AI large model in a single interaction, pass in the complete messages, return the AI reply
        and save it. Allow the tool to be called. When the call returns, return the result of the AI tool
        call as system. And it will not automatically re-call the result after the AI processing tool is invoked.

        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param show_response: (bool) 是否打印返回的 reasoning & content，默认为 True
        :param raise_error: (bool) 遇到响应问题时为抛出错误，否则打印错误。默认为 False
        :param return_all_messages: (bool) 返回内容为单次消息 response_content or 整个 messages list，默认为 True

        :return result_content: (str / list) AI 返回的单次消息 response_content or 整个 messages list
        """

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages

        # 构建 URL
        endpoint = "/chat/completions"
        self.target_url = self.base_url + endpoint

        # 构建 headers
        self.headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建请求体
        self.request_body_dict = {
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
            "stream": self.stream,
            # 种子参数
            "seed": self.seed,
            # # 敏感内容
            # "extra_body": {"safetySettings": self.safety_settings},
        }

        # 工具参数
        if self.tools:  # 只有在 self.tools 不为空时有效
            self.request_body_dict["tools"] = self.tools
            self.request_body_dict["tool_choice"] = self.tool_choice

        # 流式输出
        if self.stream:

            # 流式请求 使用 requests 以流式方式 POST 请求接口
            with requests.post(
                    url=self.target_url,
                    headers=self.headers,
                    json=self.request_body_dict,
                    stream=True
            ) as response:

                # 查看状态码
                self.response_status = response.status_code

                # 判断请求是否成功
                if self.response_status != 200:
                    # 抛出错误
                    if raise_error:
                        message = AI.status_code_messages.get(self.response_status, "Unknown Error")
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise HTTPError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                        f"request failed! status_code: {self.response_status} ({message})")

                    # 改为打印信息
                    else:
                        message = AI.status_code_messages.get(self.response_status, "Unknown Error")  # 未知错误
                        print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                              f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

                response_reasoning = ""
                response_content = ""
                tool_calls = {}  # 用于累积工具调用信息

                # 记录最后一次收到数据的时间
                last_received_time = time.time()
                break_time = 120
                # iter_lines() 会按行（以换行符分隔）逐行读取服务器返回的流数据
                for chunk in response.iter_lines():
                    if chunk:  # 过滤掉空行（有些 SSE 数据可能包含心跳或空行

                        decoded_chunk = chunk.decode("utf-8")  # 将字节数据解码成字符串
                        # OpenAI 风格的 SSE(Server-Sent Events) 数据以 "data: " 开头
                        if decoded_chunk.startswith("data: "):
                            # 去掉开头的 "data: " 前缀，获取纯 JSON 数据部分
                            json_data = decoded_chunk[len("data: "):]

                            # 如果是 "[DONE]" 表示流式输出已经结束
                            if json_data.strip() == "[DONE]":
                                break  # 跳出循环

                            # 检查是否超时 (break_time 秒没有收到数据)
                            if time.time() - last_received_time > break_time:
                                print(f"{self.system_remind}[Timeout: No new data received within "
                                      f"{break_time} seconds.]{self.end_style}")
                                break

                            try:
                                # 将 JSON 字符串解析为 Python 字典
                                content_dict = json.loads(json_data)

                                # 只在第一次解析时获取元信息
                                if self.stream_begin_output:
                                    self.response_id = content_dict.get("id")
                                    self.response_model = content_dict.get("model")
                                    self.response_object = content_dict.get("object")
                                    self.response_created = content_dict.get("created")

                                    # 打印回复
                                    if show_response:
                                        # 打印 AI 思考过程的字体
                                        if self.show_reasoning:
                                            print(
                                                f"{self.bold}{self.tool_role_color}{self.model}{self.end_style}: "
                                                f"{self.tool_content_color}", end="", flush=True)
                                        # 不打印 AI 思考过程的字体
                                        else:
                                            print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}:"
                                                  f" {self.assistant_content_color}", end="", flush=True)

                                    last_received_time = time.time()  # 刷新更新时间
                                    self.stream_begin_output = False  # 获取本次信息后关闭

                                # 处理 choices
                                choices = content_dict.get("choices", [])
                                if choices:
                                    first_choice = choices[0]
                                    delta = first_choice.get("delta", {})

                                    # 打印 AI 思考过程
                                    if self.show_reasoning:
                                        # 追加 AI 思考过程
                                        reasoning_piece = next((delta.get(key) for key in AI.ai_thinking_parameters
                                                                if delta.get(key)), None)
                                        # 如果有思考内容就打印
                                        if isinstance(reasoning_piece, str):
                                            response_reasoning += reasoning_piece
                                            # 打印回复
                                            if show_response:
                                                print(reasoning_piece, end="", flush=True)
                                            last_received_time = time.time()  # 刷新更新时间

                                    # 只在第一次收到 content 内容时转换
                                    if delta.get("content") and self.reasoning_output:

                                        # 打印回复
                                        if show_response:
                                            # 如果 response_reasoning 为 ''，打印 None
                                            if not response_reasoning and self.show_reasoning:
                                                print('None')
                                            # 如果 response_reasoning 不是以换行符结尾，则打印一个换行符
                                            elif not response_reasoning.endswith('\n') and self.show_reasoning:
                                                print('')

                                        # 打印回复
                                        if show_response:
                                            # 打印 AI 回复内容的字体
                                            if self.show_reasoning:
                                                print(f"{self.bold}{self.assistant_role_color}{self.model}"
                                                      f"{self.end_style}: {self.assistant_content_color}",
                                                      end="", flush=True)
                                        self.reasoning_output = False

                                    # 追加 AI 回复内容
                                    if "content" in delta:
                                        content_piece = delta["content"]
                                        if isinstance(content_piece, str):
                                            response_content += content_piece
                                            # 打印回复
                                            if show_response:
                                                print(content_piece, end="", flush=True)
                                            last_received_time = time.time()  # 刷新更新时间

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

                # 打印回复
                if show_response:
                    print(f"{self.end_style}\n")  # 结束颜色
                self.stream_begin_output = True  # 下一次打印时变成首次输出
                self.reasoning_output = True  # 下一次打印时变成首次输出

                # 添加 AI 回复到消息历史
                self.response_content = response_content
                self.messages.append({"role": "assistant", "content": self.response_content})

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

                            if func_name in self.tool_methods:
                                # 尝试调用工具
                                result = self.tool_methods[func_name](**args)
                            else:
                                result = f"Unknown tool: {func_name}"
                        except json.JSONDecodeError:  # JSON 输入格式错误，AI 的问题
                            print(f'{self.system_remind}[The JSON format of the input Tool is incorrect.]'
                                  f'{self.end_style}')
                            result = f"The tool cannot be executed {func_name}: {str(e)}"
                        except Exception as e:
                            result = f"The tool cannot be executed {func_name}: {str(e)}"

                        # 发送工具结果回模型
                        self.messages.append({  # 让该 messages 参与下一次 user 对话，防止直接 system 的报错
                            "role": "system",
                            "content": result
                        })

        # 非流式输出
        else:

            # 发送请求
            self.response = requests.post(
                url=self.target_url,
                headers=self.headers,
                json=self.request_body_dict
            )

            # 查看状态码
            self.response_status = self.response.status_code

            # 判断请求是否成功
            if self.response_status != 200:
                # 抛出错误
                if raise_error:
                    message = AI.status_code_messages.get(self.response_status, "Unknown Error")
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise HTTPError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                    f"request failed! status_code: {self.response_status} ({message})")

                # 改为打印信息
                else:
                    message = AI.status_code_messages.get(self.response_status, "Unknown Error")  # 未知错误
                    print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                          f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

            # 解析返回参数
            response_dict = self.response.json()
            self.response_id = response_dict.get("id")  # 本次请求的唯一标识符
            self.response_model = response_dict.get("model")  # 使用的模型名称
            self.response_object = response_dict.get("object")  # 返回对象类型，一般是 "chat.completion"
            self.response_created = response_dict.get("created")  # 创建时间，Unix 时间戳

            choices = response_dict.get("choices", [])  # 返回的回答列表，通常只有一个
            response_messages = {}
            if choices:
                first_choice = choices[0]
                response_messages = first_choice.get("message", {})  # AI 返回的 messages

                # 保存 AI 消息内容
                for key in AI.ai_thinking_parameters:  # 遍历参数列表，找到第一个非空的思考字段
                    if response_messages.get(key):
                        self.response_reasoning = response_messages[key]
                        break
                self.response_content = response_messages.get("content")  # AI 生成的回答文本
                self.response_finish_reason = first_choice.get("finish_reason")  # 结束原因，如 "stop"
                self.response_index = first_choice.get("index")  # 选项索引

            usage = response_dict.get("usage", {})  # 令牌使用情况统计
            self.response_prompt_tokens += usage.get("prompt_tokens", 0)  # 提示词消耗的 tokens 数
            self.response_completion_tokens += usage.get("completion_tokens", 0)  # 生成内容消耗的 tokens 数
            self.response_total_tokens += usage.get("total_tokens", 0)  # 总共消耗的 tokens 数

            # 保存 AI 回复为 assistant 角色追加到 self.messages
            self.messages.append({"role": "assistant", "content": self.response_content})

            # 打印回复
            if show_response:
                # 打印 AI 的思考内容
                if self.show_reasoning:
                    print(f"{self.bold}{self.tool_role_color}{self.model}{self.end_style}: "
                          f"{self.tool_content_color}{self.response_reasoning}{self.end_style}")

                # 打印 AI 的回复内容
                print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                      f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

            #  处理 tool_calls
            if "tool_calls" in response_messages:

                for call in response_messages["tool_calls"]:
                    func_name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])

                    # 动态调用对应函数
                    if func_name in self.tool_methods:  # 注意用你之前定义的工具字典
                        result = self.tool_methods[func_name](**args)
                    else:
                        result = f"Unknown tool: {func_name}"

                    # 把结果发回给模型
                    self.messages.append({  # 让该 messages 参与下一次 user 对话，防止直接 system 的报错
                        "role": "system",
                        "content": result
                    })

        if return_all_messages:  # 返回所有 messages
            result_content = self.messages
        else:
            result_content = self.response_content  # 返回单次 response_content

        return result_content

    #  与 AI 大模型持续聊天
    def continue_chat(self, system_content: Optional[str] = None, messages: Optional[List[dict]] = None,
                      end_token: str = '', stream: bool = True, raise_error: bool = False) -> List[dict]:
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
            '\n' is entered on an empty chunk or when an empty "Enter" is pressed; otherwise, it will simply move to
            the next chunk and wait for further input. In this case, the bottom newline character \n will
            not be retained.
        3.  Methods in the class Tools are allowed to be used. The sequence is: user-assistant-tool-assistant

        :param system_content: (str) 'role': 'system' 中的 content 的内容，被赋值时会消除前面的所有对话记录。
                               如果未赋值则运用初始信息，默认为初始信息
        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param end_token: (str) 输入结束 token，在检测到该 token 并后紧跟 '\n' 时结束输入过程并输入，默认为换行符 '' 代表换行符，
                                此时在检测到一个空行后紧跟一个换行符代表输入结束。此参数不允许包含换行符
        :param stream: (bool) 是否启用流输出 (逐字返回)，默认为 True
        :param raise_error: (bool) 遇到响应问题时为抛出错误，否则打印错误。默认为 False

        :return messages: (list) AI 返回的消息列表 messages
        """

        # 将 stream 记录
        self.stream = stream

        # 检查 end_token 是否包含换行符 '\n'
        if '\n' in end_token or '\r' in end_token:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"end_token cannot contain chunk breaks.")

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages
        # 检查 system_content 赋值
        if system_content is not None:
            self.messages = [{"role": "system",
                              "content": system_content}]

        # 构建 URL
        endpoint = "/chat/completions"
        self.target_url = self.base_url + endpoint

        # 构建 headers
        self.headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建请求体
        self.request_body_dict = {
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
            # # 敏感内容
            # "extra_body": {"safetySettings": self.safety_settings},
        }

        # 工具参数
        if self.tools:  # 只有在 self.tools 不为空时有效
            self.request_body_dict["tools"] = self.tools
            self.request_body_dict["tool_choice"] = self.tool_choice

        print(f"Let's start chatting! The current model is \033[31m{self.model}\033[0m.")
        self.start_time = time.time()

        # 对话循环
        while True:

            # 获取用户输入
            user_input_list = []
            prompt = f"{self.bold}{self.user_role_color}User{self.end_style}: "  # 绿色加粗 User:

            if end_token == '':
                # 空 token：空行换行直接结束
                while True:
                    chunk = input(prompt)
                    if chunk == '':  # 空行直接结束
                        break
                    user_input_list.append(chunk)
                    prompt = f"{self.bold}{self.user_role_color}----> {self.end_style}"
            else:
                # 非空 token: 以 token 结尾换行才结束
                while True:
                    chunk = input(prompt)
                    if chunk.endswith(end_token):
                        content_line = chunk[:-len(end_token)].rstrip()
                        # 如果这一行只有 token（去掉后为空），保留一个空行
                        if content_line == '':
                            user_input_list.append('')
                        else:
                            user_input_list.append(content_line)
                        break
                    user_input_list.append(chunk)
                    prompt = f"{self.bold}{self.user_role_color}----> {self.end_style}"

            user_input = "\n".join(user_input_list)

            # 退出条件
            if user_input.lower() in ['退出', 'exit', 'quit']:

                end_time = time.time()  # 记录结束时间
                running_time = end_time - self.start_time  # 计算运行时长 (单位：秒)

                # 根据运行时长选择不同的输出格式
                if running_time >= 3600:  # 大于等于1小时
                    hours = int(running_time // 3600)
                    minutes = int((running_time % 3600) // 60)
                    seconds = int(running_time % 60)
                    print(
                        f"The conversation has ended. It took {self.tool_remark_color}{hours}{self.end_style} hours, "
                        f"{self.tool_content_color}{minutes}{self.end_style} minutes and "
                        f"{self.tool_role_color}{seconds}{self.end_style} seconds.")
                elif running_time >= 60:  # 大于等于1分钟但小于1小时
                    minutes = int(running_time // 60)
                    seconds = int(running_time % 60)
                    print(f"The conversation has ended. It took {self.tool_content_color}{minutes}{self.end_style} "
                          f"minutes and {self.tool_role_color}{seconds}{self.end_style} seconds.")
                else:  # 小于1分钟
                    print(f"The conversation has ended. It took {self.tool_role_color}{int(running_time)}"
                          f"{self.end_style} seconds.")

                if stream:  # '流式'
                    print('Goodbye, see you next time ^_< !\n')
                else:  # 非'流式'
                    print(
                        f'\nIn this conversation, the input contains {self.user_role_color}'
                        f'{self.response_prompt_tokens}{self.end_style} '
                        f'characters, and the output has {self.assistant_role_color}{self.response_completion_tokens}'
                        f'{self.end_style} characters.')
                    print('Goodbye, see you next time ^_< !\n')
                break

            # 添加用户消息到对话历史
            self.messages.append({"role": "user", "content": user_input})

            # 更新请求体
            self.request_body_dict["messages"] = self.messages

            # 流式输出
            if stream:

                # 流式请求 使用 requests 以流式方式 POST 请求接口
                with (requests.post(
                        url=self.target_url,
                        headers=self.headers,
                        json=self.request_body_dict,
                        stream=True
                ) as response):

                    # 查看状态码
                    self.response_status = response.status_code

                    # 判断请求是否成功
                    if self.response_status != 200:
                        # 抛出错误
                        if raise_error:
                            message = AI.status_code_messages.get(self.response_status, "Unknown Error")
                            class_name = self.__class__.__name__  # 获取类名
                            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                            raise HTTPError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                            f"request failed! status_code: {self.response_status} ({message})")

                        # 改为打印信息
                        else:
                            message = AI.status_code_messages.get(self.response_status, "Unknown Error")  # 未知错误
                            print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                                  f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

                    response_reasoning = ""
                    response_content = ""
                    tool_calls = {}  # 用于累积工具调用信息

                    # 记录最后一次收到数据的时间
                    last_received_time = time.time()
                    break_time = 120
                    # iter_lines() 会按行（以换行符分隔）逐行读取服务器返回的流数据
                    for chunk in response.iter_lines():
                        if chunk:  # 过滤掉空行（有些 SSE 数据可能包含心跳或空行）

                            decoded_chunk = chunk.decode("utf-8")  # 将字节数据解码成字符串
                            # OpenAI 风格的 SSE(Server-Sent Events) 数据以 "data: " 开头
                            if decoded_chunk.startswith("data: "):
                                # 去掉开头的 "data: " 前缀，获取纯 JSON 数据部分
                                json_data = decoded_chunk[len("data: "):]

                                # 如果是 "[DONE]" 表示流式输出已经结束
                                if json_data.strip() == "[DONE]":
                                    break  # 跳出循环

                                # 检查是否超时 (break_time 秒没有收到数据)
                                if time.time() - last_received_time > break_time:
                                    print(f"{self.system_remind}[Timeout: No new data received within "
                                          f"{break_time} seconds.]{self.end_style}")
                                    break

                                try:
                                    # 将 JSON 字符串解析为 Python 字典
                                    content_dict = json.loads(json_data)

                                    # 只在第一次解析时获取元信息
                                    if self.stream_begin_output:
                                        self.response_id = content_dict.get("id")
                                        self.response_model = content_dict.get("model")
                                        self.response_object = content_dict.get("object")
                                        self.response_created = content_dict.get("created")

                                        # 打印 AI 思考过程的字体
                                        if self.show_reasoning:
                                            print(
                                                f"{self.bold}{self.tool_role_color}{self.model}{self.end_style}: "
                                                f"{self.tool_content_color}", end="", flush=True)
                                        # 不打印 AI 思考过程的字体
                                        else:
                                            print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}:"
                                                  f" {self.assistant_content_color}", end="", flush=True)

                                        last_received_time = time.time()  # 刷新更新时间
                                        self.stream_begin_output = False  # 获取本次信息后关闭

                                    # 处理 choices
                                    choices = content_dict.get("choices", [])
                                    if choices:
                                        first_choice = choices[0]
                                        delta = first_choice.get("delta", {})

                                        # 打印 AI 思考过程
                                        if self.show_reasoning:
                                            # 追加 AI 思考过程
                                            reasoning_piece = next((delta.get(key) for key in AI.ai_thinking_parameters
                                                               if delta.get(key)), None)
                                            # 如果有思考内容就打印
                                            if isinstance(reasoning_piece, str):
                                                response_reasoning += reasoning_piece
                                                print(reasoning_piece, end="", flush=True)
                                                last_received_time = time.time()  # 刷新更新时间

                                        # 只在第一次收到 content 内容时转换
                                        if delta.get("content") and self.reasoning_output:

                                            # 如果 response_reasoning 为 ''，打印 None
                                            if not response_reasoning and self.show_reasoning:
                                                print('None')
                                            # 如果 response_reasoning 不是以换行符结尾，则打印一个换行符
                                            elif not response_reasoning.endswith('\n') and self.show_reasoning:
                                                print('')

                                            # 打印 AI 回复内容的字体
                                            if self.show_reasoning:
                                                print(f"{self.bold}{self.assistant_role_color}{self.model}"
                                                      f"{self.end_style}: {self.assistant_content_color}",
                                                      end="", flush=True)
                                            self.reasoning_output = False

                                        # 追加 AI 回复内容
                                        if "content" in delta:
                                            content_piece = delta["content"]
                                            if isinstance(content_piece, str):
                                                response_content += content_piece
                                                print(content_piece, end="", flush=True)
                                                last_received_time = time.time()  # 刷新更新时间

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
                    self.reasoning_output = True  # 下一次打印时变成首次输出

                    # 添加AI回复到消息历史
                    self.messages.append({"role": "assistant", "content": response_content})

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

                                if func_name in self.tool_methods:
                                    # 尝试调用工具
                                    result = self.tool_methods[func_name](**args)
                                else:
                                    result = f"Unknown tool: {func_name}"
                            except json.JSONDecodeError:  # JSON 输入格式错误，AI 的问题
                                print(f'{self.system_remind}[The JSON format of the input Tool is incorrect.]'
                                      f'{self.end_style}')
                                result = f"The tool cannot be executed {func_name}: {str(e)}"
                            except Exception as e:
                                result = f"The tool cannot be executed {func_name}: {str(e)}"

                                # 发送工具结果回模型
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(obj={"result": result}, ensure_ascii=False)
                            })

                            # 尝试输入带有 tool 的 messages
                            try:
                                # 尝试继续对话
                                self.chat(raise_error=True)  # 获取调用工具后的AI回答

                            except HTTPError:
                                if self.response_status == 400:

                                    # 删除最后一条消息（tool）
                                    if self.messages:
                                        self.messages.pop()

                                    # 改成 system role 再塞进去
                                    self.messages.append({  # 让该 messages 参与下一次 user 对话，防止直接 system 的报错
                                        "role": "system",
                                        "content": result
                                    })

                                else:
                                    raise

                            except Exception as e:
                                # 捕获其他所有异常并打印
                                print(f"An unexpected error occurred: {type(e).__name__}: {e}")

            # 非流式输出
            else:
                # 发送请求
                self.response = requests.post(
                    url=self.target_url,
                    headers=self.headers,
                    json=self.request_body_dict
                )

                # 查看状态码
                self.response_status = self.response.status_code

                # 判断请求是否成功
                if self.response_status != 200:
                    # 抛出错误
                    if raise_error:
                        message = AI.status_code_messages.get(self.response_status, "Unknown Error")
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise HTTPError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                        f"request failed! status_code: {self.response_status} ({message})")

                    # 改为打印信息
                    else:
                        message = AI.status_code_messages.get(self.response_status, "Unknown Error")  # 未知错误
                        print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                              f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

                # 解析返回参数
                response_dict = self.response.json()
                self.response_id = response_dict.get("id")  # 本次请求的唯一标识符
                self.response_model = response_dict.get("model")  # 使用的模型名称
                self.response_object = response_dict.get("object")  # 返回对象类型，一般是 "chat.completion"
                self.response_created = response_dict.get("created")  # 创建时间，Unix 时间戳

                choices = response_dict.get("choices", [])
                if choices:

                    first_choice = choices[0]
                    response_messages = first_choice.get("message", {})  # AI 返回的 messages

                    # 保存 AI 消息内容
                    for key in AI.ai_thinking_parameters:  # 遍历参数列表，找到第一个非空的思考字段
                        if response_messages.get(key):
                            self.response_reasoning = response_messages[key]
                            break
                    self.response_content = response_messages.get("content")  # AI 生成的回答文本
                    self.response_finish_reason = first_choice.get("finish_reason")
                    self.response_index = first_choice.get("index")

                    # 保存 AI 回复为 assistant 角色追加到 self.messages
                    self.messages.append({"role": "assistant", "content": self.response_content})

                    # 打印 AI 的思考过程
                    if self.show_reasoning:
                        print(f"{self.bold}{self.tool_role_color}{self.model}{self.end_style}: "
                              f"{self.tool_content_color}{self.response_reasoning}{self.end_style}")
                    # 打印 AI 的回复内容
                    print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                          f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

                    #  处理 tool_calls
                    if "tool_calls" in response_messages:

                        for call in response_messages["tool_calls"]:
                            func_name = call["function"]["name"]
                            args = json.loads(call["function"]["arguments"])

                            # 动态调用对应函数
                            if func_name in self.tool_methods:  # 注意用你之前定义的工具字典
                                result = self.tool_methods[func_name](**args)
                            else:
                                result = f"Unknown tool: {func_name}"

                            # 把结果发回给模型
                            self.messages.append({
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(obj={"result": result}, ensure_ascii=False)
                            })

                            # 尝试输入带有 tool 的 messages
                            try:
                                # 尝试继续对话
                                self.chat(raise_error=True)  # 获取调用工具后的AI回答

                            except HTTPError:
                                if self.response_status == 400:

                                    # 删除最后一条消息（tool）
                                    if self.messages:
                                        self.messages.pop()

                                    # 改成 system role 再塞进去
                                    self.messages.append({  # 让该 messages 参与下一次 user 对话，防止直接 system 的报错
                                        "role": "system",
                                        "content": result
                                    })

                                else:
                                    raise

                usage = response_dict.get("usage", {})  # 令牌使用情况统计
                self.response_prompt_tokens += usage.get("prompt_tokens", 0)  # 提示词消耗的 tokens 数
                self.response_completion_tokens += usage.get("completion_tokens", 0)  # 生成内容消耗的 tokens 数
                self.response_total_tokens += usage.get("total_tokens", 0)  # 总共消耗的 tokens 数

        messages = self.messages

        return messages

    # 流式迭代器输出单次对话
    def stream_yield_chat(self, messages: Optional[List[dict]] = None, raise_error: bool = False) -> str:
        """
        为连接窗口而设计。固定为 stream 对话，输出时 yield 输出，仅支持单次对话且不会单独打印文本内容
        Designed for connecting Windows. It is fixed as a stream dialogue, with yield output during output.
        It only supports a single dialogue and does not print the text content separately.

        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param raise_error: (bool) 遇到响应问题时为抛出错误，否则打印错误。默认为 False

        :return chunk_reasoning: (str / None) 在流式，show_reasoning = True 时，流式输出每段思考内容与对话内容，
                                 输出完毕时会有一个 "reasoning_end" 的 str
        :return chunk_content: (str / None) 流式输出对话内容，输出完毕时会有一个 "content_end" 的 str
        """

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages

        # 构建 URL
        endpoint = "/chat/completions"
        self.target_url = self.base_url + endpoint

        # 构建 headers
        self.headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建请求体
        self.request_body_dict = {
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
            "stream": True,
            # 种子参数
            "seed": self.seed,
            # # 敏感内容
            # "extra_body": {"safetySettings": self.safety_settings},
        }

        # 工具参数
        if self.tools:  # 只有在 self.tools 不为空时有效
            self.request_body_dict["tools"] = self.tools
            self.request_body_dict["tool_choice"] = self.tool_choice

        # 流式请求 使用 requests 以流式方式 POST 请求接口
        with requests.post(
                url=self.target_url,
                headers=self.headers,
                json=self.request_body_dict,
                stream=True
        ) as response:

            # 查看状态码
            self.response_status = response.status_code

            # 判断请求是否成功
            if self.response_status != 200:
                # 抛出错误
                if raise_error:
                    message = AI.status_code_messages.get(self.response_status, "Unknown Error")
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise HTTPError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                    f"request failed! status_code: {self.response_status} ({message})")

                # 改为打印信息
                else:
                    message = AI.status_code_messages.get(self.response_status, "Unknown Error")  # 未知错误
                    print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                          f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

            response_reasoning = ""
            response_content = ""
            tool_calls = {}  # 用于累积工具调用信息

            # 记录最后一次收到数据的时间
            last_received_time = time.time()
            break_time = 120
            # iter_lines() 会按行（以换行符分隔）逐行读取服务器返回的流数据
            for chunk in response.iter_lines():
                if chunk:  # 过滤掉空行（有些 SSE 数据可能包含心跳或空行

                    decoded_chunk = chunk.decode("utf-8")  # 将字节数据解码成字符串
                    # OpenAI 风格的 SSE(Server-Sent Events) 数据以 "data: " 开头
                    if decoded_chunk.startswith("data: "):
                        # 去掉开头的 "data: " 前缀，获取纯 JSON 数据部分
                        json_data = decoded_chunk[len("data: "):]

                        # 如果是 "[DONE]" 表示流式输出已经结束
                        if json_data.strip() == "[DONE]":
                            break  # 跳出循环

                        # 检查是否超时 (break_time 秒没有收到数据)
                        if time.time() - last_received_time > break_time:
                            print(f"{self.system_remind}[Timeout: No new data received within "
                                  f"{break_time} seconds.]{self.end_style}")
                            break

                        try:
                            # 将 JSON 字符串解析为 Python 字典
                            content_dict = json.loads(json_data)

                            # 只在第一次解析时获取元信息
                            if self.stream_begin_output:
                                self.response_id = content_dict.get("id")
                                self.response_model = content_dict.get("model")
                                self.response_object = content_dict.get("object")
                                self.response_created = content_dict.get("created")

                                last_received_time = time.time()  # 刷新更新时间
                                self.stream_begin_output = False  # 获取本次信息后关闭

                            # 处理 choices
                            choices = content_dict.get("choices", [])
                            if choices:
                                first_choice = choices[0]
                                delta = first_choice.get("delta", {})

                                # 打印 AI 思考过程
                                if self.show_reasoning:
                                    # 追加 AI 思考过程
                                    reasoning_piece = next((delta.get(key) for key in AI.ai_thinking_parameters
                                                            if delta.get(key)), None)
                                    # 如果有思考内容就打印
                                    if isinstance(reasoning_piece, str):
                                        response_reasoning += reasoning_piece

                                        last_received_time = time.time()  # 刷新更新时间

                                    chunk_reasoning = reasoning_piece
                                    yield chunk_reasoning

                                # 只在第一次收到 content 内容时转换
                                if delta.get("content") and self.reasoning_output:

                                    if self.show_reasoning:
                                        chunk_reasoning = "reasoning_end"  # 表示 reasoning 已经输出完毕
                                        yield chunk_reasoning

                                    self.reasoning_output = False

                                # 追加 AI 回复内容
                                if "content" in delta:
                                    content_piece = delta["content"]
                                    if isinstance(content_piece, str):
                                        response_content += content_piece

                                        last_received_time = time.time()  # 刷新更新时间

                                        chunk_content = content_piece
                                        yield chunk_content

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

            self.stream_begin_output = True  # 下一次打印时变成首次输出
            self.reasoning_output = True  # 下一次打印时变成首次输出

            # 添加 AI 回复到消息历史
            self.response_content = response_content
            self.messages.append({"role": "assistant", "content": self.response_content})

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

                        if func_name in self.tool_methods:
                            # 尝试调用工具
                            result = self.tool_methods[func_name](**args)
                        else:
                            result = f"Unknown tool: {func_name}"
                    except json.JSONDecodeError:  # JSON 输入格式错误，AI 的问题
                        print(f'{self.system_remind}[The JSON format of the input Tool is incorrect.]'
                              f'{self.end_style}')
                        result = f"The tool cannot be executed {func_name}: {str(e)}"
                    except Exception as e:
                        result = f"The tool cannot be executed {func_name}: {str(e)}"

                    # 发送工具结果回模型
                    self.messages.append({  # 让该 messages 参与下一次 user 对话，防止直接 system 的报错
                        "role": "system",
                        "content": result
                    })

        chunk_content = "content_end"  # 表示 chunk_content 已输出完毕
        yield chunk_content
        return None

    # 展示当前 messages 的内容
    def show_messages(self, messages: Optional[List[dict]] = None) -> None:
        """
        打印 messages 中的内容，如果被赋值则打印，如果未被赋值则打印 self.messages
        Print the content in messages. If it is assigned a value, print it; if not, print self.messages。

        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息

        :return: None
        """

        # 检查赋值
        if messages is None:
            messages = self.messages.copy()  # 拷贝一份，避免修改外部列表
        else:
            messages = messages

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

        # 打印所有消息内容 (role 和 content），根据角色加颜色
        print("\n\033[3mAll messages\033[0m:")
        for i, msg in enumerate(messages, 1):
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

        return None

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
            self.target_url = self.base_url + endpoint

            # 构建 headers
            self.headers = {
                # "User-Agent": "<…>",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # 构建请求体
            self.request_body_dict = {
                "model": self.model,
                "messages": summary_prompt,
                "max_tokens": len_lim,
                "temperature": 0.3,
            }

            # 发送请求
            response_dict = requests.post(
                url=self.target_url,
                headers=self.headers,
                json=self.request_body_dict
            ).json()
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

    # 测试 AI 大模型每分钟相应次数
    def get_api_call_rate(self, test_api_key: Optional[str] = None, test_base_url: Optional[str] = None,
                          test_model: Optional[str] = None, max_test_time: int = 120, show_progress: bool = True)\
            -> dict:
        """
        获取过去 60 秒内 API 的调用次数
        Obtain the number of API calls in the past 60 seconds.

        :param test_api_key: (str) 需要测试的 model 所属的 API，默认为类属性中的参数
        :param test_base_url: (str) 需要测试的 model 的 base URL，默认为类属性中的参数
        :param test_model: (str) 需要测试的 model，如未输入则测试类属性中的模型 self.model，此为默认项
        :param max_test_time: (int) 最大测试时长，默认为 120 秒
        :param show_progress: (bool) 打印测试进展，默认为 True

        :return ai_qps: (dict) AI 大模型的 API 调用信息
        """

        # 检查输入情况
        if test_api_key is None:
            test_api_key = self.api_key
        if test_base_url is None:
            test_base_url = self.base_url
        if test_model is None:
            test_model = self.model

        # 构建 URL
        endpoint = "/chat/completions"
        target_url = test_base_url + endpoint

        # 构建 headers
        headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {test_api_key}"
        }

        # 构建 messages
        messages = [{"role": "system", "content": "You are a helpful assistant. You will kindly answer "
                                                  "users' messages and use tools at the appropriate time."},
                    {"role": "user", "content": "Hello!"}]

        # 构建请求体
        request_body_dict = {
            # 重要参数
            "model": test_model,
            "messages": messages,
        }

        success_count = 0
        start_time = time.time()  # 记录开始时间

        while True:
            try:
                # 判断是否超过最大时长
                elapsed = time.time() - start_time
                if elapsed >= max_test_time:
                    return {
                        "status": "timeout",
                        "success_count": success_count,
                        "elapsed_time_sec": round(elapsed, 2)
                    }

                # 发送请求
                response = requests.post(
                    url=target_url,
                    headers=headers,
                    json=request_body_dict
                )

                if response.status_code == 200:
                    success_count += 1
                    if show_progress:
                        Time_difference = time.time() - start_time  # 记录时间差
                        print(f'The number of successful tests: '
                              f'[{self.system_remark_color}{success_count:^3}{self.end_style}],'
                              f'{Time_difference: 5.2f}')

                elif response.status_code == 429:
                    if show_progress:
                        print(f'Test over [{response.status_code}]: '
                              f'{self.system_remark_color}{success_count}{self.end_style}')

                    elapsed = time.time() - start_time
                    return {
                        "status": 429,
                        "success_count": success_count,
                        "elapsed_time_sec": round(elapsed, 2)
                    }

                else:
                    if show_progress:
                        print(f'Test failed [{response.status_code}]: '
                              f'{self.system_remark_color}{success_count}{self.end_style}')

                    elapsed = time.time() - start_time
                    return {
                        "status": response.status_code,
                        "error": response.text,
                        "success_count": success_count,
                        "elapsed_time_sec": round(elapsed, 2)
                    }

            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start_time
                return {
                    "status": "network_error",
                    "error": str(e),
                    "success_count": success_count,
                    "elapsed_time_sec": round(elapsed, 2)
                }

    # 保存 self.messages 至 txt 文件
    def save_messages_to_txt(self, file_name: str) -> str:
        """
        将对话历史保存为易读的 TXT 文件，文件名需要带扩展名 '.txt'
        Save the conversation history as an easy-to-read TXT file, and the file name needs to have the extension '.txt'.

        :param file_name: (str) self.messages 保存的名称

        :return status: (str) 返回信息，让 AI 大模型明白保存是否成功
        """

        # 确保保存目录存在
        if not os.path.exists(messages_save_path) or not os.access(messages_save_path, os.R_OK | os.W_OK | os.X_OK):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the path does not exist or there is no permission to access: {messages_save_path}")

        # 完整文件路径
        file_path = os.path.join(messages_save_path, file_name)

        # 查找第一条包含历史对话加载消息的索引
        first_load_index = -1

        # 保存时删除首个 "The following is the content of the loaded {} historical dialogue" 之前的内容
        for i, msg in enumerate(self.messages):
            content = msg.get("content", "").lower()
            if (msg.get("role") == "system" and
                    "the following is the content of the loaded" in content and
                    "historical dialogue" in content):
                first_load_index = i
                break  # 找到第一条就停止搜索

        # 如果没有找到加载消息，则从开头处理所有消息
        if first_load_index == -1:
            print("No historical conversation load message found, processing all messages from the beginning")
            start_index = 0
        else:
            print(f"Found historical conversation load message at index: {first_load_index}")
            start_index = first_load_index + 1  # 从加载消息之后开始处理

        # 处理消息内容
        text_content = []
        for msg in self.messages[start_index:]:
            role = msg.get("role", "").strip()
            content = msg.get("content", "").strip()

            # 跳过空消息
            if not role or not content:
                continue

            # 添加角色行
            text_content.append(f"{role}:")

            # 添加内容行（多行处理）
            content_lines = content.splitlines()
            for line in content_lines:
                text_content.append(line.strip())

            # 添加空行作为分隔
            text_content.append("")

        # 添加保存时间记录
        if text_content:  # 只有在有内容时才添加时间戳
            save_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            text_content.append("system:")
            text_content.append(f"The save time for the above content is: {save_time}")
            text_content.append("")  # 最后添加一个空行

        # 写入文件
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(text_content))
                status = f"The conversation has been successfully saved to: {file_path}"
                print(f'{self.system_remind}[{status}]{self.end_style}\n')

            return status

        except Exception as e:
            status = f"Save failed: {str(e)}"
            return status

    # 读取与 AI 对话的 messages 信息
    def load_messages_from_txt(self, file_name: str) -> str:
        """
        从 TXT 文件加载对话历史，文件名需要带扩展名 '.txt'
        Load conversation history from a TXT file, and the file name needs to have the extension '.txt'.

        :param file_name: (str) 要加载的 TXT 文件名 (不含路径)

        :return status: (str) 返回信息，让 AI 大模型明白读取是否成功
        """

        # 检查保存目录权限
        if not os.path.exists(messages_save_path) or not os.access(messages_save_path, os.R_OK):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the directory does not exist or is not accessible: {messages_save_path}")

        # 构建完整路径
        file_path = os.path.join(messages_save_path, file_name)

        # 检查文件存在性
        if not os.path.exists(file_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the file does not exist {file_path}")

        allow_role = ["system", "user", "assistant"]
        messages = []
        current_role = None
        current_content = []

        # 添加系统通知消息
        completion_msg = {
            "role": "system",
            "content": f"The following is the content of the loaded {file_name} historical dialogue."
        }
        self.messages.append(completion_msg)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()

                    # 跳过空行
                    if not stripped_line:
                        continue

                    # 检查是否是角色行 (以冒号结尾)
                    if stripped_line.endswith(":"):
                        # 保存前一条消息
                        if current_role and current_content and current_role.lower() in allow_role:
                            content_str = "\n".join(current_content).strip()
                            messages.append({
                                "role": current_role.strip(),
                                "content": content_str
                            })

                        # 提取角色（去除末尾冒号）
                        role_candidate = stripped_line.rstrip(":").strip()

                        # 只保留需要的角色
                        if role_candidate.lower() in allow_role:
                            current_role = role_candidate
                            current_content = []
                        else:
                            current_role = None  # 忽略不需要的角色
                            current_content = []

                    # 内容行 - 只有当当前角色是我们需要的时才处理
                    elif current_role and current_role.lower() in allow_role:
                        # 保留原始行内容，去除前后空格
                        current_content.append(stripped_line)

            # 处理文件末尾可能存在的最后一条消息
            if current_role and current_content and current_role.lower() in allow_role:
                content_str = "\n".join(current_content).strip()
                messages.append({
                    "role": current_role.strip(),
                    "content": content_str
                })

            # 将读取的消息追加到self.messages
            self.messages.extend(messages)

            status = f'The historical conversation {file_name} has been read.'
            print(f'{self.system_remind}[{status}]{self.end_style}\n')

            return status

        except Exception as e:
            # 错误处理
            error_msg = {
                "role": "system",
                "content": f"An error occurred when reading historical conversation: {str(e)}"
            }
            self.messages.append(error_msg)

            status = f'The historical conversation {file_name} reading failed.'
            print(f'{self.system_remind}[{status}]{self.end_style}\n')

            return status

    # 查看当前保存的历史对话 self.messages
    def list_historical_conversations(self) -> str:
        """
        列出所有保存的历史对话文件
        List all saved historical conversation files.

        :return status: (str) 返回信息，让 AI 大模型明历史 messages 是否打印成功
        """

        # 检查保存目录权限
        if not os.path.exists(messages_save_path) or not os.access(messages_save_path, os.R_OK):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the directory does not exist or is not accessible: {messages_save_path}")

        try:
            # 获取目录中所有文件
            all_files = os.listdir(messages_save_path)

            # 筛选 TXT 文件
            txt_files = [
                f for f in all_files
                if f.lower().endswith('.txt') and os.path.isfile(os.path.join(messages_save_path, f))
            ]

            # 按修改时间排序（最新在前）
            txt_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(messages_save_path, x)
            ), reverse=True)

            # 创建一个空字典，用于存储时间和文件名的映射
            time_file_dict = {}

            for filename in txt_files:
                file_path = os.path.join(messages_save_path, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                        # 从后往前查找最后一条系统消息
                        for i in range(len(lines) - 1, -1, -1):
                            line = lines[i].strip()

                            # 检查是否是系统消息行
                            if line.startswith("system:") or line.startswith("system：") or line.startswith("SYSTEM:"):
                                # 获取保存时间内容（通常是下一行）
                                if i + 1 < len(lines):
                                    time_line = lines[i + 1].strip()

                                    # 匹配 "The save time for the above content is：2023-10-15 14:30" 或类似格式
                                    time_match = re.search(pattern=r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})',
                                                           string=time_line)
                                    if time_match:
                                        save_time = time_match.group(1)
                                        time_file_dict[save_time] = filename
                                        break  # 找到后就跳出内部循环，处理下一个文件

                except Exception as e:
                    print(f"An error occurred when parsing the file {filename}: {str(e)}")
                    continue

            # 检查字典是否为空
            if not time_file_dict:
                status = f"No valid message files found in {messages_save_path}."
                print(f'{self.system_remind}[{status}]{self.end_style}\n')

                return status

            else:
                # 获取排序后的列表（最新在前）
                sorted_items = sorted(time_file_dict.items(), reverse=True)

                # 打印标题
                print(f"The historical messages in {self.system_content_color}{messages_save_path}"
                      f"{self.end_style} are as follows:")

                # 打印每个文件
                for i, (save_time, file) in enumerate(sorted_items, 1):
                    print(f"{self.system_content_color}{i}{self.end_style}."
                          f" - {self.assistant_content_color}{file}{self.end_style}"  # <-- 已修正
                          f" - {self.user_content_color}{save_time}{self.end_style}")
                print('')

            status = (f"The historical dialogue in {messages_save_path} has been displayed to the user."
                      f"The historical dialogues are arranged in the following order: {sorted_items}")

            return status

        except Exception as e:

            status = f"Error in obtaining the historical dialogue list:{str(e)}"
            print(f'{self.system_remind}[{status}]{self.end_style}\n')

            return status


""" 真人模型 """
class Human:
    """
    此为真人交互时占用类

    It can be interaction among multiple users, achieving interaction between users and between users
    and large AI models.
    """

    # 初始化
    def __init__(self, ai_keyword: Optional[str] = None, instance_id: Optional[str] = None,
                 information: Optional[str] = None, show_reasoning: bool = False):
        """
        Human 类参数初始化，主要参数需要与类 AI 相同
        Initialize the parameters of the Human class. The main parameters need to be the same as those of the class AI.

        # 自定义参数 (4)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型
        :param show_reasoning: (bool) 是否打印推理过程，如果有推理的话。默认为 False
        """

        # 自定义参数 (4)
        self.ai_keyword = ai_keyword
        self.instance_id = instance_id
        self.information = information
        self.show_reasoning = show_reasoning

        # 输入信息
        self.messages = None

        # 输出信息
        self.response_content = None

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
        self.tool_remark_color = '\033[34;2m'  # 暗蓝色

        self.bold = '\033[1m'  # 加粗
        self.system_remind = '\033[90m'  # 亮黑色
        self.end_style = '\033[0m'  # 还原

    # 确保可以从实例变量中找到 instance_id
    def __repr__(self):
        return f"{self.instance_id}"

    # 以真人的身份与 AI 大模型对话
    def chat(self, messages: Optional[List[dict]] = None, return_all_messages: bool = True, **kwargs) \
            -> List[dict] or str:
        """
        用户收到信息，返回信息，仅一次，不会循环
        The user receives the message and returns it only once, without any loops.

        :param messages: (List[dict]) 用户收到的信息，用户收到信息中 'system' 将突出显示，'user' 将为主要内容
        :param return_all_messages: (bool) 返回内容为单次消息 response_content or 整个 messages list，默认为 True

        --- **kwargs ---

        - input_role_user: (bool) 用户输入在 messages 中记录为 'user' (True) 还是 'assistant' (False)，默认为 True
        - end_token: (str) 此参数不允许包含换行符。end_token 默认情况下，只有在空的一行输入换行符 '\n' 或空按“回车”才会将
                                内容输入，否则只是换到下一行并等待继续输入，此情况下最下面的换行符 \n 不会保留

        :return result_content: (str / list) AI 返回的单次消息 response_content or 整个 messages list
        """

        # 参数
        input_role_user = kwargs.get("input_role_user", True)
        end_token = kwargs.get("end_token", "")

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
        self.response_content = human_reply

        if return_all_messages:  # 返回所有 messages
            result_content = self.messages
        else:
            result_content = self.response_content  # 返回单次 response_content

        return result_content


""" DeepSeek 大模型 """
class DeepSeek(AI):
    """
    DeepSeek

    Use the DeepSeek model for chatting and analysis.
    """

    pass


""" Gemini 大模型 """
class Gemini(AI):
    """
    Gemini 的 AI 大模型
    Gemini's AI large model

    # 数据结构
    1.  一般文本对话时的 contents 结构:
    contents = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": "Hello!"
            }]
        },{
            "role": "model",
            "parts": [{
                "text": "Hello! What can I do for you?"}]
            }],
        "system_instruction": {
            "parts": [{
                "text": "You are a helpful assistant. You will kindly answer "
                        "users' messages and use tools at the appropriate time."
            }]
        }
    }
    2.  模型返回的工具调用请求:
    contents = {
        "role": "model",
        "parts": [{
            "functionCall": {
                "name": "my_function",
                "args": {
                    "x": 3,
                    "y": 5
                }
            }
        }]
    }
    3.  返回带有 tool 请求的 contents:
    contents = {
        "role": "function",
        "parts": [{
            "functionResponse": {
                "name": "my_function",
                "response": {
                    "content": {"result": 8}
                }
            }
        }]
    }
    """

    # Gemini AI 所需参数
    def __init__(self,

                 # 必要参数 (4)
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, messages: Optional[List[dict]] = None,

                 # 自定义参数 (4)
                 ai_keyword: Optional[str] = None, instance_id: Optional[str] = None,
                 information: Optional[str] = None, show_reasoning: bool = False,

                 # 其它参数 (3)
                 stream: bool = False, tools: Optional[list] = None, tool_methods: Optional[dict] = None):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        # 必要参数 (4)
        :param api_key: (str) 输入的 API KEY，即 API 密钥
        :param base_url: (str) 输入的 base URL
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 与 'deepseek-reasoner'
        :param messages: (dict) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息

        # 自定义参数 (4)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型
        :param show_reasoning: (bool) 是否打印推理过程，如果有推理的话。默认为 False

        # 其它参数 (3)
        :param stream: (list) 是否启用流输出 (逐字返回)，默认为 False
        :param tools: (list) 工具信息条，用于描述工具
        :param tool_methods: (dict) 工具包，放有具体工具

        --- 文本生成网址 ---
        https://ai.google.dev/api/generate-content?
        """

        # 超类初始化
        super().__init__()  # 无需超类参数

        # 必要参数 (4)
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = Gemini_api_key_1

        if base_url is not None:  # Gemini 请求用不到
            self.base_url = base_url
        else:
            self.base_url = Gemini_base_url

        if model is not None:
            self.model = model
        else:
            self.model = 'gemini-2.5-pro'

        if messages is not None:
            self.messages = messages
        else:
            self.messages = [{"role": "system", "content": "You are a helpful assistant. You will kindly answer "
                                                           "users' messages and use tools at the appropriate time."}]

        # 自定义参数 (4)
        self.ai_keyword = ai_keyword
        self.instance_id = instance_id
        self.information = information
        self.show_reasoning = show_reasoning  # Gemini 中无法单独查看 AI 的思考

        # 其它参数 (3)
        self.stream = stream
        if tools is not None:
            self.tools = tools
        else:
            self.tools = AI.toolkit
        if tool_methods is not None:
            self.tool_methods = tool_methods
        else:
            # 创建 Tools 实例
            self.tools_instance = Tools()
            self.tool_methods = {
                "save_messages_to_txt": self.save_messages_to_txt,
                "load_messages_from_txt": self.load_messages_from_txt,
                "list_historical_conversations": self.list_historical_conversations,
                "read_txt": self.tools_instance.read_txt,
                "read_excel": self.tools_instance.read_excel,
                "read_json": self.tools_instance.read_json,
                "plot_line": self.tools_instance.plot_line,
                "plot_scatter": self.tools_instance.plot_scatter,
                "generate_image": self.tools_instance.generate_image,
                "save_image": self.tools_instance.save_image,
                # "text_to_speech": tools_instance.text_to_speech,
            }

        # 时间参数
        self.start_time = None

        # 计费参数
        self.response_prompt_tokens = 0
        self.response_completion_tokens = 0
        self.response_total_tokens = 0

        # __gemini_client()
        self.client = None  # 初始

        # __get_files_from_kwargs()
        self.files_dict = {}
        self.types_dict = {}
        self.uploaded_files = {}

        # __convert_openai_messages_to_gemini()
        self.messages_gemini = None
        self.system_gemini = None

        # __convert_openai_tools_to_gemini()
        self.tookit_gemini = {}

        # chat()
        self.response_status = 0
        self.response_content = None

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
        self.tool_remark_color = '\033[34;2m'  # 暗蓝色

        self.bold = '\033[1m'  # 加粗
        self.system_remind = '\033[90m'  # 亮黑色
        self.end_style = '\033[0m'  # 还原

        # 敏感内容
        self.safety_settings = [
            {
                "category": types.HarmCategory.HARM_CATEGORY_HARASSMENT,  # 骚扰
                "threshold": types.HarmBlockThreshold.BLOCK_ONLY_HIGH,  # 仅阻止高概率有害
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,  # 仇恨言论
                "threshold": types.HarmBlockThreshold.BLOCK_ONLY_HIGH,  # 仅阻止高概率有害
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  # 性内容
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,  # 不阻止
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  # 危险内容
                "threshold": types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,  # 仅阻止中等及以上有害
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,  # 选举相关
                "threshold": types.HarmBlockThreshold.BLOCK_ONLY_HIGH,  # 仅阻止高概率有害
            },
        ]

    # 登陆 Gemini
    def __gemini_client(self) -> None:
        """
        通过显式 API 登陆 Gemini
        Log in to Gemini via an explicit API.

        :return: None
        """

        self.client = genai.Client(api_key=self.api_key)

        return None

    # 读取 **kwargs 中的信息
    def __get_files_from_kwargs(self, file_dic: dict) -> None:
        """
        从 kwargs 中读取 file_1, file_2, … 参数，返回两个字典
        Read file_1, file_2,... from kwargs Parameters, returning two dictionaries.

        1. files_dict: {'file_1': '路径1', 'file_2': '路径2', ...}
        2. types_dict: {'file_1': 'application/pdf', 'file_2': 'image/jpeg', ...}

        :param file_dic: (dict) 读取文件的 dict
        """

        files_dict = {}
        types_dict = {}

        for key, value in file_dic.items():
            if key.startswith("file_"):
                files_dict[key] = value

                # 根据路径自动猜测 mime_type
                mime_type, _ = mimetypes.guess_type(value)
                if mime_type is None:
                    mime_type = "application/octet-stream"  # 默认二进制类型
                types_dict[key] = mime_type

        # 按 file_ 序号排序
        self.files_dict = dict(sorted(files_dict.items(), key=lambda x: int(x[0].split("_")[1])))
        self.types_dict = dict(sorted(types_dict.items(), key=lambda x: int(x[0].split("_")[1])))

        # 放入 client 中
        for key in self.files_dict:
            path = self.files_dict[key]
            mime_type = self.types_dict[key]

            # 判断是 URL 还是本地路径
            if path.startswith("http://") or path.startswith("https://"):
                response = requests.get(path)
                file_io = BytesIO(response.content)
            elif os.path.exists(path):
                file_io = open(path, "rb")
            else:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"{key}the specified path is invalid:{path}")

            # 上传文件
            uploaded_file = self.client.files.upload(
                file=file_io,
                config=dict(mime_type=mime_type)
            )

            # 检查文件上传
            wait_time = 2
            while uploaded_file.state.name == "PROCESSING":
                # 每隔2秒检查一次状态
                time.sleep(wait_time)

            # 检查文件上传是否成功
            if uploaded_file.state.name == "FAILED":
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"file processing failed. Please check if {self.files_dict[key]} "
                                 f"is damaged or if the format is supported. ")

            self.uploaded_files[key] = uploaded_file

            # 如果是本地打开的文件，需要关闭
            if not isinstance(file_io, BytesIO):
                file_io.close()

        return None

    # 将 OpenAI 的 messages 转换成 Gemini 格式
    def __convert_openai_messages_to_gemini(self) -> list:
        """
        将类似 OpenAI SDK 的 messages（list of dicts with 'role' & 'content'）,
        转换为 Gemini SDK 的 Content 对象列表
        messages (list of dicts with 'role' & 'content') similar to those in the OpenAI SDK
        Convert to the list of Content objects of the Gemini SDK.

        :return messages_gemini: (list) Gemini 结构的 messages
        """

        system_instruction = ""
        messages_gemini = []

        if self.messages is not None:
            for msg in self.messages:
                role = msg["role"]
                content = msg["content"].strip()

                if role == "system":
                    system_instruction += content + "\n"
                elif role == "user":
                    messages_gemini.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                    messages_gemini.append({"role": "model", "parts": [{"text": content}]})

        # 如果有上传的文件，放到最后一个用户消息的 parts 中
        if getattr(self, "uploaded_files", {}) != {}:  # 避免报错
            # 把 self.uploaded_files 的所有值，转成 list
            uploaded_values = list(self.uploaded_files.values())

            # 插入到 self.messages_gemini 的前面
            self.messages_gemini = uploaded_values + messages_gemini

        # 没有需要上传的文件
        else:
            self.messages_gemini = messages_gemini

        self.system_gemini = system_instruction.strip()

        return messages_gemini

    # 将 OpenAI 的 tookit 转换成 Gemini格式
    def __convert_openai_tools_to_gemini(self) -> None:
        """
        将 OpenAI/DeepSeek 格式的工具列表转换为 Gemini SDK 兼容的 Tool 对象
        Convert the list of tools in OpenAI/DeepSeek format to Gemini SDK-compatible Tool objects.
        """

        openai_toolkit = AI.toolkit

        # Gemini SDK 的类型映射
        TYPE_MAP = {
            "string": Type.STRING,
            "integer": Type.INTEGER,
            "number": Type.NUMBER,
            "boolean": Type.BOOLEAN,
            "object": Type.OBJECT,
            "array": Type.ARRAY,
        }

        gemini_declarations = []
        for tool_dict in openai_toolkit:
            if tool_dict.get("type") != "function":
                continue

            func_def = tool_dict["function"]

            # 1. 解析参数 (properties)
            properties_schema = {}
            # 使用 .get() 确保在 'parameters' 或 'properties' 不存在时不会出错
            param_defs = func_def.get("parameters", {}).get("properties", {})

            for param_name, param_info in param_defs.items():
                param_type_str = param_info.get("type", "string")

                # 处理可选类型，如 ["string", "null"]
                # 在 Gemini 中，可选性由 `required` 列表决定，我们只需提取基本类型
                if isinstance(param_type_str, list):
                    primary_type_str = next((t for t in param_type_str if t != "null"), "string")
                else:
                    primary_type_str = param_type_str

                gemini_type = TYPE_MAP.get(primary_type_str, Type.STRING)  # 如果找不到类型，默认为 STRING

                properties_schema[param_name] = Schema(
                    type=gemini_type,
                    description=param_info.get("description", "")
                )

            # 2. 创建 FunctionDeclaration
            declaration = FunctionDeclaration(
                name=func_def["name"],
                description=func_def["description"],
                parameters=Schema(
                    type=Type.OBJECT,
                    properties=properties_schema,
                    required=func_def.get("parameters", {}).get("required", [])
                )
            )
            gemini_declarations.append(declaration)

        # 3. 将所有声明包装成一个 Tool 对象
        self.tookit_gemini = Tool(function_declarations=gemini_declarations)

        return None

    # 与 Gemini 大模型聊天
    def chat(self, messages: Optional[List[dict]] = None, show_response: bool = True, raise_error: bool = False,
             return_all_messages: bool = True, **kwargs) -> List[dict] or str or None:
        """
        与 AI 大模型聊天，单次交互，传入完整 messages，返回 AI 回复并保存
        Chat with the AI large model, with only one interaction, return one AI response and save it.

        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param show_response: (bool) 是否打印返回的 reasoning & content，默认为 True
        :param raise_error: (bool) 遇到响应问题时为抛出错误，否则打印错误。默认为 False
        :param return_all_messages: (bool) 返回内容为单次消息 response_content or 整个 messages list，默认为 True

        :return result_content: (str / list) AI 返回的单次消息 response_content or 整个 messages list
        """

        # 初始化
        self.__gemini_client()

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages

        # 取出参数
        file_dic = {}
        for key in list(kwargs.keys()):
            if key.startswith('file_'):
                # pop() 会查找键，返回其值，然后从字典中移除这个键值对
                file_dic[key] = kwargs.pop(key)

        # 分类并读取文件
        self.__get_files_from_kwargs(file_dic)

        # 更改请求 messages
        self.__convert_openai_messages_to_gemini()

        # 流式输出
        if self.stream:

            try:
                # 打印 AI 的回复内容
                if show_response:
                    print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                          f"{self.assistant_content_color}")

                # 发送请求
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=self.messages_gemini,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_gemini,
                        safety_settings=self.safety_settings,
                        **kwargs
                    )
                )

                # 流式打印
                response_content = ""
                for chunk in response:

                    if show_response:
                        print(chunk.text, end="")

                    if chunk.text is not None:
                        response_content += chunk.text

                # 结束打印
                if show_response:
                    print(f"{self.end_style}\n")

            except errors.ClientError as e:

                if raise_error:
                    class_name = self.__class__.__name__
                    method_name = inspect.currentframe().f_code.co_name
                    raise HTTPError(
                        f"\033[95mIn {method_name} of {class_name}\033[0m, "
                        f"{e}"
                    )
                else:
                    print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                          f"{e}")

                return []

            # 保存 AI 回复为 assistant 角色追加到 self.messages
            self.response_content = response_content
            self.messages.append({"role": "assistant", "content": self.response_content})

        # 非流式输出
        else:

            try:
                # 发送请求
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.messages_gemini,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_gemini,
                        safety_settings=self.safety_settings,
                        **kwargs
                    )
                )

            except errors.ClientError as e:

                if raise_error:
                    class_name = self.__class__.__name__
                    method_name = inspect.currentframe().f_code.co_name
                    raise HTTPError(
                        f"\033[95mIn {method_name} of {class_name}\033[0m, "
                        f"{e}"
                    )
                else:
                    print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                          f"{e}")

                return []

            # 生成的文本内容
            response_content = response.text
            self.response_content = response_content

            # 结束打印
            if show_response:
                print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                      f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

            # 保存 AI 回复为 assistant 角色追加到 self.messages
            self.messages.append({"role": "assistant", "content": self.response_content})

            # 提取 token 数据
            response_tokens = response.usage_metadata
            if response_tokens.prompt_token_count is not None:
                self.response_prompt_tokens += response_tokens.prompt_token_count
            if response_tokens.candidates_token_count is not None:
                self.response_completion_tokens += response_tokens.candidates_token_count
            if response_tokens.total_token_count is not None:
                self.response_total_tokens += response_tokens.total_token_count

        if return_all_messages:  # 返回所有 messages
            result_content = self.messages
        else:
            result_content = self.response_content  # 返回单次 response_content

        return result_content

    # 与 Gemini 大模型持续聊天
    def continue_chat(self, system_content: Optional[str] = None, messages: Optional[List[dict]] = None,
                      end_token: str = '', stream: bool = True, raise_error: bool = False, **kwargs)\
            -> List[dict]:
        """
        与 Gemini AI 大模型连续聊天
        Continuous chatting with Gemini AI large models, supporting streaming, can use the Tools in the class Tools.

        注意：
        1.  想要退出需要输入：'退出', 'exit' 或 'quit'
        2.  end_token 默认情况下，只有在空的一行输入换行符 '\n' 或空按“回车”才会将内容输入给 AI 模型，否则只是换到下一行并等待继续输入，
            此情况下最下面的换行符 \n 不会保留
        3.  允许使用类 Tools 中的方法。顺序为：user-assistant-tool-assistant

        Note:
        1.  To exit, you need to enter: '退出', 'exit' or 'quit'.
        2.  By default, the content of end_token will only be input to the AI model when a newline character
            '\n' is entered on an empty chunk or when an empty "Enter" is pressed; otherwise, it will simply move to
            the next chunk and wait for further input. In this case, the bottom newline character \n will
            not be retained.
        3.  Methods in the class Tools are allowed to be used. The sequence is: user-assistant-tool-assistant

        :param system_content: (str) 'role': 'system' 中的 content 的内容，被赋值时会消除前面的所有对话记录。
                               如果未赋值则运用初始信息，默认为初始信息
        :param messages: (List[dict]) 完整对话消息列表，包括 system、user 等角色消息
        :param end_token: (str) 输入结束 token，在检测到该 token 并后紧跟 '\n' 时结束输入过程并输入，默认为换行符 '' 代表换行符，
                                此时在检测到一个空行后紧跟一个换行符代表输入结束。此参数不允许包含换行符
        :param stream: (bool) 是否启用流输出 (逐字返回)，默认为 True
        :param raise_error: (bool) 遇到响应问题时为抛出错误，否则打印错误。默认为 False

        :return messages: (list) AI 返回的消息列表 messages
        """

        # 初始化
        self.__gemini_client()

        # 将 stream 记录
        self.stream = stream

        # 检查 end_token 是否包含换行符 '\n'
        if '\n' in end_token or '\r' in end_token:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"end_token cannot contain chunk breaks.")

        # 检查 messages 的赋值情况
        if messages is not None:
            self.messages = messages
        # 检查 system_content 赋值
        if system_content is not None:
            self.messages = [{"role": "system",
                              "content": system_content}]

        print(f"Let's start chatting! The current model is \033[31m{self.model}\033[0m.")
        self.start_time = time.time()

        # 对话循环
        while True:

            # 获取用户输入
            user_input_list = []
            prompt = f"{self.bold}{self.user_role_color}User{self.end_style}: "  # 绿色加粗 User:

            if end_token == '':
                # 空 token：空行换行直接结束
                while True:
                    chunk = input(prompt)
                    if chunk == '':  # 空行直接结束
                        break
                    user_input_list.append(chunk)
                    prompt = f"{self.bold}{self.user_role_color}----> {self.end_style}"
            else:
                # 非空 token: 以 token 结尾换行才结束
                while True:
                    chunk = input(prompt)
                    if chunk.endswith(end_token):
                        content_line = chunk[:-len(end_token)].rstrip()
                        # 如果这一行只有 token（去掉后为空），保留一个空行
                        if content_line == '':
                            user_input_list.append('')
                        else:
                            user_input_list.append(content_line)
                        break
                    user_input_list.append(chunk)
                    prompt = f"{self.bold}{self.user_role_color}----> {self.end_style}"

            user_input = "\n".join(user_input_list)

            # 退出条件
            if user_input.lower() in ['退出', 'exit', 'quit']:

                end_time = time.time()  # 记录结束时间
                running_time = end_time - self.start_time  # 计算运行时长 (单位：秒)

                # 根据运行时长选择不同的输出格式
                if running_time >= 3600:  # 大于等于1小时
                    hours = int(running_time // 3600)
                    minutes = int((running_time % 3600) // 60)
                    seconds = int(running_time % 60)
                    print(
                        f"The conversation has ended. It took {self.tool_remark_color}{hours}{self.end_style} hours, "
                        f"{self.tool_content_color}{minutes}{self.end_style} minutes and "
                        f"{self.tool_role_color}{seconds}{self.end_style} seconds.")
                elif running_time >= 60:  # 大于等于1分钟但小于1小时
                    minutes = int(running_time // 60)
                    seconds = int(running_time % 60)
                    print(f"The conversation has ended. It took {self.tool_content_color}{minutes}{self.end_style} "
                          f"minutes and {self.tool_role_color}{seconds}{self.end_style} seconds.")
                else:  # 小于1分钟
                    print(f"The conversation has ended. It took {self.tool_role_color}{int(running_time)}"
                          f"{self.end_style} seconds.")

                if stream:  # '流式'
                    print('Goodbye, see you next time ^_< !\n')
                else:  # 非'流式'
                    print(
                        f'\nIn this conversation, the input contains {self.user_role_color}'
                        f'{self.response_prompt_tokens}{self.end_style} '
                        f'characters, and the output has {self.assistant_role_color}{self.response_completion_tokens}'
                        f'{self.end_style} characters.')
                    print('Goodbye, see you next time ^_< !\n')
                break

            # 添加用户消息到对话历史
            self.messages.append({"role": "user", "content": user_input})

            # 更改请求 messages
            self.__convert_openai_messages_to_gemini()

            # 流式输出
            if stream:

                try:
                    # 打印 AI 的回复内容
                    print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                          f"{self.assistant_content_color}")

                    # 发送请求
                    response = self.client.models.generate_content_stream(
                        model=self.model,
                        contents=self.messages_gemini,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_gemini,
                            safety_settings=self.safety_settings,
                            **kwargs
                        )
                    )

                    # 流式打印
                    text_content = ""
                    for chunk in response:

                        print(chunk.text, end="")

                        if chunk.text is not None:
                            text_content += chunk.text

                    # 结束打印
                    print(f"{self.end_style}\n")

                except errors.ClientError as e:

                    if raise_error:
                        class_name = self.__class__.__name__
                        method_name = inspect.currentframe().f_code.co_name
                        raise HTTPError(
                            f"\033[95mIn {method_name} of {class_name}\033[0m, "
                            f"{e}"
                        )
                    else:
                        print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                              f"{e}")

                    return []

                self.response_content = text_content

                # 保存 AI 回复为 assistant 角色追加到 self.messages
                self.messages.append({"role": "assistant", "content": self.response_content})

            # 非流式输出
            else:

                try:
                    # 发送请求
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=self.messages_gemini,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_gemini,
                            safety_settings=self.safety_settings,
                            **kwargs
                        )
                    )

                except errors.ClientError as e:
                    self.response_status = e.status_code
                    message = AI.status_code_messages.get(self.response_status, "Unknown Error")

                    if raise_error:
                        class_name = self.__class__.__name__
                        method_name = inspect.currentframe().f_code.co_name
                        raise HTTPError(
                            f"\033[95mIn {method_name} of {class_name}\033[0m, "
                            f"request failed! status_code: {self.response_status} ({message})"
                        )
                    else:
                        print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                              f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

                    return []

                # 生成的文本内容
                self.response_content = response.text

                # 结束打印
                print(f"{self.bold}{self.assistant_role_color}{self.model}{self.end_style}: "
                      f"{self.assistant_content_color}{self.response_content}{self.end_style}\n")

                # 保存 AI 回复为 assistant 角色追加到 self.messages
                self.messages.append({"role": "assistant", "content": self.response_content})

                # 提取 token 数据
                response_tokens = response.usage_metadata
                self.response_prompt_tokens += response_tokens.prompt_token_count
                self.response_completion_tokens += response_tokens.candidates_token_count
                self.response_total_tokens += response_tokens.total_token_count

        messages = self.messages

        return messages


""" 即梦 AI 图 & 视频模型 """
class Jimeng:
    """
    Jimeng 视频生成
    """

    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None,
                 prompt: Optional[str] = None, save_path: Optional[str] = None):
        """
        初始化Jimeng视频生成器

        参数:
            access_key: 访问密钥ID
            secret_key: 秘密访问密钥
            prompt: 视频描述文本，必需输入
            save_path: 视频保存目录，必需输入
        """

        # 参数初始化
        if access_key is not None:
            self.access_key = access_key
        else:
            self.access_key = Jimeng_api_key
        if secret_key is not None:
            self.secret_key = secret_key
        else:
            self.secret_key = Jimeng_api_secret
        if prompt is not None:
            self.prompt = prompt
        else:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, the prompt cannot be None.")
        if save_path is not None:
            self.save_path = save_path
        else:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, the save_path cannot be None.")

        self.task_id = None
        self.video_url = None

        # 服务配置常量
        self.method = 'POST'
        self.host = 'visual.volcengineapi.com'
        self.region = 'cn-north-1'
        self.endpoint = 'https://visual.volcengineapi.com'
        self.service = 'cv'
        self.api_version = '2022-08-31'
        self.req_key = 'jimeng_vgfm_t2v_l20'

        # ANSI颜色代码
        self.header = '\033[95m'
        self.okblue = '\033[94m'
        self.okcyan = '\033[96m'
        self.okgreen = '\033[92m'
        self.warning = '\033[93m'
        self.fail = '\033[91m'
        self.endc = '\033[0m'
        self.bold = '\033[1m'

        # 验证输入参数
        if not all([self.access_key, self.secret_key, self.prompt, self.save_path]):
            print(self.fail + 'Error: Missing required input parameters' + self.endc)
            sys.exit(1)

    # 生成签名
    def __generate_signature(self, current_date: Optional[str] = None, date_stamp: Optional[str] = None,
                             payload_hash: Optional[str] = None, canonical_querystring: Optional[str] = '') -> str:
        """
        生成请求签名 (HMAC-SHA256)

        :param current_date: (Optional[str]) 当前时间戳，格式为 ISO 8601，如 '20250926T212000Z'。
        :param date_stamp: (Optional[str]) 日期戳，格式为 'YYYYMMDD'。
        :param payload_hash: (Optional[str]) 请求体内容的 SHA256 哈希值。
        :param canonical_querystring: (Optional[str]) 请求的规范查询字符串，默认为空。

        :return: (str) 完整的 Authorization 签名字符串。
        """

        # 构建规范请求
        canonical_uri = '/'
        signed_headers = 'content-type;host;x-content-sha256;x-date'
        content_type = 'application/json'

        canonical_headers = (
            f"content-type:{content_type}\n"
            f"host:{self.host}\n"
            f"x-content-sha256:{payload_hash}\n"
            f"x-date:{current_date}\n"
        )

        canonical_request = (
            f"{self.method}\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )

        # 构建待签名字符串
        algorithm = 'HMAC-SHA256'
        credential_scope = f"{date_stamp}/{self.region}/{self.service}/request"
        canonical_request_hash = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        string_to_sign = f"{algorithm}\n{current_date}\n{credential_scope}\n{canonical_request_hash}"

        # 生成签名密钥
        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

        k_date = sign(self.secret_key.encode('utf-8'), date_stamp)
        k_region = sign(k_date, self.region)
        k_service = sign(k_region, self.service)
        k_signing = sign(k_service, 'request')

        # 计算签名
        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

        return (f"{algorithm} Credential={self.access_key}/{credential_scope}, SignedHeaders={signed_headers}, "
                f"Signature={signature}")

    # 发送请求
    def __send_signed_request(self, action: str, body_params: Dict[str, Any]) -> requests.Response:
        """
        发送签名请求

        :param action: (str) API 接口操作名称，例如 'CVSync2AsyncGetResult'。
        :param body_params: (Dict[str, Any]) 请求体参数字典，将被 JSON 序列化后发送。

        :return: (requests.Response) requests 库返回的响应对象，包含接口返回的内容。
        """

        if not self.access_key or not self.secret_key:
            print(self.fail + 'Error: Missing API credentials' + self.endc)
            sys.exit(1)

        # 获取当前时间
        current_utc = datetime.now(timezone.utc)
        current_date = current_utc.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = current_utc.strftime('%Y%m%d')

        # 构建查询参数
        query_params = {
            'Action': action,
            'Version': self.api_version,
        }
        canonical_querystring = '&'.join([
            f"{k}={urllib.parse.quote_plus(str(v))}" for k, v in sorted(query_params.items())
        ])

        # 构建请求体
        req_body = json.dumps(body_params)
        payload_hash = hashlib.sha256(req_body.encode('utf-8')).hexdigest()

        # 生成签名
        authorization_header = self.__generate_signature(
            current_date, date_stamp, payload_hash, canonical_querystring
        )

        # 构建请求头
        headers = {
            'X-Date': current_date,
            'Authorization': authorization_header,
            'X-Content-Sha256': payload_hash,
            'Content-Type': 'application/json'
        }

        # 发送请求
        request_url = f"{self.endpoint}?{canonical_querystring}"

        return requests.post(request_url, headers=headers, data=req_body)

    # 提交任务
    def submit_task(self) -> int or None:
        """
        提交视频生成任务

        :return: (Optional[int]) 提交成功返回任务 ID (task_id)，失败返回 None。
        """

        print(self.okcyan + "\nSubmitting video generation task..." + self.endc)

        body_params = {
            "req_key": self.req_key,
            "prompt": self.prompt,
            "seed": "-1",
            "aspect_ratio": "21:9",
        }

        response = self.__send_signed_request('CVSync2AsyncSubmitTask', body_params)

        if response.status_code != 200:
            print(self.fail + f"Error: Task submission failed, status code: {response.status_code}" + self.endc)
            return None

        try:
            response_data = response.json()
            if response_data.get('code') != 10000:
                error_msg = response_data.get('message', 'Unknown error')
                print(self.fail + f"API Error: {error_msg}" + self.endc)
                return None

            self.task_id = response_data['data']['task_id']
            print(self.okgreen + f"Success: Task submitted! Task ID: {self.task_id}" + self.endc)
            return self.task_id

        except Exception as e:
            print(self.fail + f"Error: Failed to parse task ID - {e}" + self.endc)
            print(f"Raw response: {response.text[:200]}...")
            return None

    # 获取任务
    def __get_task_result(self) -> requests.Response:
        """
        获取任务结果

        :return: (dict) 任务执行结果，内容由接口返回的 JSON 数据组成。
        """

        body_params = {
            "req_key": self.req_key,
            "task_id": self.task_id
        }
        return self.__send_signed_request(action='CVSync2AsyncGetResult', body_params=body_params)

    # 下载视频
    def __download_video(self) -> bool:
        """
        下载视频文件

        :return: (bool) 下载成功返回 True，失败返回 False。
        """

        try:
            print(self.okcyan + f"Downloading video: {self.video_url}" + self.endc)
            filename = f"video_{self.task_id}.mp4"
            file_path = os.path.join(self.save_path, filename)

            response = requests.get(self.video_url, stream=True)

            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(self.okgreen + f"Success: Video saved to '{file_path}'" + self.endc)
                return True
            else:
                print(self.fail + f"Error: Download failed with status code {response.status_code}" + self.endc)
                return False

        except Exception as e:
            print(self.fail + f"Error: Exception occurred during video download - {e}" + self.endc)
            return False

    # 动态进度
    @staticmethod
    def __display_waiting(seconds: int, attempt: int, max_attempts: int) -> None:
        """
        显示动态等待进度条，用于任务轮询或延时提示。

        :param seconds: (int) 当前等待的秒数。
        :param attempt: (int) 当前尝试次数。
        :param max_attempts: (int) 最大尝试次数。

        :return: (None) 方法不返回任何值，仅在控制台显示进度。
        """

        for sec in range(1, seconds + 1):
            sys.stdout.write(f"\rattempt #{attempt}/{max_attempts} - has waited for {sec} seconds...")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

        return None

    # 查询结果
    def __poll_task_result(self, max_attempts: int = 20, wait_seconds: int = 60) -> Optional[str]:
        """
        轮询任务结果，直到任务完成或达到最大尝试次数。
        成功时下载视频文件并返回视频 URL。

        :param max_attempts: (int) 最大轮询次数，默认 20。
        :param wait_seconds: (int) 每次轮询等待时间（秒），默认 60。

        :return: (Optional[str]) 成功返回生成视频的 URL，失败或未生成返回 None。
        """
        start_time = time.time()
        print(self.okcyan + "\nStart polling the task status..." + self.endc)

        for attempt in range(1, max_attempts + 1):
            attempt_start = time.time()

            # 显示等待进度
            self.__display_waiting(wait_seconds, attempt, max_attempts)

            actual_wait = time.time() - attempt_start
            total_time = time.time() - start_time

            print(f"attempt #{attempt}/{max_attempts} to complete")
            print(f"This waiting: {actual_wait:.1f} seconds; total time: {total_time:.1f} seconds")

            # 获取任务结果
            response = self.__get_task_result()

            if response.status_code != 200:
                print(self.warning + f"Warning: Query failed. Status code: {response.status_code}" + self.endc)
                continue

            try:
                result = response.json()
                if result.get('code') != 10000:
                    error_msg = result.get('message', 'Unknown error')
                    print(self.warning + f"API Warning: {error_msg}" + self.endc)
                    continue

                task_data = result['data']
                status = task_data.get('status')
                print(f"Task status: {status}")

                if status == "done":
                    self.video_url = task_data.get('video_url')

                    if not self.video_url:
                        try:
                            resp_data = json.loads(task_data.get('resp_data', '{}'))
                            urls = resp_data.get('urls', [])
                            if urls:
                                self.video_url = urls[0]
                        except Exception as e:
                            print(self.warning + f"Warning: Parsing resp_data failed - {e}" + self.endc)

                    if self.video_url:
                        print("\n" + "=" * 60)
                        print(self.header + self.bold + "The video was generated successfully!" + self.endc)
                        print(f"Video URL: {self.okcyan}{self.video_url}{self.endc}")
                        print(f"Task ID: {self.task_id}")
                        print(f"Request ID: {result.get('request_id', 'N/A')}")
                        print("=" * 60 + "\n")

                        if self.__download_video():
                            print(self.okgreen + "Video processing completed!" + self.endc)
                        return self.video_url
                    else:
                        print(self.fail + "Error: No valid video URL was found" + self.endc)

                elif status == "failed":
                    error_info = task_data.get('error', "Unknown error")
                    print(self.fail + f"Error: Task failed - {error_info}" + self.endc)
                    break
                else:
                    print(f"Information: Task processing in progress. Current status: {status}")

            except Exception as e:
                print(self.fail + f"Error: The parsing response failed - {e}" + self.endc)
                print(f"Original response: {response.text[:500]}...")

        return None

    # 生成视频
    def run_video_generation(self) -> None:
        """
        运行视频生成工作流，包括创建保存目录、提交任务、轮询任务结果和下载视频。

        过程说明：
        1. 创建视频保存目录（如果不存在）。
        2. 打印任务信息（视频描述和保存路径）。
        3. 提交视频生成任务，并获取任务 ID。
        4. 轮询任务结果，直到生成完成或达到最大轮询次数。
        5. 下载生成的视频文件，如果任务失败或超时则给出提示。

        :return: (None) 方法不返回值，所有结果通过控制台打印或文件保存体现。
        """
        # 创建保存目录
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(self.okcyan + f"Created directory: {self.save_path}" + self.endc)

        # 打印任务信息
        print("\n" + "=" * 60)
        print(self.header + self.bold + "Start the video generation task" + self.endc)
        print(f"Video description: {self.okcyan}{self.prompt}{self.endc}")
        print(f"Save directory: {self.okcyan}{self.save_path}{self.endc}")
        print("=" * 60 + "\n")

        # 提交任务
        self.task_id = self.submit_task()
        if not self.task_id:
            print(self.fail + "Error: Task submission failed, program exits" + self.endc)
            sys.exit(1)

        # 轮询任务结果
        self.video_url = self.__poll_task_result()

        if not self.video_url:
            print(self.warning + f"\nWarning: Exceeding the maximum number of polls" + self.endc)
            print(f"You can manually query later using this task ID: {self.okcyan}{self.task_id}{self.endc}")

        return None


""" AI 大模型的应用 """
class Assist(AI):
    """
    应用各种 AI 大模型完成生产力活动

    Apply various AI large models such as ChatGPT, DeepSeek, Claude, Gemini, Grok, etc., to complete productivity
    activities such as drawing, writing articles, revising articles, and analyzing data.
    """

    # 初始化，应当包含所有 AI 大模型的参数
    def __init__(self,

                 # 必要参数 (4)
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, messages: Optional[List[dict]] = None,

                 # 自定义参数 (4)
                 ai_keyword: Optional[str] = None, instance_id: Optional[str] = None,
                 information: Optional[str] = None, show_reasoning: bool = False,

                 # 附加参数 (11)
                 max_tokens: int = 128000, temperature: float = 0.7, top_p: float = 1.0, n: int = 1,
                 stream: bool = False, stop: Union[str, list, None] = None, presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0, seed: Optional[int] = None, tools: Optional[list] = None,
                 tool_methods: Optional[dict] = None, tool_choice: str = "auto"):
        """
        推理 AI 大模型公有参数
        Public parameters of the inference AI large model.

        # 必要参数 (4)
        :param api_key: (str) 输入的 API KEY，即 API 密钥
        :param base_url: (str) 输入的 base URL
        :param model: (str) 指定使用的模型，如 'deepseek-chat' 与 'deepseek-reasoner'
        :param messages: (list) 对话消息列表，包含完整对话历史，最后一条为当前发送的信息

        # 自定义参数 (4)
        :param ai_keyword: (str) 自定义 AI 关键词，可以将 api_key 与 base_url 关联起来
        :param instance_id: (str) AI 大模型的实例化 id，该 id 可以直接被实例化对象打印
        :param information: (str) 当前 AI 被实例化后的信息，自定义输入，用于区分多个 AI 模型
        :param show_reasoning: (bool) 是否打印推理过程，如果有推理的话。默认为 False

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
        :param tools: (list) 工具信息条，用于描述工具
        :param tool_methods: (dict) 工具包，放有具体工具
        :param tool_choice: (str) 工具选取方式，"auto" 为自动选取，"none" 为决不会选取"，
                            {"type": "function", "function": {"name": "xxx"}} 为强制调用指定工具，并且只能调用它。
                            "required"(部分文档称为 {"type": "function", "function": "required"} 的形式)，
                            模型必须调用某个工具，但可以自己选择哪一个
        """

        # 超类初始化
        super().__init__(
            # 必要参数 (4)
            api_key=api_key, base_url=base_url, model=model, messages=messages,

            # 自定义参数 (4)
            ai_keyword=ai_keyword, instance_id=instance_id, information=information, show_reasoning=show_reasoning,

            # 附加参数 (11)
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, stream=stream, stop=stop,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty, seed=seed, tools=tools,
            tool_methods=tool_methods, tool_choice=tool_choice
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
        self.target_url = self.base_url + endpoint

        # 构建 headers
        self.headers = {
            # "User-Agent": "<…>",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构建请求体
        self.request_body_dict = {
            # 重要参数
            "model": model_revise_manuscript,
            "messages": self.messages,
            # 文本参数
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            # 流式输出
            "stream": self.stream,
            # 种子参数
            "seed": self.seed,
            # # 敏感内容
            # "extra_body": {"safetySettings": self.safety_settings},
        }

        # 工具参数
        if self.tools:  # 只有在 self.tools 不为空时有效
            self.request_body_dict["tools"] = self.tools
            self.request_body_dict["tool_choice"] = self.tool_choice

        # 发送请求
        self.response = requests.post(
            url=self.target_url,
            headers=self.headers,
            json=self.request_body_dict
        )

        # 查看状态码
        self.response_status = self.response.status_code

        # 判断请求是否成功
        if self.response_status != 200:
            # 抛出错误
            if raise_error:
                message = AI.status_code_messages.get(self.response_status, "Unknown Error")
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise HTTPError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                f"request failed! status_code: {self.response_status} ({message})")

            # 改为打印信息
            else:
                message = AI.status_code_messages.get(self.response_status, "Unknown Error")  # 未知错误
                print(f"{self.system_remark_color}[{self.model}]{self.end_style} "
                      f"\033[31mRequest failed!\033[0m status_code: {self.response_status} ({message})")

        # 解析返回参数
        response_dict = self.response.json()
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
        self.player_configuration = {}  # setup_environment()
        self.player_configuration_model = {}  # setup_environment_models()
        self.player_configuration_list = {}  # setup_environment_list()
        self.ai_models = []  # setup_environment_list()

        # 真人玩家参数
        self.man_number = man_number

        # AI 玩家参数
        self.ai_number = ai_number

    # 配置环境，几位真人，几个 AI
    def setup_environment(self, man_number: Optional[int] = None, ai_number: Optional[int] = None,
                          default_ai_model: str = 'deepseek', show_result: bool = False, **kwargs) -> dict:
        """
        环境配置：有几位真人玩家与几个 AI，多出的 AI 用 DeepSeek 补全
        The environmental configuration: several real players and several ais.
        The extra ais are completed with DeepSeek.

        :param man_number: (int) 真人玩家的数量，默认为 None，表示根据需要分配
        :param ai_number: (int) AI 玩家的数量，分配 AI 之和的总数需要小于等于 AI 玩家的总数。不足的用 DeepSeek AI 补全。
                                默认为 None，表示根据需要分配
        :param default_ai_model: (str) 默认的 AI 模型，默认为 deepseek-ai/DeepSeek-R1
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

        # 校验 default_ai_model（直接是模型名称）
        if not isinstance(default_ai_model, str) or not default_ai_model.strip():
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(
                f"\033[95mIn {method_name} of {class_name}\033[0m, "
                f"default_ai_model must be a non-empty string representing the AI model name."
            )

        # 获取各类型数量并校验（只校验 ai_model_map 里的）
        ai_counts = {}
        for ai_type in ai_model_map.keys():
            count = kwargs.get(f"{ai_type}_ai_number", 0) or 0
            if count < 0:
                raise ValueError(f"{ai_type}_ai_number cannot be less than 0.")
            ai_counts[ai_type] = count

        # 校验 ai_number
        total_specified = sum(ai_counts.values())
        if ai_number < total_specified:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(
                f"\033[95mIn {method_name} of {class_name}\033[0m, "
                f"ai_number ({ai_number}) must be greater than or equal to the sum of the quantities "
                f"of each type ({total_specified})."
            )

        # 分配多余的到 default_ai_model
        if ai_number > total_specified:
            ai_counts["default_ai"] = ai_counts.get("default_ai", 0) + (ai_number - total_specified)

        # 实例化 OtherAI
        for ai_type, count in ai_counts.items():
            for i in range(count):
                instance_id = f"{ai_type}_{i + 1}"
                if ai_type == "default_ai":
                    # 默认模型直接用 default_ai_model
                    model_name = default_ai_model
                else:
                    # 其他模型用映射表
                    model_name = ai_model_map[ai_type]
                instances[instance_id] = AI(instance_id=instance_id, model=model_name)

        # 可选调试输出
        if show_result:
            print("AI allocation result:")
            for ai_type, count in ai_counts.items():
                if ai_type == "default_ai":
                    print(f"default ({default_ai_model}): {count}")
                else:
                    print(f"{ai_type}: {count}")
            print(f"\nTotal: {man_number + sum(ai_counts.values())}  Human: {man_number}  AI: {ai_number}")

        self.player_configuration = instances

        return instances

    # 根据 model 来设置 AI 的环境
    def setup_environment_models(self, man_number: Optional[int] = None, show_result: bool = False, **kwargs) -> dict:
        """
        环境配置：支持无限个模型，用户通过 model_1, model_1_number, model_2, model_2_number ... 指定
        Environment configuration: Supports an unlimited number of models. Users can use model_1, model_1_number,
        model_2, model_2_number... Specified

        :param man_number: (int) 真人玩家数量
        :param show_result: (bool) 是否打印分配结果
        :param kwargs: 包含若干对 (model_x, model_x_number)

        :return: (dict) {player_id: 实例对象}
        """

        # 玩家数量分配
        if man_number is None:
            man_number = self.man_number

        if man_number < 0:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"man_number must be greater than or equal to 0.")

        instances = {}

        # 实例化 Human
        for i in range(man_number):
            human_key = f"human_{i + 1}"
            instances[human_key] = Human(instance_id=human_key)

        # 解析所有 model_x
        model_index = 1
        ai_counts = {}
        seen_models = set()  # 用来检查重复模型

        while True:
            model_key = f"model_{model_index}"
            number_key = f"model_{model_index}_number"

            if model_key not in kwargs:
                break  # 没有更多模型，跳出

            model_name = kwargs.get(model_key)
            model_number = kwargs.get(number_key, 0) or 0

            if not isinstance(model_name, str) or not model_name.strip():
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"{model_key} must be a non-empty string representing the AI model name.")
            if not isinstance(model_number, int) or model_number < 0:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"{number_key} must be a non-negative integer.")

            # 检查是否有重复模型
            if model_name in seen_models:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Duplicate model detected: '{model_name}'. Each model must be unique.")
            seen_models.add(model_name)

            ai_counts[model_name] = model_number
            model_index += 1

        # 实例化 AI
        for model_name, count in ai_counts.items():
            for i in range(count):
                instance_id = f"{model_name}_{i + 1}"
                instances[instance_id] = AI(instance_id=instance_id, model=model_name)

        # 可选调试输出
        if show_result:
            print("AI allocation result:")
            for model_name, count in ai_counts.items():
                print(f"{model_name}: {count}")
            print(
                f"\nTotal: {man_number + sum(ai_counts.values())}  Human: {man_number}  AI: {sum(ai_counts.values())}")

        self.player_configuration_model = instances
        return instances

    # 通过输入 ai_models 的 list 来配置 AI 的环境
    def setup_environment_list(self, man_number: int, ai_models: list, show_result: bool = False) -> dict:
        """
        环境配置：通过 list 指定 AI 模型
        Environment configuration: Use a list to specify AI models

        :param man_number: (int) 真人玩家数量
        :param ai_models: (list[str]) AI 模型名称的列表，如 ['gpt-oss-120b', 'GPT-5', 'gpt-oss-120b']
        :param show_result: (bool) 是否打印分配结果

        :return: (dict) {player_id: 实例对象}
        """

        if man_number < 0:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"man_number must be greater than or equal to 0.")

        if not isinstance(ai_models, list):
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"ai_models must be a list of strings representing AI model names.")

        instances = {}

        # 实例化 Human
        for i in range(man_number):
            human_key = f"human_{i + 1}"
            instances[human_key] = Human(instance_id=human_key)

        # 实例化 AI
        ai_counts = {}
        for model_name in ai_models:
            if not isinstance(model_name, str) or not model_name.strip():
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"ai_models must only contain non-empty strings.")

            ai_counts[model_name] = ai_counts.get(model_name, 0) + 1
            instance_id = f"{model_name}_{ai_counts[model_name]}"
            instances[instance_id] = AI(instance_id=instance_id, model=model_name)

        # 可选调试输出
        if show_result:
            print("AI allocation result (list mode):")
            for model_name, count in ai_counts.items():
                print(f"{model_name}: {count}")
            print(
                f"\nTotal: {man_number + len(ai_models)}  Human: {man_number}  AI: {len(ai_models)}"
            )

        self.player_configuration_list = instances
        return instances


""" 应用 Duck Typing 来调用实例中的 API 与 URL """
def set_api_config(ai_instance: object, api_url_pair: str):
    """
    可以改动已实例化的 AI 模型中的 API 配置
    The API configuration in the instantiated AI model can be modified.

    支持修改的输入的 api_url_pair 参数: ['1', 'ds', 'gm']

    :param ai_instance: (object) 已实例化的 AI 模型对象
    :param api_url_pair: (str) API 与 URL 配对的名称
    """

    # 获取当前函数名（用于错误信息）
    method_name = inspect.currentframe().f_code.co_name

    # 预定义的API配置 - 确保这些变量在作用域内
    try:
        api_configs = {
            # 渠道 AI
            '1': {"api_key": api_key_1, "base_url": base_url_1, "model": "gpt-oss-120b"},
            # 渠道 AI 高级卡
            '2': {"api_key": api_key_2, "base_url": base_url_1, "model": "gpt-oss-120b"},
            # DeepSeek 官方
            'ds': {"api_key": DeepSeek_api_key, "base_url": DeepSeek_base_url, "model": "deepseek-reasoner"},
            # Gemini 渠道 1
            'gm': {"api_key": Gemini_api_key_1, "base_url": Gemini_base_url, "model": "gemini-2.5-pro"},
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
