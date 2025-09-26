"""
magiclib / general

------------------------------------------------------------------------------------------------------------------------
magiclib / general is the core of the magiclib library, providing essential functionality for file handling, data
saving, visualization, and data optimization. It serves as the foundation for the library, enabling seamless reading and
writing of various file formats, generating insightful plots, and performing preprocessing or enhancement on datasets
to streamline subsequent analysis and processing tasks.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""


import re
import os
import copy
import time
import json
import shutil
import random
import chardet
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from io import StringIO
import scipy.stats as stats
from pandas import DataFrame
from scipy.stats import norm
from datetime import datetime
from natsort import natsorted
from collections import Counter
from sympy import sympify, symbols
from scipy.signal import find_peaks
from collections import OrderedDict
from PyPDF2.errors import DependencyError
from matplotlib import pyplot as plt, patches
from scipy.interpolate import UnivariateSpline
from typing import Union, Tuple, Optional, List, Dict


""" 关键常量 """
current_dir = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
database_path = os.path.join(current_dir, '../Database')  # 相对于库根目录 ('magiclib') 获取 'Magic_Database' 路径
database_path = os.path.abspath(database_path)  # 确保路径为绝对路径

Magic_Database = os.path.join(database_path, 'Magic_Database')  # 魔法库路径
Standard_Database = os.path.join(database_path, 'Standard_Database')  # 标准库路径
Pottery_Database = os.path.join(database_path, 'Pottery_Database')  # 陶器基因库路径

Category_Index = 'category'  # 分类索引
interval_time = 0.5  # 程序休息时间


""" 函数系统 """
class Function:
    """
    函数系统

    The function system can add additional functionality to the main function
    and other functions, some of which are necessary and some of which are optional.
    """

    # 初始化
    def __init__(self, data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                 title: Optional[str] = None, x_list: Optional[list] = None, y_list: Optional[list] = None,
                 x_label: str = None, y_label: str = None):
        """
        Function 函数系统为被引系统，无需赋值初始化

        输入数据 DataFrame，dict，x_list and y_list 其中的一个，title 为非必要数据
                x_list & y_list 的优先级最高，DataFrame 其次，dict 最低

        :param data_dic: (dict) 输入 dict，key 为 title，value 为 data_dic
        :param data_df:  (DataFrame) 输入 DataFrame
        :param title: (str) 数据的 title
        :param x_list: (list) x坐标的 list
        :param y_list: (list) y坐标的 list
        :param x_label: (str) x坐标的 label
        :param y_label: (str) y坐标的 label
        """

        # 接收参数初始化
        self.data_dic = data_dic
        self.data_df = data_df
        self.title = title
        self.x_list = x_list
        self.y_list = y_list
        self.x_label = x_label
        self.y_label = y_label

        # 数据初始化分配
        if type(self) == Function:  # 当 self 为 Function 的直接实例时为真
            self.data_init()

        # 以下数据为添加填，在该类初始化中可自动生成，其它子类中无需再添加
        self.category_dic = None  # (dict) value 为包含 self.Category_Index 值的 list，不含标题
        self.precision_data_dic = None  # (dict) 原数据的精度
        self.point_scale_dic = None  # (dict) value 是标准化前数据在横纵坐标上的最大和最小的值

    # 数据初始化分配模组
    def data_init(self) -> None:
        """
        数据的初始化分配：小组大，大分小
        The initialization allocation of data. Put the whole together from the pieces, then break the whole into pieces.

        思路：从零散组成数据，再从整体分配到零散
        x_list & y_list 优先度最高，data_df 其次，data_dic 最低
        但单独的 x_list or y_list 无法改变 data_dic，如改变则需要自定 title

        Approach: Start by composing data from fragmented pieces, then distribute from the whole to the fragmented.
        x_list & y_list have the highest priority, followed by data_df, and data_dic has the lowest priority.
        However, a standalone x_list or y_list cannot change data_dic;
        if a change is required, a custom title must be specified
        """

        # 检查默认值
        if self.title is not None:
            title = self.title
        else:
            title = 'Untitled'

        if self.x_label is not None:
            x_label = self.x_label
        else:
            x_label = 'X'

        if self.y_label is not None:
            y_label = self.y_label
        else:
            y_label = 'Y'

        # 优先使用 x_list & y_list
        if self.x_list is not None and self.y_list is not None:
            # 组装成 data_df
            self.data_df = pd.DataFrame({x_label: self.x_list, y_label: self.y_list})
            # 组装成 data_dic
            self.data_dic = {title: self.data_df}

        # 有 data_df 被赋值的同时 x_list or y_list 也被赋值，data_df 相应部分会被覆盖
        elif self.data_df is not None:
            # 如果有 DataFrame，但同时存在 x_list 或 y_list，则替换相应的列
            if self.x_list is not None:
                self.data_df.iloc[:, 0] = self.x_list
            if self.y_list is not None:
                self.data_df.iloc[:, 1] = self.y_list

            # 只有在 DataFrame 表格有两列时更新列名
            if len(self.data_df.columns) == 2:
                self.data_df.columns = [x_label, y_label]  # 更新列名
            # 组装成 data_dic
            self.data_dic = {title: self.data_df}

        # data_dic 不会被 x_list & y_list 覆盖
        elif self.data_dic is not None:
            # dict 类型数据的处理
            title, data_df = random.choice(list(self.data_dic.items()))
            self.title = title  # 为随机抽取
            self.data_df = data_df  # 为随机抽取

            # 直接在排序中使用正则表达式和列表推导式
            self.data_dic = OrderedDict(sorted(self.data_dic.items(),
                                               key=lambda item: [int(text) if text.isdigit() else text.lower() for text
                                                                 in re.split(pattern='([0-9]+)', string=item[0])]))

            # 更改数据类型，将数字转化为 float，其余为 str
            for data_df in self.data_dic.values():

                for col in data_df.columns:

                    if data_df[col].map(
                            lambda x: isinstance(x, (int, float)) or
                                      str(x).replace('.', '', 1).isdigit()).all():
                        data_df[col] = data_df[col].astype(float)
                    else:
                        data_df[col] = data_df[col].astype(str)

        # 将 DataFrame 数据转换为 x_list & y_list 和 x_label & y_label
        if self.data_df is not None and len(self.data_df.columns) == 2:
            column_names = self.data_df.columns.tolist()  # 将列名转换为列表
            # 将列名赋值给相应的变量
            self.x_label = column_names[0]
            self.y_label = column_names[1]

            self.x_list = self.data_df[column_names[0]].tolist()  # 提取第一列数据，并转换为列表
            self.y_list = self.data_df[column_names[1]].tolist()  # 提取第二列数据，并转换为列表

        #  遍历 data_dic，提取名为 self.Category_Index 的列为列表，存入 self.category_dic
        if self.data_dic is not None:
            category_dic = {}
            for title, data_df in self.data_dic.items():
                if self.Category_Index in self.data_df.columns:
                    category_dic[title] = data_df[self.Category_Index].tolist()
                else:
                    category_dic[title] = None
            self.category_dic = category_dic

        # 只有是 Magic 及其子类时才会为 True
        if isinstance(self, Magic):

            # 计算数据的精度
            if self.data_dic is not None:

                precision_data_dic = {}  # 创建空 dict 以接收数据
                for title, data_df in self.data_dic.items():
                    column_names = data_df.columns.tolist()  # 将列名转换为列表
                    x_list = data_df[column_names[0]].tolist()  # 提取第一列数据，并转换为列表
                    total_difference = 0

                    for i in range(1, len(x_list)):
                        difference = x_list[i] - x_list[i - 1]
                        total_difference += difference
                    precision_data = total_difference / (len(x_list) - 1)
                    precision_data_dic[title] = precision_data

                self.precision_data_dic = precision_data_dic

            # 寻找最大值和最小值点
            if self.data_dic is not None:

                point_scale_dic = {}
                for title, data_df in self.data_dic.items():
                    # 寻找最大值和最小值点
                    x_min = data_df.iloc[:, 0].min()  # 获取横坐标的最小值
                    x_max = data_df.iloc[:, 0].max()  # 获取横坐标的最大值
                    y_min = data_df.iloc[:, 1].min()  # 获取纵坐标的最小值
                    y_max = data_df.iloc[:, 1].max()  # 获取纵坐标的最大值
                    point_scale_dic[title] = {'x_min': x_min,
                                              'x_max': x_max,
                                              'y_min': y_min,
                                              'y_max': y_max}

                self.point_scale_dic = point_scale_dic

        return None

    # 储存数据清空
    def clear_data(self) -> None:
        """
        删除已经储存的数据，用 None 将其覆盖。目的是防止多次数据读取时后面的数据会因为 x_list, y_list, x_label 和 y_label 的优化使用，
        从而覆盖前面的数据
        Delete the data already stored, overwriting it with None.
        The purpose is to prevent the following data from being optimized for
        x_list, y_list, x_label and y_label. Thus overwriting the previous data.
        """

        self.data_dic = None
        self.data_df = None
        self.title = None
        self.x_list = None
        self.y_list = None
        self.x_label = None
        self.y_label = None

        return None

    # 更改图片的背景
    @staticmethod
    def change_imshow(background_color: str, background_transparency: float, show_in_one: bool = False,
                      x_min: Optional[float] = None, x_max: Optional[float] = None,
                      y_min: Optional[float] = None, y_max: Optional[float] = None) -> None:
        """
        更改图片的背景
        Change the background of the picture.

        :param background_color: (str) 需要更改的背景名
        :param background_transparency: (float) 背景图片的透明度，最小越透明
        :param show_in_one: (bool) 是否显示在一张图中
        :param x_min: (float) 横坐标的最小值，只有 show_in_one == True 时才有意义
        :param x_max: (float) 横坐标的最大值，只有 show_in_one == True 时才有意义
        :param y_min: (float) 纵坐标的最小值，只有 show_in_one == True 时才有意义
        :param y_max: (float) 纵坐标的最大值，只有 show_in_one == True 时才有意义

        :return: None
        """
        gradient = np.linspace(start=0, stop=1, num=100).reshape(-1, 1)
        np.vstack((np.array([1, 1, 0]), np.array([0.5, 0.7, 1])))

        if not show_in_one:
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()

        plt.imshow(gradient, aspect='auto', cmap=background_color, vmin=0, vmax=1,
                   extent=(x_min, x_max, y_min, y_max), alpha=background_transparency)

        return None

    # 检查 TXT 文件的编码类型
    @staticmethod
    def detect_encodings(path: str, show: bool = True) -> str:
        """
        检查 TXT 文件或一个目录下所有 TXT 文件的编码类型
        Check the encoding types of TXT files or all TXT files in a directory.

        :param path: (str) TXT 文件路径或目录路径
        :param show: (bool) 是否打印，默认为 True

        :return results: (str) 返回文件编码的结果
        """

        # 初始化空字符串来存储结果
        formatted_results = ""

        # 检测单个文件的编码
        def get_encoding(file_path):
            with open(file_path, 'rb') as f:
                # 读取 TXT 以得到其编码形式
                result = chardet.detect(f.read())
                return result['encoding']

        # 检查给定的路径是文件还是目录
        if os.path.isfile(path):
            # 如果是文件，则仅检测该文件
            filename = os.path.basename(path)
            encoding = get_encoding(path)
            # 加入到结果字符串
            formatted_results += f"The encoding of \033[94m{filename}\033[0m is \033[91m{encoding}\033[0m\n"
            if show:
                print(f"The encoding of \033[94m{filename}\033[0m is \033[91m{encoding}\033[0m")

        elif os.path.isdir(path):
            # 如果是目录，遍历其中的所有文件并检测每一个 TXT 文件
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.txt') or file.endswith('.TXT'):
                        full_path = os.path.join(root, file)
                        encoding = get_encoding(full_path)
                        # 加入到结果字符串
                        formatted_results += f"The encoding of \033[94m{file}\033[0m is \033[91m{encoding}\033[0m\n"
                        if show:
                            print(f"The encoding of \033[94m{file}\033[0m is \033[91m{encoding}\033[0m")

        else:
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"033[95mIn {method_name}\033[0m, "
                             f"\033[94m{path}\033[0m is neither a valid file nor directory.")

        results = formatted_results.strip()  # 返回去除末尾换行符的结果字符串

        return results

    # 更改 TXT 文件的编码类型
    @staticmethod
    def convert_encoding(path: str, target_encoding: str = 'UTF-16', show: bool = True) -> None:
        """
        更改 TXT 文件或一个目录下所有 TXT 文件的编码类型
        Change the encoding type of a TXT file or all TXT files in a directory.

        :param path: (str) TXT 文件路径或目录路径
        :param target_encoding: (str) 目标编码类型，默认为 UTF-16
        :param show: (bool)  是否打印更改编码类型的文件，默认为 True

        :return: None
        """

        # 读取文件并检测当前编码
        def read_and_detect_encoding(file_path):
            with open(file_path, 'rb') as f:
                content_original = f.read()
                detected = chardet.detect(content_original)
                current_encoding = detected['encoding']
                decoded_content = content_original.decode(current_encoding)
                return decoded_content

        # 写回文件使用目标编码
        def write_with_target_encoding(file_path, content_file):
            with open(file_path, 'w', encoding=target_encoding) as f:
                f.write(content_file)

        # 检查路径是文件还是目录
        if os.path.isfile(path):
            # 如果是文件，则仅转换该文件
            content = read_and_detect_encoding(path)
            write_with_target_encoding(path, content)

        elif os.path.isdir(path):
            # 如果是目录，遍历其中的所有文件并转换每一个.txt文件
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.txt') or file.endswith('.TXT'):
                        full_path = os.path.join(root, file)
                        content = read_and_detect_encoding(full_path)
                        write_with_target_encoding(full_path, content)
                        if show:
                            print(f"\033[94m{os.path.basename(full_path)}\033[0m "
                                  f"has been converted to \033[91m{target_encoding}\033[0m.")

        # 不为 TXT 文件且不为路径的情况则报错
        else:
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"033[95mIn {method_name}\033[0m, '{path}' is neither a valid file nor directory.")

        return None

    # 更改扩展名
    @staticmethod
    def rename_extension(path: str, old_extension: str, new_extension: str, show: bool = True) -> None:
        """
        更改一个文件夹下所有目标文件的扩展名
        Change the file extension of all target files in a folder.

        :param path: (str) 需要更改扩展名的文件的目录路径
        :param old_extension: (str) 旧扩展名，无需加点
        :param new_extension: (str) 新扩展名，无需加点
        :param show: (bool) 是否打印更改扩展名的文件，默认为 True

        :return: None
        """
        # 检查路径是否存在
        if not os.path.exists(path):
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise FileNotFoundError(f"In {method_name}, The path '{path}' does not exist!")

        # 遍历路径下的文件
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            # 检查文件是否是指定的旧扩展名
            if os.path.isfile(file_path) and file_path.endswith(old_extension):
                # 构造新的文件名
                new_file_name = os.path.splitext(file)[0] + '.' + new_extension
                new_file_path = os.path.join(path, new_file_name)
                # 重命名文件
                os.rename(file_path, new_file_path)
                if show:
                    print(f"File \033[94m{file}\033[0m changed to \033[91m{new_file_name}\033[0m")
        return None

    # 抽取正态分布随机点
    @staticmethod
    def get_point_normal(mean: float, std: float, number: int, show: bool = False) -> Union[List[float], object]:
        """
        生成服从正态分布的随机点，并且可以绘制图形
        Generate random points following a normal distribution and show the graph.

        :param mean: (float) 均值
        :param std: (float) 标准差
        :param number: (int) 生成的数量
        :param show: (bool) 是否绘图，默认为否

        :return point_normal_list: (list) 生成的正态分布随机点的 list
        """

        point_normal_array = np.random.normal(mean, std, number)

        if show:
            # 绘制直方图
            plt.hist(point_normal_array, bins=200, density=True, alpha=0.7)
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Normal Distribution')
            plt.grid(True)
            plt.show()
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        # 将 numpy 数组转换为列表
        point_normal_list = point_normal_array.tolist()

        return point_normal_list

    # 计算正态分布标准差
    @staticmethod
    def calculate_std(interval_symmetric: Union[tuple, list], probability: float, show: bool = False) -> float:
        """
        根据以 0 为均值的对称区间和概率来计算标准差
        Calculate the standard deviation based on a symmetric interval with
        a mean of 0 and the corresponding probabilities.

        :param interval_symmetric: (tuple / list) 以 0 为均值的对称区间
        :param probability: (float) 概率
        :param show: (bool) 是否绘图，默认为否

        :return std: (float) 标准差
        """

        # 检查对称区间
        lower_bound, upper_bound = interval_symmetric
        if abs(lower_bound + upper_bound) > 0.0001:
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"033[95mIn {method_name}\033[0m, interval_symmetric is not value expected.")

        # 计算正态分布的标准差
        interval_length = upper_bound - lower_bound
        std = interval_length / (2 * norm.ppf((1 + probability) / 2))

        # 生成正态分布数据
        x = np.linspace(0 - 3 * std, 0 + 3 * std, num=100)
        y = norm.pdf(x, 0, std)

        if show:
            # 绘制正态分布曲线
            plt.plot(x, y, label=f"Mean={0}, Std={std}")
            plt.fill_between(x, y, where=(x >= interval_symmetric[0]) &
                                         (x <= interval_symmetric[1]), alpha=0.5, color='green')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Normal Distribution')
            plt.legend()
            plt.grid(True)
            plt.show()
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return std

    # 计算正态分布的概率
    @staticmethod
    def calculate_probability(mean: float, interval: Union[tuple, list], std: float, show: bool = False) -> float:
        """
        在正态分布中，根据均值，区间和标准差来计算概率
        In a normal distribution, calculate the probability based on
        the mean, interval, and standard deviation.

        :param mean: (float) 均值
        :param interval: (tuple / list) 区间
        :param std: (float) 标准差
        :param show: (bool) 是否绘图，默认为否

        :return probability: (float) 概率
        """

        # 计算区间的上下限对应的 z-scores
        lower_z = (interval[0] - mean) / std
        upper_z = (interval[1] - mean) / std

        # 使用标准正态分布的累积分布函数计算概率
        probability = stats.norm.cdf(upper_z) - stats.norm.cdf(lower_z)

        # 生成正态分布数据
        x = np.linspace(mean - 3 * std, mean + 3 * std, num=100)
        y = stats.norm.pdf(x, mean, std)

        if show:
            # 绘制正态分布曲线
            plt.plot(x, y, label=f"Mean={mean}, Std={std}")
            plt.fill_between(x, y, where=(x >= interval[0]) & (x <= interval[1]), alpha=0.5, color='green')
            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.title('Normal Distribution')
            plt.legend()
            plt.grid(True)
            plt.show()
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return probability

    # 获取文件夹下所有子文件夹的名称
    @staticmethod
    def get_subdirectories(folder_path: str) -> tuple:
        """
        获取目录文件夹下所有子文件夹的名称
        Retrieve the names of all subfolders within a directory.

        :param folder_path: (str) 文件夹的路径

        :return result: (tuple) 子文件夹的名称
        """

        subdirectories = []
        for item in os.listdir(folder_path):

            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subdirectories.append(item)
        result = tuple(subdirectories)

        return result

    # 获取文件夹下所有目标子文件的名称
    @staticmethod
    def get_files(folder_path, target_suffix: [None, list, str] = None, seek_directory: bool = True, show: bool = True)\
            -> List[str]:
        """
        获取目录文件夹下所有子文件夹或指定类型文件的名称
        Retrieve the names of all subfolders and/or files with specific suffixes within a directory.

        :param folder_path: (str) 文件夹的路径
        :param target_suffix: (str / list / None) 指定扩展名（例如 ".pdf"），也可以是字符串列表，None 表示不过滤
        :param seek_directory: (bool) 是否查找子目录，True 表示包含目录，False 表示只返回文件
        :param show: (bool) 是否在控制台打印结果，True 表示打印，False 表示静默返回

        :return files_list: (list) 子文件夹和 / 或目录名称
        """

        # ANSI 颜色列表：用于控制台高亮不同扩展名的文件
        color_list = [
            '\033[91m',  # 红色
            '\033[92m',  # 绿色
            '\033[93m',  # 黄色
            '\033[94m',  # 蓝色
            '\033[95m',  # 紫色
            '\033[96m',  # 青色
            '\033[90m',  # 灰色
        ]
        reset = '\033[0m'  # 重置颜色

        files_list = []  # 存储所有符合条件的结果（绝对路径）

        # 标准化 target_suffix：统一转为小写列表，方便后续匹配
        if isinstance(target_suffix, str):
            target_suffix = [target_suffix.lower()]
        elif isinstance(target_suffix, list):
            if not all(isinstance(s, str) for s in target_suffix):
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"033[95mIn {method_name}\033[0m,"
                                 f"all elements in the target_suffix list must be strings")
            target_suffix = [s.lower() for s in target_suffix]
        elif target_suffix is not None:
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"033[95mIn {method_name}\033[0m,"
                             f"target_suffix must be a string, a list of strings, or None")

        directories = [] if seek_directory else None  # 用于记录子目录（如果开启）
        files = []  # 用于记录符合条件的文件（仅文件名）

        # 遍历当前目录下的所有子项
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # 处理子目录
            if os.path.isdir(item_path) and seek_directory:
                directories.append(item)
                files_list.append(item_path)

            # 处理文件并按扩展名筛选
            elif os.path.isfile(item_path):
                if target_suffix is None or any(item.lower().endswith(suf) for suf in target_suffix):
                    files.append(item)
                    files_list.append(item_path)

        # 扩展名颜色映射（自动为每种扩展名分配颜色）
        ext_color_map = {}
        color_index = 0

        def get_color(external):
            nonlocal color_index
            if external not in ext_color_map:
                if color_index < len(color_list):
                    ext_color_map[external] = color_list[color_index]
                    color_index += 1
                else:
                    ext_color_map[external] = '\033[97m'  # 白色
            return ext_color_map[external]

        # 控制是否打印结果
        if show:
            print(f"\nDirectory: \033[4m{folder_path}{reset}")

            # 打印子目录（若有且允许）
            if directories:
                print(f"Subdirectories ({len(directories)}):")
                for idx, d in enumerate(directories, 1):
                    print(f"   {idx:>2}. \033[1m{d}{reset}")

            # 打印文件（若有）
            if files:
                print(f"Files ({len(files)}):")
                for idx, f in enumerate(files, 1):
                    ext = os.path.splitext(f)[1].lower()
                    color = get_color(ext)
                    print(f"   {idx:>2}. {color}{f}{reset}")

        return files_list

    # 在 list 中对文件名排序
    @staticmethod
    def sort_file_list(file_list: List[str], show=True) -> List[str]:
        """
        对一个包含文件名或目录名的字符串列表进行分类和排序（符合 macOS 排序习惯）
        Classify and sort a list of strings containing file names or directory names
        (in line with macOS sorting habits).

        :param file_list: (list) 要排序的文件名/目录名列表，元素必须是字符串
        :param show: (bool) 是否打印排序结果

        :return sorted_list: (list) 排序后的列表，目录在前，文件按扩展名分类后自然排序
        """

        # ANSI 颜色列表：用于控制台高亮不同扩展名的文件
        color_list = [
            '\033[91m',  # 红色
            '\033[92m',  # 绿色
            '\033[93m',  # 黄色
            '\033[94m',  # 蓝色
            '\033[95m',  # 紫色
            '\033[96m',  # 青色
            '\033[90m',  # 灰色
        ]
        reset = '\033[0m'  # 重置颜色

        # 检查是否为字符串列表
        if not isinstance(file_list, list):
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"033[95mIn {method_name}\033[0m, file_list must be of the list type")
        if not all(isinstance(item, str) for item in file_list):
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"033[95mIn {method_name}\033[0m, all elements in file_list must be of type str")

        directories = []
        files_by_extension = {}

        # 分类：目录和按扩展名分组的文件
        for name in file_list:
            if '.' not in name or name.startswith('.'):
                directories.append(name)
            else:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                files_by_extension.setdefault(ext, []).append(name)

        sorted_dirs = natsorted(directories)
        sorted_files = []

        # 颜色分配字典
        ext_color_map = {}
        color_index = 0

        # 为扩展名分配颜色
        for ext in sorted(files_by_extension.keys()):
            if ext not in ext_color_map:
                if color_index < len(color_list):
                    ext_color_map[ext] = color_list[color_index]
                    color_index += 1
                else:
                    ext_color_map[ext] = '\033[97m'  # 超出颜色列表用白色
            sorted_files.extend(natsorted(files_by_extension[ext]))

        sorted_list = sorted_dirs + sorted_files

        if show:
            print("Sorted result:\n")
            if sorted_dirs:
                print("Directories:")
                for d in sorted_dirs:
                    print(f"  - {color_list[0]}{d}{reset}")  # 目录统一用第一个颜色
            if sorted_files:
                print("\nFiles:")
                for f in sorted_files:
                    ext = os.path.splitext(f)[1].lower()
                    color = ext_color_map.get(ext, '\033[97m')
                    print(f"  - {color}{f}{reset}")
            print()

        return sorted_list

    # 使 Excel 数据重新排序
    @staticmethod
    def sort_data(directory_path: str, sort_axis: str = 'row') -> None:
        """
        对 DataFrame 进行从小到大的排序
        Sort the DataFrame from smallest to largest.

        :param directory_path: (str) 文件路径
        :param sort_axis: (str) 排序的方向，默认为行

        :return: None
        """

        # 列出目录下的所有文件
        for filename in os.listdir(directory_path):
            # 检查文件是否是Excel文件
            if filename.endswith('.xlsx'):
                file_path = os.path.join(directory_path, filename)

                # 读取Excel文件
                df = pd.read_excel(file_path, engine='openpyxl')

                # 横向排序
                if sort_axis == 'row':
                    df_sorted = df.apply(lambda row: pd.Series(sorted(row)), axis=1)
                elif sort_axis == 'column':
                    df_sorted = df.apply(lambda row: pd.Series(sorted(row)), axis=0)
                else:
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of sort_data\033[0m, accepts only: 'row', 'column'.")

                # 分割文件名和扩展名
                basename, file_extension = os.path.splitext(filename)

                # 添加后缀"_sorted"到文件名，并再次附加扩展名
                new_filename = basename + '_sorted' + file_extension

                # 合并目录路径和新文件名
                output_path = os.path.join(directory_path, new_filename)

                # 保存为新的Excel文件
                df_sorted.to_excel(output_path, index=False, engine='openpyxl')

                print(f"\033[94m{filename}\033[0m is sorted.")

        return None

    # 分析一张图片中像素最多的颜色
    @staticmethod
    def find_most_color_pixel(image_path: str, print_details: bool = True, show_image: bool = False) -> Tuple[str, str]:
        """
        分析一张图片中像素最多的颜色，返回其 RGB 值和十六进制，也可以显示出来
        Analyze the color with the most pixels in an image and return its RGB value and hexadecimal,
        which can also be displayed.

        :param image_path: (str) 图片的路径
        :param print_details: (bool) 是否打印 RGB 值和十六进制，默认为 True
        :param show_image: (bool) 是否显示图片，默认为 False

        :return rgb_value: (str) 图片中像素最多的颜色的 RGB 值
        :return hex_value: (str) 图片中像素最多的颜色的十六进制
        """

        # 打开图像文件
        img = Image.open(image_path)

        # 获取图像中所有像素的列表
        pixels = list(img.getdata())

        # 计算每个像素出现的频率
        pixel_freq = Counter(pixels)

        # 找到出现频率最高的像素
        most_common_pixel = pixel_freq.most_common(1)[0][0]

        # 获取该像素的RGB值
        rgb_value = most_common_pixel

        # 将RGB值转换为十六进制
        hex_value = "#{:02x}{:02x}{:02x}".format(rgb_value[0], rgb_value[1], rgb_value[2])

        if show_image:
            # 使用matplotlib显示颜色
            fig, ax = plt.subplots()
            ax.add_patch(patches.Rectangle(xy=(0, 0), width=1, height=1, color=hex_value))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.axis('off')  # 关闭坐标轴
            plt.show()

        if print_details:
            print("Most common RGB value:", rgb_value)
            print("Corresponding hex value:", hex_value)

        return rgb_value, hex_value

    # 复制文件到目标文件夹 (不包括文件夹)
    @staticmethod
    def copy_files(source_path: str, target_path: str, search_all: bool = False, ignore_list: list = None) -> None:
        """
        根据指定的参数将文件从源目录复制到目标目录。
        Copies files from the source directory to the destination directory based on the specified parameters.

        :param source_path: (str) 文件 (夹) 来源的目录，此路径必需是目录
        :param target_path: (str) 文件 (夹) 复制到的路径，此路径必需是目录
        :param search_all: (bool) 是否搜寻来源目录的所有子文件夹，默认为 False.
        :param ignore_list: (list) 需要忽略的文件或文件夹，默认为 None。如果提供，将自动包含 '.DS_Store'。
                            该忽略如果是路径的情况下只会记录最后的文件 / 文件夹名 (而无视其路径)，如果忽略了一个文件夹也只算一次忽略。
                            如果搜寻所有文件，也可以输入目录名称来忽略这个目录下的所有文件。

        :return: None
        """

        # 检查 source_path 是否为文件夹
        if not os.path.isdir(source_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"source path '{source_path}' must be a directory.")

        # 检查 target_path 是否为文件夹
        if not os.path.isdir(target_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"destination path '{target_path}' must be a directory.")

        # 初始化忽略列表，自动包含 '.DS_Store'
        if ignore_list is None:
            ignore_list = []
        # 提取 '.DS_Store' 的标准形式并加入忽略列表
        ds_store_path = os.path.join(source_path, '.DS_Store')  # 构建完整路径
        if ds_store_path not in ignore_list and '.DS_Store' not in ignore_list:
            ignore_list.append('.DS_Store')
        # 将忽略项标准化为文件名或相对于 source_path 的路径
        normalized_ignore_list = set()  # 使用 set 避免重复
        for item in ignore_list:
            # 如果是完整路径，提取文件名
            if os.path.isabs(item):
                normalized_ignore_list.add(os.path.basename(item))
            else:
                normalized_ignore_list.add(item)
        ignore_set = list(normalized_ignore_list)

        # 初始化文件列表和忽略计数
        files_to_copy = []
        ignored_items = []

        # 搜索文件
        if not search_all:  # 只在顶层寻找文件
            for entry in os.listdir(source_path):
                entry_path = os.path.join(source_path, entry)
                if os.path.isfile(entry_path) and entry not in ignore_set:
                    files_to_copy.append(entry_path)
                elif entry in ignore_set and entry != '.DS_Store':  # 忽略但不包括 .DS_Store
                    ignored_items.append(entry_path)
        else:  # 递归搜索所有子文件夹
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file not in ignore_set:
                        files_to_copy.append(os.path.join(root, file))
                    elif file != '.DS_Store':  # 忽略但不包括 .DS_Store
                        ignored_items.append(os.path.join(root, file))

        # 检查是否有文件需要复制
        if not files_to_copy:
            print(f"\033[31mNo files to copy from \033[34m{source_path}\033[0m to \033[34m{target_path}\033[0m.")
            return None

        # 复制文件到目标路径
        copied_count = 0
        for file in files_to_copy:
            base_name = os.path.basename(file)
            destination_file = os.path.join(target_path, base_name)

            # 检查文件名冲突
            if os.path.exists(destination_file):
                base, ext = os.path.splitext(base_name)
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = os.path.join(target_path, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.copy2(file, destination_file)  # 使用 copy2 保留文件元数据
            copied_count += 1

        # 打印忽略和复制信息
        if ignored_items:
            print(
                f"Copied \033[32m{copied_count}\033[0m files to \033[34m{target_path}\033[0m, "
                f"and ignored \033[33m{len(ignored_items)}\033[0m other items."
            )
        else:
            print(
                f"Copied \033[32m{copied_count}\033[0m files to \033[34m{target_path}\033[0m."
            )

        return None


""" 优化系统 """
class Optimizer(Function):
    """
    对 data_dic 数据进行优化

    Optimize the data in the variable data_dic to achieve the desired outcome.
    The results can either be saved or used directly for plotting.

    注意：
    默认数据只会对 self.data_dic 进行操作

    Attention:
    The default data only operates on self.data_dic
    """

    # 使用全局变量作为类属性的默认值
    Magic_Database = Magic_Database
    Category_Index = Category_Index

    # 初始化
    def __init__(self, data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                 title: Optional[str] = None, x_list: Optional[list] = None, y_list: Optional[list] = None,
                 x_label: str = None, y_label: str = None):
        """
        Function 函数系统为被引系统，无需赋值初始化

        输入数据 DataFrame，dict，x_list and y_list 其中的一个，title 为非必要数据
                x_list & y_list 的优先级最高，DataFrame 其次，dict 最低

        :param data_dic: (dict) 输入 dict，key 为 title，value 为 data_dic
        :param data_df:  (DataFrame) 输入 DataFrame
        :param title: (str) 数据的 title
        :param x_list: (list) x坐标的 list
        :param y_list: (list) y坐标的 list
        :param x_label: (str) x坐标的 label
        :param y_label: (str) y坐标的 label
        """

        # 超类初始化
        super().__init__(data_dic=data_dic, data_df=data_df, title=title, x_list=x_list, y_list=y_list,
                         x_label=x_label, y_label=y_label)  # 数据初始化时自动完成数据分配

        # 接收参数初始化
        self.data_dic = data_dic
        self.data_df = data_df
        self.title = title
        self.x_list = x_list
        self.y_list = y_list
        self.x_label = x_label
        self.y_label = y_label

        self.current_dic = 'data_dic'

        # 变量初始化，所有变量均为 dict
        self.extended_dic = None  # generate_rows_to_df()
        self.noised_dic = None  # add_noise_to_dataframe()
        self.randomized_dic = None  # change_to_random()
        self.normalized_dic = None  # normalize_columns()
        self.scaled_dic = None  # scale_df()
        self.separated_dic = None  # separate_df_by_category()
        self.many_df_dict_by_row = None  # split_df_by_row()
        self.many_df_dict_by_column = None  # split_df_by_column()
        self.one_column_dict = None  # merge_dfs_to_single_df_wide()
        self.one_column_long_dict = None  # merge_dfs_to_single_df_long()
        self.converted_dic = None  # convert_one_length_dict()
        self.inserted_dic = None  # insert_data_to_df()
        self.sorted_dic = None  # sort_df()
        self.calculated_dic = None  # calculate_statistics()
        self.without_category_dic = None  # remove_category()
        self.add_category_dic = None  # add_category()
        self.split_by_category_dic = None  # split_df_by_category()
        self.merge_by_category_dic = None  # merge_df_by_category()

        # 数据初始化分配 和 数据类型导入
        if type(self) == Optimizer:  # 当 self 为 Optimizer 的直接实例时为真
            self.data_init()

    # 扩展 DataFrame 表格
    def generate_rows_to_df(self, data_dic: Optional[Dict[str, DataFrame]] = None, number: Optional[int] = None,
                            tolerance: float = 0.05, strict: bool = False, show: bool = True,
                            overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        根据给定的 dict 中的每一个 DataFrame，生成指定数量的行，
        每行的值在原始数据的基础上 (根据原数据的最大、最小值和标准差计算得出) 增加随机噪声
        Generates the specified number of rows for each DataFrame in the given dict,
        The value of each row increases random noise based on the original data (calculated from the maximum,
        minimum, and standard deviation of the original data).

        :param data_dic: (dict) 包含多个 DataFrame 的 dict 。键为描述性标题，值为 DataFrame
        :param number: (int) 期望生成的每个 DataFrame 的总行数，此项为必输入项
        :param tolerance: (float) 允许的最大 / 最小值容差范围的百分比，默认为 0.05
        :param strict: (bool) 严格模式，如果原始 DataFrame 的行数超过了指定的 number，将会引发错误
        :param show: (bool) 显示进度消息，默认为 True
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return extended_dic: (dict) 扩展后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        if number is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'number'.")

        # 初始化输出 dict
        extended_dic = {}

        # 遍历每一个 DataFrame
        for key, data_df in data_dic.items():

            # 尝试将 DataFrame 中所有内容转换为数值，如果失败就报错，ValueError
            try:
                data_df = data_df.map(lambda x: pd.to_numeric(x, errors='raise'))
            except ValueError as e:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"DataFrame contains non-numeric values.") from e

            # 验证 tolerance值
            if tolerance < 0:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The 'tolerance' parameter must be non-negative.")

            current_rows = len(data_df)
            if current_rows > number and strict:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The dataframe {key} already has {current_rows} rows, "
                                 f"which exceeds the desired {number} rows.")

            # 计算需要添加的行数
            rows_to_add = number - current_rows
            new_rows_list = []

            for _ in range(rows_to_add):
                chosen_row_idx = np.random.choice(data_df.index)
                chosen_row = data_df.iloc[chosen_row_idx].copy()

                for column in data_df.columns:
                    column_std_dev = data_df[column].std()  # 根据正态分布生成数据
                    column_max = data_df[column].max() * (1 + tolerance)  # 计算向上包容值
                    column_min = data_df[column].min() * (1 - tolerance)  # 计算向下包容值

                    noise = np.random.normal(loc=0, scale=column_std_dev)  # 根据包容值计算噪声
                    new_value = chosen_row[column] + noise  # 添加噪声至生成的数据

                    # 如果新值超出范围，则重新生成噪声，直到其值落入范围内
                    while new_value > column_max or new_value < column_min:
                        noise = np.random.normal(loc=0, scale=column_std_dev)
                        new_value = chosen_row[column] + noise

                    chosen_row[column] = new_value

                new_rows_list.append(chosen_row)

            # 添加新行
            updated_df = pd.concat([data_df, pd.DataFrame(new_rows_list)], ignore_index=True)

            if show:
                print(f"The dataframe \033[92m{key}\033[0m now has \033[94m{number}\033[0m rows.")

            extended_dic[key] = updated_df

        # 覆盖原数据
        if overwrite:
            self.data_dic = extended_dic

        self.extended_dic = extended_dic
        self.current_dic = 'extended_dic'

        return extended_dic

    # 添加噪声到 DataFrame 表格
    def add_noise_to_dataframe(self, data_dic: Optional[Dict[str, DataFrame]] = None, columns: Optional[list] = None,
                               noise_percentage: Optional[float] = 0.05, tolerance: Optional[float] = None,
                               show: bool = True, overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        向指定的 DataFrame 的列添加相对于列均值的噪声
        Add noise to the specified columns of the DataFrame relative to the mean of the column.

        :param data_dic: (dict) 包含多个 DataFrame 的 dict 。键为描述性标题，值为 DataFrame
        :param noise_percentage: (float) 作为噪声标准差的列均值的百分比
        :param columns: (list) 需要添加噪声的列的名称列表。默认为 None，表示对所有列添加噪声
        :param tolerance: (float) 允许的最大 / 最小值的容差范围。如果为 None，则不限制
        :param show: (bool) 是否显示进度消息，默认为 True
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return noised_dic: (dict) 添加噪声后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        noised_dic = {}

        for key, data_df in data_dic.items():

            # 如果没有指定列，则为所有列添加噪声
            if columns is None:
                columns_to_modify = data_df.columns
            else:
                columns_to_modify = columns

            noisy_df = data_df.copy()

            for col in columns_to_modify:

                # 检查并删除 self.Category_Index 列
                if self.Category_Index in data_df.columns:
                    data_df = data_df.drop(columns=self.Category_Index)

                mean_val = noisy_df[col].mean()  # 计算列的均值
                noise_scale = mean_val * noise_percentage  # 根据均值计算噪声的标准差

                # 如果设置了 tolerance, 计算新的最大值和最小值
                if tolerance is not None:
                    column_max = noisy_df[col].max() * (1 + tolerance)
                    column_min = noisy_df[col].min() * (1 - tolerance)
                else:
                    # 否则，没有限制
                    column_max = float('inf')
                    column_min = float('-inf')

                noisy_values = noisy_df[col].copy()  # 创建一个该列的拷贝

                # 对每个值进行检查并添加噪声，直到它在允许的范围内
                for i in range(len(noisy_values)):
                    while True:
                        noise = np.random.normal(0, noise_scale)  # 生成噪声
                        new_value = noisy_values.iloc[i] + noise  # 加上噪声

                        # 检查新值是否在指定的范围内
                        if column_min <= new_value <= column_max:
                            noisy_values.iloc[i] = new_value  # 如果是，使用新值
                            break  # 并退出循环

                noisy_df[col] = noisy_values  # 将添加了噪声的列保存到 DataFrame 中

                if show:
                    print(f"The column \033[92m{col}\033[0m "
                          f"in the dataframe \033[94m{key}\033[0m has been noise-augmented.")

                noised_dic[key] = noisy_df

            # 覆盖原数据
            if overwrite:
                self.data_dic = noised_dic

            self.noised_dic = noised_dic
            self.current_dic = 'noised_dic'

            return noised_dic

    # 更改表格的行或列成给定随机值
    def change_to_random(self, data_dic: Optional[Dict[str, DataFrame]] = None, position: Optional[int] = None,
                         random_number: Optional[list] = None, which_position: str = 'column',
                         overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        根据指定的位置更改 DataFrame 的值为给定列表中的随机值
        Modify the values of a DataFrame at the specified position with random values from a given list.

        :param data_dic: (dict) 输入的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param position: (int) 要修改的目标位置（ 0 为第一列 / 行，1 为第二列 / 行，-1 为最后一列 / 行等此规律），此项为必输入项
        :param random_number: (list) 用于随机选择的值列表，此项为必输入项
        :param which_position: (str) 选择更改列或行，默认为 column，可选 row
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return randomized_dic: (dict) 修改后的数据 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        if position is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'position'.")

        if random_number is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'random_number'.")

        randomized_dic = {}

        for key, df in data_dic.items():
            if which_position == 'column':
                if position >= len(df.columns) or position < -len(df.columns):
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"'{key}' has {len(df.columns)} columns, "
                                     f"but the input value is {position}.")
                # 创建与目标列长度相同的随机数组
                random_values = np.random.choice(random_number, len(df))
                # 使用iloc替换目标列的值
                df.iloc[:, position] = random_values

            elif which_position == 'row':
                if position >= len(df) or position < -len(df):
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"'{key}' has {len(df)} rows, "
                                     f"but the input value is {position}.")
                # 创建与目标行长度相同的随机数组
                random_values = np.random.choice(random_number, len(df.columns))
                # 使用iloc替换目标行的值
                df.iloc[position, :] = random_values

            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Invalid value for 'which_position'."
                                 f"Only 'column' or 'row' are accepted.")

            randomized_dic[key] = df

        # 覆盖原数据
        if overwrite:
            self.data_dic = randomized_dic

        self.randomized_dic = randomized_dic
        self.current_dic = 'randomized_dic'

        return randomized_dic

    # 将数据正态化
    def normalize_columns(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                          col_indices: Optional[List[int]] = None, mean: Optional[float] = None,
                          std: Optional[float] = None, min_val: Optional[float] = None,
                          max_val: Optional[float] = None, median_val: Optional[float] = None,
                          overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        此函数将字典中每个 DataFrame 指定的多列数据替换为正态分布的数据，同时考虑了标准差、最大值、最小值、均值和中位数的限制
        This function replaces multiple columns of data in each DataFrame within a dictionary
        with normally distributed data, while also taking into account restrictions on standard deviation,
        maximum value, minimum value, mean, and median.

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param col_indices: (int) 需要正态化的列的索引，此项为必输入项
        :param mean: (float) 正态分布的期望值，默认为 None，此时会使用各列数据的均值
        :param std: (float) 正态分布的标准差，默认为 None，此时会使用各列数据的标准差
        :param min_val: (float) 数据的最小值限制，默认为 None，此时会使用各列数据的最小值
        :param max_val: (float) 数据的最大值限制，默认为 None，此时会使用各列数据的大值
        :param median_val: (float) 数据的中位数限制，默认为 None，此时会使用各列数据的中值
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return normalized_dic: (dict) 正态化后的数据 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        if col_indices is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'col_indices'.")

        normalized_dic = {}
        for title, data_df in data_dic.items():
            # 复制 DataFrame 以避免修改原始数据
            normalized_df = data_df.copy()

            # 确保提供的列索引都在 DataFrame 的列数范围内
            max_index = normalized_df.shape[1] - 1
            if any(i > max_index or i < 0 for i in col_indices):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Column index out of range for DataFrame '{title}'.")

            for col_index in col_indices:

                # 如果没有指定 mean 和 std，就使用原列数据的均值、标准差、最小值、最大值和中位数
                column_mean = mean if mean is not None else data_df.iloc[:, col_index].mean()
                column_std = std if std is not None else data_df.iloc[:, col_index].std()
                column_min_val = min_val if min_val is not None else data_df.iloc[:, col_index].min()
                column_max_val = max_val if max_val is not None else data_df.iloc[:, col_index].max()
                column_median_val = median_val if median_val is not None else data_df.iloc[:, col_index].median()

                # 不断生成数据直到所有值都在指定的范围内
                while True:
                    normalized_data = np.random.normal(loc=column_mean,
                                                       scale=column_std,
                                                       size=normalized_df.shape[0])

                    median_adjustment = column_median_val - np.median(normalized_data)
                    normalized_data += median_adjustment

                    # 检查数据是否在指定的范围内
                    if all(column_min_val <= data <= column_max_val for data in normalized_data):
                        break

                # 替换指定列的数据
                normalized_df.iloc[:, col_index] = normalized_data

            # 更新 normalized_dic 字典
            normalized_dic[title] = normalized_df

        # 覆盖原数据
        if overwrite:
            self.data_dic = normalized_dic

        self.normalized_dic = normalized_dic
        self.current_dic = 'normalized_dic'

        return normalized_dic

    # 将行 / 列数据放缩到给定值
    def scale_df(self, data_dic: Optional[Dict[str, DataFrame]] = None, target_number: float = 100,
                 noise_section: Optional[list[float]] = None, sort_axis: str = 'row', overwrite: bool = True)\
            -> Dict[str, DataFrame]:
        """
        将 DataFrame 中的行或列按照给定的数进行放缩
        Shrink the rows or columns in the DataFrame by the given number.

        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据
        :param target_number: (float) 放缩后的数据之和，默认为 100
        :param noise_section: (list) 浮动的数值，使得数据放缩到给定值的附近随机数，分别为其的第一个和和二个元素，默认无浮动
        :param sort_axis: (str) 以行 / 列进行放缩，默认为 'row'
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return scaled_dic: (dict) 经过放缩计算后的 dict
        """

        # 检查 data_dic 的赋值
        if data_dic is not None:
            data_dic = data_dic
        elif self.data_dic is not None:
            data_dic = copy.deepcopy(self.data_dic)
        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"There is no valid data_dic value for statistical calculation.")

        if len(noise_section) != 2:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The length of the noise_section must be 2, and the gain length is {len(noise_section)}")

        lower_number, upper_number = noise_section
        scaled_dic = {}

        # 如果是按行缩放
        if sort_axis == 'row':

            for title, data_df in data_dic.items():

                # 对每一行进行操作
                for idx in data_df.index:

                    random_number = random.uniform(lower_number, upper_number)  # 取随机数
                    number = target_number + random_number  # 目标数值
                    row_sum = data_df.loc[idx].sum()

                    # 防止除以零
                    if row_sum != 0:
                        data_df.loc[idx] = data_df.loc[idx].apply(lambda x: x / row_sum * number)
                    else:
                        data_df.loc[idx] = number / len(data_df.columns)  # 如果原行和为零，则平均分配

                scaled_dic[title] = data_df

        # 如果是按列缩放
        elif sort_axis == 'column':

            for title, data_df in data_dic.items():

                # 对每一列进行操作
                for col in data_df.columns:

                    random_number = random.uniform(lower_number, upper_number)  # 取随机数
                    number = target_number + random_number  # 目标数值
                    col_sum = data_df[col].sum()

                    # 防止除以零
                    if col_sum != 0:
                        data_df[col] = data_df[col].apply(lambda x: x / col_sum * number)
                    else:
                        data_df[col] = number / len(data_df.index)  # 如果原列和为零，则平均分配

                scaled_dic[title] = data_df

        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"sort_axis must be 'row' or 'column'.")

        # 覆盖原数据
        if overwrite:
            self.data_dic = scaled_dic

        self.scaled_dic = scaled_dic
        self.current_dic = 'scaled_dic'

        return scaled_dic

    # 将长度为 1 的 dict 中的 DataFrame 按照 self.Category_Index 进行分开
    def separate_df_by_category(self, data_dic: Optional[Dict[str, DataFrame]] = None, overwrite: bool = True)\
            -> Dict[str, DataFrame]:
        """
        将 dict 中的数据按照 self.Category_Index 分开
        Separate the data in the dict by self.Category_Index.

        :param data_dic: (dict) 长度为 1 需要分开的 dict
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return separated_dic: (dict) 分开后多长度的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 dict 长度是否为1
        if len(data_dic) != 1:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Error: Input dictionary must have exactly one key.")

        # 获取 dict 的唯一键
        data_df = list(data_dic.values())[0]

        # 检查是否存在 self.Category_Index 列
        if self.Category_Index not in data_df.columns:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Error: {self.Category_Index} column not found in the table.")

        # 按 self.Category_Index 列的值分割 DataFrame
        categories = data_df[self.Category_Index].unique()
        separated_dic = {cat: data_df[data_df[self.Category_Index] == cat].
        drop(columns=self.Category_Index).reset_index(drop=True) for cat in
                              categories}

        # 覆盖原数据
        if overwrite:
            self.data_dic = separated_dic

        self.separated_dic = separated_dic
        self.current_dic = 'separated_dic'

        return separated_dic

    # 将长度为 1 的 dict (宽格式) 分开成长度大于 1 的 dict，通过行
    def split_df_by_row(self, data_dic: Optional[Dict[str, DataFrame]] = None, title_list: Optional[List[str]] = None,
                        overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        将包含单个 DataFrame 的字典按行拆分成多个 DataFrame。每个新的 DataFrame 包含原始 DataFrame 的一行数据
        可以通过 title_list 指定每个拆分后的 DataFrame 的标题
        Split a dictionary containing a single DataFrame into multiple dataframes by row.
        Each new DataFrame contains a row of data from the original DataFrame. title_list specifies the
        title of each split DataFrame.

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param title_list: (list) 数据的标题的列表，默认为 None，表示原 row 索引
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return many_df_dict_by_row: (dict) 分割后长度大于 1 的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        x_label = self.x_label if self.x_label is not None else 'X'
        y_label = self.y_label if self.y_label is not None else 'Y'

        # 提取 data_df
        data_df = list(data_dic.values())[0]

        # 使用 shape 获取 DataFrame 的行数和列数
        num_rows, _ = data_df.shape

        if len(data_dic) != 1:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The dictionary must contain exactly one key-value pair.")

        # 如果 content_list 被赋值，检查 content_list 的长度是否与原数据的行数一致
        if title_list is not None and not num_rows == len(title_list):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"title_list is not consistent with the original data, "
                             f"the original data has {num_rows} rows, "
                             f"but the length of 'title_list' is {len(title_list)}.")

        # 当 title_list 为 None 时，使用原 DataFrame 的行索引
        if title_list is None:
            title_list = data_df.index.tolist()

        # 创建一个新的字典以存储分割后的 DataFrame
        many_df_dict_by_row = {}

        for index in range(len(data_df)):
            row = data_df.iloc[index]
            row_key = title_list[index]
            row_data = {x_label: [], y_label: []}

            for col_name in data_df.columns:
                # 将列名转换为浮点数
                try:
                    float_col_name = float(col_name)
                except ValueError:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"Column name '{col_name}' cannot be converted to a float.")

                row_data[x_label].append(float_col_name)
                row_data[y_label].append(row[col_name])

            # 将每行的数据转换为新的 DataFrame 并存入字典
            many_df_dict_by_row[str(row_key)] = pd.DataFrame(row_data)

        # 覆盖原数据
        if overwrite:
            self.data_dic = many_df_dict_by_row

        self.many_df_dict_by_row = many_df_dict_by_row
        self.current_dic = 'many_df_dict_by_row'

        return many_df_dict_by_row

    # 将长度为 1 的 dict (宽格式) 分开成长度大于 1 的 dict，通过行
    def split_df_by_column(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                           title_list: Optional[List[str]] = None, overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        将包含单个 DataFrame 的字典按列拆分成多个 DataFrame。每个新的 DataFrame 包含原始 DataFrame 的一列数据
        可以通过 title_list 指定每个拆分后的 DataFrame 的标题，该数据只有一列
        Split a dictionary containing a single DataFrame into multiple dataframes by column.
        Each new DataFrame contains a column of data from the original DataFrame. title_list specifies
        the title of each split DataFrame. The data has only one column.

        :param data_dic: (dict) 输入的数据 dict，其中 key 是数据的名称，value 是 DataFrame 表格
        :param title_list: (list) 每个拆分后的 DataFrame 的标题列表
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return column_dict: (dict) 按列拆分后的 DataFrame 字典
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 提取 data_df
        data_df = list(data_dic.values())[0]

        # 使用 shape 获取 DataFrame 的行数和列数
        _, num_columns = data_df.shape

        if len(data_dic) != 1:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The dictionary must contain exactly one key-value pair.")

        # 检查 title_list 的长度是否与 DataFrame 的列数相同
        if title_list is not None and num_columns != len(title_list):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"title_list is not consistent with the original data, "
                             f"the original data has {num_columns} columns, "
                             f"but the length of 'title_list' is {len(title_list)}.")

        # 使用 DataFrame 的列名作为默认标题列表
        if title_list is None:
            title_list = data_df.columns.tolist()

        # 创建一个新的字典以存储按列拆分的 DataFrame
        many_df_dict_by_column = {}

        for col_name, title in zip(data_df.columns, title_list):

            # 将每列的数据转换为新的 DataFrame 并存入字典
            many_df_dict_by_column[str(title)] = pd.DataFrame(data_df[col_name])

        # 覆盖原数据
        if overwrite:
            self.data_dic = many_df_dict_by_column

        self.many_df_dict_by_column = many_df_dict_by_column
        self.current_dic = 'many_df_dict_by_column'

        return many_df_dict_by_column

    # 将大于 1 长度的 dict 合并成一个长度为 1 的 dict (宽格式)
    def merge_dfs_to_single_df_wide(self, data_dic: Optional[Dict[str, pd.DataFrame]] = None,
                                    title: Optional[str] = None, row_list: Optional[list] = None,
                                    overwrite: bool = True) -> Dict[str, pd.DataFrame]:
        """
        将多个 DataFrame 合并成一个 DataFrame，合并后的 DataFrame 为宽格式
        第一个 DataFrame 的第一列的值将被用作新 DataFrame 的列名
        其余的 DataFrame 数据将按照行添加到新的 DataFrame 中，行索引为原始 DataFrame 的键
        Merge multiple DataFrames into a single DataFrame, with the merged DataFrame in a wide format.
        The values in the first column of the first DataFrame will be used as column names for the new DataFrame.
        The data from the remaining DataFrames will be added to the new DataFrame row by row,
        with the row index corresponding to the keys of the original DataFrames

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param title: (str) 数据的标题，默认为 None，表示 self.title
        :param row_list: (list) 行的索引名称，默认为 None，表示原 subtitle 索引
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return one_column_dict: (dict) 合并后长度为 1 的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 title 的赋值
        if title is not None:
            title = title
        elif self.title is not None:
            title = self.title
        else:
            title = 'Untitled'

        # 检查输入的 data_dic 是否为空
        if len(data_dic) <= 1:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The length of the data_dic should be greater than 1.")

        # 检查所有 DataFrame 第一列的值是否一致
        first_col_values = [df.iloc[:, 0].tolist() for df in data_dic.values()]
        if not all(first_col_values[0] == values for values in first_col_values):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"First column values must be the same in all DataFrames.")

        # 检查 row_list 与 data_dic 的长度是否一致
        if row_list is not None and not len(row_list) == len(data_dic):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The length of row_list must be the same as that of data_dic.")

        # 使用第一个 DataFrame 的第一列的值作为新 DataFrame 的列名
        new_col_names = first_col_values[0]

        # 创建一个空的 DataFrame，用于存放合并后的数据
        merged_df = pd.DataFrame(columns=new_col_names)

        # 遍历字典，将每个 DataFrame 的数据（除第一列外）添加到新的 DataFrame
        for index, (key, df) in enumerate(data_dic.items()):
            # 跳过第一列，将其余数据转换为一行，然后添加到 merged_df 中
            row_data = df.iloc[:, 1:].values.flatten().tolist()
            row_index = row_list[int(index)] if row_list is not None else key
            merged_df.loc[row_index] = row_data

        # 将合并后的 DataFrame 存入字典
        one_column_dict = {title: merged_df}

        # 覆盖原数据
        if overwrite:
            self.data_dic = one_column_dict

        self.one_column_dict = one_column_dict
        self.current_dic = 'one_column_dict'

        return one_column_dict

    # 将大于 1 长度的 dict 合并成一个长度为 1 的 dict (长格式)
    def merge_dfs_to_single_df_long(self, data_dic: Optional[Dict[str, DataFrame]] = None, title: Optional[str] = None,
                                    keep_common_only: bool = False, overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        将大于 1 长度的 dict 合并成一个长度为 1 的 dict，合并后的 DataFrame 为长格式
        Merge a dictionary with more than one entry into a single-entry dictionary,
        combining the DataFrames into a long format.

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param title: (str) 数据的标题，默认为 None，表示 self.title
        :param keep_common_only: (bool) 是否只保留公共列，默认为 False
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return one_column_dict: (dict) 合并后长度为 1 的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 title 的赋值
        if title is not None:
            title = title
        elif self.title is not None:
            title = self.title
        else:
            title = 'Untitled'

        dfs = []  # 用于存储更新后的DataFrame列表

        # 提取共有列，如果keep_common_only为True
        if keep_common_only:
            # 使用集合的交集来找出所有DataFrame的共有列，并转换为列表
            common_cols = list(set.intersection(*[set(df.columns) for df in data_dic.values()]))
        else:
            # 如果不仅保留共有列，那么就不需要进行列的筛选
            common_cols = None

        for category, df in data_dic.items():
            # 如果有需要，筛选出共有列
            if common_cols is not None:
                df = df[common_cols]

            # 重置索引，避免在concat时产生NaN
            df.reset_index(drop=True, inplace=True)
            # 添加类别列
            df[self.Category_Index] = category
            # 将更新后的DataFrame添加到列表中
            dfs.append(df)

        # 合并所有DataFrame
        merged_df = pd.concat(dfs, ignore_index=True)

        # 如果提供了标题，则使用标题作为键名，创建一个新的字典
        one_column_long_dict = {title: merged_df}

        # 覆盖原数据
        if overwrite:
            self.data_dic = one_column_long_dict

        self.one_column_long_dict = one_column_long_dict
        self.current_dic = 'one_column_long_dict'

        return one_column_long_dict

    # 在宽格式和长格式之间转换
    def convert_one_length_dict(self, data_dic: Optional[Dict[str, DataFrame]] = None, overwrite: bool = True) \
            -> Dict[str, DataFrame]:
        """
        转换长度为 1 的 dict，在宽格式和长格式之间自动转换
        Converts a dict of length 1 with Manager.
        self.Category_Index to automatically convert between wide and long formats.

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return converted_dic: (dict) 转换形式之后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 获取字典中的 DataFrame 和键
        key = list(data_dic.keys())[0]
        df = list(data_dic.values())[0]

        # 如果存在 self.Category_Index 列，则假定 DataFrame 为长格式，需要转换为宽格式
        if self.Category_Index in df.columns:
            # 使用 pivot 方法将长格式转为宽格式
            wide_df = df.pivot_table(index=df.index, columns=self.Category_Index, values=df.columns[0],
                                     aggfunc='first')

            # 重设索引，并将可能产生的多级列简化为单级
            wide_df = wide_df.reset_index(drop=True)
            wide_df.columns.name = None

            # 将数据向上移动，如果前一个数据是 NaN 的话
            wide_df = wide_df.apply(lambda col: pd.Series(col.dropna().values))

            # 删除所有值均为 NaN 的行
            wide_df = wide_df.dropna(how='all')

            # 构建新的字典
            converted_dic = {key: wide_df}

        # 如果不存在 self.Category_Index 列，则假定 DataFrame 为宽格式，需要转换为长格式
        else:
            # 从宽格式转换到长格式
            long_df = df.melt(var_name=self.Category_Index, value_name='content')

            # 按 self.Category_Index 分组，然后裁剪，以确保所有类别长度一致
            grouped = long_df.groupby(self.Category_Index)
            min_length = grouped.size().min()
            adjusted_long_df = pd.concat([group_df.iloc[:min_length] for _, group_df in grouped])

            # 调整列的顺序，确保 self.Category_Index 列在第二列
            cols = adjusted_long_df.columns.drop(self.Category_Index).tolist()
            # 将 self.Category_Index 列插入到第一列的位置，即索引 1（Python 是从 0 开始计数的）
            cols.insert(1, self.Category_Index)
            adjusted_long_df = adjusted_long_df[cols]

            # 删除旧索引，创建一个从 0 开始的新索引
            adjusted_long_df = adjusted_long_df.reset_index(drop=True)

            # 构建新的字典
            converted_dic = {key: adjusted_long_df}

        # 覆盖原数据
        if overwrite:
            self.data_dic = converted_dic

        self.converted_dic = converted_dic
        self.current_dic = 'converted_dic'

        return converted_dic

    # 插入或替换 DataFrame 中的数据
    def insert_data_to_df(self, data_dic: Optional[Dict[str, DataFrame]] = None, axis: Optional[str] = None,
                          position: Optional[int] = None, insert_data: Union[float, str, None] = None,
                          insert_column_name: Optional[str] = None, replace: bool = False,
                          index: Optional[List[int]] = None, overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        在 DataFrame 中插入或替换一整行或一整列，将其所有值设置为给定的数据
        Inserts or replaces an entire row or column in a DataFrame, setting all its values to the given data.

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param axis: (str) 选择 'row' 或 'column'，此项为必输入项
        :param position: (int) 插入或替换的位置 (第几行 / 列)，默认为 None，表示最后一行 / 列
        :param insert_data: (float / str) 插入或替换的数据 (字符串或浮点数)，此项为必输入项
        :param insert_column_name: (str) 若插入列，则为新列的名称，只有在插入列时才有意义，默认为 None，表示原位置
        :param replace: (bool) 是否为替换，默认为 False
        :param index: (list) 插入或替换的行 / 列的起始和结束位置，长度必须为2，默认为 None，表示所有
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return inserted_dic: (dict) 插入或修改数据之后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        if axis is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'axis'.")

        if insert_data is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'insert_data'.")

        inserted_dic = {}
        for title, data_df in data_dic.items():

            # 处理 position 参数
            if position is None:
                if axis == 'row':
                    position = len(data_df)  # 设置为最后一行
                elif axis == 'column':
                    position = len(data_df.columns)  # 设置为最后一列

            # 处理 index 参数
            if index is not None:
                if len(index) != 2:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"ihe Index list must contain two elements, "
                                     f"representing the start and end positions.")
                else:
                    start_idx, end_idx = index[0], index[1] + 1
            else:
                start_idx, end_idx = 0, len(data_df.columns) if axis == 'row' else len(data_df)

            if axis == 'row':
                # 创建与指定列数相同长度的 DataFrame
                new_data = pd.DataFrame([insert_data] * len(data_df.columns[start_idx:end_idx])).T
                new_data.columns = data_df.columns[start_idx:end_idx]

                if replace:
                    # 替换行
                    data_df.iloc[position, start_idx:end_idx] = new_data.iloc[0]
                else:
                    # 插入行
                    data_df = pd.concat([data_df.iloc[:position], new_data,
                                         data_df.iloc[position:]]).reset_index(drop=True)

            elif axis == 'column':
                # 处理插入列的名称
                if insert_column_name is not None:
                    column_name = insert_column_name
                else:
                    column_name = position
                new_data = pd.DataFrame([insert_data] * len(data_df.iloc[start_idx:end_idx]), columns=[column_name])

                if replace:
                    # 替换列
                    data_df.iloc[start_idx:end_idx, position] = new_data[column_name]
                else:
                    # 插入列
                    data_df = pd.concat(objs=[data_df.iloc[:, :position], new_data, data_df.iloc[:, position:]], axis=1)

            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Axis must be either 'row' or 'column'.")

            inserted_dic[title] = data_df

        # 覆盖原数据
        if overwrite:
            self.data_dic = inserted_dic

        self.inserted_dic = inserted_dic
        self.current_dic = 'inserted_dic'

        return inserted_dic

    # 对 DataFrame 中的数据进行排序
    def sort_df(self, data_dic: Optional[Dict[str, DataFrame]] = None, sort_axis: str = 'row',
                ascending: bool = True, position: [Union[List[int], Tuple[int], None]] = None,
                overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        对 DataFrame 进行排序
        Sort the DataFrame.

        :param data_dic: (dict) 输入长度为 1 的数据 dict ，其中 key 是数据的名称，value 是 DataFrame 表格
        :param sort_axis: (str) 排序的方向，'row' 或 'column'，默认为 'row'
        :param ascending: (bool) 是否为升序，默认为 True
        :param position: (list / tuple) 需要进行排序的行 / 列，默认为 None，表示所有的行 / 列
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return sorted_dic: (dict) 排序之后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        sorted_dic = {}
        for title, data_df in data_dic.items():

            # 初始化 sorted_df 为原 DataFrame 的副本
            sorted_df = data_df.copy()

            # 确保 position 是一个 list 类型，支持 list 或 tuple
            positions = list(position) if position is not None else range(len(data_df))

            # 对 DataFrame 进行排序
            if sort_axis == 'row':
                # 如果提供了 position 参数，排序指定的行
                for i in positions:
                    if i < len(data_df):  # 防止越界
                        # 按行进行排序，确保排序后是 pandas Series
                        sorted_df.iloc[i, :] = pd.Series(sorted(data_df.iloc[i, :], reverse=not ascending),
                                                         index=data_df.columns)
            elif sort_axis == 'column':
                # 如果提供了 position 参数，排序指定的列
                for i in positions:
                    if i < len(data_df.columns):  # 防止越界
                        # 按列进行排序，确保排序后是 pandas Series
                        sorted_df.iloc[:, i] = pd.Series(sorted(data_df.iloc[:, i], reverse=not ascending),
                                                         index=data_df.index)
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Sort_axis must be either 'row' or 'column'.")

            sorted_dic[title] = sorted_df

        # 覆盖原数据
        if overwrite:
            self.data_dic = sorted_dic

        self.sorted_dic = sorted_dic
        self.current_dic = 'sorted_dic'

        return sorted_dic

    # 计算 DataFrame 表格的各项统计学参数
    def calculate_statistics(self, data_dic: Optional[Dict[str, DataFrame]] = None, type_stat: Optional[list] = None,
                             sort_axis: str = 'column', overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        计算 DataFrame 表格中各列的的统计学参数
        Calculate the statistical parameters for each column in the DataFrame table.

        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据
        :param type_stat: (list) 需要计算的统计学参数，默认为 ['max', 'min', 'med', 'std', 'mean']
                        包括：['max', 'min', 'med', 'std', 'mean', 'var', 'sum', 'count', 'mode', 'abs', 'cumsum',
                              'cummax', 'cummin', 'cumprod', 'quantile']
        :param sort_axis: (str) 以行 / 列进行放缩，默认为 'column'
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return calculated_dic: (dict) 经过统计学计算后的 dict
        """

        # 确保 sort_axis 有效
        if sort_axis not in ['column', 'row']:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"sort_axis must be 'row' or 'column'.")

        # 将字符串映射到实际的 pandas 函数
        stat_functions = {
            'max': lambda df: df.max(),  # 最大值
            'min': lambda df: df.min(),  # 最小值
            'med': lambda df: df.median(),  # 中位数
            'std': lambda df: df.std(),  # 标准差
            'mean': lambda df: df.mean(),  # 平均值
            'var': lambda df: df.var(),  # 方差
            'sum': lambda df: df.sum(),  # 总和
            'count': lambda df: df.count(),  # 非空值计数
            'mode': lambda df: df.mode().iloc[0],  # 众数
            'abs': lambda df: df.abs().sum(),  # 绝对值的总和
            'cumsum': lambda df: df.cumsum().iloc[-1],  # 累积和的最后一个值
            'cummax': lambda df: df.cummax().iloc[-1],  # 累积最大值的最后一个值
            'cummin': lambda df: df.cummin().iloc[-1],  # 累积最小值的最后一个值
            'cumprod': lambda df: df.cumprod().iloc[-1],  # 累积乘积的最后一个值
            'quantile': lambda df: df.quantile(0.5)  # 分位数（默认为50%分位数，即中位数）
        }

        # 检查 data_dic 的赋值
        if data_dic is not None:
            data_dic = data_dic
        elif self.data_dic is not None:
            data_dic = copy.deepcopy(self.data_dic)
        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"There is no valid data_dic value for statistical calculation.")

        # 检查 type_stat 的赋值
        if type_stat is None:
            type_stat = ['max', 'min', 'med', 'std', 'mean']
        else:
            # 检查是否有无效的统计请求
            for stat in type_stat:
                if stat not in stat_functions.keys():
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"Invalid statistic requested: {stat}")

        calculated_dic = {}
        for title, data_df in data_dic.items():
            # 创建一个空字典以存储统计结果
            stats_dic = {}

            # 只计算 type_stat 中的统计量
            for stat in type_stat:
                if stat in stat_functions:
                    # 根据 sort_axis 应用统计函数于行或列
                    if sort_axis == 'column':
                        stats_dic[stat] = stat_functions[stat](data_df)
                    elif sort_axis == 'row':
                        stats_dic[stat] = data_df.apply(stat_functions[stat], axis=1)

            # 创建一个汇总 DataFrame
            if sort_axis == 'column':
                summary_df = pd.DataFrame(stats_dic).transpose()
            else:
                summary_df = pd.DataFrame(stats_dic)

            # 将汇总添加到原始 DataFrame
            if sort_axis == 'column':
                result_df = pd.concat([data_df, summary_df])
            else:
                result_df = pd.concat(objs=[data_df, summary_df], axis=1)

            # 将结果存储在字典中
            calculated_dic[title] = result_df

        # 覆盖原数据
        if overwrite:
            self.data_dic = calculated_dic

        self.calculated_dic = calculated_dic
        self.current_dic = 'calculated_dic'

        return calculated_dic

    # 删除 Category_Index 列
    def remove_category(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                        category_index: Union[str, int, None] = None, sort_axis: str = 'column', show: bool = True,
                        overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        在目标 dict 中删除名为 category_index 这一行 / 列，通常用作删除 Category_Index 列
        Remove the column/row named category_index in the target dictionary,
        usually used to delete the Category_Index column.

        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据
        :param category_index: (str / int) 为 str 时，为需要删除行 / 列的名称，默认为 self.Category_Index
                                           为 int 时，为需要删除行 / 列的行索引名称
        :param sort_axis: (str) 以行 / 列进行检索，默认为 'column'
        :param show: (bool) 是否显示进度消息，默认为 True
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return without_category_dic: (dict) 删除后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 category_index 的赋值
        if category_index is not None:
            category_index = category_index
        else:
            category_index = self.Category_Index

        # 确保 sort_axis 有效
        if sort_axis not in ['column', 'row']:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"sort_axis must be 'row' or 'column'.")

        # 遍历 dict
        for title, data_df in data_dic.items():
            if sort_axis == 'column':
                if category_index in data_df.columns:
                    data_dic[title] = data_df.drop(columns=[category_index])  # 删除列
                    if show:
                        print(f"In \033[92m{title}\033[0m, the column \033[31m{category_index}\033[0m "
                              f"has been removed.")
                else:
                    if show:
                        print(f"Column \033[31m{category_index}\033[0m not found in \033[92m{title}\033[0m, skipping.")

            elif sort_axis == 'row':
                if category_index in data_df.index:
                    data_dic[title] = data_df.drop(index=[category_index])  # 删除行
                    if show:
                        print(f"In \033[92m{title}\033[0m, the row \033[31m{category_index}\033[0m has been removed.")
                else:
                    if show:
                        print(f"Row \033[31m{category_index}\033[0m not found in \033[92m{title}\033[0m, skipping.")

        without_category_dic = data_dic

        # 覆盖原数据
        if overwrite:
            self.data_dic = without_category_dic

        self.without_category_dic = without_category_dic
        self.current_dic = 'without_category_dic'

        return without_category_dic

    # 在最后一行按照要求添加 Category_Index 列
    def add_category(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                     category_index: Optional[str] = None, category_name: Optional[list] = None,
                     category_number: Optional[list] = None, sort_axis='column', show: bool = True,
                     overwrite: bool = True) -> Dict[str, DataFrame]:
        """
        在最后一行 / 列添加新的行 / 列，但添加的内容只能为 str，
        如果 category_name 与 category_number 均未被赋值，则用 self.category_dic，但长度不一致时仍会报错
        Add a new row/column at the end, but the added content must be of type str.
        If category_name and category_number are both unassigned, use self.category_dic.
        However, an error will still be raised if the lengths do not match.

        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据
        :param category_index: (str) 添加新行 / 列的标题，默认为 self.Category_Index
        :param category_name: (list) 添加新行 / 列的内容，均为字符串
        :param category_number: (list) 添加新行 / 列的数量，均为整形，
                                category_number 的长度需要与 category_name 一致，同时相加的值需要与 DataFrame 行 / 列相等
        :param sort_axis: (str) 以行 / 列进行检索，默认为 'column'
        :param show: (bool) 是否显示进度消息，默认为 True
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return add_category_dic: (dict) 添加新 category 后的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 category_index 的赋值
        if category_index is not None:
            category_index = category_index
        else:
            category_index = self.Category_Index

        # 确保 sort_axis 有效
        if sort_axis not in ['column', 'row']:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"sort_axis must be 'row' or 'column'.")

        # 检查 category_name 和 category_number 是否符合要求
        total_count = None
        if category_name is not None or category_number is not None:
            if not (isinstance(category_name, list) and isinstance(category_number, list)):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"both category_name and category_number must be lists.")
            if len(category_name) != len(category_number):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"category_name and category_number must have the same length.")
            if not all(isinstance(name, str) for name in category_name):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"all elements in category_name must be strings.")
            if not all(isinstance(num, int) and num > 0 for num in category_number):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"all elements in category_number must be positive integers.")

            # 计算总数量
            total_count = sum(category_number)

        # 遍历字典以按照要求删除行 / 列
        for title, data_df in data_dic.items():
            if sort_axis == 'column':  # 添加列
                if category_name is None and category_number is None:
                    if title not in self.category_dic:
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"\033[95merror:\033[0m '{title}' not found in self.category_dic.")
                    if len(self.category_dic[title]) != len(data_df):
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"({len(self.category_dic[title])}) does not match the number of rows "
                                         f"({len(data_df)}) in '{title}'.")
                    category_values = self.category_dic[title]
                else:
                    if len(data_df) != total_count:
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"total category count ({total_count}) does not match the number of rows "
                                         f"({len(data_df)}) in '{title}'.")

                    # 创建分类列
                    category_values = []
                    for name, num in zip(category_name, category_number):
                        category_values.extend([name] * num)

                # 添加新列
                data_dic[title][category_index] = category_values

                # 如果 show == True，则打印信息
                if show:
                    print(f"In \033[92m{title}\033[0m, the column \033[31m{category_index}\033[0m has been added.")

            elif sort_axis == 'row':  # 添加行
                if category_name is None and category_number is None:
                    if title not in self.category_dic:
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"\033[95merror:\033[0m '{title}' not found in self.category_dic.")
                    if len(self.category_dic[title]) != len(data_df.columns):
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"\033[95merror:\033[0m Length mismatch: self.category_dic['{title}'] "
                                         f"({len(self.category_dic[title])}) does not match the number of columns "
                                         f"({len(data_df.columns)}) in '{title}'.")
                    category_values = self.category_dic[title]
                else:
                    if len(data_df.columns) != total_count:
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"Total category count ({total_count}) does not match the number of columns "
                                         f"({len(data_df.columns)}) in '{title}'.")

                    # 创建分类行
                    category_values = []
                    for name, num in zip(category_name, category_number):
                        category_values.extend([name] * num)

                # 转换为 DataFrame 并添加到原 DataFrame 末尾
                category_df = pd.DataFrame(data=[category_values], columns=data_df.columns, index=[category_index])
                data_dic[title] = pd.concat([data_df, category_df])

                # 如果 show == True，则打印信息
                if show:
                    print(f"In \033[92m{title}\033[0m, the row \033[31m{category_index}\033[0m has been added.")

        add_category_dic = data_dic

        # 覆盖原数据
        if overwrite:
            self.data_dic = add_category_dic

        self.add_category_dic = add_category_dic
        self.current_dic = 'add_category_dic'

        return add_category_dic

    # 根据 category_index 来拆分一个 "ALL" DataFrame 成多个 DataFrame
    def split_df_by_category(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                             category_index: Optional[str] = None, show: bool = True, overwrite: bool = True)\
            -> Dict[str, DataFrame]:
        """
        根据 category_index 来拆分一个 DataFrame 成多个 DataFrame，拆分后每个原本的 category_index 成为了其 key
        Split a DataFrame into multiple DataFrames based on category_index, where each unique value of
        category_index becomes a key.

        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据
        :param category_index: (str) 添加新行 / 列的标题，默认为 self.Category_Index
        :param show: (bool) 是否显示进度消息，默认为 True
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return split_by_category_dic: (dict) 拆分后的 dict，其包含多个 key-value，且 key 为原 category_index 值
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 category_index 的赋值
        if category_index is not None:
            category_index = category_index
        else:
            category_index = self.Category_Index

        # 1. 检查 data_dic 是否为空
        if not data_dic or not isinstance(data_dic, dict):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"data_dic cannot be empty and must be a dictionary.")

        # 2. 如果 data_dic 的长度大于 1，则直接打印并返回
        if len(data_dic) > 1:
            split_by_category_dic = data_dic
            print(f"The dictionary already contains multiple key-value pairs: \033[31m{list(data_dic.keys())}\033[0m.")

            # 覆盖原数据
            if overwrite:
                self.data_dic = split_by_category_dic

            self.split_by_category_dic = split_by_category_dic
            self.current_dic = 'split_by_category_dic'

            return split_by_category_dic

        # 3. 获取唯一的 DataFrame
        title, data_df = next(iter(data_dic.items()))

        # 4. 检查 category_index 是否存在
        if category_index not in data_df.columns:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Column '{category_index}' not found in DataFrame.")

        # 5. 按 category_index 拆分 DataFrame
        split_by_category_dic = {cat: data_df[data_df[category_index] == cat].copy()
                                 for cat in data_df[category_index].unique()}

        # 6. 打印拆分后的结果
        if show:
            print(f"The DataFrame has been split by \033[32m{category_index}\033[0m into "
                  f"\033[34m{len(split_by_category_dic)}\033[0m groups: "
                  f"Category: \033[31m{list(split_by_category_dic.keys())}\033[0m.")

        # 覆盖原数据
        if overwrite:
            self.data_dic = split_by_category_dic

        self.split_by_category_dic = split_by_category_dic
        self.current_dic = 'split_by_category_dic'

        return split_by_category_dic

    # 根据 category_index 来整合多个 DataFrame 成一个 "ALL" DataFrame
    def merge_df_by_category(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                             category_index: Optional[str] = None, title: Optional[str] = None,
                             keep_common_only: bool = False, show: bool = True, overwrite: bool = True)\
            -> Dict[str, DataFrame]:
        """
        根据 category_index 来将多个 DataFrame 整合成一个 DataFrame，此时 category_index 加在最后一列 (相对第一个 DataFrame 的)，
        且其值为原本的 key
        Multiple dataframes are combined into a single DataFrame using category_index. In this case,
        category_index is added to the last column (Relative to the first DataFrame) with the original key value.

        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据
        :param category_index: (str) 添加新行 / 列的标题，默认为 self.Category_Index
        :param title: (str) 整合成一个 DataFrame 时，对应 key 的 title，默认为 'ALL'
        :param keep_common_only: (bool) 是否仅保留共有列，默认为 False
        :param show: (bool) 是否显示进度消息，默认为 True
        :param overwrite: (bool) 是否覆盖原 data_dic 数据，默认为 True

        :return merge_by_category_dic: (dict) 整合后带有 category_index 的 dict
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 检查 category_index 的赋值
        if category_index is not None:
            category_index = category_index
        else:
            category_index = self.Category_Index

        # 检查 title 的赋值
        if title is not None:
            title = title
        else:
            title = 'ALL'

        # 检查 data_dic 中的 value 是否均为 DataFrame 格式
        if not isinstance(data_dic, dict) or not all(isinstance(v, pd.DataFrame) for v in data_dic.values()):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"data_dic must be a dictionary that contains a DataFrame as a value.")

        # 如果 data_dic 只有一个键值对，则直接返回原字典
        if len(data_dic) == 1:

            merge_by_category_dic = data_dic
            if show:
                print(f"The \033[31m{list(data_dic.keys())[0]}\033[0m contains only a single key-value pair.")

            # 覆盖原数据
            if overwrite:
                self.data_dic = merge_by_category_dic

            self.merge_by_category_dic = merge_by_category_dic
            self.current_dic = 'merge_by_category_dic'

            return merge_by_category_dic

        # 处理多个 DataFrame
        merge_by_category_dic = {}
        merged_list = []
        common_columns = set.intersection(*(set(df.columns) for df in data_dic.values())) if keep_common_only else None

        for df_title, data_df in data_dic.items():
            temp_df = data_df.copy()
            if keep_common_only:
                temp_df = temp_df[list(common_columns)]  # 仅保留共有列
            temp_df[category_index] = df_title  # 添加类别索引列
            merged_list.append(temp_df)

        merged_df = pd.concat(merged_list, ignore_index=True)
        # 在原字典中新增合并后的 DataFrame
        merge_by_category_dic[title] = merged_df

        if show:
            print(f"The dictionary has been merged, and its title is \033[31m{title}\033[0m.")

        # 覆盖原数据
        if overwrite:
            self.data_dic = merge_by_category_dic

        self.merge_by_category_dic = merge_by_category_dic
        self.current_dic = 'merge_by_category_dic'

        return merge_by_category_dic


""" 管理系统 """
class Manager(Optimizer):
    """
    文件系统

    Some basic operations, such as opening files, saving files, and also having the ability to show graphs.
    It is worth noting that the parameters for reading and storing data can be saved. Additionally, the
    data can be stored in a database.
    """

    # 使用全局变量作为类属性的默认值
    Magic_Database = Magic_Database
    Category_Index = Category_Index

    # 字体，大小及加粗
    font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 22}
    font_ticket = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18}
    font_legend = {'family': 'Times New Roman', 'weight': 'bold', 'size': 16}
    font_mark = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}

    # 程序休息时间
    interval_time = interval_time

    # 0 初始化
    def __init__(self,

                 # 接收参数 (7)
                 data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None, title: Optional[str] = None,
                 x_list: Optional[list] = None, y_list: Optional[list] = None, x_label: str = None, y_label: str = None,

                 # 关键参数 (7)
                 txt_path: Optional[str] = None, excel_path: Optional[str] = None, json_path: Optional[str] = None,
                 keyword: Optional[str] = None, file_pattern: Optional[str] = None, save_path: Optional[str] = None,
                 magic_database: Optional[str] = None,
                 ):
        """
        # 接收参数 (7)
        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据，当传入的 dict 长度不为1时，
                         需要每个 DataFrame 中的列都一样 (即同类型的)
        :param data_df:  (DataFrame) 输入 DataFrame
        :param title: (str) 数据的 title
        :param x_list: (list) x坐标的 list
        :param y_list: (list) y坐标的 list
        :param x_label: (str) x坐标的 label
        :param y_label: (str) y坐标的 label

        # 文件打开路径，名称匹配，已有 dict (7)
        :param txt_path: (str) TXT 文件路径，可以是文件路径，也可以是目录
        :param excel_path: (str) Excel 文件路径，可以是文件路径，也可以是目录
        :param json_path: (str) JSON 文件路径，也可以是目录
        :param keyword: (str) 关键词，为需要目标数据的类型
        :param file_pattern: (str) 当 path 为目录时，file_pattern 为目录下文件的正则表达式匹配，只有 path 为目录时才有意义
        :param save_path: (str) 保存路径
        :param magic_database: (str) 数据库的位置
        """

        # 超类初始化
        super().__init__(data_dic=data_dic, data_df=data_df, title=title, x_list=x_list, y_list=y_list,
                         x_label=x_label, y_label=y_label)  # 数据初始化时自动完成数据分配

        # 接收参数 (7)
        self.data_dic = data_dic
        self.data_df = data_df
        self.title = title
        self.x_list = x_list
        self.y_list = y_list
        self.x_label = x_label
        self.y_label = y_label

        # 文件打开路径，名称匹配 (7)
        self.txt_path = txt_path
        self.excel_path = excel_path
        self.json_path = json_path
        self.keyword = keyword  # 如果接入了 keyword， 那么输入的其它 to_magic_dic 中有的数据将会被 to_magic_dic 覆盖
        self.file_pattern = file_pattern
        self.save_path = save_path
        if magic_database is not None:
            self.magic_database = magic_database
        else:
            self.magic_database = self.Magic_Database

        # file 共有参数 (5，算上 x_label & y_label)
        self.increasing_order = None
        self.delete_nan = None
        self.swap_column = None

        # read_txt() 文件相关参数 (2)
        self.delimiter = None
        self.columns_txt = None

        # read_excel() 文件相关参数 (5)
        self.sheet = None
        self.header = None
        self.index = None
        self.columns_excel = None
        self.rows_excel = None

        # read_json() 文件相关参数 (1)
        self.key_point_dic = None  # 为读取到的数据

        # save_json() 存储相关参数 (6)
        self.sample_json_dic = None  # (dict) key 为 title，value 为需要保存的参数
        self.title_tuple = None  # (tuple) 为需要保存数据的 title 的 tuple
        self.sampled_dic = None  # (dict) value 为归一化后的 DataFrame 数据
        self.key_sampled_dic = None  # (dict) value 为特殊点 key 归一化后的 DataFrame 数据
        self.sample_rule_dic = None  # (dict) value 为归一化的规则
        self.interval_sample = None  # (float) 为归一化后的间隔

        # show 共有参数 () (13，算上 save_path)
        self.show_in_one = None  # 是否绘制在同一张图片中
        self.dpi = None  # 默认所有图片的保存图像均为 dpi=600
        self.width_height = None  # 图片的宽和高
        self.grid = None  # 图片是否加风格
        self.alpha = None  # 曲线的透明程序，1 为不透明，0 为完全透明
        self.show_label = None  # 是否显示线条注解
        self.x_min = None  # X 轴显示的最小值
        self.x_max = None  # X 轴显示的最大值
        self.y_min = None  # Y 轴显示的最小值
        self.y_max = None  # Y 轴显示的最大值
        self.background_color = None  # 背景颜色
        self.background_transparency = None  # 背景色的透明度

        # plot_line() 相关参数 (3)
        self.line_color = None
        self.line_style = None
        self.line_width = None

        # plot_scatter() 相关参数 (3)
        self.point_color = None
        self.point_style = None
        self.point_size = None

        # 二十种常用的配色方案
        self.color_palette = [
            '#1f77b4',  # 蓝色 | Blue (Tableau Blue)
            '#ff7f0e',  # 橙色 | Orange (Tableau Orange)
            '#2ca02c',  # 绿色 | Green (Tableau Green)
            '#d62728',  # 红色 | Red (Tableau Red)
            '#9467bd',  # 紫色 | Purple (Tableau Purple)
            '#8c564b',  # 棕色 | Brown (Tableau Brown)
            '#e377c2',  # 粉色 | Pink (Tableau Pink)
            '#7f7f7f',  # 灰色 | Gray (Tableau Gray)
            '#bcbd22',  # 橄榄绿 | Olive (Tableau Olive)
            '#17becf',  # 青色 | Cyan (Tableau Cyan)
            '#005AB5',  # 深蓝色 | Deep Blue (Color Universal Design)
            '#DC3220',  # 深红色 | Deep Red (Color Universal Design)
            '#A6611A',  # 赭石色 | Ochre
            '#E7298A',  # 品红色 | Magenta
            '#6A3D9A',  # 深紫色 | Dark Purple
            '#FF8C00',  # 深橙色 | Dark Orange
            '#66C2A5',  # 浅绿色 | Light Green
            '#FC8D62',  # 珊瑚色 | Coral
            '#8DA0CB',  # 暗蓝色 | Muted Blue
        ]

        # 数据初始化分配 和 数据类型导入
        if type(self) == Manager:  # 当 self 为 Manager 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # Manager 及其子类需要调用以初始化属性

        # 此三行目的是使 DataFrame 中的数据显示全
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.max_rows', None)  # 显示所有行
        pd.set_option('display.width', 500)  # 显示宽度

        # 此三行目的是使角标的字体也改成 Times New Roman
        plt.rcParams["mathtext.fontset"] = "custom"  # 设置数字文本的字体，使其可以自定义
        plt.rcParams["mathtext.rm"] = "Times New Roman"  # 设置正常数字文本的字体为 Times New Roman
        plt.rcParams["mathtext.it"] = "Times New Roman:italic"  # 设置斜体数字文本的字体为 Times New Roman

        # 1  smooth_curve()   /* 变量声明 */
        self.spline_smoothing_dic = None  # (dict) 平滑曲线的 dict，value 为 spline
        # 2  locate_point()
        self.key_point_dic = None  # (dict) 保存的特殊点的 dict，value 为一个包含特殊点的 dict
        # 3  improve_precision()
        self.smoothing_dic = None  # (dict) value 为高精度 DataFrame 数据，精度与 precision_smoothing 有关
        self.precision_smoothing = None  # (float) smoothing_dic 中 DataFrame 的精度
        # 4  reduce_point()
        self.dispersed_dic = None  # (dict) value 为低精度 DataFrame 数据，精度与 interval_disperse 有关
        self.interval_disperse = None  # (float) dispersed_dic 每两个点间的间隔需要大于该值
        # 5  normalized_data()
        self.normalized_dic = None  # (dict) value 为标准化后的 DataFrame 数据，与接入的参数有关
        self.key_normalized_dic = None  # (dict) value 为标准化后的特殊点 key 的数据
        self.normalize_rule_dic = None  # (dict) value 为坐标标准化变化规则，无法直接自乘
        # 6  adjust_data()
        self.adjusted_dic = None  # (dict) value 为调整后的 DataFrame 数据，与接入的参数有关
        self.key_adjusted_dic = None  # (dict) value 为调整后的特殊点 key 的数据
        self.adjust_rule_dic = None  # (dict) value 为坐标调整变化规则
        # 7  assign_weight()
        self.balanced_dic = None  # (dict) value 为整合后的 DataFrame 数据
        self.weight_list_dic = None  # (dict) value 为数据权重的 list
        self.weight_dic = None  # (dict) key 为特殊点 or 普通点，value 为坐标权重的 dict
        # 8  fit_curve()
        self.spline_fitting_dic = None  # (dict) 拟合曲线的 dict，value 为 spline
        # 9  restore_precision()
        self.fitting_dic = None  # (dict) value 为拟合后原精度的 DataFrame 数据
        # 10 realize_data()
        self.realized_dic = None  # (dict) value 为真实后的 DataFrame 数据

    # 1 TXT 文件的读取
    def read_txt(self, txt_path: Optional[str] = None, file_pattern: Optional[str] = None,
                 x_label: str = None, y_label: str = None, increasing_order: Optional[int] = None,
                 delete_nan: Optional[bool] = None, swap_column: Optional[bool] = None,
                 delimiter: Optional[str] = None, columns_txt: Union[list, None] = None,
                 update_to: Optional[bool] = None) -> Dict[str, DataFrame]:
        """
        获取 TXT 文件的方法，返回的数据为一个 dict
        Method to retrieve a TXT file, returning data as a dictionary.

        # 关键参数 (2)
        :param txt_path: (str) TXT 文件路径，可以是文件路径，也可以是目录，若被赋值则对该路径文件 / 目录进行处理
        :param file_pattern: (str) 当 path 为目录时，file_pattern 为目录下文件的正则表达式匹配，只有 path 为目录时才有意义

        # file 共有参数 (5)
        :param x_label: (str) X 坐标的标题
        :param y_label: (str) Y 坐标的标题
        :param increasing_order: (int) 以第几列为顺序排列并更新行索引，默认为 None 表示原顺序排列
        :param delete_nan: (bool) 是否删除全为 NaN 的行和列，默认为 False
        :param swap_column: (bool) 交换第一列和第二列，默认为 False，表示不交换

        # read_txt() 相关参数 (2)
        :param delimiter: (str) TXT 文件中的分割符，用以分列
        :param columns_txt: (list) TXT 文件所用的列，以 0 开头

        # 临时变量 (1)
        :param update_to: (bool) 是否更新至原来的 data_dic，如果 key 有重名则覆盖

        # 返回值 (1)
        :return data_dic: (dict) 返回该 TXT 文件的 dict，key 为文件名的测TXT的名称，value 为 DataFrame (该文件的有效数据)
                 如果初始化时为 TXT 文件路径，则返回的 dict 中仅有该文件一个数据的 key & value
                 如果初始化时为 TXT 文件的目录路径，则返回的dict中含有该目录下所有 TXT 文件数据的 key & value
        DataFrame: column1:       float64
                   column2:       float64
                   dtype:         object
        """

        # 检查赋值 (9)
        if True:

            if txt_path is not None:
                txt_path = txt_path
            elif self.txt_path is not None:
                txt_path = self.txt_path
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"No task path input provided.")

            if file_pattern is not None:
                file_pattern = file_pattern
            elif self.file_pattern is not None:
                file_pattern = self.file_pattern
            else:
                file_pattern = r'.*'

            if x_label is not None:
                x_label = x_label
            elif self.x_label is not None:
                x_label = self.x_label
            else:
                x_label = None

            if y_label is not None:
                y_label = y_label
            elif self.y_label is not None:
                y_label = self.y_label
            else:
                y_label = None

            if increasing_order is not None:
                increasing_order = increasing_order
            elif self.increasing_order is not None:
                increasing_order = self.increasing_order
            else:
                increasing_order = None

            if delete_nan is not None:
                delete_nan = delete_nan
            elif self.delete_nan is not None:
                delete_nan = self.delete_nan
            else:
                delete_nan = False

            if swap_column is not None:
                swap_column = swap_column
            elif self.swap_column is not None:
                swap_column = self.swap_column
            else:
                swap_column = False

            if delimiter is not None:
                delimiter = delimiter
            elif self.delimiter is not None:
                delimiter = self.delimiter
            else:
                delimiter = r' '

            if columns_txt is not None:
                columns_txt = columns_txt
            elif self.delimiter is not None:
                columns_txt = self.columns_txt
            else:
                columns_txt = [0, 1]

            if update_to is not None:
                update_to = update_to
            else:
                update_to = False

        # 检查路径是否是绝对路径
        if not os.path.isabs(txt_path):
            # 如果是相对路径，则将其转换为绝对路径
            txt_path = os.path.abspath(txt_path)

        # 收到的 txt_path 为 TXT 文件路径的情况
        if os.path.isfile(txt_path) and (txt_path.endswith('.txt') or txt_path.endswith('.TXT')):

            # 检查 TXT 文件的编码类型
            with open(txt_path, 'rb') as f:

                # 获取文件的编码类型
                result = chardet.detect(f.read())
                encoding = result['encoding']

            # 打开 TXT 文件
            with open(txt_path, 'r', encoding=encoding) as f:
                lines = f.readlines()  # 读取 TXT 文件

            # 找到有效行
            pattern = re.compile(pattern=r'\s*-?\d+(\.\d+)?')  # 空格开头，匹配带正负号的小数数字
            valid_lines = [line for line in lines if pattern.match(line)]  # 运用正则表达式找到以数字开头的行
            # 将读取到的为 DataFrame 数据
            data_df = pd.DataFrame([re.split(delimiter, line.strip()) for line in valid_lines])

            # 只保留需要留下的列
            data_df = data_df.iloc[:, columns_txt]  # 默认为前两列

            if delete_nan:
                data_df = data_df.dropna(how='all')  # 删除一行中全部是 NaN 值的行
                data_df = data_df.dropna(axis=1, how='all')  # 删除一列中全部是 NaN 值的列

            # 更改分割符和列索引的名称
            if x_label is not None:
                data_df = data_df.rename(columns={data_df.columns[0]: x_label})
            if y_label is not None:
                data_df = data_df.rename(columns={data_df.columns[1]: y_label})

            # 获取所有列的名字
            columns = data_df.columns
            # 创建一个字典，所有列都设置为 float，除了 self.Category_Index 列
            convert_dict = {}

            for col in columns:
                if col == self.Category_Index:
                    convert_dict[col] = str
                else:
                    try:
                        # 尝试将每一列转换为 float
                        data_df[col].astype(float)
                        convert_dict[col] = float
                    except ValueError:
                        convert_dict[col] = str

            # 应用astype()，传递字典
            data_df = data_df.astype(convert_dict)

            # 交换第一列和第二列
            if swap_column:

                # 保存原始列名
                col1_name = data_df.columns[0]  # 第一列的名称
                col2_name = data_df.columns[1]  # 第二列的名称

                # 交换列数据
                data_df[col1_name], data_df[col2_name] = data_df[col2_name].copy(), data_df[col1_name].copy()

                # 重新命名列
                data_df.rename(columns={col1_name: col2_name, col2_name: col1_name}, inplace=True)

            if increasing_order is not None:

                # 重新排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[increasing_order])
                data_df.reset_index(drop=True, inplace=True)

            # 返回该 TXT 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
            data_dic = {re.match(pattern=r'(\.?\/.*\/)(?P<name>[^\/]+)\..*$', string=txt_path).group('name'): data_df}

            # 是否更新
            if not update_to:
                self.data_dic = data_dic  # 将得到的数据的 dict 传给 self.data_dic
            else:
                self.data_dic.update(data_dic)  # 将得到的数据的 dict 更新至 self.data_dic

            return data_dic

        # 收到的 txt_path 为 TXT 文件目录路径的情况
        elif os.path.isdir(txt_path):

            # 创造一个空的 dict 以存储数据
            data_dic = {}
            for file in os.listdir(txt_path):
                file_path = os.path.join(txt_path, file)  # 目录下所有的文件的路径
                if re.search(file_pattern, file) and (file_path.endswith('.txt') or file_path.endswith('.TXT')):

                    # 检查 TXT 文件的编码类型
                    with open(file_path, 'rb') as f:

                        # 获取文件的编码类型
                        result = chardet.detect(f.read())
                        encoding = result['encoding']

                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()  # 读取 TXT 文件

                    # 找到有效行
                    pattern = re.compile(pattern=r'\s*-?\d+(\.\d+)?')  # 空格开头，匹配带正负号的小数数字
                    valid_lines = [line for line in lines if pattern.match(line)]  # 运用正则表达式找到以数字开头的行
                    # 从有效行创建 DataFrame
                    data_df = pd.DataFrame([re.split(delimiter, line.strip()) for line in valid_lines])

                    # 只保留需要留下的列
                    data_df = data_df.iloc[:, columns_txt]  # 默认为前两列

                    if delete_nan:
                        data_df = data_df.dropna(how='all')  # 删除一行中全部是 NaN 值的行
                        data_df = data_df.dropna(axis=1, how='all')  # 删除一列中全部是 NaN 值的列

                    # 更改分割符和列索引的名称
                    if x_label is not None:
                        data_df = data_df.rename(columns={data_df.columns[0]: x_label})
                    if y_label is not None:
                        data_df = data_df.rename(columns={data_df.columns[1]: y_label})

                    # 获取所有列的名字
                    columns = data_df.columns
                    # 创建一个字典，所有列都设置为 float，除了 self.Category_Index 列
                    convert_dict = {}

                    for col in columns:
                        if col == self.Category_Index:
                            convert_dict[col] = str
                        else:
                            try:
                                # 尝试将每一列转换为 float
                                data_df[col].astype(float)
                                convert_dict[col] = float
                            except ValueError:
                                convert_dict[col] = str

                    # 应用astype()，传递字典
                    data_df = data_df.astype(convert_dict)

                    # 交换第一列和第二列
                    if swap_column:

                        # 保存原始列名
                        col1_name = data_df.columns[0]  # 第一列的名称
                        col2_name = data_df.columns[1]  # 第二列的名称

                        # 交换列数据
                        data_df[col1_name], data_df[col2_name] = data_df[col2_name].copy(), data_df[col1_name].copy()

                        # 重新命名列
                        data_df.rename(columns={col1_name: col2_name, col2_name: col1_name}, inplace=True)

                    if increasing_order is not None:

                        # 重新排序并重新设置行索引
                        data_df = data_df.sort_values(by=data_df.columns[increasing_order])
                        data_df.reset_index(drop=True, inplace=True)

                    # 返回该 TXT 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
                    data_dic[re.match(pattern=r'(\.?\/.*\/)(?P<name>[^\/]+)\..*$',
                                      string=file_path).group('name')] = data_df

            # 是否更新
            if not update_to:
                self.data_dic = data_dic  # 将得到的数据的 dict 传给 self.data_dic
            else:
                self.data_dic.update(data_dic)  # 将得到的数据的 dict 更新至 self.data_dic

            return data_dic

        # 不为以上情况时报错
        else:
            # 如果给定的路径既不是一个文件，也不是一个目录，则引发值错误异常并给出错误提示
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Invalid txt_path: {txt_path}")

    # 2 Excel 文件的读取
    def read_excel(self, excel_path: Optional[str] = None, file_pattern: Optional[str] = None,
                   x_label: str = None, y_label: str = None, increasing_order: Optional[int] = None,
                   delete_nan: Optional[bool] = None, swap_column: Optional[bool] = None,
                   sheet: Union[int, str, None] = 0, header: Optional[int] = None, index: Optional[int] = None,
                   columns_excel: Union[int, list, None] = None, rows_excel: Optional[int] = None,
                   update_to: Optional[bool] = None) -> Dict[str, DataFrame]:
        """
        获取 Excel 表格的方法，返回的数据为一个 dict
        Method to retrieve an Excel file, returning data as a dictionary.

        # 关键参数 (2)
        :param excel_path: (str) Excel 文件路径，可以是文件路径，也可以是目录
        :param file_pattern: (str) 当 path 为目录时，file_pattern 为目录下文件的正则表达式匹配，只有 path 为目录时才有意义

        # file 共有参数 (5)
        :param x_label: (str) X 坐标的标题
        :param y_label: (str) Y 坐标的标题
        :param increasing_order: (int) 以第几列为顺序排列并更新行索引，默认为 None 表示原顺序排列
        :param delete_nan: (bool) 是否删除全为 NaN 的行和列，默认为 False
        :param swap_column: (bool) 交换第一列和第二列，默认为 False，表示不交换

        # read_excel() 相关参数 (5)
        :param sheet: (int / str) 文读取的 Excel 表格中第几个工作表的数据，
                      默认值为 0（int）表示第一页，也可以输入工作工作表的名字 (str)
        :param header: (int) 以第几行为表的列标，默认值为 O，表示第一行
        :param index: (int) 第几列数据为行索引，默认为 None，表示无行索引
        :param columns_excel: (int / list) 使用第哪些列的数据，默认为 None，表示所有列
        :param rows_excel: (int) 使用几行数据，默认为 None，表示表头以下所有行

        # 临时变量 (1)
        :param update_to: (bool) 是否更新至原来的 data_dic，如果 key 有重名则覆盖

        # 返回值 (1)
        :return data_dic: (dict) 返回该 Excel 文件的 dict，key 为文件名的测 Excel 的名称，value 为 DataFrame (该文件的有效数据)
                 如果初始化时为 Excel 文件路径，则返回的 dict 中仅有该文件一个数据的 key & value
                 如果初始化时为 Excel 文件的目录路径，则返回的dict中含有该目录下所有 Excel 文件数据的 key & value
        DataFrame: column1:       float64
                   column2:       float64
                   dtype:         object
        """

        # 检查赋值 (12)
        if True:

            if excel_path is not None:
                excel_path = excel_path
            elif self.excel_path is not None:
                excel_path = self.excel_path
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"No task path input provided.")

            if file_pattern is not None:
                file_pattern = file_pattern
            elif self.file_pattern is not None:
                file_pattern = self.file_pattern
            else:
                file_pattern = r'.*'

            if x_label is not None:
                x_label = x_label
            elif self.x_label is not None:
                x_label = self.x_label
            else:
                x_label = None

            if y_label is not None:
                y_label = y_label
            elif self.y_label is not None:
                y_label = self.y_label
            else:
                y_label = None

            if increasing_order is not None:
                increasing_order = increasing_order
            elif self.increasing_order is not None:
                increasing_order = self.increasing_order
            else:
                increasing_order = None

            if delete_nan is not None:
                delete_nan = delete_nan
            elif self.delete_nan is not None:
                delete_nan = self.delete_nan
            else:
                delete_nan = None

            if swap_column is not None:
                swap_column = swap_column
            elif self.swap_column is not None:
                swap_column = self.swap_column
            else:
                swap_column = None

            if sheet is not None:
                sheet = sheet
            elif self.sheet is not None:
                sheet = self.sheet
            else:
                sheet = 0

            if header is not None:
                header = header
            elif self.header is not None:
                header = self.header
            else:
                header = 0

            if index is not None:
                index = index
            elif self.index is not None:
                index = self.index
            else:
                index = None

            if columns_excel is not None:
                columns_excel = columns_excel
            elif self.columns_excel is not None:
                columns_excel = self.columns_excel
            else:
                columns_excel = None

            if rows_excel is not None:
                rows_excel = rows_excel
            elif self.rows_excel is not None:
                rows_excel = self.rows_excel
            else:
                rows_excel = None

            if update_to is not None:
                update_to = update_to
            else:
                update_to = False

        # 检查路径是否是绝对路径
        if not os.path.isabs(excel_path):
            # 如果是相对路径，则将其转换为绝对路径
            excel_path = os.path.abspath(excel_path)

        # 收到的 excel_path 为 Excel 文件路径的情况
        if os.path.isfile(excel_path) and excel_path.endswith('.xlsx'):
            data_dic = {}

            # 读取 Excel 文件并将数据赋给 data_dic
            data_df = pd.read_excel(excel_path, sheet_name=sheet, header=header, index_col=index,
                                    usecols=columns_excel, nrows=rows_excel)

            if delete_nan:
                data_df = data_df.dropna(how='all')  # 删除一行中全部是 NaN 值的行
                data_df = data_df.dropna(axis=1, how='all')  # 删除一列中全部是 NaN 值的列

            # 更改分割符和列索引的名称
            if x_label is not None:
                data_df = data_df.rename(columns={data_df.columns[0]: x_label})
            if y_label is not None:
                data_df = data_df.rename(columns={data_df.columns[1]: y_label})

            # 获取所有列的名字
            columns = data_df.columns
            # 创建一个字典，所有列都设置为 float，除了 self.Category_Index 列
            convert_dict = {}

            for col in columns:
                if col == self.Category_Index:
                    convert_dict[col] = str
                else:
                    try:
                        # 尝试将每一列转换为 float
                        data_df[col].astype(float)
                        convert_dict[col] = float
                    except ValueError:
                        convert_dict[col] = str

            # 应用astype()，传递字典
            data_df = data_df.astype(convert_dict)

            # 交换第一列和第二列
            if swap_column:

                # 保存原始列名
                col1_name = data_df.columns[0]  # 第一列的名称
                col2_name = data_df.columns[1]  # 第二列的名称

                # 交换列数据
                data_df[col1_name], data_df[col2_name] = data_df[col2_name].copy(), data_df[col1_name].copy()

                # 重新命名列
                data_df.rename(columns={col1_name: col2_name, col2_name: col1_name}, inplace=True)

            if increasing_order is not None:
                # 重新排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[increasing_order])
            data_df.reset_index(drop=True, inplace=True)

            # 返回该 Excel 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
            data_dic[re.match(pattern=r'(\.?\/.*\/)(?P<name>[^\/]+)\..*$', string=excel_path).group('name')] = data_df

            # 是否更新
            if not update_to:
                self.data_dic = data_dic  # 将得到的数据的 dict 传给 self.data_dic
            else:
                self.data_dic.update(data_dic)  # 将得到的数据的 dict 更新至 self.data_dic

            return data_dic

        # 收到的 excel_path 为 Excel 文件目录路径的情况
        elif os.path.isdir(excel_path):

            # 创造一个空的 dict 以存储数据
            data_dic = {}
            for file in os.listdir(excel_path):
                file_path = os.path.join(excel_path, file)  # 目录下所有文件的路径

                if re.search(file_pattern, file) and file_path.endswith('.xlsx'):

                    # 读取 Excel 文件并将数据赋给 data_dic
                    data_df = pd.read_excel(file_path, sheet_name=sheet, header=header, index_col=index,
                                            usecols=columns_excel, nrows=rows_excel)

                    if delete_nan:
                        data_df = data_df.dropna(how='all')  # 删除一行中全部是 NaN 值的行
                        data_df = data_df.dropna(axis=1, how='all')  # 删除一列中全部是 NaN 值的列

                    # 更改分割符和列索引的名称
                    if x_label is not None:
                        data_df = data_df.rename(columns={data_df.columns[0]: x_label})
                    if y_label is not None:
                        data_df = data_df.rename(columns={data_df.columns[1]: y_label})

                    # 获取所有列的名字
                    columns = data_df.columns
                    # 创建一个字典，所有列都设置为 float，除了 self.Category_Index 列
                    convert_dict = {}

                    for col in columns:
                        if col == self.Category_Index:
                            convert_dict[col] = str
                        else:
                            try:
                                # 尝试将每一列转换为 float
                                data_df[col].astype(float)
                                convert_dict[col] = float
                            except ValueError:
                                convert_dict[col] = str

                    # 应用astype()，传递字典
                    data_df = data_df.astype(convert_dict)

                    # 交换第一列和第二列
                    if swap_column:
                        
                        # 保存原始列名
                        col1_name = data_df.columns[0]  # 第一列的名称
                        col2_name = data_df.columns[1]  # 第二列的名称

                        # 交换列数据
                        data_df[col1_name], data_df[col2_name] = data_df[col2_name].copy(), data_df[col1_name].copy()

                        # 重新命名列
                        data_df.rename(columns={col1_name: col2_name, col2_name: col1_name}, inplace=True)

                    if increasing_order is not None:
                        # 重新排序并重新设置行索引
                        data_df = data_df.sort_values(by=data_df.columns[increasing_order])
                    data_df.reset_index(drop=True, inplace=True)

                    # 返回该 Excel 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
                    data_dic[re.match(pattern=r'(\.?\/.*\/)(?P<name>[^\/]+)\..*$',
                                      string=file_path).group('name')] = data_df

            # 是否更新
            if not update_to:
                self.data_dic = data_dic  # 将得到的数据的 dict 传给 self.data_dic
            else:
                self.data_dic.update(data_dic)  # 将得到的数据的 dict 更新至 self.data_dic

            return data_dic

        # 不为以上情况时报错
        else:
            # 如果给定的路径既不是一个文件，也不是一个目录，则引发值错误异常并给出错误提示
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, Invalid excel_path: {excel_path}")

    # 3 读取 JSON 文件的读取
    def read_json(self, keyword: Optional[str] = None, json_path: Optional[str] = None,
                  magic_database: Optional[str] = None, update_to: Optional[bool] = None)\
            -> Tuple[Dict[str, DataFrame], Dict[str, any], Dict[str, any]]:
        """
        获取 JSON 表格的方法，返回的数据为一个 dict
        Method to retrieve a JSON file, returning data as a dictionary.

        注意：keyword 与 json_path 只能有一个被赋值
        Attention: Only one of keyword and json_path can be assigned a value.

        # 关键参数 (3)
        :param keyword: (str) 关键字，根据关键字来寻找数据库中的数据
        :param json_path: (str) JSON 文件的路径或目录路径
        :param magic_database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置

        # 临时变量 (1)
        :param update_to: (bool) 是否更新至原来的 data_dic，如果 key 有重名则覆盖

        # 返回值 (3)
        :return data_dic: (dict) 返回该 JSON 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
        :return key_point_dic: (dict) 返回该 JSON 文件的 dict，key 为文件名，value 为 DataFrame (该文件的关键点)
        :return sampled_dic: (dict) 返回该 JSON 文件的 dict，key 为文件名，value 为 DataFrame (有效数据 + 关键点)
                            如果初始化时为文件路径，则返回的 dict 中仅有该文件一个数据的 key & value
                            如果初始化时为文件的目录路径，则返回的 dict 中含有该目录下所有 JSON 文件数据的 key & value

        DataFrame: column1:       float64
                   column2:       float64
                   dtype:         object

        an example of sampled_dic is
                {
                file_name1: { 'title': title1 (str), 'data_df': data_df (DataFrame), 'key_dic': key_dic1
                (dict (str: DataFrame)), 'rule_dic': rule_dic1 (dict (str: float)), 'interval': interval1 (float)},
                file_name2: { 'title': title2 (str), 'data_df': data_df2 (DataFrame), 'key_dic': key_dic2
                (dict (str: DataFrame)), 'rule_dic': rule_dic2 (dict (str: float)), 'interval': interval2 (float)}
                }
        """

        if True:

            # 检查 keyword 是否被赋值
            if keyword is not None:
                keyword = keyword
            else:
                keyword = self.keyword

            # 检查 json_path 是否被赋值
            if json_path is not None:
                json_path = json_path
            else:
                json_path = self.json_path

            # 确保 keyword 与 json_path 只有一个被赋值
            if keyword is not None and json_path is not None:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Either 'keyword' or 'json_path' must be assigned, but not both. "
                                 f"Please provide only one.")

            elif keyword is None and json_path is None:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Either 'keyword' or 'json_path' must be assigned, but not both. "
                                 f"Please provide only one.")

            # 检查 magic_database 是否被赋值
            if magic_database is not None:
                magic_database = magic_database
            else:
                magic_database = self.magic_database

            if update_to is not None:
                update_to = update_to
            else:
                update_to = False

        data_dic = {}
        key_point_dic = {}
        sampled_dic = {}

        # keyword 被赋值的情况
        if keyword is not None:

            target_database = os.path.join(magic_database, keyword)

            for filename in os.listdir(target_database):

                sample_single_dic = {}
                file_path = os.path.join(target_database, filename)

                # 是 JSON 文件的情况
                if filename.endswith('.json') and os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='UTF-8') as file:
                        file_content = file.read()

                    # 使用 json.loads() 解析文件内容为 dict
                    sample = json.loads(file_content)

                    # 获取 key 对应的 value 的 python 格式，而不是 JSON 格式
                    title = sample.get('title')

                    data_df_json = sample.get('data_df')
                    data_df = pd.read_json(StringIO(data_df_json))

                    key_dic_json = sample.get('key_dic')
                    key_dic = {k: pd.read_json(StringIO(v)) if isinstance(v, str) else
                    pd.read_json(StringIO(json.dumps(v))) for k, v in key_dic_json.items()}

                    rule_dic_json = sample.get('rule_dic')
                    rule_dic = {k: float(v) for k, v in rule_dic_json.items()}
                    
                    interval_json = sample.get('interval')
                    interval = float(interval_json)

                    # data_dic
                    data_dic[title] = data_df

                    # key_point_dic
                    key_point_dic[title] = key_dic

                    # sample_single_dic
                    sample_single_dic['title'] = title
                    sample_single_dic['data_df'] = data_df
                    sample_single_dic['key_dic'] = key_dic
                    sample_single_dic['rule_dic'] = rule_dic
                    sample_single_dic['interval'] = interval

                # 如果不是 JSON 文件
                else:
                    continue

                # 使用 os.path.splitext() 分离文件名和扩展名
                name_without_extension, _ = os.path.splitext(filename)
                sampled_dic[name_without_extension] = sample_single_dic

        # json_path 被赋值的情况
        if json_path is not None:

            # 检查路径是否是绝对路径
            if not os.path.isabs(json_path):
                # 如果是相对路径，则将其转换为绝对路径
                json_path = os.path.abspath(json_path)

            # json_path 是目录的情况
            if os.path.isdir(json_path):

                for filename in os.listdir(json_path):

                    sample_single_dic = {}
                    file_path = os.path.join(json_path, filename)

                    # 是 JSON 文件的情况
                    if filename.endswith('.json') and os.path.isfile(file_path):
                        with open(file_path, 'r', encoding='UTF-8') as file:
                            file_content = file.read()

                        # 使用 json.loads() 解析文件内容为 dict
                        sample = json.loads(file_content)

                        # 获取 key 对应的 value 的 python 格式，而不是 JSON 格式
                        title = sample.get('title')
                        data_df_json = sample.get('data_df')
                        data_df = pd.read_json(StringIO(data_df_json))
                        key_dic_json = sample.get('key_dic')
                        key_dic = {k: pd.read_json(StringIO(v)) for k, v in key_dic_json.items()}
                        rule_dic_json = sample.get('rule_dic')
                        rule_dic = {k: float(v) for k, v in rule_dic_json.items()}
                        interval_json = sample.get('interval')
                        interval = float(interval_json)

                        # data_dic
                        data_dic[title] = data_df

                        # key_point_dic
                        key_point_dic[title] = key_dic

                        # sample_single_dic
                        sample_single_dic['title'] = title
                        sample_single_dic['data_df'] = data_df
                        sample_single_dic['key_dic'] = key_dic
                        sample_single_dic['rule_dic'] = rule_dic
                        sample_single_dic['interval'] = interval

                    # 如果不是 JSON 文件
                    else:
                        continue

                    sampled_dic[filename] = sample_single_dic

            # 是 JSON 文件的情况
            sample_single_dic = {}
            if json_path.endswith('.json') and os.path.isfile(json_path):
                with open(json_path, 'r', encoding='UTF-8') as file:
                    file_content = file.read()

                # 使用 json.loads() 解析文件内容为 dict
                sample = json.loads(file_content)

                # 获取 key 对应的 value 的 python 格式，而不是 JSON 格式
                title = sample.get('title')
                data_df_json = sample.get('data_df')
                data_df = pd.read_json(StringIO(data_df_json))
                key_dic_json = sample.get('key_dic')
                key_dic = {k: pd.read_json(StringIO(v)) for k, v in key_dic_json.items()}
                rule_dic_json = sample.get('rule_dic')
                rule_dic = {k: float(v) for k, v in rule_dic_json.items()}
                interval_json = sample.get('interval')
                interval = float(interval_json)

                # data_dic
                data_dic[title] = data_df

                # key_point_dic
                key_point_dic[title] = key_dic

                # sample_single_dic
                sample_single_dic['title'] = title
                sample_single_dic['data_df'] = data_df
                sample_single_dic['key_dic'] = key_dic
                sample_single_dic['rule_dic'] = rule_dic
                sample_single_dic['interval'] = interval

                file_name = os.path.splitext(os.path.basename(json_path))[0]
                sampled_dic[file_name] = sample_single_dic

            if not os.path.exists(json_path):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"json_path is not a path to a JSON file or folder.")

            if not json_path.endswith('.json') and os.path.isfile(json_path):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"json_path is not a path to a JSON file or folder.")

        # 是否更新
        if not update_to:
            self.data_dic = data_dic  # 将得到的数据的 dict 传给 self.data_dic，下同
            self.key_point_dic = key_point_dic
            self.sampled_dic = sampled_dic
        else:
            self.data_dic.update(data_dic)  # 将得到的数据的 dict 更新至 self.data_dic，下同
            self.key_point_dic.update(key_point_dic)
            self.sampled_dic.update(sampled_dic)

        return data_dic, key_point_dic, sampled_dic

    # 4 存储为 TXT 类型文件
    def save_txt(self, data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                 save_path: Union[bool, str] = True, keyword: Optional[str] = None, delimiter: Optional[str] = None,
                 float_precision: int = None, remark: Optional[str] = None, show_result: bool = True) -> None:
        """
        将数据保存为 TXT 文件
        Save the data as an TXT file.

        注意：remark 除第一行外，不能以数字开头
        Attention: remark Except the first line, cannot start with a number.

        :param data_dic: (dict) 若被赋值，则会对该 dic 进行保存
        :param data_df: (DataFrame) 若被赋值，则会对该 Dataframe 进行保存
        :param save_path: (str) 保存目录，若无赋值则用初始化中的 self.save_path，若为 False 或 None 则为不保存
                          注意：应为保存的目录而非具体路径
        :param keyword: (str) 关键字，将关键字信息保存至 TXT 文件
        :param delimiter: (str) 分割符，默认为 '   '(制表符)
        :param float_precision: (int) 保留小数的位数，默认为 2
        :param remark: (str) 保存时的留言，默认为无留言
        :param show_result: (bool) 是否打印保存的路径信息，默认为 True

        :return: None
        """

        # 检查赋值
        if True:

            # data_dic 与 data_df 被赋值的情况
            if data_dic is not None and data_df is not None:  # 两个均被赋值时
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Either 'data_dic' or 'data_df' must be assigned, but not both. "
                                 f"Please provide only one.")

            elif data_dic is not None:  # 只有 data_dic 被赋值时
                data_dic = copy.deepcopy(data_dic)
            elif data_df is not None:  # 只有 data_df 被赋值时
                data_dic = {'Untitled': data_df}
            else:
                # 使用 getattr 来动态获取属性
                data_dic = copy.deepcopy(getattr(self, self.current_dic))

            if data_dic is None:  # 两个均未被赋值时
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Either 'data_dic' or 'data_df' must be assigned, but not both. "
                                 f"Please provide only one.")

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，line_save_path 将会被其赋值，若 save_path == '' 则保存在运行的py文件的目录下
            else:
                save_path = save_path

            if keyword is not None:
                keyword = keyword
            elif self.keyword is not None:
                keyword = self.keyword
            else:
                keyword = 'Not classified'

            if delimiter is not None:
                delimiter = delimiter
            else:
                delimiter = r'   '

            if float_precision is not None:
                float_precision = float_precision
            else:
                float_precision = 2

        for title, data_df in data_dic.items():

            # 确保 save_path 是有效的路径，并且有权限写入
            if save_path is not None:  # 如果 save_path 的值不为 None，则保存
                file_name = title + ".txt"  # 初始文件名为 'title.txt'
                full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = title + f"_{count}.txt"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = title + f"_{count}.txt"
                        full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                # 打开一个文件用于写入
                with open(full_file_path, 'w') as file:
                    # 写入文件的标题
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 获取当前时间
                    file.write('Title: ' + title + '\n')  # 写入文件标题
                    file.write('Keyword: ' + keyword + '\n')  # 写入文件类型
                    file.write('Time: ' + current_time + '\n')  # 写入当前时间
                    if remark is not None:
                        file.write('Remark: ' + remark + '\n')  # 写入备注
                    file.write('\n')  # 写入一个空行作为数据的开始前的分隔

                    # 写入列标题，列与列之间用 delimiter 分隔
                    header_row = delimiter.join(str(data_df.columns[i])
                                                if i % 2 == 0 else str(data_df.columns[i]) + delimiter
                                                for i in range(len(data_df.columns)))

                    file.write('Columns: ' + header_row + '\n')  # 写入列标题

                    # 遍历DataFrame中的每一行数据
                    for index, row in data_df.iterrows():
                        row_data = []  # 用于存储处理后的行数据
                        for i, item in enumerate(row):  # 遍历行中的每个元素
                            # 如果是浮点数，保留指定的小数位数
                            formatted_item = f'{item:.{float_precision}f}' if isinstance(item, float) else str(item)
                            row_data.append(formatted_item)  # 添加到行数据列表
                            # 如果不是最后一项数据，在数据后面添加 delimiter 分隔符
                            if i < len(row) - 1:  # 每两列之后添加两个 delimiter 分隔符，其他情况添加一个delimiter分隔符
                                row_data.append(delimiter if i % 2 == 0 else delimiter + delimiter)

                        file.write(''.join(row_data) + '\n')  # 将处理后的行数据写入文件

                # 打印结果
                if show_result:
                    print(f"File saved to \033[94m{full_file_path}\033[0m")

        return None

    # 5 存储为 Excel 类型文件
    def save_excel(self, data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                   save_path: Union[bool, str] = True, show_result: bool = True) -> None:
        """
        将数据保存为 Excel 文件
        Save the data as an Excel file.

        :param data_dic: (dict) 若被赋值，则会对该 dic 进行保存
        :param data_df: (DataFrame) 若被赋值，则会对该 Dataframe 进行保存
        :param save_path: (str) 保存目录，若无赋值则用初始化中的 self.save_path，若为 False 或 None 则为不保存
                          注意：应为保存的目录而非具体路径
        :param show_result: (bool) 是否打印保存的路径信息，默认为 True

        :return: None
        """

        # 检查赋值
        if True:

            # data_dic 与 data_df 被赋值的情况
            if data_dic is not None and data_df is not None:  # 两个均被赋值时
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Either 'data_dic' or 'data_df' must be assigned, but not both. "
                                 f"Please provide only one.")

            elif data_dic is not None:  # 只有 data_dic 被赋值时
                data_dic = copy.deepcopy(data_dic)
            elif data_df is not None:  # 只有 data_df 被赋值时
                data_dic = {'Untitled': data_df}
            else:
                # 使用 getattr 来动态获取属性
                data_dic = copy.deepcopy(getattr(self, self.current_dic))

            if data_dic is None:  # 两个均未被赋值时
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Either 'data_dic' or 'data_df' must be assigned, but not both. "
                                 f"Please provide only one.")

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，line_save_path 将会被其赋值，若 save_path == '' 则保存在运行的py文件的目录下
            else:
                save_path = save_path

        for title, data_df in data_dic.items():

            # 确保 save_path 是有效的路径，并且有权限写入
            if save_path is not None:  # 如果 save_path 的值不为 None，则保存
                file_name = title + ".xlsx"  # 初始文件名为 'title.xlsx'
                full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = title + f"_{count}.xlsx"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = title + f"_{count}.xlsx"
                        full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                data_df.to_excel(full_file_path, index=False)  # 使用完整路径将 DataFrame 保存到 Excel 文件

                # 打印结果
                if show_result:
                    print(f"File saved to \033[94m{full_file_path}\033[0m")

        return None

    # 6 存储为 JSON 类型文件
    def save_json(self, keyword: Optional[str] = None, data_dic: Optional[Dict[str, DataFrame]] = None,
                  key_sampled_dic: Optional[Dict[str, DataFrame]] = None, interval_sample: Optional[float] = 0.01,
                  magic_database: Optional[str] = None, to_save: bool = True) -> Dict[str, any]:
        """
        将数据以 JSON 的形式入库存储
        Store the data in the form of JSON.

        自动检查是否已储存过，若已储存，则会跳过该组数据 (通过 title 进行检查)
        Automatically checks if it has been saved, and if it has, the set of data is skipped (checked by title).

        :param keyword: (str) 存储的实验数据的类型，默认为 self.keyword 中的类型
        :param data_dic: (dict)  key 为 title， value 数据的 DataFrame，默认为 smoothing_dic 数据优先
        :param key_sampled_dic: (dict)  key 为 title， value 为归一化后特殊点 key 的 DataFrame
        :param interval_sample: (float) 间隔，间隔将大于 interval_sample ，若为 None，刚保持原精度，默认为 0.01
        :param magic_database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param to_save:  (bool) 是否存储，默认为 True

        :return sample_json_dic: (dict) dict 中 key 为 title， value 为 JSON 格式数据，即储存的数据
        an example of sample_json_dic is
                {
                file_name1: { 'title': title1 (str), 'data_df': data_df (DataFrame), 'key_dic': key_dic1
                (dict (str: DataFrame)), 'rule_dic': rule_dic1 (dict (str: float)), 'interval': interval1 (float)},
                file_name2: { 'title': title2 (str), 'data_df': data_df2 (DataFrame), 'key_dic': key_dic2
                (dict (str: DataFrame)), 'rule_dic': rule_dic2 (dict (str: float)), 'interval': interval2 (float)}
                }
        Please note that the type should be JSON.
        """

        # 判断 keyword 是否有输入
        if keyword is not None:
            keyword = keyword
        else:
            if self.keyword is not None:
                keyword = self.keyword
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, No available keyword value.")

        # 检查 magic_database 是否被赋值
        if magic_database is not None:
            magic_database = magic_database
        else:
            magic_database = self.magic_database

        # 判断 keyword 是否在 keyword_tuple 中
        keyword_tuple = self.get_subdirectories(folder_path=magic_database)
        if keyword not in keyword_tuple:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(
                f"\033[95mIn {method_name} of {class_name}\033[0m, "
                f"an unexpected one matching keyword be found: {keyword}. "
                f"Available types: {[key for key in keyword_tuple]}")

        # 判断是用有输入的 data_dic
        sampled_before_normalize_dic = None
        if data_dic is not None:
            sampled_before_normalize_dic = data_dic
        else:
            for attr_name in ['smoothing_dic', 'data_dic']:
                selected_data_dic = getattr(self, attr_name)
                if selected_data_dic is not None:
                    sampled_before_normalize_dic = copy.deepcopy(selected_data_dic)
                    break

        # 判断是用有输入的 key_sampled_dic
        if key_sampled_dic is not None:
            key_dic = key_sampled_dic
        else:
            key_dic = copy.deepcopy(self.key_point_dic)

        self.interval_sample = interval_sample
        title_list = []
        point_current_scale_dic = {}
        sampled_before_interval_dic = {}
        sample_rule_dic = {}

        # data_dic 归一化部分
        for title, data_df in sampled_before_normalize_dic.items():
            point_current_scale = {}
            data_df = copy.deepcopy(data_df)
            title_list.append(title)

            # 寻找最大值和最小值点，不能用 self.point_scale_dic，因为通过调整精度会改变最大最小值
            x_min = data_df.iloc[:, 0].min()
            x_max = data_df.iloc[:, 0].max()
            y_min = data_df.iloc[:, 1].min()
            y_max = data_df.iloc[:, 1].max()

            # 存入 point_current_scale 中
            point_current_scale['x_min'] = x_min
            # point_current_scale['x_max'] = x_max
            point_current_scale['y_min'] = y_min
            # point_current_scale['y_max'] = y_max

            # 将每个 title 下的 point_current_scale 存入 point_current_scale_dic
            point_current_scale_dic[title] = point_current_scale

            # 计算变化规则
            x_sample_rule = 1 / (x_max - x_min)
            y_sample_rule = 1 / (y_max - y_min)

            # 将标准化规则加入到 dict 中
            sample_rule_dic[title] = {
                'x_rule': x_sample_rule,
                'y_rule': y_sample_rule}

            # 应用变换规则，得到标准化后的数据
            data_df.iloc[:, 0] = (data_df.iloc[:, 0] - x_min) * x_sample_rule
            data_df.iloc[:, 1] = (data_df.iloc[:, 1] - y_min) * y_sample_rule
            # 以行坐标排序并重新设置行索引
            data_df = data_df.sort_values(by=data_df.columns[0])
            data_df.reset_index(drop=True, inplace=True)
            sampled_before_interval_dic[title] = data_df

        self.title_tuple = tuple(title_list)  # 转化为 tuple 后保存
        self.sample_rule_dic = sample_rule_dic

        key_sampled_dic = {}

        if key_dic is not None:  # 当特殊点存在时
            # 特殊点 key 处理部分
            for title, key_point in key_dic.items():

                key_sampled_single_dic = {}
                x_sample_rule = sample_rule_dic[title]['x_rule']
                y_sample_rule = sample_rule_dic[title]['y_rule']
                x_min = point_current_scale_dic[title]['x_min']
                y_min = point_current_scale_dic[title]['y_min']

                # extremum 进行标准化
                if key_point['extremum'] is not None:
                    extremum_sampled_df = key_point['extremum'].copy()
                    # 应用变换规则，得到标准化后的数据
                    extremum_sampled_df.iloc[:, 0] = \
                        (extremum_sampled_df.iloc[:, 0] - x_min) * x_sample_rule
                    extremum_sampled_df.iloc[:, 1] = \
                        (extremum_sampled_df.iloc[:, 1] - y_min) * y_sample_rule

                    # 以行坐标排序并重新设置行索引
                    extremum_sampled_df = extremum_sampled_df.sort_values(
                        by=extremum_sampled_df.columns[0])
                    extremum_sampled_df.reset_index(drop=True, inplace=True)
                    key_sampled_single_dic['extremum'] = extremum_sampled_df

                # extremum 不存在的情况
                else:
                    key_sampled_single_dic['extremum'] = None

                # inflection 进行标准化
                if key_point['inflection'] is not None:
                    inflection_sampled_df = key_point['inflection'].copy()
                    # 应用变换规则，得到标准化后的数据
                    inflection_sampled_df.iloc[:, 0] = \
                        (inflection_sampled_df.iloc[:, 0] - x_min) * x_sample_rule
                    inflection_sampled_df.iloc[:, 1] = \
                        (inflection_sampled_df.iloc[:, 1] - y_min) * y_sample_rule

                    # 以行坐标排序并重新设置行索引
                    inflection_sampled_df = \
                        inflection_sampled_df.sort_values(by=inflection_sampled_df.columns[0])
                    inflection_sampled_df.reset_index(drop=True, inplace=True)
                    key_sampled_single_dic['inflection'] = inflection_sampled_df

                # inflection 不存在的情况
                else:
                    key_sampled_single_dic['inflection'] = None

                # max 进行标准化
                if key_point['max'] is not None:
                    max_sampled_df = key_point['max'].copy()
                    # 应用变换规则，得到标准化后的数据
                    max_sampled_df.iloc[:, 0] = \
                        (max_sampled_df.iloc[:, 0] - x_min) * x_sample_rule
                    max_sampled_df.iloc[:, 1] = \
                        (max_sampled_df.iloc[:, 1] - y_min) * y_sample_rule

                    # 以行坐标排序并重新设置行索引
                    max_sampled_df = max_sampled_df.sort_values(by=max_sampled_df.columns[0])
                    max_sampled_df.reset_index(drop=True, inplace=True)
                    key_sampled_single_dic['max'] = max_sampled_df

                # max 不存在的情况
                else:
                    key_sampled_single_dic['max'] = None

                # min 进行标准化
                if key_point['min'] is not None:
                    min_sampled_df = key_point['min'].copy()
                    # 应用变换规则，得到标准化后的数据
                    min_sampled_df.iloc[:, 0] = \
                        (min_sampled_df.iloc[:, 0] - x_min) * x_sample_rule
                    min_sampled_df.iloc[:, 1] = \
                        (min_sampled_df.iloc[:, 1] - y_min) * y_sample_rule

                    # 以行坐标排序并重新设置行索引
                    min_sampled_df = min_sampled_df.sort_values(by=min_sampled_df.columns[0])
                    min_sampled_df.reset_index(drop=True, inplace=True)
                    key_sampled_single_dic['min'] = min_sampled_df

                # min 不存在的情况
                else:
                    key_sampled_single_dic['min'] = None

                key_sampled_dic[title] = key_sampled_single_dic

            self.key_sampled_dic = key_sampled_dic

        # data_dic 改动精度部分
        if interval_sample is not None:

            sampled_dic = {}

            # 遍历 dict 以完成所有数据的处理
            for title, data_df in sampled_before_interval_dic.items():

                data_df = copy.deepcopy(data_df)

                # 删除 self.smoothing_df 中的部分数据，使得横坐标间隔等于或大于 interval_disperse
                last_x = data_df.iloc[0, 0]  # 上一个横坐标值
                for _, row in data_df.iterrows():
                    current_x = row.iloc[0]
                    if current_x - last_x >= interval_sample:
                        last_x = current_x
                    else:
                        data_df = data_df.drop(index=row.name)

                # 以行坐标排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[0])
                data_df.reset_index(drop=True, inplace=True)
                sampled_dic[title] = data_df
        else:
            sampled_dic = sampled_before_interval_dic

        self.sampled_dic = sampled_dic

        # 将数据储存为 JSON 格式
        sample_json_dic = {}
        for title in title_list:

            saved_sampled_single_dic = {}

            # 获取相应的数据
            saved_data_df = copy.deepcopy(sampled_dic[title])
            if key_dic is not None:  # 当特殊点存在时
                saved_key_dic = copy.deepcopy(key_sampled_dic[title])
            else:
                saved_key_dic = None
            saved_rule_dic = copy.deepcopy(sample_rule_dic[title])

            # 将 title 转化为 JSON 格式
            title_json = title.replace("'", "\"")
            # 将 saved_data_df 转化为 JSON 格式
            saved_data_df_json = saved_data_df.to_json(orient='records')
            # 将 saved_key_dic 转化为 JSON 格式
            saved_key_dic_json = {}
            if key_dic is not None:  # 当特殊点存在时
                for key_type, key_df in saved_key_dic.items():  # 遍历原始 dict ，将每个 DataFrame 转换为 JSON 格式的字符串
                    saved_key_dic_json[key_type.replace("'", "\"")] = key_df.to_json(orient='records')
            # 将 saved_rule_dic 转化为 JSON 格式
            saved_rule_dic_json = {}
            for rule_name, rule_number in saved_rule_dic.items():  # 遍历原始 dict ，将每个 DataFrame 转换为 JSON 格式的字符串
                saved_rule_dic_json[rule_name.replace("'", "\"")] = json.dumps(rule_number)
            # 将 interval_sample 转化为 JSON 格式
            interval_sample_json = json.dumps(interval_sample)

            # 将数据保存在单独的 dict 中
            saved_sampled_single_dic["title"] = title_json
            saved_sampled_single_dic["data_df"] = saved_data_df_json
            saved_sampled_single_dic["key_dic"] = saved_key_dic_json
            saved_sampled_single_dic["rule_dic"] = saved_rule_dic_json
            saved_sampled_single_dic["interval"] = interval_sample_json

            # 将数据汇总在 sample_json_dic 中
            sample_json_dic[title] = json.dumps(saved_sampled_single_dic)

        self.sample_json_dic = sample_json_dic

        target_database = os.path.join(magic_database, keyword)

        # 寻找目标库中所有的 title，目的是查看是否已经入库
        title_values = []
        for filename in os.listdir(target_database):
            file_path = os.path.join(target_database, filename)
            if filename.endswith('.json') and os.path.isfile(file_path):
                with open(file_path, 'r', encoding='UTF-8') as file:
                    file_content = file.read()

                # 使用 json.loads() 解析文件内容为 dict
                sample = json.loads(file_content)
                # 获取 title 键对应的值
                title_value = sample.get('title')
                title_values.append(title_value)

        # 保存数据至库中
        if to_save:
            for title, sample_json in sample_json_dic.items():

                # 若已经入库，则进入下一个循环
                if title in title_values:
                    continue

                count = 1
                magic_name = keyword + f"_{count}.json"  # 初始文件名为读取数据时的文件名
                save_path = os.path.join(target_database, magic_name)
                # 找是否存在，并不断 +1，直到不重复
                while os.path.exists(save_path):
                    count += 1
                    magic_name = keyword + f"_{count}.json"  # 初始文件名为读取数据时的文件名
                    save_path = os.path.join(target_database, magic_name)

                # 将 JSON 字符串保存到文件中
                with open(save_path, 'w') as file:
                    file.write(sample_json)

        return sample_json_dic

    # 7 从数据类型快速获取读取和绘图参数 (自动运行并且识别)
    def to_magic(self, keyword: Optional[str] = None) -> Tuple[dict, dict]:
        """
        根据目标数据快速找到接口
        Find the interface quickly based on the target data.

        :param keyword: (str) 关键词，为需要目标数据的类型

        :return target_file_dic: (DataFrame) 目标词条下读取参数的 dict
        :return target_plot_dic: (DataFrame) 目标词条下绘制参数的 dict
        """

        # 检查是否有新输入的 keyword
        if keyword is not None:
            keyword = keyword
            self.keyword = keyword
        else:
            keyword = self.keyword

        # 参数 dict
        to_magic_dic = {

            # 拉伸(应力-应变) 曲线
            'tensile': {
                'file_type': 'excel',  # file 类型
                'plot_type': 'line',  # show 类型

                'excel': {
                    # file 共有参数 (5)
                    'x_label': 'Strain / %',
                    'y_label': 'Stress / MPa',
                    'increasing_order': 0,
                    'delete_nan': True,
                    'swap_column': True,

                    # read_excel() 相关参数 (5)
                    'sheet': 0,
                    'header': 9,
                    'index': None,
                    'columns_excel': [6, 7],
                    'rows_excel': None
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (6, 4.5),
                    'show_grid': True,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 0,
                    'x_max': None,
                    'y_min': 0,
                    'y_max': None,
                    'background_color': 'GnBu',
                    'background_transparency': 0.15,

                    # plot_line() 相关参数 (3)
                    'line_color': ['#ff9914'],
                    'line_style': '-',
                    'line_width': 3,
                },

                'remark': 'Tensile curve with first digit data only.'},

            # 压缩(应力-应变) 曲线
            'compression': {
                'file_type': 'excel',  # file 类型
                'plot_type': 'line',  # show 类型

                'excel': {
                    # file 共有参数 (5)
                    'x_label': 'Strain / %',
                    'y_label': 'Stress / MPa',
                    'increasing_order': 0,
                    'delete_nan': True,
                    'swap_column': True,

                    # read_excel() 相关参数 (5)
                    'sheet': 0,
                    'header': 9,
                    'index': None,
                    'columns_excel': [6, 7],
                    'rows_excel': None
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (6, 4.5),
                    'show_grid': True,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 0,
                    'x_max': None,
                    'y_min': 0,
                    'y_max': None,
                    'background_color': 'afmhot',
                    'background_transparency': 0.15,

                    # plot_line() 相关参数 (3)
                    'line_color': ['#1E90FF'],
                    'line_style': '-',
                    'line_width': 3,
                },

                'remark': 'Tensile curve with first digit data only.'},

            # 扭转(应力-应变) 曲线
            'torsion': {
                'file_type': 'excel',  # file 类型
                'plot_type': 'line',  # show 类型

                'excel': {
                    # file 共有参数 (5)
                    'x_label': 'Load / N',
                    'y_label': 'Displacement / mm',
                    'increasing_order': 0,
                    'delete_nan': True,
                    'swap_column': True,

                    # read_excel() 相关参数 (5)
                    'sheet': 0,
                    'header': 9,
                    'index': None,
                    'columns_excel': [1, 2],
                    'rows_excel': None
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (6, 4.5),
                    'show_grid': True,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 0,
                    'x_max': None,
                    'y_min': 0,
                    'y_max': None,
                    'background_color': 'BrBG',
                    'background_transparency': 0.15,

                    # plot_line() 相关参数 (3)
                    'line_color': ['#9400D3'],
                    'line_style': '-',
                    'line_width': 3,
                },

                'remark': 'Tensile curve with first digit data only.'},

            # 拉曼 曲线
            'Raman': {
                'file_type': 'txt',  # file 类型
                'plot_type': 'line',  # show 类型

                'txt': {
                    # file 共有参数 (5)
                    'x_label': 'Wavenumber',
                    'y_label': 'Intensity',
                    'increasing_order': 0,
                    'delete_nan': False,
                    'swap_column': False,

                    # read_txt() 相关参数 (2)
                    'delimiter': r'\s+',
                    'columns_txt': [0, 1]
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (6, 4.5),
                    'show_grid': False,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 100,
                    'x_max': 2000,
                    'y_min': 0,
                    'y_max': None,
                    'background_color': 'GnBu',
                    'background_transparency': 0.15,

                    # plot_line() 相关参数 (3)
                    'line_color': ['#ff9914'],
                    'line_style': '-',
                    'line_width': 3,
                },

                'remark': 'The parameters of the Raman graph are drawn.'},

            # XRD 曲线
            'XRD': {
                'file_type': 'txt',  # file 类型
                'plot_type': 'line',  # show 类型

                'txt': {
                    # file 共有参数 (5)
                    'x_label': '2θ / degree',
                    'y_label': 'Intensity / (a.u.)',
                    'increasing_order': None,
                    'delete_nan': False,
                    'swap_column': False,

                    # read_txt() 相关参数 (2)
                    'delimiter': r'\s+',
                    'columns_txt': [0, 1]
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (6, 4.5),
                    'show_grid': False,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 5,
                    'x_max': 90,
                    'y_min': 0,
                    'y_max': None,
                    'background_color': 'GnBu',
                    'background_transparency': 0.15,

                    # plot_line() 相关参数 (3)
                    'line_color': ['#C04040'],
                    'line_style': '-',
                    'line_width': 3
                },

                'remark': 'The parameters of the XRD graph are drawn.'
                          'This attribute is recommended for reading data, but not advised for plotting.'},

            # 热膨胀 曲线
            'DIL': {
                'file_type': 'txt',  # file 类型
                'plot_type': 'line',  # show 类型

                'txt': {
                    # file 共有参数 (5)
                    'x_label': 'Temperature / \N{DEGREE SIGN}C',  # 用转译
                    'y_label': 'dL / Lo',
                    'increasing_order': None,
                    'delete_nan': False,
                    'swap_column': False,

                    # read_excel() 相关参数 (2)
                    'delimiter': r';\s*',
                    'columns_txt': [0, 2]
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': True,
                    'width_height': (6, 4.5),
                    'show_grid': True,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 0,
                    'x_max': None,
                    'y_min': None,
                    'y_max': None,
                    'background_color': 'afmhot',
                    'background_transparency': 0.15,

                    # plot_line() 相关参数 (3)
                    'line_color': ['#ff9914'],
                    'line_style': '-',
                    'line_width': 3,
                },

                'remark': 'Plotting of the thermal expansion curve,'
                          'but unable to change the values in the second column.'},

            # XPS 曲线
            'XPS': {
                'file_type': 'excel',  # file 类型
                'plot_type': 'scatter',  # show 类型

                'excel': {
                    # file 共有参数 (5)
                    'x_label': 'Binding energy / (eV)',
                    'y_label': 'Intensity / (a.u.)',
                    'increasing_order': 0,
                    'delete_nan': False,
                    'swap_column': False,

                    # read_excel() 相关参数 (5)
                    'sheet': 0,
                    'header': 15,
                    'index': None,
                    'columns_excel': [0, 2, 4],
                    'rows_excel': None
                },

                'scatter': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (6, 4.5),
                    'show_grid': False,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': None,
                    'x_max': None,
                    'y_min': None,
                    'y_max': None,
                    'background_color': 'afmhot',
                    'background_transparency': 0.15,

                    # plot_scatter() 相关参数 (3)
                    'point_color': ['black'],
                    'point_style': 'o',
                    'point_size': 3,
                },

                'remark': 'This is XPS data, consisting of three columns. The first column is the horizontal axis, '
                          'the second column represents the data coordinates, and the third column is '
                          'for blank coordinates. The horizontal axis values are in descending order.'
                          'This attribute is recommended for reading data, but not advised for plotting.'},

            # SEM-EDS Mapping
            'Mapping': {
                'file_type': 'txt',  # file 类型
                'plot_type': 'line',  # show 类型

                'txt': {
                    # file 共有参数 (5)
                    'x_label': 'keV',
                    'y_label': 'cps / eV',
                    'increasing_order': None,
                    'delete_nan': False,
                    'swap_column': False,

                    # read_txt() 相关参数 (2)
                    'delimiter': r'\s*,\s*',
                    'columns_txt': [0, 1]
                },

                'line': {
                    # plot 共有参数 (11)
                    'show_in_one': False,
                    'width_height': (8, 4.5),
                    'show_grid': True,
                    'alpha': 1,
                    'show_label': False,
                    'x_min': 0,
                    'x_max': None,
                    'y_min': 0,
                    'y_max': None,
                    'background_color': sns.light_palette(color='#ef5350', as_cmap=True),
                    'background_transparency': 0.1,

                    # plot_line() 相关参数 (3)
                    'line_color': ['purple'],
                    'line_style': '-',
                    'line_width': 1.5
                },

                'remark': 'The energy spectrum of the txt file in the scanned image of SEM-EDS Mapping is drawn. '
                          'Please change the parameters and separators as required.'},
        }

        # 当 keyword 与 to_magic_dic 中的 key 均不相同时报错
        if keyword not in list(to_magic_dic.keys()):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(
                f"\033[95mIn {method_name} of {class_name}\033[0m, "
                f"an unexpected one matching keyword be found: {keyword}."
                f" Available types: {[key for key in to_magic_dic]}")

        target_file_dic = None
        target_plot_dic = None

        # 从 to_magic_dic 中找出该词条
        for key_title in to_magic_dic:
            if key_title == keyword:  # 关键词与 to_magic_dic 中的词条对应时

                # 寻找 file 的类型
                file_type = to_magic_dic[key_title]['file_type']
                plot_type = to_magic_dic[key_title]['plot_type']
                # 取出该类型 dict 的内容
                target_file_dic = to_magic_dic[key_title][file_type]
                target_plot_dic = to_magic_dic[key_title][plot_type]

                if file_type == 'txt':  # 目标数据类型为 TXT 的情况
                    # file 共有属性 (5)
                    self.x_label = target_file_dic['x_label']
                    self.y_label = target_file_dic['y_label']
                    self.increasing_order = target_file_dic['increasing_order']
                    self.delete_nan = target_file_dic['delete_nan']
                    self.swap_column = target_file_dic['swap_column']

                    # read_txt() 相关参数 (2)
                    self.delimiter = target_file_dic['delimiter']
                    self.columns_txt = target_file_dic['columns_txt']

                elif file_type == 'excel':  # 目标数据类型为 Excel 的情况
                    # file 共有属性 (5)
                    self.x_label = target_file_dic['x_label']
                    self.y_label = target_file_dic['y_label']
                    self.increasing_order = target_file_dic['increasing_order']
                    self.delete_nan = target_file_dic['delete_nan']
                    self.swap_column = target_file_dic['swap_column']

                    # read_excel() 相关参数 (5)
                    self.sheet = target_file_dic['sheet']
                    self.header = target_file_dic['header']
                    self.index = target_file_dic['index']
                    self.columns_excel = target_file_dic['columns_excel']
                    self.rows_excel = target_file_dic['rows_excel']

                if plot_type == 'line':  # 绘图类型为 line 的情况
                    # plot 共有属性 (11)
                    self.show_in_one = target_plot_dic['show_in_one']
                    self.width_height = target_plot_dic['width_height']
                    self.grid = target_plot_dic['show_grid']
                    self.alpha = target_plot_dic['alpha']
                    self.show_label = target_plot_dic['show_label']
                    self.x_min = target_plot_dic['x_min']
                    self.x_max = target_plot_dic['x_max']
                    self.y_min = target_plot_dic['y_min']
                    self.y_max = target_plot_dic['y_max']
                    self.background_color = target_plot_dic['background_color']
                    self.background_transparency = target_plot_dic['background_transparency']

                    # plot_line() 相关参数 (3)
                    self.line_color = target_plot_dic['line_color']
                    self.line_style = target_plot_dic['line_style']
                    self.line_width = target_plot_dic['line_width']

                elif plot_type == 'scatter':
                    # plot 共有属性 (11)
                    self.show_in_one = target_plot_dic['show_in_one']
                    self.width_height = target_plot_dic['width_height']
                    self.grid = target_plot_dic['show_grid']
                    self.alpha = target_plot_dic['alpha']
                    self.show_label = target_plot_dic['show_label']
                    self.x_min = target_plot_dic['x_min']
                    self.x_max = target_plot_dic['x_max']
                    self.y_min = target_plot_dic['y_min']
                    self.y_max = target_plot_dic['y_max']
                    self.background_color = target_plot_dic['background_color']
                    self.background_transparency = target_plot_dic['background_transparency']

                    # plot_scatter() 相关参数 (3)
                    self.point_color = target_plot_dic['point_color']
                    self.point_style = target_plot_dic['point_style']
                    self.point_size = target_plot_dic['point_size']

                break  # 数据接入完成后退出循环

        return target_file_dic, target_plot_dic

    # 8 打印变量名称和内容
    def to_reality(self, variable: Union[str, bool, None] = None, search: str = 'current', color: str = 'brightmagenta',
                   total_width: int = 120, name_width: int = 20, only_df: bool = True,
                   print_what: str = 'all', print_result: bool = True, print_max_rows: Optional[int] = None) -> str:
        """
        用醒目的文字打印变量名称和内容
        Print variable names and contents in bold text.

        文本颜色改变：
        \033[31m (红色)，(所变颜色) + Hello (需要变色的文本内容）+ \033[0m

        :param variable: (str, bool) 需要打印名称及内容的变量，默认为 self.data_dic，只有当 search != 'current' 时才有效，
                         当 variable == True，且 search 为 'class', 'local' 或 'global' 时，为查找当前环境下的变量
                         在填写参数时，只需要填写变量名即可，无需带上 'self'，如 'data_dic' 或 'data_df'
        :param search: (str) 检索范围，有 'current', 'class', 'local' 和 'global'，分别表示类内部，局部和全局，默认为 'current'
        :param color: (str) 字体的颜色，默认为亮洋红色
        :param total_width: (int) 标题总宽度
        :param name_width: (int) 中间标题内容的宽度
        :param only_df: (bool) 是否打印 data_dic 中的 DataFrame 表格，只有当 data_dic 长度为 1 时才有效，默认为 True
        :param print_what: (str) 有 'all', 'name' 和 'value'，分别为变量的装饰后变量名和值，装饰后的变量名，变量值，默认为 'all'
        :param print_result: (bool) 是否打印修改后的文本内容，默认为 True
        :param print_max_rows: (bool) 单个 DataFrame 显示的最大行数，最大值为 10，默认为 None 表示无限制

        :return formatted_text: (str) 修改后的文本
        """

        # 检查变量赋值
        if variable is not None:
            variable = variable
        else:
            variable = self.current_dic

        # 更改显示的最大行数
        max_rows = pd.get_option('display.max_rows')  # 查看当前最大显示行数的配置
        if print_max_rows is not None:
            pd.set_option('display.max_rows', print_max_rows)
        else:
            pd.set_option('display.max_rows', None)

        # 颜色对应字典
        ansi_colors = {
            "black": "\033[30m",  # 黑色
            "red": "\033[31m",  # 红色
            "green": "\033[32m",  # 绿色
            "yellow": "\033[33m",  # 黄色
            "blue": "\033[34m",  # 蓝色
            "magenta": "\033[35m",  # 品红色/洋红色
            "cyan": "\033[36m",  # 青色
            "white": "\033[37m",  # 白色
            "darkred": "\033[31;2m",  # 暗红色
            "darkgreen": "\033[32;2m",  # 暗绿色
            "darkyellow": "\033[33;2m",  # 暗黄色
            "darkblue": "\033[34;2m",  # 暗蓝色
            "darkmagenta": "\033[35;2m",  # 暗品红色 / 暗洋红色
            "darkcyan": "\033[36;2m",  # 暗青色
            "darkwhite": "\033[37;2m",  # 暗白色
            "brightblack": "\033[90m",  # 亮黑色
            "brightred": "\033[91m",  # 亮红色
            "brightgreen": "\033[92m",  # 亮绿色
            "brightyellow": "\033[93m",  # 亮黄色
            "brightblue": "\033[94m",  # 亮蓝色
            "brightmagenta": "\033[95m",  # 亮品红色 / 亮洋红色
            "brightcyan": "\033[96m",  # 亮青色
            "brightwhite": "\033[97m",  # 亮白色

            # 背景色
            "bg_black": "\033[40m",  # 黑色背景
            "bg_red": "\033[41m",  # 红色背景
            "bg_green": "\033[42m",  # 绿色背景
            "bg_yellow": "\033[43m",  # 黄色背景
            "bg_blue": "\033[44m",  # 蓝色背景
            "bg_magenta": "\033[45m",  # 品红色背景 / 洋红色背景
            "bg_cyan": "\033[46m",  # 青色背景
            "bg_white": "\033[47m",  # 白色背景
            "bg_brightblack": "\033[100m",  # 亮黑色背景
            "bg_brightred": "\033[101m",  # 亮红色背景
            "bg_brightgreen": "\033[102m",  # 亮绿色背景
            "bg_brightyellow": "\033[103m",  # 亮黄色背景
            "bg_brightblue": "\033[104m",  # 亮蓝色背景
            "bg_brightmagenta": "\033[105m",  # 亮品红色背景 / 亮洋红色背景
            "bg_brightcyan": "\033[106m",  # 亮青色背景
            "bg_brightwhite": "\033[107m",  # 亮白色背景

            # 高亮色（Text attributes）
            "bold": "\033[1m",  # 粗体
            "dim": "\033[2m",  # 细体
            "italic": "\033[3m",  # 斜体
            "underline": "\033[4m",  # 下划线
            "blink": "\033[5m",  # 闪烁
            "reverse": "\033[7m",  # 反色
            "hidden": "\033[8m",  # 隐藏
            "reset": "\033[0m",  # 重置（恢复默认颜色）
        }

        print_name = None
        print_value = None
        name_str = ''

        # 在类中寻找变量时
        if search == 'current':

            # 使用 getattr 来动态获取属性
            selected_data_dic = copy.deepcopy(getattr(self, self.current_dic))
            print_name = self.current_dic

            # 如果值是可以被直接打印的，就打印出来
            if isinstance(selected_data_dic, dict):
                print_value = dict(selected_data_dic)  # 如果已经是字典，直接使用
            else:
                print_value = selected_data_dic  # 对于其他类型，直接打印

        # 在类中寻找变量时
        elif search == 'class':

            for name, value in vars(self).items():

                if variable is True and not isinstance(variable, str):  # variable 为 True 时，查找当前环境下所有变量名
                    name_str += name + '\n'  # 将 name 追加到字符串并换行

                elif name == variable:  # 比较属性名是否等于传入的变量名
                    print_name = self.__class__.__name__ + "." + name

                    # 如果值是可以被直接打印的，就打印出来
                    if isinstance(value, dict):
                        print_value = dict(value)  # 如果已经是字典，直接使用
                    elif isinstance(value, pd.DataFrame):  # 如果是 DataFrame 类型，直接使用
                        print_value = value
                    elif hasattr(value, 'to_dict'):  # 对于支持 to_dict 的对象，例如 Pandas DataFrame
                        print_value = value.to_dict()
                    else:
                        print_value = value  # 对于其他类型，直接打印

        # 在局部中寻找变量时
        elif search == 'local':

            for name, value in locals().items():

                if variable is True and not isinstance(variable, str):  # variable 为 True 时，查找当前环境下所有变量名
                    name_str += name + '\n'  # 将 name 追加到字符串并换行

                elif name == variable:  # 比较属性名是否等于传入的变量名
                    print_name = self.__class__.__name__ + "." + name

                    # 如果值是可以被直接打印的，就打印出来
                    if isinstance(value, dict):
                        print_value = dict(value)  # 如果已经是字典，直接使用
                    elif isinstance(value, pd.DataFrame):  # 如果是 DataFrame 类型，直接使用
                        print_value = value
                    elif hasattr(value, 'to_dict'):  # 对于支持 to_dict 的对象，例如 Pandas DataFrame
                        print_value = value.to_dict()
                    else:
                        print_value = value  # 对于其他类型，直接打印

        # 在全局中寻找变量时
        elif search == 'global':

            for name, value in globals().items():

                if variable is True and not isinstance(variable, str):  # variable 为 True 时，查找当前环境下所有变量名
                    name_str += name + '\n'  # 将 name 追加到字符串并换行

                elif name == variable:  # 比较属性名是否等于传入的变量名
                    print_name = self.__class__.__name__ + "." + name

                    # 如果值是可以被直接打印的，就打印出来
                    if isinstance(value, dict):
                        print_value = dict(value)  # 如果已经是字典，直接使用
                    elif isinstance(value, pd.DataFrame):  # 如果是 DataFrame 类型，直接使用
                        print_value = value
                    elif hasattr(value, 'to_dict'):  # 对于支持 to_dict 的对象，例如 Pandas DataFrame
                        print_value = value.to_dict()
                    else:
                        print_value = value  # 对于其他类型，直接打印

        # 都不为时则报错
        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the variable 'search' can only be one of 'current', 'class', 'local', or 'global'.")

        if print_name is None and variable is isinstance(variable, str):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the following variable was not found in the {search} variables:\n{variable}")

        if variable is True and not isinstance(variable, str) and search != 'current':
            print_name = f'variable in {search}'
            print_value = name_str

        # 检查 print_what 的赋值
        if print_what not in ['all', 'name', 'value']:
            raise ValueError("print_what must be 'all', 'name', or 'value'")

        name_text = None
        value_text = None

        # 变量标题的装饰
        if print_what == 'all' or print_what == 'name':

            start_color = ansi_colors[color]
            end_color = '\033[0m'

            # 计算前后字符的长度
            side_width = (total_width - name_width) // 2

            # 构造格式化字符串
            name_text = (
                            f"{'-' * side_width}{start_color}{print_name: ^{name_width}}{end_color}{'-' * side_width}"
                        ) + "\n"

        elif print_what == 'value':

            name_text = ""

        # 变量值
        if print_what == 'all' or print_what == 'value':

            if only_df and isinstance(variable, dict) and len(variable) == 1:  # 只打印 DataFrame 的情况
                title = list(variable.keys())[0]
                data_df = list(variable.values())[0]
                value_text = "title: " + str(title) + "\n" + str(data_df) + "\n"
            else:  # 打印全变量时
                value_text = str(print_value) + "\n"

        elif print_what == 'name':

            value_text = "\n"

        result_text = name_text + value_text

        if print_result:
            print(result_text)

        # 改回显示的最大行数
        pd.set_option('display.max_rows', max_rows)

        return result_text

    # 9 绘制折线统计图
    def plot_line(self,

                  data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                  save_path: Union[bool, str] = True, dpi: int = 600, image_title: str = '', show: bool = True,

                  show_in_one: Optional[bool] = None, width_height: Optional[tuple] = None,
                  show_grid: Optional[bool] = None, alpha: Optional[float] = None, show_label: Optional[bool] = None,
                  x_label: Optional[str] = None, y_label: Optional[str] = None,
                  x_min: Union[int, float, None] = None, x_max: Union[int, float, None] = None,
                  y_min: Union[int, float, None] = None, y_max: Union[int, float, None] = None,
                  background_color: Optional[str] = None, background_transparency: Optional[float] = None,

                  line_color: Union[list, tuple, None] = None, line_style: Optional[str] = None,
                  line_width: Optional[float] = None,

                  **kwargs) -> None:
        """
        绘制线形统计图
        Draw a line chart.

        # 关键参数 (4)
        :param data_dic: (dict) 若被赋值，则会对该 dic 进行绘图
        :param data_df: (DataFrame) 若被赋值，则会对该 Dataframe 进行绘图
        :param save_path: (str) 保存目录，若无赋值则用初始化中的 self.save_path，若为 False 或 None 则为不保存
                          注意：应为保存的目录而非具体路径
        :param dpi: (int) 图片保存的精度，只有在需要保存时才有意义，默认为 dpi=600，无赋值侧沿用初始化中的 dpi
        :param image_title: (str) 图的标题，默认为无标题
        :param show: (bool) 是否显示图片，默认为 True

        # 绘制参数 (13)
        :param show_in_one: (bool) 是否将所有数据展示在一张图片里，默认为 False
        :param width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)
        :param show_grid: (bool) 是否显示网格，默认不显示
        :param alpha: (float) 线条的透明度，默认为 1，表示不透明
        :param show_label: (bool) 是否展示线条注解，默认为否
        :param x_label: (str) X 轴的标题，默认为原列中的标题，若有则为该标题
        :param y_label: (str) Y 轴的标题，默认为原列中的标题，若有则为该标题
        :param x_min: (int / float) 横坐标的最小值
        :param x_max: (int / float) 横坐标的最大值
        :param y_min: (int / float) 纵坐标的最小值
        :param y_max: (int / float) 纵坐标的最大值
        :param background_color: (str / tuple) 设置图片的背景颜色，默认为无
        :param background_transparency: (float) 背景色的透明程度，最小越透明，只有存在背景色时才有意义，默认为 0.15

        # 线条参数 (3)
        :param line_color: (list / tuple) 线条的颜色，即 line 图像线条的颜色
                          line_color 为长度大于 1 的 list / tuple 时，只有 show_in_one == True 时才有意义
        :param line_style: (str) 最上方线条的风格，即 line 图像线条的风格，默认为线状
        :param line_width: (float) 线条的粗细程度，默认为 3

        :return: None

        --- **kwargs ---

        # 坐标轴部分 (6)
        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度旋转的角度
        - y_rotation: (float) Y 轴刻度旋转的角度
        - invert_x: (bool) 反转 X 轴，默认为 False
        - invert_y: (bool) 反转 Y 轴，默认为 False

        # 标记注释 (2)
        - target_index: (int / list) 标注点 list，或 int (为索引)
        - annotate_text: (str / list) 标注文字 list，或 str，只有在 target_index 存在的情况下才有意义，
                             且与target_index一一对应

        # 填充部分 (3)
        - fill_area: (bool) 是否填充，默认为 False
        - fill_color: (tuple, str) 填充颜色，默认为线条的颜色，如为 show_in_one 则只能为线条色，只有在 fill_area 为 True 时才有意义
        - fill_alpha: (float) 填充透明度，默认为 0.4，只有在 fill_area 为 True 时才有意义

        # 标记部分 (27)
        - x_marker_range: (tuple) 以 X 轴区域标记，可接受 list 和 tuple，当为 list 时表示闭区间，当为 tuple 时表示开区间
        - x_marker: (str) 以 X 轴区域标记时的标记符号，只有当 x_marker_range 被赋值时才有意义
        - x_marker_color: (str / tuple) 以 X 轴区域标记时的颜色，默认为红色，只有当 x_marker_range 被赋值时才有意义
        - x_marker_size: (int) 以 X 轴区域标记时的标记大小，只有当 x_marker_range 被赋值时才有意义
        - x_marker_interval: (int) 对于以 X 轴区域标记，每几个点标记一个，默认为 1 表示全标记，
                                 只有当 x_marker_range 被赋值时才有意义
        - y_marker_range: (tuple) Y 轴区域标记，可接受 list 和 tuple，当为 list 时表示闭区间，当为 tuple 时表示开区间
        - y_marker: (str) 以y轴区域标记时的标记符号，只有当 y_marker_range 被赋值时才有意义
        - y_marker_color: (str / tuple) 以 Y 轴区域标记时的颜色，默认为蓝色，只有当 y_marker_range 被赋值时才有意义
        - y_marker_size: (int) 以 Y 轴区域标记时的标记大小，只有当 y_marker_range 被赋值时才有意义
        - y_marker_interval: (int) 对于以 Y 轴区域标记，每几个点标记一个，默认为 1 表示全标记，
                                 只有当 y_marker_range 被赋值时才有意义
        - expression: (str) 表达式，用以表达式标记
        - expression_marker: (str) 表达式标记时的符号，只有当 expression 被赋值时才有意义
        - expression_marker_color: (str / tuple) 以表达式记时的颜色，默认为绿色，只有当 expression 被赋值时才有意义
        - expression_marker_size: (int) 以表达式标记时的标记大小，只有当 expression 被赋值时才有意义
        - expression_marker_interval: (int) 对于以表达式标记，每几个点标记一个，默认为 1 表示全标记，
                                          只有当 expression 被赋值时才有意义
        - max_show:  (bool) 是否显示最大值，默认为 False
        - max_show_marker: (str) 最大值的标记符号，只有 max_show 为 True 时才有意义
        - max_show_label: (str) 最大值的标签名，默认不显示，只有 max_show 为 True 时才有意义
        - max_show_color: (str / tuple) 最大值符号的颜色，默认为红色，只有 max_show 为 True 时才有意义
        - max_show_size:  (int) 最大值符号的大小，默认为 20，只有 max_show 为 True 时才有意义
        - max_show_text: (bool) 最大值标出，默认不显示，只有 max_show 为 True 时才有意义

        - min_show:  (bool) 是否显示最小值，默认为 False
        - min_show_marker: (str) 最小值的标记符号，只有 min_show 为 True 时才有意义
        - min_show_label: (str) 最小值的标签名，默认不显示，只有 min_show 为 True 时才有意义
        - min_show_color: (str / tuple) 最小值符号的颜色，默认为绿色，只有 min_show 为 True 时才有意义
        - min_show_size: (int) 最小值符号的大小，默认为 20，只有 min_show为True时才有意义
        - min_show_text: (bool) 最小值标出，默认不显示，只有 max_show 为 True 时才有意义
        """

        # 检查赋值 (2 + 16，算上 label)
        if True:

            # data_dic 与 data_df 被赋值的情况
            if data_dic is not None and data_df is not None:  # 两个均被赋值时
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"data_dic and data_dic should not be assigned at the same time.")
            elif data_dic is not None:  # 只有 data_dic 被赋值时
                data_dic = copy.deepcopy(data_dic)
            elif data_df is not None:  # 只有 data_df 被赋值时
                data_dic = {'Untitled': data_df}
            else:
                # 使用 getattr 来动态获取属性
                data_dic = copy.deepcopy(getattr(self, self.current_dic))

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                line_save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                line_save_path = None
            # 当有指定的 save_path 时，line_save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                line_save_path = save_path

            if show_in_one is not None:
                show_in_one = show_in_one
            elif self.show_in_one is not None:
                show_in_one = self.show_in_one
            else:
                show_in_one = False

            if width_height is not None:
                width_height = width_height
            elif self.width_height is not None:
                width_height = self.width_height
            else:
                width_height = (6, 4.5)

            if show_grid is not None:
                show_grid = show_grid
            elif self.grid is not None:
                show_grid = self.grid
            else:
                show_grid = False

            if alpha is not None:
                alpha = alpha
            elif self.alpha is not None:
                alpha = self.alpha
            else:
                alpha = 1

            if show_label is not None:
                show_label = show_label
            elif self.show_label is not None:
                show_label = self.show_label
            else:
                show_label = False

            if x_label is not None:
                x_label = x_label
            elif self.x_label is not None:
                x_label = self.x_label
            else:
                x_label = 'X'

            if y_label is not None:
                y_label = y_label
            elif self.y_label is not None:
                y_label = self.y_label
            else:
                y_label = 'Y'

            if x_min is not None:
                x_min = x_min
            elif self.x_min is not None:
                x_min = self.x_min
            else:
                x_min = None

            if x_max is not None:
                x_max = x_max
            elif self.x_max is not None:
                x_max = self.x_max
            else:
                x_max = None

            if y_min is not None:
                y_min = y_min
            elif self.y_min is not None:
                y_min = self.y_min
            else:
                y_min = None

            if y_max is not None:
                y_max = y_max
            elif self.y_max is not None:
                y_max = self.y_max
            else:
                y_max = None

            if background_color is not None:
                background_color = background_color
            elif self.background_color is not None:
                background_color = self.background_color
            else:
                background_color = None

            if background_transparency is not None:
                background_transparency = background_transparency
            elif self.background_transparency is not None:
                background_transparency = self.background_transparency
            else:
                background_transparency = 0.15

            if line_color is not None:
                line_color_single = line_color[0]
            elif self.line_color is not None:
                line_color_single = self.line_color[0]
            else:
                line_color_single = self.color_palette[0]

            if line_style is not None:
                line_style = line_style
            elif self.line_style is not None:
                line_style = self.line_style
            else:
                line_style = '-'

            if line_width is not None:
                line_width = line_width
            elif self.line_width is not None:
                line_width = self.line_width
            else:
                line_width = 3

        # 检查关键字参数
        if True:

            # 检查关键字参数的 list
            expected_kwargs = [

                # 坐标轴部分 (6)
                "custom_xticks", "custom_yticks", "x_rotation", "y_rotation", "invert_x", "invert_y",

                # 标记注释 (2)
                "target_index", "annotate_text",

                # 填充部分 (3)
                "fill_area", "fill_color", "fill_alpha",

                # 标记部分 (27)
                "x_marker_range", "x_marker", "x_marker_color", "x_marker_size", "x_marker_interval",

                "y_marker_range", "y_marker", "y_marker_color", "y_marker_size", "y_marker_interval",

                "expression", "expression_marker", "expression_marker_color", "expression_marker_size",
                "expression_marker_interval",

                "max_show", "max_show_marker", "max_show_label", "max_show_color", "max_show_size", "max_show_text",

                "min_show", "min_show_marker", "min_show_label", "min_show_color", "min_show_size", "min_show_text"
            ]

            # 检查是否所有的关键字参数均可用
            for kw in kwargs:
                if kw not in expected_kwargs:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(
                        f"\033[95mIn {method_name} of {class_name}\033[0m, "
                        f"Invalid keyword argument: '{kw}'. "
                        f"Allowed keyword arguments are: {', '.join(expected_kwargs)}")

            # 坐标轴部分 (6)  /* 默认参数 */
            custom_xticks = kwargs.get('custom_xticks', None)
            custom_yticks = kwargs.get('custom_yticks', None)
            x_rotation = kwargs.get('x_rotation', None)
            y_rotation = kwargs.get('y_rotation', None)
            invert_x = kwargs.get('invert_x', False)
            invert_y = kwargs.get('invert_y', False)

            # 标记注释 (2)
            target_index = kwargs.get('target_index', None)
            annotate_text = kwargs.get('annotate_text', None)

            # 填充部分 (3)
            fill_area = kwargs.get('fill_area', False)
            fill_color = kwargs.get('fill_color', line_color_single)
            fill_alpha = kwargs.get('fill_alpha', 0.4)

            # 根据 X 范围寻找 (5)
            x_marker_range = kwargs.get('x_marker_range', None)
            x_marker = kwargs.get('x_marker', '^')
            x_marker_color = kwargs.get('x_marker_color', 'red')
            x_marker_size = kwargs.get('x_marker_size', 20)
            x_marker_interval = kwargs.get('x_marker_interval', 1)

            # 根据 Y 范围寻找 (5)
            y_marker_range = kwargs.get('y_marker_range', None)
            y_marker = kwargs.get('y_marker', 'v')
            y_marker_color = kwargs.get('y_marker_color', 'blue')
            y_marker_size = kwargs.get('y_marker_size', 20)
            y_marker_interval = kwargs.get('y_marker_interval', 1)

            # 根据表达式寻找 (5)
            expression = kwargs.get('expression', None)
            expression_marker = kwargs.get('expression_marker', 'o')
            expression_marker_color = kwargs.get('expression_marker_color', 'green')
            expression_marker_size = kwargs.get('expression_marker_size', 20)
            expression_marker_interval = kwargs.get('expression_marker_interval', 1)

            # 寻找最大值 (6)
            max_show = kwargs.get('max_show', False)
            max_show_marker = kwargs.get('max_show_marker', '^')
            max_show_label = kwargs.get('max_show_label', None)
            max_show_color = kwargs.get('max_show_color', 'red')
            max_show_size = kwargs.get('max_show_size', 20)
            max_show_text = kwargs.get('max_show_text', False)

            # 寻找最小值 (6)
            min_show = kwargs.get('min_show', False)
            min_show_marker = kwargs.get('min_show_marker', 'v')
            min_show_label = kwargs.get('min_show_label', None)
            min_show_color = kwargs.get('min_show_color', 'green')
            min_show_size = kwargs.get('min_show_size', 20)
            min_show_text = kwargs.get('min_show_text', False)

        # 将所有图片展示在同一张图中
        if show_in_one:
            # 创建新的图形窗口
            plt.figure(figsize=width_height, dpi=200)

            # 展示在同一张图中时使用迭代器
            if line_color is not None:
                line_color_iteration = iter(line_color)
            else:
                line_color_iteration = iter(self.color_palette)

            # 初始化全局的最大和最小值
            x_min_global = float('inf')
            x_max_global = float('-inf')
            y_min_global = float('inf')
            y_max_global = float('-inf')

            # 遍历 dict ，更新最大和最小值
            for df in data_dic.values():
                x_min_global = min(x_min_global, df.iloc[:, 0].min())
                x_max_global = max(x_max_global, df.iloc[:, 0].max())
                y_min_global = min(y_min_global, df.iloc[:, 1].min())
                y_max_global = max(y_max_global, df.iloc[:, 1].max())

        else:
            # 不需要迭代器
            line_color_iteration = None
            x_min_global = None
            x_max_global = None
            y_min_global = None
            y_max_global = None

        # 遍历 dict 以获取数据
        for title, data_df in data_dic.items():

            # 不绘制在同一张图中
            if not show_in_one:
                # 创建新的图形窗口
                plt.figure(figsize=width_height, dpi=200)

                # 绘制线型统计图
                plt.plot(data_df.iloc[:, 0], data_df.iloc[:, 1],
                         color=line_color_single,
                         linestyle=line_style,
                         linewidth=line_width,
                         zorder=1,
                         alpha=alpha,
                         label=title)

                # 填充折线图下方的区域
                if fill_area:
                    plt.fill_between(x=data_df.iloc[:, 0],
                                     y1=data_df.iloc[:, 1],
                                     color=fill_color,
                                     alpha=fill_alpha)

                plt.autoscale()

                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))
                if custom_xticks is not None:
                    plt.xticks(custom_xticks)
                if custom_yticks is not None:
                    plt.yticks(custom_yticks)

                x_lower_limit = x_min
                x_upper_limit = x_max
                y_lower_limit = y_min
                y_upper_limit = y_max

            else:  # 绘制在同一张图中
                current_color = next(line_color_iteration)

                # 绘制线型统计图
                plt.plot(data_df.iloc[:, 0], data_df.iloc[:, 1],
                         color=current_color,
                         linestyle=line_style,
                         linewidth=line_width,
                         zorder=1,
                         alpha=alpha,
                         label=title)

                # 填充折线图下方的区域
                if fill_area:
                    plt.fill_between(x=data_df.iloc[:, 0],
                                     y1=data_df.iloc[:, 1],
                                     color=current_color,
                                     alpha=fill_alpha)

                plt.autoscale()

                # 使用条件表达式来设置边界
                x_lower_limit = x_min if (x_min is not None) else x_min_global
                x_upper_limit = x_max if (x_max is not None) else x_max_global
                y_lower_limit = y_min if (y_min is not None) else y_min_global
                y_upper_limit = y_max if (y_max is not None) else y_max_global

                plt.xlim((x_lower_limit, x_upper_limit))
                plt.ylim((y_lower_limit, y_upper_limit))

                if custom_xticks is not None:
                    plt.xticks(custom_xticks)
                if custom_yticks is not None:
                    plt.yticks(custom_yticks)

            # 获取 xlabel_original & ylabel_original
            xlabel_original, ylabel_original = data_df.columns[:2].tolist()

            # X 轴标题
            if x_label is not None:
                xlabel_plot = x_label
            else:
                xlabel_plot = xlabel_original

            # Y 轴标题
            if y_label is not None:
                ylabel_plot = y_label
            else:
                ylabel_plot = ylabel_original

            # 展示线条注解
            if show_label:
                plt.legend(prop=self.font_legend)

            # 网格
            plt.grid(show_grid)

            # 设置坐标轴字体
            if image_title != '':
                plt.title(image_title, fontdict=self.font_title)
            plt.xlabel(xlabel_plot, fontdict=self.font_title)
            plt.ylabel(ylabel_plot, fontdict=self.font_title)

            # 标注索引
            if target_index is not None:
                target_x = data_df.iloc[target_index, 0]
                target_y = data_df.iloc[target_index, 1]
                radius = 1
                circle = patches.Circle(xy=(target_x, target_y), radius=radius, color='red', fill=False)
                plt.gca().add_patch(circle)

                if annotate_text is not None:
                    plt.annotate(annotate_text, xy=(target_x, target_y),
                                 xytext=(target_x + 0.5, target_y + 1),
                                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontproperties=self.font_mark)

            # 背景
            if background_color is not None:

                # 调用函数加背景，防止刻度被锁住
                self.change_imshow(background_color=background_color,
                                   background_transparency=background_transparency,
                                   show_in_one=show_in_one,
                                   x_min=np.float64(x_lower_limit), x_max=np.float64(x_upper_limit),
                                   y_min=np.float64(y_lower_limit), y_max=np.float64(y_upper_limit))

                # 如果 show_in_one，将 background_color 设置为 None，防止多次加背景
                if show_in_one:
                    background_color = None

            plt.xticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'],
                       rotation=x_rotation)
            plt.yticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'],
                       rotation=y_rotation)

            # 显示标记
            if True:

                # 添加标记
                if x_marker_range is not None:
                    x_indices = []
                    if isinstance(x_marker_range, list):  # 闭区间
                        for i in range(len(data_df[x_label])):
                            if x_marker_range[0] <= data_df[x_label][i] <= x_marker_range[1] \
                                    and i % x_marker_interval == 0:
                                x_indices.append(i)
                    elif isinstance(x_marker_range, tuple):  # 开区间
                        x_indices = []
                        for i in range(len(data_df[x_label])):
                            if x_marker_range[0] < data_df[x_label][i] < x_marker_range[1] \
                                    and i % x_marker_interval == 0:
                                x_indices.append(i)

                    # 标记符合 X 范围的点
                    plt.scatter(np.array(data_df[x_label])[x_indices],
                                np.array(data_df[y_label])[x_indices],
                                marker=x_marker,  # 设置标记的符号
                                s=x_marker_size,  # 设置标记的大小
                                facecolor='none',  # 使标记内部为空心
                                edgecolors=x_marker_color,  # 设置标记的颜色
                                zorder=2)  # 设置权重

                if y_marker_range is not None:
                    y_indices = []
                    if isinstance(y_marker_range, list):  # 闭区间

                        for i in range(len(data_df[y_label])):
                            if y_marker_range[0] <= data_df[y_label][i] <= y_marker_range[1] \
                                    and i % y_marker_interval == 0:
                                y_indices.append(i)
                    elif isinstance(y_marker_range, tuple):  # 开区间
                        y_indices = []
                        for i in range(len(data_df[y_label])):
                            if y_marker_range[0] < data_df[y_label][i] < y_marker_range[1] \
                                    and i % y_marker_interval == 0:
                                y_indices.append(i)

                    # 标记符合 Y 范围的点
                    plt.scatter(np.array(data_df[x_label])[y_indices],
                                np.array(data_df[y_label])[y_indices],
                                marker=y_marker,  # 设置标记的符号
                                s=y_marker_size,  # 设置标记的大小
                                facecolor='none',  # 使标记内部为空心
                                edgecolors=y_marker_color,  # 设置标记的颜色
                                zorder=2)  # 设置权重

                if expression is not None:
                    expr = sympify(expression)
                    x_sym = symbols('x')
                    y_sym = symbols('y')
                    expression_indices = []

                    for i in range(len(data_df[x_label])):
                        if expr.subs([(x_sym, data_df[x_label][i]),
                                      (y_sym, data_df[y_label][i])]) and i % expression_marker_interval == 0:
                            expression_indices.append(i)

                    # 标记符合表达式的点
                    plt.scatter(np.array(data_df[x_label])[expression_indices],
                                np.array(data_df[y_label])[expression_indices],
                                marker=expression_marker,  # 设置标记的符号
                                s=expression_marker_size,  # 设置标记的大小
                                facecolor='none',  # 使标记内部为空心
                                edgecolors=expression_marker_color,  # 设置标记的颜色
                                zorder=4)  # 设置权重

                # 找到最大值和最小值的索引
                if max_show:
                    max_index = np.argmax(data_df[y_label])
                    plt.scatter(data_df[x_label][max_index], data_df[y_label][max_index],
                                marker=max_show_marker,  # 设置标记的符号
                                color=max_show_color,  # 设置标记的大小
                                s=max_show_size,  # 使标记内部为空心
                                label=max_show_label,  # 设置标记的颜色
                                zorder=3)  # 设置权重
                    if max_show_text:
                        # 在最大值附近添加最大值的大小文本标签
                        plt.text(data_df[x_label][max_index], data_df[y_label][max_index],
                                 s=f'Max Value: {data_df[y_label][max_index]:.2f}',
                                 ha='center', va='bottom', color='red',
                                 fontname=self.font_legend['family'],
                                 weight=self.font_legend['weight'],
                                 fontsize=self.font_legend['size'])

                if min_show:
                    min_index = np.argmin(data_df[y_label])
                    plt.scatter(data_df[x_label][min_index], data_df[y_label][min_index],
                                marker=min_show_marker,  # 设置标记的符号
                                color=min_show_color,  # 设置标记的大小
                                s=min_show_size,  # 使标记内部为空心
                                label=min_show_label,  # 设置标记的颜色
                                zorder=3)  # 设置权重
                    if min_show_text:
                        # 在最小值附近添加最大值的大小文本标签
                        plt.text(data_df[x_label][min_index], data_df[y_label][min_index],
                                 s=f'Min Value: {data_df[y_label][min_index]:.2f}',
                                 ha='left', va='bottom', color='red',
                                 fontname=self.font_legend['family'],
                                 weight=self.font_legend['weight'],
                                 fontsize=self.font_legend['size'])

                if (max_show is True and max_show_label is not None) or \
                        (min_show is True and min_show_label is not None):

                    # 显示图例
                    plt.legend(prop=self.font_legend)

            # 反转 X 轴或 Y 轴
            if invert_x:
                plt.gca().invert_xaxis()
            if invert_y:
                plt.gca().invert_yaxis()

            plt.tight_layout()  # 调整

            if not show_in_one:

                if line_save_path is not None:  # 如果 line_save_path 的值不为 None，则保存
                    file_name = title + ".png"  # 初始文件名为 "title.png"
                    full_file_path = os.path.join(line_save_path, file_name)  # 创建完整的文件路径

                    if os.path.exists(full_file_path):  # 查看该文件名是否存在
                        count = 1
                        file_name = title + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                        full_file_path = os.path.join(line_save_path, file_name)  # 更新完整的文件路径

                        while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                            count += 1
                            file_name = title + f"_{count}.png"
                            full_file_path = os.path.join(line_save_path, file_name)  # 更新完整的文件路径

                    plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

                # 判断是否显示图像
                if show:
                    plt.show()  # 显示图像
                    time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃
                else:
                    plt.close()  # 清楚图像，以免对之后的作图进行干扰

        if show_in_one:

            if line_save_path is not None:  # 如果 line_save_path 的值不为 None，则保存
                file_name = "Image.png"  # 初始文件名为 "Image.png"
                full_file_path = os.path.join(line_save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = "Image" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(line_save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = "Image" + f"_{count}.png"
                        full_file_path = os.path.join(line_save_path, file_name)  # 更新完整的文件路径

                plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将图表保存到指定的路径

            # 判断是否显示图像
            if show:
                plt.show()  # 显示图像
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃
            else:
                plt.close()  # 清楚图像，以免对之后的作图进行干扰

        return None

    # 10 绘制散点图统计图
    def plot_scatter(self,

                     data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                     save_path: Union[bool, str] = True, dpi: int = 600, image_title: str = '', show: bool = True,

                     show_in_one: Optional[bool] = None, width_height: Optional[tuple] = None,
                     show_grid: Optional[bool] = None, alpha: Optional[float] = None, show_label: Optional[bool] = None,
                     x_label: Optional[str] = None, y_label: Optional[str] = None,
                     x_min: Union[int, float, None] = None, x_max: Union[int, float, None] = None,
                     y_min: Union[int, float, None] = None, y_max: Union[int, float, None] = None,
                     background_color: Optional[str] = None, background_transparency: Optional[float] = None,

                     point_color: Union[list, tuple, None] = None, point_style: Optional[str] = None,
                     point_size: Optional[float] = None,

                     **kwargs) -> None:
        """
        绘制散点统计图
        Draw a scatter chart.

        # 关键参数 (4)
        :param data_dic: (dict) 若被赋值，则会对该 dic 进行绘图
        :param data_df: (DataFrame) 若被赋值，则会对该 Dataframe 进行绘图
        :param save_path: (str) 保存目录，若无赋值则用初始化中的 self.save_path，若为 False 或 None 则为不保存
                          注意：应为保存的目录而非具体路径
        :param dpi: (int) 图片保存的精度，只有在需要保存时才有意义，默认为 dpi=600，无赋值侧沿用初始化中的 dpi
        :param image_title: (str) 图的标题，默认为无标题
        :param show: (bool) 是否显示图片，默认为 True

        # 绘制参数 (13)
        :param show_in_one: (bool) 是否将所有数据展示在一张图片里，默认为 False
        :param width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)
        :param show_grid: (bool) 是否显示网格，默认不显示
        :param alpha: (float) 线条的透明度，默认为 1，表示不透明
        :param show_label: (bool) 是否展示线条注解，默认为否
        :param x_label: (str) X 轴的标题，默认为原列中的标题，若有则为该标题
        :param y_label: (str) Y 轴的标题，默认为原列中的标题，若有则为该标题
        :param x_min: (int / float) 横坐标的最小值
        :param x_max: (int / float) 横坐标的最大值
        :param y_min: (int / float) 纵坐标的最小值
        :param y_max: (int / float) 纵坐标的最大值
        :param background_color: (str / tuple) 设置图片的背景颜色，默认为无
        :param background_transparency: (float) 背景色的透明程度，最小越透明，只有存在背景色时才有意义，默认为 0.15

        # 散点参数 (3)
        :param point_color: (list / tuple) 散点的颜色，即 scatter 图像散点的颜色，透明色为 'none'，
                           point_color 为长度大于 1 的 list / tuple 时，只有 show_in_one == True 时才有意义
        :param point_style: (str) 最上方散点的风格，即 scatter 图像散点的风格，默认为点状
        :param point_size: (float) 散点的大小，默认为 3

        :return: None

        --- **kwargs ---

        # 坐标轴部分 (6)
        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度旋转的角度
        - y_rotation: (float) Y 轴刻度旋转的角度
        - invert_x: (bool) 反转 X 轴，默认为 False
        - invert_y: (bool) 反转 Y 轴，默认为 False

        # 标记注释 (2)
        - target_index: (int / list) 标注点 list，或 int (为索引)
        - annotate_text: (str / list) 标注文字 list，或 str，只有在 target_index 存在的情况下才有意义，
                             且与target_index一一对应

        # 绘图部分 (2)
        - face_colors: (tuple / list) 散点的外圈颜色，透明色为 'none'
        - edge_colors: (tuple / list) 散点的内部颜色，透明色为 'none'

        # 标记部分 (27)
        - x_marker_range: (tuple) 以 X 轴区域标记，可接受 list 和 tuple，当为 list 时表示闭区间，当为 tuple 时表示开区间
        - x_marker: (str) 以 X 轴区域标记时的标记符号，只有当 x_marker_range 被赋值时才有意义
        - x_marker_color: (str / tuple) 以 X 轴区域标记时的颜色，默认为红色，只有当 x_marker_range 被赋值时才有意义
        - x_marker_size: (int) 以 X 轴区域标记时的标记大小，只有当 x_marker_range 被赋值时才有意义
        - x_marker_interval: (int) 对于以 X 轴区域标记，每几个点标记一个，默认为 1 表示全标记，
                                 只有当 x_marker_range 被赋值时才有意义
        - y_marker_range: (tuple) Y 轴区域标记，可接受 list 和 tuple，当为 list 时表示闭区间，当为 tuple 时表示开区间
        - y_marker: (str) 以y轴区域标记时的标记符号，只有当 y_marker_range 被赋值时才有意义
        - y_marker_color: (str / tuple) 以 Y 轴区域标记时的颜色，默认为蓝色，只有当 y_marker_range 被赋值时才有意义
        - y_marker_size: (int) 以 Y 轴区域标记时的标记大小，只有当 y_marker_range 被赋值时才有意义
        - y_marker_interval: (int) 对于以 Y 轴区域标记，每几个点标记一个，默认为 1 表示全标记，
                                 只有当 y_marker_range 被赋值时才有意义
        - expression: (str) 表达式，用以表达式标记
        - expression_marker: (str) 表达式标记时的符号，只有当 expression 被赋值时才有意义
        - expression_marker_color: (str / tuple) 以表达式记时的颜色，默认为绿色，只有当 expression 被赋值时才有意义
        - expression_marker_size: (int) 以表达式标记时的标记大小，只有当 expression 被赋值时才有意义
        - expression_marker_interval: (int) 对于以表达式标记，每几个点标记一个，默认为 1 表示全标记，
                                          只有当 expression 被赋值时才有意义
        - max_show:  (bool) 是否显示最大值，默认为 False
        - max_show_marker: (str) 最大值的标记符号，只有 max_show 为 True 时才有意义
        - max_show_label: (str) 最大值的标签名，默认不显示，只有 max_show 为 True 时才有意义
        - max_show_color: (str / tuple) 最大值符号的颜色，默认为红色，只有 max_show 为 True 时才有意义
        - max_show_size:  (int) 最大值符号的大小，默认为 20，只有 max_show 为 True 时才有意义
        - max_show_text: (bool) 最大值标出，默认不显示，只有 max_show 为 True 时才有意义

        - min_show:  (bool) 是否显示最小值，默认为 False
        - min_show_marker: (str) 最小值的标记符号，只有 min_show 为 True 时才有意义
        - min_show_label: (str) 最小值的标签名，默认不显示，只有 min_show 为 True 时才有意义
        - min_show_color: (str / tuple) 最小值符号的颜色，默认为绿色，只有 min_show 为 True 时才有意义
        - min_show_size: (int) 最小值符号的大小，默认为 20，只有 min_show为True时才有意义
        - min_show_text: (bool) 最小值标出，默认不显示，只有 max_show 为 True 时才有意义
        """

        # 检查赋值 (2 + 16，算上 label)
        if True:

            # data_dic 与 data_df 被赋值的情况
            if data_dic is not None and data_df is not None:  # 两个均被赋值时
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"data_dic and data_dic should not be assigned at the same time.")
            elif data_dic is not None:  # 只有 data_dic 被赋值时
                data_dic = copy.deepcopy(data_dic)
            elif data_df is not None:  # 只有 data_df 被赋值时
                data_dic = {'Untitled': data_df}
            else:
                # 使用 getattr 来动态获取属性
                data_dic = copy.deepcopy(getattr(self, self.current_dic))

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                scatter_save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                scatter_save_path = None
            # 当有指定的 save_path 时，scatter_save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                scatter_save_path = save_path

            if show_in_one is not None:
                show_in_one = show_in_one
            elif self.show_in_one is not None:
                show_in_one = self.show_in_one
            else:
                show_in_one = False

            if width_height is not None:
                width_height = width_height
            elif self.width_height is not None:
                width_height = self.width_height
            else:
                width_height = (6, 4.5)

            if show_grid is not None:
                show_grid = show_grid
            elif self.grid is not None:
                show_grid = self.grid
            else:
                show_grid = False

            if alpha is not None:
                alpha = alpha
            elif self.alpha is not None:
                alpha = self.alpha
            else:
                alpha = 1

            if show_label is not None:
                show_label = show_label
            elif self.show_label is not None:
                show_label = self.show_label
            else:
                show_label = False

            if x_label is not None:
                x_label = x_label
            elif self.x_label is not None:
                x_label = self.x_label
            else:
                x_label = 'X'

            if y_label is not None:
                y_label = y_label
            elif self.y_label is not None:
                y_label = self.y_label
            else:
                y_label = 'Y'

            if x_min is not None:
                x_min = x_min
            elif self.x_min is not None:
                x_min = self.x_min
            else:
                x_min = None

            if x_max is not None:
                x_max = x_max
            elif self.x_max is not None:
                x_max = self.x_max
            else:
                x_max = None

            if y_min is not None:
                y_min = y_min
            elif self.y_min is not None:
                y_min = self.y_min
            else:
                y_min = None

            if y_max is not None:
                y_max = y_max
            elif self.y_max is not None:
                y_max = self.y_max
            else:
                y_max = None

            if background_color is not None:
                background_color = background_color
            elif self.background_color is not None:
                background_color = self.background_color
            else:
                background_color = None

            if background_transparency is not None:
                background_transparency = background_transparency
            elif self.background_transparency is not None:
                background_transparency = self.background_transparency
            else:
                background_transparency = 0.15

            if point_color is not None:
                point_color_single = point_color[0]
            elif self.point_color is not None:
                point_color_single = self.point_color[0]
            else:
                point_color_single = self.color_palette[0]

            if point_style is not None:
                point_style = point_style
            elif self.point_style is not None:
                point_style = self.point_style
            else:
                point_style = 'o'

            if point_size is not None:
                point_size = point_size
            elif self.point_size is not None:
                point_size = self.point_size
            else:
                point_size = 3

        # 检查关键字参数及其初始化
        if True:

            # 检查关键字参数的 list
            expected_kwargs = [

                # 坐标轴部分 (6)
                "custom_xticks", "custom_yticks", "x_rotation", "y_rotation", "invert_x", "invert_y",

                # 标记注释 (2)
                "target_index", "annotate_text",

                # 绘图部分 (2)
                "face_colors", "edge_colors",

                # 标记部分 (27)
                "x_marker_range", "x_marker", "x_marker_color", "x_marker_size", "x_marker_interval",

                "y_marker_range", "y_marker", "y_marker_color", "y_marker_size", "y_marker_interval",

                "expression", "expression_marker", "expression_marker_color", "expression_marker_size",
                "expression_marker_interval",

                "max_show", "max_show_marker", "max_show_label", "max_show_color", "max_show_size", "max_show_text",

                "min_show", "min_show_marker", "min_show_label", "min_show_color", "min_show_size", "min_show_text"
            ]

            # 检查是否所有的关键字参数均可用
            for kw in kwargs:
                if kw not in expected_kwargs:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(
                        f"\033[95mIn {method_name} of {class_name}\033[0m, "
                        f"Invalid keyword argument: '{kw}'. "
                        f"Allowed keyword arguments are: {', '.join(expected_kwargs)}")

            # 坐标轴部分 (6)  /* 默认参数 */
            custom_xticks = kwargs.get('custom_xticks', None)
            custom_yticks = kwargs.get('custom_yticks', None)
            x_rotation = kwargs.get('x_rotation', None)
            y_rotation = kwargs.get('y_rotation', None)
            invert_x = kwargs.get('invert_x', False)
            invert_y = kwargs.get('invert_y', False)

            # 标记注释 (2)
            target_index = kwargs.get('target_index', None)
            annotate_text = kwargs.get('annotate_text', None)

            # 绘图部分 (2)
            face_colors = kwargs.get('face_colors', None)
            edge_colors = kwargs.get('edge_colors', None)

            # 根据 X 范围寻找 (5)
            x_marker_range = kwargs.get('x_marker_range', None)
            x_marker = kwargs.get('x_marker', '^')
            x_marker_color = kwargs.get('x_marker_color', 'red')
            x_marker_size = kwargs.get('x_marker_size', 20)
            x_marker_interval = kwargs.get('x_marker_interval', 1)

            # 根据 Y 范围寻找 (5)
            y_marker_range = kwargs.get('y_marker_range', None)
            y_marker = kwargs.get('y_marker', 'v')
            y_marker_color = kwargs.get('y_marker_color', 'blue')
            y_marker_size = kwargs.get('y_marker_size', 20)
            y_marker_interval = kwargs.get('y_marker_interval', 1)

            # 根据表达式寻找 (5)
            expression = kwargs.get('expression', None)
            expression_marker = kwargs.get('expression_marker', 'o')
            expression_marker_color = kwargs.get('expression_marker_color', 'green')
            expression_marker_size = kwargs.get('expression_marker_size', 20)
            expression_marker_interval = kwargs.get('expression_marker_interval', 1)

            # 寻找最大值 (6)
            max_show = kwargs.get('max_show', False)
            max_show_marker = kwargs.get('max_show_marker', '^')
            max_show_label = kwargs.get('max_show_label', None)
            max_show_color = kwargs.get('max_show_color', 'red')
            max_show_size = kwargs.get('max_show_size', 20)
            max_show_text = kwargs.get('max_show_text', False)

            # 寻找最小值 (6)
            min_show = kwargs.get('min_show', False)
            min_show_marker = kwargs.get('min_show_marker', 'v')
            min_show_label = kwargs.get('min_show_label', None)
            min_show_color = kwargs.get('min_show_color', 'green')
            min_show_size = kwargs.get('min_show_size', 20)
            min_show_text = kwargs.get('min_show_text', False)

        # 将所有图片展示在同一张图中
        if show_in_one:
            # 创建新的图形窗口
            plt.figure(figsize=width_height, dpi=200)

            # 展示在同一张图中时使用迭代器
            if point_color is not None:
                point_color_iteration = iter(point_color)
            else:
                point_color_iteration = iter(self.color_palette)

            # 初始化全局的最大和最小值
            x_min_global = float('inf')
            x_max_global = float('-inf')
            y_min_global = float('inf')
            y_max_global = float('-inf')

            # 遍历 dict ，更新最大和最小值
            for df in data_dic.values():
                x_min_global = min(x_min_global, df.iloc[:, 0].min())
                x_max_global = max(x_max_global, df.iloc[:, 0].max())
                y_min_global = min(y_min_global, df.iloc[:, 1].min())
                y_max_global = max(y_max_global, df.iloc[:, 1].max())

        else:
            # 不需要迭代器
            point_color_iteration = None
            x_min_global = None
            x_max_global = None
            y_min_global = None
            y_max_global = None

        # 遍历 dict 以获取数据
        for title, data_df in data_dic.items():

            # 不绘制在同一张图中
            if not show_in_one:

                # 创建新的图形窗口
                plt.figure(figsize=width_height, dpi=200)

                # 绘制散点统计图
                plt.scatter(data_df.iloc[:, 0], data_df.iloc[:, 1],
                            c=point_color_single,
                            facecolors=face_colors,  # 内圈颜色
                            edgecolors=edge_colors,  # 外圈颜色
                            marker=point_style,
                            s=point_size,
                            zorder=1,
                            alpha=alpha,
                            label=title)

                plt.autoscale()

                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))
                if custom_xticks is not None:
                    plt.xticks(custom_xticks)
                if custom_yticks is not None:
                    plt.yticks(custom_yticks)

                x_lower_limit = x_min
                x_upper_limit = x_max
                y_lower_limit = y_min
                y_upper_limit = y_max

            else:  # 绘制在同一张图中
                current_color = next(point_color_iteration)

                # 绘制散点统计图
                plt.scatter(data_df.iloc[:, 0], data_df.iloc[:, 1],
                            c=current_color,
                            facecolors=face_colors,  # 内圈颜色
                            edgecolors=edge_colors,  # 外圈颜色
                            marker=point_style,
                            s=point_size,
                            zorder=1,
                            alpha=alpha,
                            label=title)

                plt.autoscale()

                # 使用条件表达式来设置边界
                x_lower_limit = x_min if (x_min is not None) else x_min_global
                x_upper_limit = x_max if (x_max is not None) else x_max_global
                y_lower_limit = y_min if (y_min is not None) else y_min_global
                y_upper_limit = y_max if (y_max is not None) else y_max_global

                plt.xlim((x_lower_limit, x_upper_limit))
                plt.ylim((y_lower_limit, y_upper_limit))

                if custom_xticks is not None:
                    plt.xticks(custom_xticks)
                if custom_yticks is not None:
                    plt.yticks(custom_yticks)

            # 获取 xlabel_original & ylabel_original
            xlabel_original, ylabel_original = data_df.columns[:2].tolist()

            # X 轴标题
            if x_label is not None:
                xlabel_plot = x_label
            else:
                xlabel_plot = xlabel_original

            # Y 轴标题
            if y_label is not None:
                ylabel_plot = y_label
            else:
                ylabel_plot = ylabel_original

            # 展示线条注解
            if show_label:
                plt.legend(prop=self.font_legend)

            # 网格
            plt.grid(show_grid)

            # 设置坐标轴字体
            if image_title != '':
                plt.title(image_title, fontdict=self.font_title)
            plt.xlabel(xlabel_plot, fontdict=self.font_title)
            plt.ylabel(ylabel_plot, fontdict=self.font_title)

            # 标注索引
            if target_index is not None:
                target_x = data_df.iloc[target_index, 0]
                target_y = data_df.iloc[target_index, 1]
                radius = 1
                circle = patches.Circle(xy=(target_x, target_y), radius=radius, color='red', fill=False)
                plt.gca().add_patch(circle)

                if annotate_text is not None:
                    plt.annotate(annotate_text,
                                 xy=(target_x, target_y),
                                 xytext=(target_x + 0.5, target_y + 1),
                                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontproperties=self.font_title)

            # 设置刻度
            plt.autoscale()
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            if custom_xticks is not None:
                plt.xticks(custom_xticks)
            if custom_yticks is not None:
                plt.yticks(custom_yticks)

            # 背景
            if background_color is not None:

                # 调用函数加背景，防止刻度被锁住
                self.change_imshow(background_color=background_color,
                                   background_transparency=background_transparency,
                                   show_in_one=show_in_one,
                                   x_min=np.float64(x_lower_limit), x_max=np.float64(x_upper_limit),
                                   y_min=np.float64(y_lower_limit), y_max=np.float64(y_upper_limit))

                # 如果 show_in_one，将 background_color 设置为 None，防止多次加背景
                if show_in_one:
                    background_color = None

            plt.xticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'],
                       rotation=x_rotation)
            plt.yticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'],
                       rotation=y_rotation)

            # 检查标记
            if True:

                # 添加标记
                if x_marker_range is not None:
                    x_indices = []
                    if isinstance(x_marker_range, list):  # 闭区间
                        for i in range(len(data_df[x_label])):
                            if x_marker_range[0] <= data_df[x_label][i] <= x_marker_range[1] \
                                    and i % x_marker_interval == 0:
                                x_indices.append(i)
                    elif isinstance(x_marker_range, tuple):  # 开区间
                        x_indices = []
                        for i in range(len(data_df[x_label])):
                            if x_marker_range[0] < data_df[x_label][i] < x_marker_range[1] \
                                    and i % x_marker_interval == 0:
                                x_indices.append(i)

                    # 标记符合 X 范围的点
                    plt.scatter(np.array(data_df[x_label])[x_indices],
                                np.array(data_df[y_label])[x_indices],
                                marker=x_marker,  # 设置标记的符号
                                s=x_marker_size,  # 设置标记的大小
                                facecolor='none',  # 使标记内部为空心
                                edgecolors=x_marker_color,  # 设置标记的颜色
                                zorder=2)  # 设置权重

                if y_marker_range is not None:
                    y_indices = []
                    if isinstance(y_marker_range, list):  # 闭区间

                        for i in range(len(data_df[y_label])):
                            if y_marker_range[0] <= data_df[y_label][i] <= y_marker_range[1] \
                                    and i % y_marker_interval == 0:
                                y_indices.append(i)
                    elif isinstance(y_marker_range, tuple):  # 开区间
                        y_indices = []
                        for i in range(len(data_df[y_label])):
                            if y_marker_range[0] < data_df[y_label][i] < y_marker_range[1] \
                                    and i % y_marker_interval == 0:
                                y_indices.append(i)

                    # 标记符合 Y 范围的点
                    plt.scatter(np.array(data_df[x_label])[y_indices],
                                np.array(data_df[y_label])[y_indices],
                                marker=y_marker,  # 设置标记的符号
                                s=y_marker_size,  # 设置标记的大小
                                facecolor='none',  # 使标记内部为空心
                                edgecolors=y_marker_color,  # 设置标记的颜色
                                zorder=2)  # 设置权重

                if expression is not None:
                    expr = sympify(expression)
                    x_sym = symbols('x')
                    y_sym = symbols('y')
                    expression_indices = []

                    for i in range(len(data_df[x_label])):
                        if expr.subs([(x_sym, data_df[x_label][i]),
                                      (y_sym, data_df[y_label][i])]) and i % expression_marker_interval == 0:
                            expression_indices.append(i)

                    # 标记符合表达式的点
                    plt.scatter(np.array(data_df[x_label])[expression_indices],
                                np.array(data_df[y_label])[expression_indices],
                                marker=expression_marker,  # 设置标记的符号
                                s=expression_marker_size,  # 设置标记的大小
                                facecolor='none',  # 使标记内部为空心
                                edgecolors=expression_marker_color,  # 设置标记的颜色
                                zorder=4)  # 设置权重

                # 找到最大值和最小值的索引
                if max_show:
                    max_index = np.argmax(data_df[y_label])
                    plt.scatter(data_df[x_label][max_index], data_df[y_label][max_index],
                                marker=max_show_marker,  # 设置标记的符号
                                color=max_show_color,  # 设置标记的大小
                                s=max_show_size,  # 使标记内部为空心
                                label=max_show_label,  # 设置标记的颜色
                                zorder=3)  # 设置权重
                    if max_show_text:
                        # 在最大值附近添加最大值的大小文本标签
                        plt.text(data_df[x_label][max_index], data_df[y_label][max_index],
                                 f'Max Value: {data_df[y_label][max_index]:.2f}',
                                 ha='center', va='bottom', color='red',
                                 fontname=self.font_legend['family'],
                                 weight=self.font_legend['weight'],
                                 fontsize=self.font_legend['size'])

                if min_show:
                    min_index = np.argmin(data_df[y_label])
                    plt.scatter(data_df[x_label][min_index], data_df[y_label][min_index],
                                marker=min_show_marker,  # 设置标记的符号
                                color=min_show_color,  # 设置标记的大小
                                s=min_show_size,  # 使标记内部为空心
                                label=min_show_label,  # 设置标记的颜色
                                zorder=3)  # 设置权重
                    if min_show_text:
                        # 在最小值附近添加最大值的大小文本标签
                        plt.text(data_df[x_label][min_index], data_df[y_label][min_index],
                                 f'Min Value: {data_df[y_label][min_index]:.2f}',
                                 ha='left', va='bottom', color='red',
                                 fontname=self.font_legend['family'],
                                 weight=self.font_legend['weight'],
                                 fontsize=self.font_legend['size'])

                if (max_show is True and max_show_label is not None) or \
                        (min_show is True and min_show_label is not None):

                    # 显示图例
                    plt.legend(prop=self.font_legend)

            # 反转 X 轴或 Y 轴
            if invert_x:
                plt.gca().invert_xaxis()
            if invert_y:
                plt.gca().invert_yaxis()

            plt.tight_layout()  # 调整

            if not show_in_one:

                if scatter_save_path is not None:  # 如果 scatter_save_path 的值不为 None，则保存
                    file_name = title + ".png"  # 初始文件名为 "title.png"
                    full_file_path = os.path.join(scatter_save_path, file_name)  # 创建完整的文件路径

                    if os.path.exists(full_file_path):  # 查看该文件名是否存在
                        count = 1
                        file_name = title + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                        full_file_path = os.path.join(scatter_save_path, file_name)  # 更新完整的文件路径

                        while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                            count += 1
                            file_name = title + f"_{count}.png"
                            full_file_path = os.path.join(scatter_save_path, file_name)  # 更新完整的文件路径

                    plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

                # 判断是否显示图像
                if show:
                    plt.show()  # 显示图像
                    time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃
                else:
                    plt.close()  # 清楚图像，以免对之后的作图进行干扰

        if show_in_one:

            if scatter_save_path is not None:  # 如果 scatter_save_path 的值不为 None，则保存
                file_name = "Image.png"  # 初始文件名为 "Image.png"
                full_file_path = os.path.join(scatter_save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = "Image" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(scatter_save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = "Image" + f"_{count}.png"
                        full_file_path = os.path.join(scatter_save_path, file_name)  # 更新完整的文件路径

                plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将图表保存到指定的路径

            # 判断是否显示图像
            if show:
                plt.show()  # 显示图像
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃
            else:
                plt.close()  # 清楚图像，以免对之后的作图进行干扰

        return None


""" 模组系统 """
class Module(Optimizer):
    """
    模组系统

    Modules can add additional functionality to the main function
    and other functions, some of which are necessary and some of
    which are optional. The addition of modules can make the
    functionality of functions more powerful and detailed.And the
    modules performed on the DataFrame include adding coordinates,
    removing coordinates, automatically removing points based on
    probability, and adding coordinate points based on a normal
    distribution. Can be used to search for special points or
    intervals within the DataFrame, such as locating peaks,
    locating plateaus, and so on.

    注意：
    1.  删除点时，返回删除后的表格(重要)，记录删除的点
        添加点时，返回添加点的表格，记录添加的点(重要，与原表格)
    2.  分开添加的点是为了加权
    3.  顺序：手动 -> 自动 -> 寻找
    4.  用 data_dic 代表 DataFrame 表格，只在最后转换为需要的变量名
    5.  输出前均需要进行排序和更新行索引
    6.  如果坐标输出多项，则用 tuple 输出
    7.  手动点用 data_dic，自动随机用 data_dic
    8.  数据的检索顺序与 self.current_dic 相关，且不会改变 self.data_dic

    Note:
    1.  When deleting a point, return to the deleted table (important) and record the deleted point
        When adding a point, return the table where the point was added, recording the added point
        (important, with the original table)
    2.  The points added separately are for weighting
    3.  Sequence: Manual -&gt; Automatic -&gt; Look for
    4.  Use data_dic to represent the DataFrame table and only convert to the desired variable name at the end
    5.  Sort and update the row index before output
    6.  If the coordinates output multiple items, use the tuple output
    7.  data_dic is selected manually and data_dic is selected automatically
    8.  The retrieval order of the data is related to self.current_dic and does not change self.data_dic
    """

    # 初始化
    def __init__(self,

                 # 接收的参数
                 data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None,
                 title: Optional[str] = None, x_list: Optional[list] = None, y_list: Optional[list] = None,
                 x_label: str = None, y_label: str = None):

        # 超类初始化
        super().__init__(data_dic=data_dic, data_df=data_df, title=title, x_list=x_list, y_list=y_list,
                         x_label=x_label, y_label=y_label)  # 数据初始化时自动完成数据分配

        # 数据初始化分配
        if type(self) == Module:  # 当 self 为 Module 的直接实例时为真
            self.data_init()

        # custom_point()
        self.custom_dic = None  # (dict) key 为所用数据的 title，value 为标记数据的 DataFrame
        # remove_point()
        self.reserved_dic = None  # (dict) key 为所用数据的 title，value 为移除坐标后的 DataFrame
        self.removed_certain_dic = None  # (dict) key 为所用数据的 title，value 为手动选定移除坐标的 DataFrame
        self.removed_percent_dic = None  # (dict) key 为所用数据的 title，value 为随机移坐标的 DataFrame
        # append_points()
        self.appended_dic = None  # (dict) key 为所用数据的 title，value 为添加坐标后的 DataFrame
        self.append_certain_dic = None  # (dict) key 为所用数据的 title，value 为手动选定添加坐标的 DataFrame
        self.append_scope_dic = None  # (dict) key 为所用数据的 title，value 为范围内随机添加坐标的 DataFrame
        # get_peak() & find_data_peaks()
        self.peak_dic = None  # (dict) key 为所用数据的 title，value 为每个数据的峰值点
        # get_plateau()
        self.fragment_dic = None  # (dict) key 为所用数据的 title，value 为每个数据的 list 形式的平台区间
        # SF_remove()
        self.sf_removed_dic = None  # (dict) key 为所用数据的 title，value 为删除随机区间坐标后的 DataFrame
        self.sf_remove_section_dic = None  # dict  key 为所用数据的 title，value 为被移除区间坐标的 DataFrame 组成的 list
        # SF_append()
        self.sf_appended_dic = None  # (dict) key 为所用数据的 title，value 为根据正态分布添加随机坐标后的 DataFrame
        self.sf_append_point_dic = None  # (dict) key 为所用数据的 title，value 为添加的坐标的 DataFrame 组成的 list
        # random_dict()
        self.random_dic = None  # (dict) key 和 value 与 data_dic 一致，长度与 num_pairs 参数有关
        # move_points()
        self.moved_dic = None  # (dict) key 为所用数据的 title，value 移动后数据的 DataFrame
        # handle_duplicate_x()
        self.handled_dic = None  # (dict)  key 为所用数据的 title，value 处理相同 X 数据后的 DataFrame

    # 自定义选点
    def custom_point(self, data_dic: Optional[dict] = None, **kwargs: Union[Tuple[float, float]]) \
            -> Dict[str, Optional[DataFrame]]:
        r"""
        找出与自定义坐标最近的坐标点，并对该点进行标记
        Find the coordinate point closest to the custom coordinate and mark that point.

        Recommended parameters and usage:
            After function improve_precision.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param kwargs: -1- (Tuple) pattern = r'^custom_point(_\d+)?$'，为变量名，
                      值为 Tuple，代表需要标记的坐标。例如：-1- custom_point = (1, 2)

        :return custom_dic： (dict) key 为所用数据的 title，value 为标记数据的 DataFrame
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 检查关键字，并提取坐标
        pattern = r'^custom_point(_\d+)?$'  # 正则表达式
        custom_point_list = []  # 初始化列表，用来存储点
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
        for key, value in kwargs.items():
            if re.search(pattern, key):  # 判断是否满足 pattern
                if isinstance(value, tuple) and len(value) == 2:  # 判断是否为长度为 2 的元组
                    for item in value:
                        # 判断内部元素是否为 float or int，在 Python 中 True 的类型是 bool，它同时也是 int 的一个子类
                        if not isinstance(item, (float, int)) or isinstance(item, bool):
                            raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                            f" {value} expects to receive two float or int values.")
                    custom_point_list.append(value)
                else:
                    raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                    f" expected variable {key} to be a tuple of length 2,"
                                    f" but got a {type(value)} of length {len(key)} instead.")
            else:
                raise ValueError(
                    f"\033[95mIn {method_name} of {class_name}\033[0m, "
                    f"Expected one matching keyword argument for pattern"
                    f" '{pattern}', but found {key}")

        custom_dic = {}
        for title, data_df in data_dic.items():
            # 寻找距离最近的点
            custom_point_result = []
            for custom_point in custom_point_list:
                min_dist = np.inf
                closest_point = None
                for _, row in data_df.iterrows():
                    # 计算当前数据点与自定义坐标的距离
                    dist = np.sqrt((row[0] - custom_point[0]) ** 2 + (row[1] - custom_point[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = tuple(row)
                custom_point_result.append(closest_point)

            # 整合成 DataFrame的格式
            custom_df = pd.DataFrame(custom_point_result, columns=[self.x_label, self.y_label])
            # 根据横坐标重新排序并重新设置行索引
            custom_df = custom_df.sort_values(by=data_df.columns[0])
            custom_df = custom_df.reset_index(drop=True)
            custom_dic[title] = custom_df

        self.custom_dic = custom_dic
        self.current_dic = 'custom_dic'

        return custom_dic

    # 移除点
    def remove_point(self, data_dic: Optional[dict] = None, remove_percent: Optional[float] = None,
                     even: bool = False, **kwargs: [Tuple[float, float]]) \
            -> Tuple[Dict[str, DataFrame], Dict[str, Optional[DataFrame]], Dict[str, Optional[DataFrame]]]:
        r"""
        从 DataFrame 中根据不同的条件移除坐标点
        Removes coordinate points from the DataFrame based on different conditions.

        Recommended parameters and usage:
            After function reduce_precision.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param remove_percent: -2- (float) 随机删除的数据点百分比，该项落后于手动执行
        :param even: (bool) 是否均匀删除数据点的标志，只有在 remove_percent 不为None时才有意义
        :param kwargs: -1- (Tuple) pattern = r'^remove_point(_\d+)?$'，为变量名，值为 Tuple，
                      代表需要移除的点，该项优先执行。例如：-1- remove_point = (1, 2)

        :return reserved_dic: (dict) key 为 title， value 为移除坐标后的 DataFrame
        :return removed_certain_dic: (dict) key 为 title， value 为手动移除坐标的 DataFrame
        :return removed_percent_dic: (dict) key 为 title， value 为随机移除坐标的 DataFrame
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 检查关键字，并提取坐标
        pattern = r'^remove_point(_\d+)?$'  # 正则表达式
        remove_point_list = []  # 初始化列表，用来存储点
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
        for key, value in kwargs.items():
            if re.search(pattern, key):  # 判断是否满足 pattern
                if isinstance(value, tuple) and len(value) == 2:  # 判断是否为长度为 2 的元组
                    for item in value:
                        # 判断内部元素是否为 float or int，在 Python 中 True 的类型是 bool，它同时也是 int 的一个子类
                        if not isinstance(item, (float, int)) or isinstance(item, bool):
                            raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                            f" {value} expects to receive two float or int values.")
                    remove_point_list.append(value)
                else:
                    raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                    f" expected variable {key} to be a tuple of length 2,"
                                    f" but got a {type(value)} of length {len(key)} instead.")
            else:
                raise ValueError(
                    f"\033[95mIn {method_name} of {class_name}\033[0m, "
                    f"Expected one matching keyword argument for pattern '{pattern}',"
                    f" but found {key}")

        reserved_dic = {}
        removed_certain_dic = {}
        removed_percent_dic = {}
        for title, data_df in data_dic.items():

            # 1 移除用户指定的数据点
            if remove_point_list is not None:
                custom_list = []
                for remove_point in remove_point_list:
                    min_dist = np.inf
                    closest_point = None
                    for _, row in data_df.iterrows():
                        # 计算当前数据点与自定义坐标的距离
                        dist = np.sqrt((row[0] - remove_point[0]) ** 2 + (row[1] - remove_point[1]) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = tuple(row)
                    custom_list.append(closest_point)
                # 从DataFrame中移除最近的数据点
                removed_points = data_df[data_df.apply(lambda remove_row: tuple(remove_row) in custom_list, axis=1)]
                data_df = data_df.drop(removed_points.index)

                # 根据横坐标重新排序并重新设置行索引
                removed_points = removed_points.sort_values(by=data_df.columns[0])
                removed_points = removed_points.reset_index(drop=True)
                removed_certain_dic[title] = removed_points

            # 根据横坐标重新排序并重新设置行索引
            data_df = data_df.sort_values(by=data_df.columns[0])
            data_df = data_df.reset_index(drop=True)

            # 2 随机删除百分比数量的数据点
            if remove_percent is not None:
                # 创建空 DataFrame
                removed_percent_points = pd.DataFrame(columns=data_df.columns)
                num_points = int(len(data_df) * remove_percent)
                if num_points > 0:
                    if even:  # 均匀
                        indices = np.linspace(0, len(data_df) - 1, num=num_points, dtype=int)
                        indices = np.round(indices).astype(int)
                        sorted_indices = sorted(indices, reverse=True)
                        for idx in sorted_indices:
                            removed_point = data_df.iloc[idx]
                            data_df = data_df.drop(idx)
                            removed_percent_points = pd.concat([removed_percent_points,
                                                                removed_point.to_frame().transpose()],
                                                               ignore_index=True)

                    else:  # 不均匀
                        indices = list(range(len(data_df)))
                        random.shuffle(indices)
                        indices = indices[:num_points]
                        removed_points = data_df.iloc[indices]
                        data_df = data_df.drop(indices)
                        removed_percent_points = removed_points

                # 根据横坐标重新排序并重新设置行索引
                removed_percent_points = removed_percent_points.sort_values(by=data_df.columns[0])
                removed_percent_points = removed_percent_points.reset_index(drop=True)
                removed_percent_dic[title] = removed_percent_points

            # 根据横坐标重新排序并重新设置行索引
            reserved_df = data_df.sort_values(by=data_df.columns[0])
            reserved_df = reserved_df.reset_index(drop=True)
            reserved_dic[title] = reserved_df

        self.reserved_dic = reserved_dic
        self.removed_certain_dic = removed_certain_dic
        self.removed_percent_dic = removed_percent_dic
        self.current_dic = 'reserved_dic'

        return reserved_dic, removed_certain_dic, removed_percent_dic

    # 移除区间
    def remove_section(self):
        pass

    # 添加点
    def append_point(self, data_dic: Optional[dict] = None, **kwargs: Union[Tuple[float, float], List[float]]) \
            -> Tuple[Dict[str, DataFrame], Dict[str, Optional[DataFrame]], Dict[str, Optional[DataFrame]]]:
        r"""
        加入坐标到 DataFrame 中
        Manually or automatically add coordinates to a DataFrame.

        Recommended parameters and usage:
            After function reduce_precision.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param kwargs: -1- / -2- (Tuple) pattern = r'^append(_scope)?_point(_\d+)?$'，为变量名，值为 Tuple，
                      代表需要添加的点。例如：-1- append_point = (1, 2)
                                         -2- append_scope_point = [(2,2.5),(3,3.5),4]

        :return appended_dic: (dict) key 为 title， value 为添加坐标后的 DataFrame
        :return append_certain_dic: (dict) key 为 title， value 为手动添加坐标的 DataFrame
        :return append_scope_dic: (dict) key 为 title， value 为范围内随机添加坐标的 DataFrame
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 检查关键字，并提取坐标
        pattern = r'^append(_scope)?_point(_\d+)?$'  # 正则表达式
        append_point_list = []  # 初始化列表，用来存储点
        append_scope_point_list_raw = []  # 初始化列表，用来存储点
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
        for key, value in kwargs.items():
            if re.search(pattern, key):  # 判断是否满足 pattern
                if isinstance(value, tuple) and len(value) == 2:  # 判断是否为长度为 2 的元组
                    for item in value:
                        # 判断内部元素是否为 float or int，在 Python 中 True 的类型是 bool，它同时也是 int 的一个子类
                        if not isinstance(item, (float, int)) or isinstance(item, bool):
                            raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                            f"{key} = {value} is not be expected.")
                    append_point_list.append(value)
                elif isinstance(value, list) and len(value) == 3:  # 判断是否为长度为 3 的列表
                    for i, item_outer in enumerate(value):
                        if (i == 0 or i == 1) and isinstance(item_outer, tuple):
                            for item_inner in item_outer:
                                # 判断内部元素是否为 float or int，在 Python 中 True 的类型是 bool，它同时也是 int 的一个子类
                                if not isinstance(item_inner, (float, int)) or isinstance(item_inner, bool):
                                    raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                                    f" {key} = {value} is not be expected.")
                        elif i == 2 and (isinstance(item_outer, (float, int)) and not isinstance(item_outer, bool)):
                            append_scope_point_list_raw.append(value)  # 在最后一个判断结束后加入列表
                        else:
                            raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                            f"{key} = {value} is not be expected.")
                else:
                    raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                    f"{key} = {value} is not be expected.")
            else:
                raise ValueError(
                    f"\033[95mIn {method_name} of {class_name}\033[0m, "
                    f"Expected one matching keyword argument for pattern '{pattern}',"
                    f" but found {key}")

        appended_dic = {}
        append_certain_dic = {}
        append_scope_dic = {}
        for title, data_df in data_dic.items():

            # 1 将用户输入的坐标添加到原始数据集中
            if len(append_point_list) > 0:
                data_df = pd.concat(
                    objs=[data_df, pd.DataFrame(append_point_list, columns=data_df.columns)], ignore_index=True)
                # 根据横坐标重新排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[0])
                data_df = data_df.reset_index(drop=True)

                append_points = pd.DataFrame(append_point_list, columns=data_df.columns)
                # 根据横坐标重新排序并重新设置行索引
                append_points = append_points.sort_values(by=data_df.columns[0])
                append_points = append_points.reset_index(drop=True)
                append_certain_dic[title] = append_points

            # 将自动生成的坐标转化成点并加入列表
            append_scope_point_list = []
            for item in append_scope_point_list_raw:
                append_scope_point_list.extend([[random.uniform(item[0][0], item[0][1]),
                                                 random.uniform(item[1][0], item[1][1])] for _ in range(item[2])])

            # 2 将自动输入的点添加到原始数据集中
            if len(append_scope_point_list) > 0:
                data_df = pd.concat(
                    objs=[data_df, pd.DataFrame(append_scope_point_list, columns=data_df.columns)], ignore_index=True)
                # 根据横坐标重新排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[0])
                data_df = data_df.reset_index(drop=True)

                append_scope_points = pd.DataFrame(append_scope_point_list, columns=data_df.columns)
                # 根据横坐标重新排序并重新设置行索引
                append_scope_points = append_scope_points.sort_values(by=data_df.columns[0])
                append_scope_points = append_scope_points.reset_index(drop=True)
                append_scope_dic[title] = append_scope_points

            # 根据横坐标重新排序并重新设置行索引
            appended_df = data_df.sort_values(by=data_df.columns[0])
            appended_df = appended_df.reset_index(drop=True)
            appended_dic[title] = appended_df

        self.appended_dic = appended_dic
        self.append_certain_dic = append_certain_dic
        self.append_scope_dic = append_scope_dic
        self.current_dic = 'appended_dic'

        return appended_dic, append_certain_dic, append_scope_dic

    # 在区间上加点
    def append_section(self):
        pass

    # 寻峰
    def get_peak(self, data_dic: Optional[Dict[str, DataFrame]] = None, x_threshold: float = 0.05,
                 y_threshold: float = 0.1, show=True) -> Dict[str, List[int]]:
        """
        确定给定数据曲线中的峰值部分，其中峰值的高度比例为可以调整整体数据，以及它所占用的水平 x_threshold
        Identify the peak sections in a given data curve, where the height proportion of the peak to
        the overall data can be adjusted, as well as the horizontal x_threshold it occupies.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param x_threshold: (float) 差值的搜索范围，满足条件的点应当在这个范围内达到大于 y_threshold 的要求。默认为 0.05
        :param y_threshold: (float) 区间内最大最小值的差与整个函数最大最小值的差，需要大于这个比例。默认为 0.1
        :param show: (bool)是否展示图像，默认为 True

        :return peak_dic: (dict) 输出的 dict ，key 为 title，value 为每个数据的峰值点的 X 坐标
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        peak_dic = {}

        for title, data_df in data_dic.items():
            x = data_df.iloc[:, 0]
            y = data_df.iloc[:, 1]

            min_y = np.min(y)
            max_y = np.max(y)

            x_peaks = []
            idx = 0

            while idx < len(y) - 1:
                if y[idx + 1] <= y[idx]:  # 寻找下一个低点
                    idx += 1
                    continue

                start = idx
                while idx < len(y) - 1 and y[idx + 1] > y[idx]:
                    idx += 1

                # 验证是否满足条件的峰值
                peak_range = y[start:idx + 1]
                if np.max(peak_range) - np.min(peak_range) >= y_threshold * (max_y - min_y) and len(
                        peak_range) < x_threshold * len(y):
                    x_peaks.append(x[np.argmax(peak_range) + start])

            peak_dic[title] = x_peaks
            self.peak_dic = peak_dic
            self.current_dic = 'peak_dic'

            # 如果 show == True，绘制图像
            if show:
                plt.figure(dpi=200)
                plt.plot(x, y, color='#FFA500')
                for x_peak in x_peaks:
                    # 设置外部颜色为亮红色
                    plt.scatter(x_peak, y[np.where(x == x_peak)[0][0]], color='#FF0000', s=120, marker='^')

                # 展示标题
                plt.title(f'{title}')
                # 展示图像
                plt.show()
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return peak_dic

    # 寻峰2
    def find_data_peaks(self, data_dic: Dict[str, pd.DataFrame],
                        height: Optional[Union[float, Tuple[float, float]]] = None,
                        distance: Optional[int] = None, threshold: Optional[float] = None,
                        prominence: Optional[float] = None, width: Optional[float] = None,
                        show: bool = True, **kwargs) -> Dict[str, List[int]]:
        """
        使用 scipy 的 find_peaks 方法，在给定的数据曲线中识别峰值部分
        Use the find_peaks method from scipy to identify peak parts in the given data curve

        :param data_dic: (dict) 数据 dict ，其中 key 为数据的标题，value 为包含数据的 DataFrame
        :param height: (float / tuple) 用于筛选峰值的高度。可以是单一数字（最小高度），None（无限制），
                       或一个包含最小和最大高度的两元素序列
        :param distance: (int) 相邻峰值之间的最小水平距离 (以样本为单位)。用于防止峰值过于接近
        :param threshold: (float) 相邻样本中峰值的最小垂直距离。此参数可以帮助忽略接近平坦部分的峰值
        :param prominence: (float) 峰值的最小突出度。突出度较高的峰值意味着这些峰值是显著的、明显的
        :param width: (float) 峰值的最小宽度 (以样本为单位)。此参数可用于筛选出基于宽度的峰值
        :param show: (bool) 是否在完成后显示每个数据曲线的图像，默认为 True
        :param kwargs: (dict) 传递给 find_peaks() 的其他参数

        :return peak_dic: (dict) 输出的 dict ，key 为数据的标题，value 为每个数据峰值的X坐标
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        peak_dic = {}

        for title, data_df in data_dic.items():
            x = data_df.iloc[:, 0]
            y = data_df.iloc[:, 1]

            peaks, _ = find_peaks(y, height=height, distance=distance, threshold=threshold,
                                  prominence=prominence, width=width, **kwargs)

            peak_dic[title] = x[peaks].tolist()
            self.peak_dic = peak_dic
            self.current_dic = 'peak_dic'

            # 如果 show == True，则显示图像
            if show:
                plt.figure(dpi=200)
                plt.plot(x, y, color='#FFA500')
                plt.scatter(x[peaks], y[peaks], color='red', marker='^')  # 高亮显示峰值
                plt.title(f'{title}')
                plt.show()
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return peak_dic

    # 峰控制
    def control_peak(self):
        pass

    # 寻平台
    def get_plateau(self, data_dic: Optional[Dict[str, DataFrame]] = None, x_threshold: float = 0.20,
                    y_threshold: float = 0.05, show=True) -> Dict[str, List[tuple]]:
        """
        寻找给定数据曲线中的平台部分，可以调整平台高低与整体的比例和占横向阈值
        Identify the plateau sections in a given data curve, where the height proportion of the plateau to
        the overall data can be adjusted, as well as the horizontal x_threshold it occupies.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param x_threshold: (float) 平台最小的 X 轴向距离，为比例，默认为 0.1
        :param y_threshold: (float) 平台最大最小值差与函数最大最小差的比例，需要小于这个值，默认为 0.05
        :param show: (bool)是否展示图像，默认为 True

        :return fragment_dic: (dict) 输出的 dict ，key 为 title，value 为每个数据的 list 形式的平台区间的 X 坐标
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 初始化最终的 dict 结果
        fragment_dic = {}

        # 对每一个 DataFrame 进行处理
        for title, data_df in data_dic.items():
            # 提取横坐标和纵坐标列
            x = data_df.iloc[:, 0]
            y = data_df.iloc[:, 1]

            # 计算纵坐标最小值和最大值
            min_y = np.min(y)
            max_y = np.max(y)

            # 初始化平台区间列表
            plateau_ranges = []

            # 遍历区间
            x_start = 0
            x_end = 1
            while x_end < len(data_df):
                # 检查区间内的最小值和最大值差是否小于阈值
                y_plateau = y[x_start:x_end]
                diff = np.max(y_plateau) - np.min(y_plateau)
                if diff < y_threshold * (max_y - min_y):
                    x_end += 1
                else:
                    if x[x_end] - x[x_start] >= x_threshold:
                        plateau_ranges.append((x[x_start], x[x_end - 1]))
                    x_start = x_end
                    x_end += 1

            if x[x_end - 1] - x[x_start] >= x_threshold:
                plateau_ranges.append((x[x_start], x[x_end - 1]))

            # 存储当前 DataFrame 的结果到 dict
            fragment_dic[title] = plateau_ranges
            self.fragment_dic = fragment_dic
            self.current_dic = 'fragment_dic'

            # 如果 show == True，绘制图像
            if show:
                plt.figure(dpi=200)  # 为每个图创建一个新窗口
                plt.plot(x, y, color='#FFA500')
                for start_x, end_x in plateau_ranges:
                    plt.axvspan(start_x, end_x, color='#008000', alpha=0.25, linewidth=0)

                # 展示标题
                plt.title(f'{title}')
                # 展示图像
                plt.show()
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return fragment_dic

    # 平台控制
    def control_plateau(self):
        pass

    # 利用紧缩因子方法 (SF_method) 移除随机横坐标区间
    def SF_remove(self, remove_section: Union[Tuple[Tuple[float, float], float, int] or
                                              List[Tuple[float, float], float, int]],
                  data_dic: Optional[Dict[str, DataFrame]] = None) \
            -> Tuple[Dict[str, DataFrame], Dict[str, List[DataFrame]]]:
        """
        利用紧缩因子方法 (SF_method) 移除随机横坐标区间
        Remove random horizontal intervals using the shrinkage factor method (SF_method).

        Recommended parameters and usage:
                    After function improve_precision.

        :param remove_section: (tuple / list) 需要被移除的随机区间
                              例如：remove_section = [(0.1, 0.8), 0.2, 2]，代表在前 10% 至 20% 区间内移除 20% 总长度的 2 个区间
        :param data_dic: (dict) key 为 title，value 为 DataFrame

        :return sf_removed_dic: (dict) key 为 title，value 为 data_dic 被移除后剩下的坐标组成的 DataFrame
        :return sf_remove_section_dic: (dict) key 为 title，value 为被移除坐标的 DataFrame 组成的 tuple
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 解析 remove_section
        (percent_start, percent_end), percent_remove, num_sections = remove_section

        # 判断是否可行
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
        if (percent_remove * num_sections) >= (percent_end - percent_start):
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"remove_section = {remove_section} is unreasonable, "
                             f"(percent_remove * num_sections) >= (percent_end - percent_start) is been expected.")

        sf_removed_dic = {}  # 存储剩下的坐标的 dict
        sf_remove_section_dic = {}  # 存储被移除的点的 dict

        for title, data_df in data_dic.items():

            removed_total_df = pd.DataFrame(columns=data_df.columns)  # 存储所有被移除的坐标
            removed_points_list = []  # 存储当前标题下被移除的坐标
            removed_indices = set()  # 存储已移除的区间的索引

            x_max = data_df.iloc[:, 0].max()  # 获取横坐标的最大值
            x_min = data_df.iloc[:, 0].min()  # 获取纵坐标的最大值

            # 计算大区间的起始和结束索引
            index_start = float(x_max - x_min) * percent_start + x_min
            index_end = float(x_max - x_min) * percent_end + x_min

            # 计算小区间的长度
            section_length = float((x_max - x_min) * percent_remove)

            while len(removed_indices) < num_sections:

                # 随机选择小区间的起始索引
                section_start = index_start + (index_end - index_start - section_length) * np.random.rand()
                # 计算小区间的结束索引
                section_end = section_start + section_length

                # 检查小区间是否与已移除的区间相交
                if any(section_start <= former_end and section_end >= former_start
                       for former_start, former_end in removed_indices):
                    continue

                # 将小区间添加到已移除的区间集合中
                removed_indices.add((section_start, section_end))

            # 根据已移除的区间构建新的 DataFrame
            for section_start, section_end in removed_indices:
                # 存储当前标题下被移除的坐标至 list
                removed_points = pd.DataFrame(columns=data_df.columns)
                removed_points = pd.concat([removed_points, data_df[(data_df[data_df.columns[0]] >= section_start) &
                                                                    (data_df[data_df.columns[0]] <= section_end)]])
                # 根据横坐标重新排序并重新设置行索引
                removed_points = removed_points.sort_values(by=data_df.columns[0])
                removed_points = removed_points.reset_index(drop=True)
                removed_points_list.append(removed_points)

                # 所有被移除的坐标
                removed_total_df = pd.concat([removed_total_df, data_df[(data_df[data_df.columns[0]] >= section_start) &
                                                                        (data_df[data_df.columns[0]] <= section_end)]])

            # 将剩余的点添加到SF_removed_dic中
            remaining_points = data_df[~data_df.index.isin(removed_total_df.index)]
            # 根据横坐标重新排序并重新设置行索引
            remaining_points = remaining_points.sort_values(by=data_df.columns[0])
            remaining_points = remaining_points.reset_index(drop=True)
            sf_removed_dic[title] = remaining_points

            # 将已移除的点添加到 SF_remove_section_dic 中，需要放在 sf_removed_dic 之后，否则会乱点
            sf_remove_section_dic[title] = removed_points_list

        self.sf_removed_dic = sf_removed_dic
        self.sf_remove_section_dic = sf_remove_section_dic
        self.current_dic = 'sf_removed_dic'

        return sf_removed_dic, sf_remove_section_dic

    # 利用紧缩因子方法 (SF_method) 添加坐标
    def SF_append(self, x_shrinking_factor: float = 0.95, y_shrinking_factor: float = 0.95,
                  x_distribution_limit: Optional[float] = None, y_distribution_limit: Optional[float] = None,
                  tendency: Optional[str] = None) -> Tuple[Dict[str, DataFrame], Dict[str, List[DataFrame]]]:
        """
        利用紧缩因子方法 (SF_method) 添加随机坐标至被移除的区间
        Add random coordinates to the removed remove_section using the shrinking factor method (SF_method).

        Recommended parameters and usage:
                    After function SF_remove.

        注意： reduce_precision 后删除的横坐标区间仅为一个点 -> 提高 interval_disperse 的精度
        Attention: After reduce_precision, the deleted x-coordinate interval is only one point
                  -> Increase the precision of interval_disperse.

        :param x_shrinking_factor: (float) 横坐标的紧缩因子，应为 0 到 1 之间，数值越大，点越紧密
        :param y_shrinking_factor: (float) 纵坐标的紧缩因子，应为 0 到 1 之间，数值越大，点越紧密
        :param x_distribution_limit: (float) 横坐标最大允许限度，建议为 0 到 1 之间，默认为原区间
        :param y_distribution_limit: (float) 纵坐标最大允许限度，默认为原区间
        :param tendency: (bool) 纵坐标的趋势，为 'up' 时上升，为 'down' 时下降

        :return sf_appended_dic: (dict) key 为 title，value 为 sf_removed_dic 添加随机坐标后的 DataFrame
        :return sf_append_point_dic: (dict) key 为 title，value 为添加坐标的 DataFrame 组成的 tuple，每个DataFrame长度均为 1
        """

        # 检查是否已用了 SF_remove 方法
        if self.sf_removed_dic is not None and self.sf_remove_section_dic is not None:
            sf_removed_dic = copy.deepcopy(self.sf_removed_dic)
            sf_remove_section_dic = copy.deepcopy(self.sf_remove_section_dic)
        else:
            raise DependencyError("SF_remove method needs to be executed first.")

        # 检查 shrinking_factor 的值
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
        if not 0 < x_shrinking_factor < 1:  # 当 x_shrinking_factor 不在 0 和 1 之间时
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The value of x_shrinking_factor should be between 0 and 1.")
        elif not 0 < y_shrinking_factor < 1:  # 当 y_shrinking_factor 不在 0 和 1 之间时
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The value of y_shrinking_factor should be between 0 and 1.")

        # 检查 tendency 的值
        if tendency not in ['up', 'down', None]:  # 当 tendency 不为'up', 'down', None 时
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The options for tendency are up, down and None.")

        sf_appended_dic = {}
        sf_append_point_dic = {}

        # 遍历总 dict
        for title, sf_remove_section in sf_remove_section_dic.items():

            sf_removed = sf_removed_dic[title]
            x_selected_list = []
            y_selected_list = []

            # 遍历一组数据内被移除区间的 dict
            for sf_remove_df in sf_remove_section:

                # X 横坐标
                # 计算中点
                first_column = sf_remove_df.iloc[:, 0]  # 使用 iloc 方法选取第一列
                x_min = first_column.min()  # 获取第一列的最小值
                x_max = first_column.max()  # 获取第一列的最大值
                x_midpoint = (x_max + x_min) / 2

                # 计算横坐标正态分布区间 (以 0 为均值的对称区间)
                x_min_normal = x_min - x_midpoint
                x_max_normal = x_max - x_midpoint

                # 计算横坐标紧缩区间 (以 0 为均值的对称区间)
                x_shrink_section = (x_max - x_min) * ((1 - x_shrinking_factor) / 2)
                x_min_sf = x_min_normal + x_shrink_section
                x_max_sf = x_max_normal - x_shrink_section

                # 计算横坐标标准差
                x_std = self.calculate_std(interval_symmetric=(x_min_sf, x_max_sf), probability=x_shrinking_factor)

                # 横坐标筛选区间
                if x_distribution_limit is not None:
                    x_removed_limit = (x_max - x_min) * ((1 - x_distribution_limit) / 2)
                    x_min_limit = x_min_normal + x_removed_limit
                    x_max_limit = x_max_normal - x_removed_limit
                else:
                    x_min_limit = x_min_normal
                    x_max_limit = x_max_normal

                # 随机选择 1 个横坐标 (正态分布)
                x_zero_selected = None
                # 坐标需要在被移除的区间内
                while x_zero_selected is None or x_zero_selected[0] <= x_min_limit or x_zero_selected[0] >= x_max_limit:
                    x_zero_selected = self.get_point_normal(mean=0, std=x_std, number=1)
                x_selected = x_zero_selected[0] + x_midpoint
                x_selected_list.append(x_selected)

                # Y 纵坐标
                # 找出距离被选择点最近的纵坐标
                diff = np.abs(sf_remove_df.iloc[:, 0] - x_selected)  # 计算第一列与目标值的差值
                nearest_indexes = np.where(diff == np.min(diff))[0]  # 找到距离最近的值的索引
                nearest_values = sf_remove_df.iloc[nearest_indexes, 1]  # 获取距离最近的值的索引对应的第二列的值
                y_midpoint = nearest_values.mean()  # 如果有两个点距离 x_selected 都最近，那么计算其第二列值的均值

                # 找出被移除区间纵坐标的极差
                second_column = sf_remove_df.iloc[:, 1]  # 使用 iloc 方法选取第二列
                y_min = second_column.min()  # 获取第二列的最小值
                y_max = second_column.max()  # 获取第二列的最大值
                y_range = y_max - y_min  # 得出极差

                # 计算纵坐标正态分布区间 (以 0 为均值的对称区间)
                y_min_normal = 0 - y_range
                y_max_normal = y_range

                # 计算纵坐标紧缩区间 (以 0 为均值的对称区间)
                y_shrink_section = ((1 - y_shrinking_factor) / 2) * (y_max - y_min)

                y_min_sf = y_min_normal + y_shrink_section
                y_max_sf = y_max_normal - y_shrink_section

                # 计算横坐标标准差
                y_std = self.calculate_std(interval_symmetric=(y_min_sf, y_max_sf), probability=y_shrinking_factor)
                # 横坐标筛选区间
                if y_distribution_limit is not None:
                    y_removed_limit = ((1 - y_distribution_limit) / 2) * (y_max - y_min)
                    y_min_limit = y_min_normal + y_removed_limit
                    y_max_limit = y_max_normal - y_removed_limit
                else:
                    y_min_limit = y_min_normal
                    y_max_limit = y_max_normal

                # 随机选择 1 个纵坐标 (正态分布)
                while True:  # 不断生成点，直到在需要的区间内
                    y_zero_selected = self.get_point_normal(mean=0, std=y_std, number=1)  # 为长度为 1 的 list
                    if tendency == 'up' and 0 < y_zero_selected[0] < y_max_limit:
                        y_selected = y_zero_selected + y_midpoint
                        break
                    elif tendency == 'down' and y_min_limit < y_zero_selected[0] < 0:
                        y_selected = y_zero_selected + y_midpoint
                        break
                    elif tendency is None and y_min_limit < y_zero_selected[0] < y_max_limit:
                        y_selected = y_zero_selected + y_midpoint
                        break

                y_selected_list.append(y_selected)

            # 使用zip函数将x_list和y_list逐对组合
            sf_append_df_list = []
            for x_selected, y_selected in zip(x_selected_list, y_selected_list):
                sf_append_df = pd.DataFrame([[x_selected, y_selected]], columns=sf_removed.columns)
                # 将指定列的对象类型转换为浮点型
                sf_append_df.iloc[:, 0] = sf_append_df.iloc[:, 0].astype(float)
                sf_append_df.iloc[:, 1] = sf_append_df.iloc[:, 1].astype(float)
                sf_append_df_list.append(sf_append_df)
            sf_append_point_dic[title] = sf_append_df_list

            # 将数据逐行插入DataFrame表格
            sf_append = pd.concat([sf_removed] + sf_append_df_list, axis=0)

            # 根据横坐标重新排序并重新设置行索引
            sf_append = sf_append.sort_values(by=sf_append.columns[0])
            sf_append = sf_append.reset_index(drop=True)
            sf_appended_dic[title] = sf_append

        self.sf_appended_dic = sf_appended_dic
        self.sf_append_point_dic = sf_append_point_dic
        self.current_dic = 'sf_appended_dic'

        return sf_appended_dic, sf_append_point_dic

    # 随机取键值对在 dict 中
    def random_dict(self, data_dic: Optional[DataFrame] = None, num_pairs: int = 1) -> Dict[str, DataFrame]:
        """
        在 dict 中随机取键值对
        Randomly select key-value pairs from a dictionary.

        :param data_dic: (dict) 若被赋值，则会对该 dict 操作
        :param num_pairs: (int) 匹配的数量

        :return random_dic: (dict) key 为 title，value 为 DataFrame，长度为 num_pairs
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 随机抽取
        keys = list(data_dic.keys())
        random_pairs = random.sample(keys, num_pairs)
        random_dic = {key: data_dic[key] for key in random_pairs}
        self.random_dic = random_dic
        self.current_dic = 'random_dic'

        return random_dic

    # 移动一个方法的点
    def move_points(self, data_dic: Optional[DataFrame] = None, peg_x: Optional[float] = None,
                    direction: Optional[str] = None, moving_direction: Optional[str] = None,
                    difference_value: Optional[float] = None) -> Dict[str, DataFrame]:
        """
        根据给定的 peg_x 参考点、方向、移动方向以及差值对点进行移动
        The points are moved based on the given peg_x reference point, direction, movement direction, and difference.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param peg_x: (float) 用于筛选的横坐标参考点
        :param direction: (str) 筛选方向，只能是 'up', 'down', 'left', 'right'
        :param moving_direction: (str) 移动的方向，只能是 'up', 'down', 'left', 'right'
        :param difference_value: (float) 移动的距离

        :return moved_dic: (dict) key 为 title，value 为移动后的 DataFrame
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        valid_directions = ['up', 'down', 'left', 'right']

        # 检查最后四个变量是否被赋值
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
        if peg_x is None or direction is None or moving_direction is None or difference_value is None:
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"All of the following parameters must be provided: peg_x, direction, moving_direction, "
                             f"and difference_value.")

        # 检查输入的方向是否合法
        if direction not in valid_directions:
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Invalid direction: {direction}. Valid directions are {valid_directions}.")
        if moving_direction not in valid_directions:
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Invalid moving direction: {moving_direction}. Valid directions are {valid_directions}.")
        if not isinstance(difference_value, (float, int)):
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"difference_value must be a float or int.")

        moved_dic = {}  # 用于存储移动后的 DataFrame

        for title, data_df in data_dic.items():

            # 确保 DataFrame 至少有两列，一列是横坐标，一列是纵坐标
            if data_df.shape[1] < 2:
                raise ValueError(f"DataFrame for {title} must have at least two columns (x and y coordinates).")

            # 找到与 peg_x 最接近的点
            closest_idx = (data_df.iloc[:, 0] - peg_x).abs().idxmin()
            closest_x = data_df.iloc[closest_idx, 0]

            # 根据 direction 选择点：'up' 和 'down' 根据纵坐标选择，'left' 和 'right' 根据横坐标选择
            selected_df = None
            if direction == 'up':
                selected_df = data_df[data_df.iloc[:, 1] > data_df.iloc[closest_idx, 1]]  # 选择纵坐标比参考点大的点
            elif direction == 'down':
                selected_df = data_df[data_df.iloc[:, 1] < data_df.iloc[closest_idx, 1]]  # 选择纵坐标比参考点小的点
            elif direction == 'left':
                selected_df = data_df[data_df.iloc[:, 0] < closest_x]  # 选择横坐标比参考点小的点
            elif direction == 'right':
                selected_df = data_df[data_df.iloc[:, 0] > closest_x]  # 选择横坐标比参考点大的点

            # 根据 moving_direction 移动点
            if moving_direction == 'up':
                selected_df.loc[:, selected_df.columns[1]] += difference_value  # 纵坐标加上差值
            elif moving_direction == 'down':
                selected_df.loc[:, selected_df.columns[1]] -= difference_value  # 纵坐标减去差值
            elif moving_direction == 'left':
                selected_df.loc[:, selected_df.columns[0]] -= difference_value  # 横坐标减去差值
            elif moving_direction == 'right':
                selected_df.loc[:, selected_df.columns[0]] += difference_value  # 横坐标加上差值

            # 将移动后的点更新到原始 DataFrame 中
            data_df.update(selected_df)
            moved_dic[str(title)] = pd.DataFrame(data_df).reset_index(drop=True)  # 确保是 DataFrame，并重新设置索引

            self.moved_dic = moved_dic
            self.current_dic = 'moved_dic'

        return moved_dic

    # 处理 X 值相同的点
    def handle_duplicate_x(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                           operation: Optional[str] = None) -> Dict[str, DataFrame]:
        """
        根据 operation 参数处理 DataFrame 中重复的 x 值。
        Depending on the operation ('up', 'down', 'delete'), it either keeps the upper, lower point or deletes both.

        :param data_dic: (dict) key 为 title，value 为 DataFrame
        :param operation: (str) 操作类型，只能是 'up', 'down', 'delete'

        :return handled_dic: (dict) key 为 title，value 为处理后的 DataFrame
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 检查输入的 operation 是否合法
        valid_operations = ['up', 'down', 'delete']
        class_name = self.__class__.__name__  # 获取类名
        method_name = inspect.currentframe().f_code.co_name  # 获取方法名

        if operation not in valid_operations:
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Invalid operation: {operation}. Valid operations are {valid_operations}.")

        handled_dic = {}

        for title, data_df in data_dic.items():
            # 确保 DataFrame 至少有两列，一列是 x 坐标，一列是 y 坐标
            if data_df.shape[1] < 2:
                raise ValueError(f"DataFrame for {title} must have at least two columns (x and y coordinates).")

            # 找到重复的 x 值
            duplicated_x = data_df[data_df.duplicated(subset=data_df.columns[0], keep=False)]

            if not duplicated_x.empty:
                # 如果 operation 是 'up'，保留上面的点 (保留第一个)
                if operation == 'up':
                    data_df = data_df.drop_duplicates(subset=data_df.columns[0], keep='first')

                # 如果 operation 是 'down'，保留下面的点 (保留最后一个)
                elif operation == 'down':
                    data_df = data_df.drop_duplicates(subset=data_df.columns[0], keep='last')

                # 如果 operation 是 'delete'，删除重复的点
                elif operation == 'delete':
                    data_df = data_df.drop_duplicates(subset=data_df.columns[0], keep=False)

            # 更新处理后的 DataFrame 到结果字典
            handled_dic[str(title)] = pd.DataFrame(data_df).reset_index(drop=True)
            self.handled_dic = handled_dic
            self.current_dic = 'handled_dic'

        return handled_dic


""" 魔法方法 """
class Magic(Manager, Module):
    """
    中心函数区
    数据生成和操纵的核心，对 dict 数据进行操作，其中 key 为数据的 title，value 为 数据的 DataFrame 表格

    The main part of the function library is divided into ten sections based on
    the precision of curve coordinates and the exploration of key data points
    with their weights. These functions are crucial for data generation and processing.
    Additionally, it is possible to enhance the functionality of each function by
    inserting modules before and after them according to specific needs, in order
    to achieve better results. A powerful function library provides flexible operations
    and effective data handling methods, enabling users to accomplish various data
    optimization tasks.

    注意：
    1.  所有方法中先运用临时变量来存储结果，来提高代码的可读性和可维护性
    2.  返回的结果只是为了其它数据获取的需要，类属性中的结果对曲线的操作要更重要
    3.  所有点输入均为 list 包含 tuple 类型，返回均为 DataFrame 类型
    4.  只有输入的参数不为 dict 其余均为 dict 格式
    5.  数据的检索顺序与 self.current_dic 相关，且不会改变 self.data_dic

    Note:
    1.  All methods use temporary variables to store results first to improve the readability
        and maintainability of the code
    2.  The returned result is only for the need of other data acquisition,
        and the result in the class attribute is more important for the operation of the curve
    3.  All entries are of the tuple type and the DataFrame type is returned
    4.  Only the entered parameters are not dict. All other parameters are dict
    5.  The retrieval order of the data is related to self.current_dic and does not change self.data_dic
    """

    # 0 初始化
    def __init__(self,

                 # 接收参数 (7)
                 data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None, title: Optional[str] = None,
                 x_list: Optional[list] = None, y_list: Optional[list] = None, x_label: str = None, y_label: str = None,

                 # 关键参数 (7)
                 txt_path: Optional[str] = None, excel_path: Optional[str] = None, json_path: Optional[str] = None,
                 keyword: Optional[str] = None, file_pattern: Optional[str] = None, save_path: Optional[str] = None,
                 magic_database: Optional[str] = None,
                 ):
        """
        # 接收参数 (7)
        :param data_dic: (dict) 文件的 dict，key 为文件名，value 为 DataFrame 数据，当传入的 dict 长度不为1时，
                         需要每个 DataFrame 中的列都一样 (即同类型的)
        :param data_df:  (DataFrame) 输入 DataFrame
        :param title: (str) 数据的 title
        :param x_list: (list) x坐标的 list
        :param y_list: (list) y坐标的 list
        :param x_label: (str) x坐标的 label
        :param y_label: (str) y坐标的 label

        # 文件打开路径，名称匹配，已有 dict (7)
        :param txt_path: (str) TXT 文件路径，可以是文件路径，也可以是目录
        :param excel_path: (str) Excel 文件路径，可以是文件路径，也可以是目录
        :param json_path: (str) JSON 文件路径，也可以是目录
        :param keyword: (str) 关键词，为需要目标数据的类型
        :param file_pattern: (str) 当 path 为目录时，file_pattern 为目录下文件的正则表达式匹配，只有 path 为目录时才有意义
        :param save_path: (str) 保存路径
        :param magic_database: (str) 数据库的位置
        """

        # 超类初始化
        super().__init__(
                         # 接收参数 (7)
                         data_dic=data_dic, data_df=data_df,
                         title=title, x_list=x_list, y_list=y_list, x_label=x_label, y_label=y_label,

                         # 关键参数 (6)
                         txt_path=txt_path, excel_path=excel_path, json_path=json_path, keyword=keyword,
                         file_pattern=file_pattern, save_path=save_path, magic_database=magic_database
                         )

        # 接收参数 (7)
        self.data_dic = data_dic
        self.data_df = data_df
        self.title = title
        self.x_list = x_list
        self.y_list = y_list
        self.x_label = x_label
        self.y_label = y_label

        # 文件打开路径，名称匹配 (7)
        self.txt_path = txt_path
        self.excel_path = excel_path
        self.json_path = json_path
        self.keyword = keyword  # 如果接入了 keyword， 那么输入的其它 to_magic_dic 中有的数据将会被 to_magic_dic 覆盖
        self.file_pattern = file_pattern
        self.save_path = save_path
        if magic_database is not None:
            self.magic_database = magic_database
        else:
            self.magic_database = self.Magic_Database

        # 数据初始化分配 和 数据类型导入
        if type(self) == Magic:  # 当 self 为 Magic 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # Manager 及其子类需要调用以初始化属性

        # /* 方法所用属性  数据类型 说明 (此处可以不声明，因为超类中已声明过，但为了方便调度所以加上) */
        self.current_dic = 'data_dic'  # 当前数据的名称

        # 1  smooth_curve()
        self.spline_smoothing_dic = None  # (dict) 平滑曲线的 dict，value 为 spline
        # 2  locate_point()
        self.key_point_dic = None  # (dict) 保存的特殊点的 dict，value 为一个包含特殊点的 dict
        # 3  improve_precision()
        self.smoothing_dic = None  # (dict) value 为高精度 DataFrame 数据，精度与 precision_smoothing 有关
        self.precision_smoothing = None  # (float) smoothing_dic 中 DataFrame 的精度
        # 4  reduce_point()
        self.dispersed_dic = None  # (dict) value 为低精度 DataFrame 数据，精度与 interval_disperse 有关
        self.interval_disperse = None  # (float) dispersed_dic 每两个点间的间隔需要大于该值
        # 5  normalized_data()
        self.normalized_dic = None  # (dict) value 为标准化后的 DataFrame 数据，与接入的参数有关
        self.key_normalized_dic = None  # (dict) value 为标准化后的特殊点 key 的数据
        self.normalize_rule_dic = None  # (dict) value 为坐标标准化变化规则，无法直接自乘
        # 6  adjust_data()
        self.adjusted_dic = None  # (dict) value 为调整后的 DataFrame 数据，与接入的参数有关
        self.key_adjusted_dic = None  # (dict) value 为调整后的特殊点 key 的数据
        self.adjust_rule_dic = None  # (dict) value 为坐标调整变化规则
        # 7  assign_weight()
        self.balanced_dic = None  # (dict) value 为整合后的 DataFrame 数据
        self.weight_list_dic = None  # (dict) value 为数据权重的 list
        self.weight_dic = None  # (dict) key 为特殊点 or 普通点，value 为坐标权重的 dict
        # 8  fit_curve()
        self.spline_fitting_dic = None  # (dict) 拟合曲线的 dict，value 为 spline
        # 9  restore_precision()
        self.fitting_dic = None  # (dict) value 为拟合后原精度的 DataFrame 数据
        # 10 realize_data()
        self.realized_dic = None  # (dict) value 为真实后的 DataFrame 数据

    # 1 平滑
    def smooth_curve(self, data_dic: Optional[Dict[str, DataFrame]] = None, degree_smoothing: int = 3,
                     smooth_smoothing: Optional[float] = 0.05, weight_dic: Optional[Dict[str, list]] = None) \
            -> Dict[str, any]:
        """
        使用 UnivariateSpline 进行曲线拟合
        Perform curve fitting using UnivariateSpline.

        STEP ONE： (Recommended parameter Settings for this method:)
                 degree_fitting = 3, smooth_fitting = 0.05, weight = None

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param degree_smoothing: (int) 曲线阶数，越大越平滑，1 <= degree_fitting <= 5，推荐为 3
        :param smooth_smoothing: (float) 平滑因子，越小越接近原数据， 推荐越小越好:0, 0.05
        :param weight_dic: (dict) 权重的 dict，若被赋值，则会用该权重进行拟合

        :return spline_smoothing_dic: (dict) 曲线的 dict ，其中 key 为数据的 title，value 为曲线的 scipy
        例如：spline_smoothing_dic = {'title1': scipy1, 'title2': scipy2}
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        spline_smoothing_dic = {}

        # 遍历 dict 以完成所有数据的处理
        for title, data_df in data_dic.items():

            data_df = copy.deepcopy(data_df)

            # 加权的 dict
            if weight_dic is not None:
                weight = weight_dic['title']
            else:
                weight = None

            # 获取 DataFrame 中的 X 和 Y 列数据
            x_section = data_df.iloc[:, 0].values  # 获取 DataFrame 的 X 列数据
            y_section = data_df.iloc[:, 1].values  # 获取 DataFrame 的 Y 列数据

            # 检查 x_section 是否有重复值
            unique_x_section, counts = np.unique(x_section, return_counts=True)
            duplicates = unique_x_section[counts > 1]

            if len(duplicates) > 0:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"x_section contains duplicate values: {duplicates}")

            # 添加检查权重列表的长度是否与数据量一致
            if weight is not None and len(weight) != len(x_section):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The length of weight_list for '{title}' ({len(weight_list)}) does not match "
                                 f"the data length ({len(x_section)}).")

            # 使用 UnivariateSpline 进行曲线拟合
            smoothing_spline = UnivariateSpline(x_section, y_section, k=degree_smoothing, s=smooth_smoothing, w=weight)
            spline_smoothing_dic[title] = smoothing_spline  # 将平滑后的曲线添加到 dict 中

        self.spline_smoothing_dic = spline_smoothing_dic
        self.current_dic = 'spline_smoothing_dic'

        return spline_smoothing_dic

    # 2 定位
    def locate_point(self, spline_dic=None, precision_dxdy: float = 0.01, locate_extremum: bool = True,
                     locate_inflection: bool = True, locate_max: bool = True, locate_min: bool = True) \
            -> dict[str, Dict[str, Optional[DataFrame]]]:
        """
        寻找极值点、拐点、最大值点和最小值在输入的 DataFrame 中
        Find extrema, inflection points, maximum points, and minimum points in the input DataFrame.

        STEP TWO： (Recommended parameter Settings for this method:)
                 precision_dxdy = 0.01, locate_inflection = True, locate_extremum = True,
                 locate_max = True, locate_min = True

        :param spline_dic: (scipy) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 scipy 曲线
        :param precision_dxdy: (float) 导函数的精度，默认为0.01
        :param locate_extremum: (bool) 是否寻找极值点，默认为 True，为 False 时输出该项为 None
        :param locate_inflection: (bool) 是否寻找拐点，默认为 True，为 False 时输出该项为 None
        :param locate_max: (bool) 是否寻找最大值，默认为 True，为 False 时输出该项为 None
        :param locate_min: (bool) 是否寻找最小值，默认为 True，为 False 时输出该项为 None

        :return key_point_dic：(dict) 包含特殊点的 dict，key 为特殊点的类型，value 为对应的 DataFrame
        例如：key_point_dic = {'title1': {'extremum': DataFrame1, 'inflection': DataFrame2,
                                                            {'max': DataFrame3, 'min': None},
                              'title2': {'extremum': DataFrame4, 'inflection': DataFrame5,
                                                            {'max': DataFrame6, 'min': None} }
                                                            此为 locate_min = False 情况
        """

        # 将需要处理的数据赋给 spline_smoothing
        if spline_dic is not None:
            spline_dic = copy.deepcopy(spline_dic)
        else:
            spline_dic = copy.deepcopy(self.spline_smoothing_dic)

        key_point_dic = {}  # 初始化

        # 遍历 dict 以完成所有数据的处理
        for title, spline in spline_dic.items():
            key_point_single_dic = {}  # 每个 spline 开始前初始化

            # 计算一阶导数和二阶导数
            dy_dx = spline.derivative(n=1)
            d2y_dx2 = spline.derivative(n=2)

            # 获取插值曲线的节点位置
            knots = spline.get_knots()
            first_knot = float(knots[0])  # 获取第一个节点的数值
            last_knot = float(knots[-1])  # 获取最后一个节点的数值

            # 创建等间距的 X 轴坐标
            x_section = np.linspace(first_knot, last_knot, int((last_knot - first_knot) / precision_dxdy))
            dy_dx_value = dy_dx(x_section)  # 一阶导数值
            d2y_dx2_value = d2y_dx2(x_section)  # 二阶导数值

            # 寻找 extremum
            extremum_list = []  # 极值列表
            if locate_extremum:
                for i in range(1, len(dy_dx_value) - 1):
                    # 前一点与后一点异号
                    if (dy_dx_value[i] > 0 > dy_dx_value[i - 1]) or (dy_dx_value[i] < 0 < dy_dx_value[i - 1]):
                        # 加入绝对值小的点
                        if abs(spline(x_section[i - 1])) < abs(spline(x_section[i])):
                            extremum_list.append((x_section[i - 1], spline(x_section[i - 1]).item()))
                        else:
                            extremum_list.append((x_section[i], spline(x_section[i]).item()))
                    elif dy_dx_value[i] == 0 and np.sign(dy_dx_value[i - 1]) != np.sign(dy_dx_value[i + 1]):
                        # 加入前后导数异号的零点
                        extremum_list.append((x_section[i], spline(x_section[i]).item()))

                # 最后一个点与倒数第二个点的比较
                if (dy_dx_value[-1] > 0 > dy_dx_value[-2]) or (dy_dx_value[-1] < 0 < dy_dx_value[-2]):
                    # 加入绝对值小的点
                    if abs(spline(x_section[-2])) < abs(spline(x_section[-1])):
                        extremum_list.append((x_section[-2], spline(x_section[-2]).item()))
                    else:
                        extremum_list.append((x_section[-1], spline(x_section[-1]).item()))

                # 转换成 DataFrame 类型
                extremum_df = pd.DataFrame(extremum_list, columns=[self.x_label, self.y_label])
                # 以行坐标排序并重新设置行索引
                extremum_df = extremum_df.sort_values(by=extremum_df.columns[0])
                extremum_df.reset_index(drop=True, inplace=True)
                key_point_single_dic['extremum'] = extremum_df

            # 不寻找 extremum 的情况
            else:
                key_point_single_dic['extremum'] = None

            # 寻找 inflection
            inflection_list = []  # 拐点列表
            if locate_inflection:
                for i in range(1, len(d2y_dx2_value) - 1):
                    # 前一点与后一点异号
                    if (d2y_dx2_value[i] > 0 > d2y_dx2_value[i - 1]) or (d2y_dx2_value[i] < 0 < d2y_dx2_value[i - 1]):
                        # 加入绝对值小的点
                        if abs(spline(x_section[i - 1])) < abs(spline(x_section[i])):
                            inflection_list.append((x_section[i - 1], spline(x_section[i - 1]).item()))
                        else:
                            inflection_list.append((x_section[i], spline(x_section[i]).item()))
                    elif d2y_dx2_value[i] == 0 and np.sign(d2y_dx2_value[i - 1]) != np.sign(d2y_dx2_value[i + 1]):
                        # 加入前后导数异号的零点
                        inflection_list.append((x_section[i], spline(x_section[i]).item()))

                # 最后一个点与倒数第二个点的比较
                if (d2y_dx2_value[-1] > 0 > d2y_dx2_value[-2]) or (d2y_dx2_value[-1] < 0 < d2y_dx2_value[-2]):
                    # 加入绝对值小的点
                    if abs(spline(x_section[-2])) < abs(spline(x_section[-1])):
                        inflection_list.append((x_section[-2], spline(x_section[-2]).item()))
                    else:
                        inflection_list.append((x_section[-1], spline(x_section[-1]).item()))

                # 转换成 DataFrame 类型
                inflection_df = pd.DataFrame(inflection_list, columns=[self.x_label, self.y_label])
                # 以行坐标排序并重新设置行索引
                inflection_df = inflection_df.sort_values(by=inflection_df.columns[0])
                inflection_df.reset_index(drop=True, inplace=True)
                key_point_single_dic['inflection'] = inflection_df

            # 不寻找 inflection 的情况
            else:
                key_point_single_dic['inflection'] = None

            # 寻找 max
            if locate_max:
                max_idx = np.argmax(spline(x_section))
                max_point = (x_section[max_idx], spline(x_section[max_idx]).item())
                # 转换成 DataFrame 类型
                max_df = pd.DataFrame(data=[max_point], columns=[self.x_label, self.y_label])
                key_point_single_dic['max'] = max_df

            # 不寻找 max 的情况
            else:
                key_point_single_dic['max'] = None

            # 寻找 min
            if locate_min:
                min_idx = np.argmin(spline(x_section))
                min_point = (x_section[min_idx], spline(x_section[min_idx]).item())
                # 转换成 DataFrame 类型
                min_df = pd.DataFrame(data=[min_point], columns=[self.x_label, self.y_label])
                key_point_single_dic['min'] = min_df

            # 不寻找 min 的情况
            else:
                key_point_single_dic['min'] = None

            # 将每条曲线的特殊点 dict 数据加入总体 dict
            key_point_dic[title] = key_point_single_dic

        self.key_point_dic = key_point_dic
        self.current_dic = 'key_point_dic'

        return key_point_dic

    # 3 高精度
    def improve_precision(self, spline_dic=None, precision_smoothing: float = 0.1) -> Dict[str, DataFrame]:
        """
        将曲线实例化高精度的 DataFrame 数据
        Instantiate the curve as a high-precision DataFrame data.

        STEP THREE： (Recommended parameter Settings for this method:)
                  precision_smoothing = 0.1

        :param spline_dic: (scipy) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 scipy 曲线
        :param precision_smoothing: (float) 高精度曲线的精度，默认精度为 0.1

        :return smoothing_dic：(dict) 包含高精度曲线表格的 dict，key 为特殊点的类型，value 为对应的高精度 DataFrame
        例如：smoothing_dic = {'title1': DataFrame1, 'title2': DataFrame2}
        """

        if spline_dic is not None:
            spline_dic = copy.deepcopy(spline_dic)
        else:
            spline_dic = copy.deepcopy(self.spline_smoothing_dic)

        self.precision_smoothing = precision_smoothing
        smoothing_dic = {}

        # 遍历 dict 以完成所有数据的处理
        for title, spline in spline_dic.items():
            # 获取插值曲线的节点位置
            knots = spline.get_knots()
            first_knot = float(knots[0])  # 获取第一个节点的数值
            last_knot = float(knots[-1])  # 获取最后一个节点的数值
            # 创建等间距的 X 轴坐标
            x_smoothing = np.linspace(first_knot, last_knot, int((last_knot - first_knot) / precision_smoothing) + 1)
            y_smoothing = spline(x_smoothing)

            # 创建 DataFrame 表格
            smoothing_df = pd.DataFrame({self.x_label: x_smoothing, self.y_label: y_smoothing})
            # 以行坐标排序并重新设置行索引
            smoothing_df = smoothing_df.sort_values(by=smoothing_df.columns[0])
            smoothing_df.reset_index(drop=True, inplace=True)
            smoothing_dic[title] = smoothing_df

        self.smoothing_dic = smoothing_dic
        self.current_dic = 'smoothing_dic'

        return smoothing_dic

    # 4 低精度
    def reduce_precision(self, data_dic: Optional[Dict[str, DataFrame]] = None,
                         interval_disperse: Optional[float] = 0.5) -> Dict[str, DataFrame]:
        """
        将原始数据进行处理，删除部分数据使得横坐标间隔大于等于指定的间隔
        Process the raw data by removing some data points to ensure that the
        horizontal coordinate interval is greater than or equal to the specified interval.

        STEP FORE： (Recommended parameter Settings for this method:)
                  interval_disperse = 0.5

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param interval_disperse: (float) 横坐标的间隔，删除数据使得横坐标间隔大于等于该值，为 None 时不删除，默认为 0.5

        :return dispersed_dic: (dict) 曲线的 dict，其中 key 为数据的 title，value 为对应的低精度 DataFrame
        例如：dispersed_dic = {'title1': DataFrame1, 'title2': DataFrame2}
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        self.interval_disperse = interval_disperse
        dispersed_dic = {}

        # interval_disperse 不为 None 时进行删除坐标
        if interval_disperse is not None:
            # 遍历 dict 以完成所有数据的处理
            for title, data_df in data_dic.items():

                data_df = copy.deepcopy(data_df)

                # 删除 self.smoothing_df 中的部分数据，使得横坐标间隔等于或大于 interval_disperse
                last_x = data_df.iloc[0, 0]  # 上一个横坐标值
                for _, row in data_df.iterrows():
                    current_x = row.iloc[0]
                    if current_x - last_x >= interval_disperse:
                        last_x = current_x
                    else:
                        data_df = data_df.drop(index=row.name)

                # 以行坐标排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[0])
                data_df.reset_index(drop=True, inplace=True)
                dispersed_dic[title] = data_df

        # interval_disperse 为 None 时不删除
        else:
            dispersed_dic = data_dic

        self.dispersed_dic = dispersed_dic
        self.current_dic = 'dispersed_dic'

        return dispersed_dic

    # 5 标准化
    def normalize_data(self, data_dic: Optional[Dict[str, DataFrame]] = None, key_point_dic: Optional[dict] = None,
                       x_min_limit: Optional[float] = None, x_max_limit: Optional[float] = None,
                       y_min_limit: Optional[float] = None, y_max_limit: Optional[float] = None,
                       normalize_key: bool = True) \
            -> Tuple[Dict[str, DataFrame], Optional[Dict[str, Optional[Dict[str, DataFrame]]]]]:
        """
        对数据进行标准化
        Normalize the data.

        STEP FIVE： (Recommended parameter Settings for this method:)
                 point_original = None, x_normal = 1, y_normal = 1

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param key_point_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为一个包含特殊点的 dict
        :param x_min_limit: (float) 横坐标标准化的目标值
        :param x_max_limit: (float) 横坐标标准化的目标值
        :param y_min_limit: (float) 横坐标标准化的目标值
        :param y_max_limit: (float) 纵坐标标准化的目标值
        :param normalize_key: (bool) 是否标准化特殊点 key，默认为 True

        :return normalized_dic: (dic) 标准化后的 dic，其中 key 为数据的 title，value 为标准后的 DataFrame
        :return key_normalized_dic: (dic) 特殊点 key 标准化后的 dic，其中 key 为数据的 title，
                                          value 为特殊点 key 标准后的 DataFrame
        例如：dispersed_dic = {'title1': DataFrame1, 'title2': DataFrame2}
             key_normalized_dic = {'title1': {'extremum': DataFrame1, 'inflection': DataFrame2,
                                                            {'max': DataFrame3, 'min': None},
                                            'title2': {'extremum': DataFrame4, 'inflection': DataFrame5,
                                                            {'max': DataFrame6, 'min': None} } 此为无 min 情况
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 将需要处理的 key_point 赋给 key_point_dic
        if key_point_dic is not None:
            key_point_dic = copy.deepcopy(key_point_dic)
        else:
            key_point_dic = copy.deepcopy(self.key_point_dic)

        point_current_scale_dic = {}
        normalized_dic = {}
        normalize_rule_dic = {}
        normal_dic = {}

        # 遍历 dict 以完成所有数据的处理
        for title, data_df in data_dic.items():

            normal_single_dic = {}
            point_current_scale = {}
            data_df = copy.deepcopy(data_df)

            # 寻找最大值和最小值点，不能用 self.point_scale_dic，因为通过调整精度会改变最大最小值
            x_min = data_df.iloc[:, 0].min()
            x_max = data_df.iloc[:, 0].max()
            y_min = data_df.iloc[:, 1].min()
            y_max = data_df.iloc[:, 1].max()

            # 存入 point_current_scale 中
            point_current_scale['x_min'] = x_min
            point_current_scale['x_max'] = x_max
            point_current_scale['y_min'] = y_min
            point_current_scale['y_max'] = y_max

            # 将每个 title 下的 point_current_scale 存入 point_current_scale_dic
            point_current_scale_dic[title] = point_current_scale

            # coordinate_normal
            if x_min_limit is not None:
                x_min_normal = x_min_limit
            else:
                x_min_normal = x_min
            normal_single_dic['x_min'] = x_min_normal

            if x_max_limit is not None:
                x_max_normal = x_max_limit
            else:
                x_max_normal = x_max
            normal_single_dic['x_max'] = x_max_normal

            if y_min_limit is not None:
                y_min_normal = y_min_limit
            else:
                y_min_normal = y_min
            normal_single_dic['y_min'] = y_min_normal

            if y_max_limit is not None:
                y_max_normal = y_max_limit
            else:
                y_max_normal = y_max
            normal_single_dic['y_max'] = y_max_normal

            normal_dic[title] = normal_single_dic

            # 计算变化规则
            x_normalize_rule = (x_max_normal - x_min_normal) / (x_max - x_min)
            y_normalize_rule = (y_max_normal - y_min_normal) / (y_max - y_min)

            # 将标准化规则加入到 dict 中
            normalize_rule_dic[title] = {
                'x_rule': x_normalize_rule,
                'y_rule': y_normalize_rule}

            # 应用变换规则，得到标准化后的数据
            data_df.iloc[:, 0] = (data_df.iloc[:, 0] - x_min) * x_normalize_rule + x_min_normal
            data_df.iloc[:, 1] = (data_df.iloc[:, 1] - y_min) * y_normalize_rule + y_min_normal
            # 以行坐标排序并重新设置行索引
            data_df = data_df.sort_values(by=data_df.columns[0])
            data_df.reset_index(drop=True, inplace=True)
            normalized_dic[title] = data_df

        self.normalize_rule_dic = normalize_rule_dic
        self.normalized_dic = normalized_dic
        self.current_dic = 'normalized_dic'

        key_normalized_dic = {}

        # 标准化特殊点 key 时
        if normalize_key:

            # 遍历 dict 以完成所有数据的处理
            for title, key_point in key_point_dic.items():

                key_normalized_single_dic = {}
                x_min_normal = normal_dic[title]['x_min']
                y_min_normal = normal_dic[title]['y_min']
                x_normalize_rule = normalize_rule_dic[title]['x_rule']
                y_normalize_rule = normalize_rule_dic[title]['y_rule']
                x_min = point_current_scale_dic[title]['x_min']
                y_min = point_current_scale_dic[title]['y_min']

                # extremum 进行标准化
                if key_point['extremum'] is not None:
                    extremum_normalized_df = key_point['extremum'].copy()
                    # 应用变换规则，得到标准化后的数据
                    extremum_normalized_df.iloc[:, 0] = \
                        (extremum_normalized_df.iloc[:, 0] - x_min) * x_normalize_rule + x_min_normal
                    extremum_normalized_df.iloc[:, 1] = \
                        (extremum_normalized_df.iloc[:, 1] - y_min) * y_normalize_rule + y_min_normal

                    # 以行坐标排序并重新设置行索引
                    extremum_normalized_df = extremum_normalized_df.sort_values(by=extremum_normalized_df.columns[0])
                    extremum_normalized_df.reset_index(drop=True, inplace=True)
                    key_normalized_single_dic['extremum'] = extremum_normalized_df

                # extremum 不存在的情况
                else:
                    key_normalized_single_dic['extremum'] = None

                # inflection 进行标准化
                if key_point['inflection'] is not None:
                    inflection_normalized_df = key_point['inflection'].copy()
                    # 应用变换规则，得到标准化后的数据
                    inflection_normalized_df.iloc[:, 0] = \
                        (inflection_normalized_df.iloc[:, 0] - x_min) * x_normalize_rule + x_min_normal
                    inflection_normalized_df.iloc[:, 1] = \
                        (inflection_normalized_df.iloc[:, 1] - y_min) * y_normalize_rule + y_min_normal

                    # 以行坐标排序并重新设置行索引
                    inflection_normalized_df = \
                        inflection_normalized_df.sort_values(by=inflection_normalized_df.columns[0])
                    inflection_normalized_df.reset_index(drop=True, inplace=True)
                    key_normalized_single_dic['inflection'] = inflection_normalized_df

                # inflection 不存在的情况
                else:
                    key_normalized_single_dic['inflection'] = None

                # max 进行标准化
                if key_point['max'] is not None:
                    max_normalized_df = key_point['max'].copy()
                    # 应用变换规则，得到标准化后的数据
                    max_normalized_df.iloc[:, 0] = \
                        (max_normalized_df.iloc[:, 0] - x_min) * x_normalize_rule + x_min_normal
                    max_normalized_df.iloc[:, 1] = \
                        (max_normalized_df.iloc[:, 1] - y_min) * y_normalize_rule + y_min_normal

                    # 以行坐标排序并重新设置行索引
                    max_normalized_df = max_normalized_df.sort_values(by=max_normalized_df.columns[0])
                    max_normalized_df.reset_index(drop=True, inplace=True)
                    key_normalized_single_dic['max'] = max_normalized_df

                # max 不存在的情况
                else:
                    key_normalized_single_dic['max'] = None

                # min 进行标准化
                if key_point['min'] is not None:
                    min_normalized_df = key_point['min'].copy()
                    # 应用变换规则，得到标准化后的数据
                    min_normalized_df.iloc[:, 0] = \
                        (min_normalized_df.iloc[:, 0] - x_min) * x_normalize_rule + x_min_normal
                    min_normalized_df.iloc[:, 1] = \
                        (min_normalized_df.iloc[:, 1] - y_min) * y_normalize_rule + y_min_normal

                    # 以行坐标排序并重新设置行索引
                    min_normalized_df = min_normalized_df.sort_values(by=min_normalized_df.columns[0])
                    min_normalized_df.reset_index(drop=True, inplace=True)
                    key_normalized_single_dic['min'] = min_normalized_df

                # min 不存在的情况
                else:
                    key_normalized_single_dic['min'] = None

                key_normalized_dic[title] = key_normalized_single_dic

        # 不标准化特殊点 key 时
        else:
            key_normalized_dic = None

        self.key_normalized_dic = key_normalized_dic

        return normalized_dic, key_normalized_dic

    # 6 放缩
    def adjust_data(self, data_dic: Optional[Dict[str, DataFrame]] = None, key_point_dic: Optional[dict] = None,
                    point_original: Optional[Tuple[Optional[float], Optional[float]]] = None,
                    locate_point_original: bool = True,
                    point_target: Optional[Tuple[Optional[float], Optional[float]]] = None, adjust_key: [bool] = True) \
            -> Tuple[Dict[str, DataFrame], Optional[Dict[str, Optional[Dict[str, DataFrame]]]]]:
        """
        对数据按原点为中心进行放缩
        Scale the data.

        STEP SIX： (Recommended parameter Settings for this method:)
                 All parameters are subject to conditions.

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param key_point_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为一个包含特殊点的 dict
        :param point_original: (float) 放缩的标准点，与 point_target 中允许 1 个元素值为 None，默认为 Y 最大值点对应的坐标
        :param locate_point_original: (bool) 是否开起寻点，默认为 True
        :param point_target: (tuple) 需要放缩到的目标坐标，与 point_original 中允许 1 个元素值为 None，默认为 Y 最大值点对应的坐标
        :param adjust_key: (bool) 是否放缩特殊点 key，默认为 True

        :return adjusted_dic: (dic) 放缩后的 dic，其中 key 为数据的 title，value 为放缩后的 DataFrame
        :return key_adjusted_dic: (dic) 特殊点 key 放缩后的 dic，其中 key 为数据的 title，value 为特殊点 key 放缩后的 DataFrame
        例如：adjusted_dic = {'title1': DataFrame1, 'title2': DataFrame2}
             key_adjusted_dic = {'title1': {'extremum': DataFrame1, 'inflection': DataFrame2,
                                                            {'max': DataFrame3, 'min': None},
                                 'title2': {'extremum': DataFrame4, 'inflection': DataFrame5,
                                                            {'max': DataFrame6, 'min': None} } 此为无 min 情况
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        adjusted_dic = {}
        adjust_rule_dic = {}

        # 将需要处理的 key_point 赋给 key_point_dic
        if key_point_dic is not None:
            key_point_dic = copy.deepcopy(key_point_dic)
        else:
            key_point_dic = copy.deepcopy(self.key_normalized_dic)

        # 遍历 dict 以完成所有数据的处理
        for title, data_df in data_dic.items():

            data_df = copy.deepcopy(data_df)

            # point_original
            if point_original is not None:
                coordinate_original = point_original
                # 寻点
                if locate_point_original:
                    min_dist = np.inf
                    closest_point = None
                    for _, row in data_df.iterrows():
                        # 计算当前数据点与自定义坐标的距离
                        dist = np.sqrt((row.iloc[0] - coordinate_original[0]) ** 2 +
                                       (row.iloc[1] - coordinate_original[1]) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = tuple(row)

                    coordinate_original = closest_point

            # 原始数据
            else:
                # 获取第二列中最大值所在的行索引
                max_index = data_df.iloc[:, 1].idxmax()
                # 根据索引提取最大值所在的那一行
                coordinate_original = data_df.loc[max_index]

            # point_target
            if point_target is not None:
                coordinate_target = point_target
            else:
                # 获取第二列中最大值所在的行索引
                max_index = data_df.iloc[:, 1].idxmax()
                # 根据索引提取最大值所在的那一行
                coordinate_target = data_df.loc[max_index]

            # 得到变化前后的坐标
            x_original, y_original = coordinate_original
            x_target, y_target = coordinate_target

            # 计算放缩比例，包括了可能含有 None 的情况
            if x_original is None:
                x_adjust_rule = y_target / y_original
                y_adjust_rule = y_target / y_original
            elif y_original is None:
                x_adjust_rule = x_target / x_original
                y_adjust_rule = x_target / x_original
            elif x_target is None:
                x_adjust_rule = y_target / y_original
                y_adjust_rule = y_target / y_original
            elif y_target is None:
                x_adjust_rule = x_target / x_original
                y_adjust_rule = x_target / x_original
            elif x_original is not None and y_original is not None:
                x_adjust_rule = x_target / x_original
                y_adjust_rule = y_target / y_original
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The values of point_original or point_target are incorrect.")

            # 将放缩规则加入到 dict 中
            adjust_rule_dic[title] = {
                'x_rule': x_adjust_rule,
                'y_rule': y_adjust_rule}

            # 应用放缩比例，对 DataFrame 进行放缩
            data_df.iloc[:, 0] *= x_adjust_rule
            data_df.iloc[:, 1] *= y_adjust_rule
            # 以行坐标排序并重新设置行索引
            data_df = data_df.sort_values(by=data_df.columns[0])
            data_df.reset_index(drop=True, inplace=True)
            adjusted_dic[title] = data_df

        self.adjust_rule_dic = adjust_rule_dic
        self.adjusted_dic = adjusted_dic
        self.current_dic = 'adjusted_dic'

        key_adjusted_dic = {}

        # 放缩特殊点 key 的情况
        if adjust_key:
            # 遍历 dict 以完成所有数据的处理
            for title, point_original in key_point_dic.items():

                key_adjusted_single_dic = {}
                x_adjust_rule = adjust_rule_dic[title]['x_rule']
                y_adjust_rule = adjust_rule_dic[title]['y_rule']

                # extremum 放缩
                if point_original['extremum'] is not None:
                    extremum_adjusted_df = point_original['extremum'].copy()
                    # 应用放缩比例，对 extremum 进行放缩
                    extremum_adjusted_df.iloc[:, 0] *= x_adjust_rule
                    extremum_adjusted_df.iloc[:, 1] *= y_adjust_rule
                    # 以行坐标排序并重新设置行索引
                    extremum_adjusted_df = extremum_adjusted_df.sort_values(by=extremum_adjusted_df.columns[0])
                    extremum_adjusted_df.reset_index(drop=True, inplace=True)
                    key_adjusted_single_dic['extremum'] = extremum_adjusted_df
                # extremum 不存在的情况
                else:
                    key_adjusted_single_dic['extremum'] = None

                # inflection 放缩
                if point_original['inflection'] is not None:
                    inflection_adjusted_df = point_original['inflection'].copy()
                    # 应用放缩比例，对 inflection 进行放缩
                    inflection_adjusted_df.iloc[:, 0] *= x_adjust_rule
                    inflection_adjusted_df.iloc[:, 1] *= y_adjust_rule
                    # 以行坐标排序并重新设置行索引
                    inflection_adjusted_df = inflection_adjusted_df.sort_values(by=inflection_adjusted_df.columns[0])
                    inflection_adjusted_df.reset_index(drop=True, inplace=True)
                    key_adjusted_single_dic['inflection'] = inflection_adjusted_df
                # inflection 不存在的情况
                else:
                    key_adjusted_single_dic['inflection'] = None

                # max 放缩
                if point_original['max'] is not None:
                    max_adjusted_df = point_original['max'].copy()
                    # 应用放缩比例，对 max 进行放缩
                    max_adjusted_df.iloc[:, 0] *= x_adjust_rule
                    max_adjusted_df.iloc[:, 1] *= y_adjust_rule
                    # 以行坐标排序并重新设置行索引
                    max_adjusted_df = max_adjusted_df.sort_values(by=max_adjusted_df.columns[0])
                    max_adjusted_df.reset_index(drop=True, inplace=True)
                    key_adjusted_single_dic['max'] = max_adjusted_df
                # max 不存在的情况
                else:
                    key_adjusted_single_dic['max'] = None

                # min 放缩
                if point_original['min'] is not None:
                    min_adjusted_df = point_original['min'].copy()
                    # 应用放缩比例，对 min 进行放缩
                    min_adjusted_df.iloc[:, 0] *= x_adjust_rule
                    min_adjusted_df.iloc[:, 1] *= y_adjust_rule
                    # 以行坐标排序并重新设置行索引
                    min_adjusted_df = min_adjusted_df.sort_values(by=min_adjusted_df.columns[0])
                    min_adjusted_df.reset_index(drop=True, inplace=True)
                    key_adjusted_single_dic['min'] = min_adjusted_df
                # max 不存在的情况
                else:
                    key_adjusted_single_dic['min'] = None

                key_adjusted_dic[title] = key_adjusted_single_dic

        # 不放缩特殊点 key 的情况
        else:
            key_adjusted_dic = None

        self.key_adjusted_dic = key_adjusted_dic

        return adjusted_dic, key_adjusted_dic

    # 7 加权
    def assign_weight(self, data_dic: Optional[Dict[str, DataFrame]] = None, add_key_dic: bool = True,
                      key_point_dic: Optional[dict] = None, common_weight: float = 1, extremum_weight: float = 1.5,
                      inflection_weight: float = 1.2, max_weight: float = 2, min_weight: float = 2,
                      assign_key: bool = True) -> Tuple[Dict[str, DataFrame], Optional[Dict[str, list]]]:
        """
        将所有特殊点与普通点整合到一个 DataFrame 表格中，并根据输入的参数计算权重
        Integrate all special points and normal points into one DataFrame table,
        and calculate weights based on the input parameters.

        STEP SEVEN： (Recommended parameter Settings for this method:)
                    point_weight = 1, extremum_weight = 1.5, inflection_weight = 1.2,
                    max_weight = 2, min_weight = 2

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param add_key_dic: (bool) 是否添加权重点，以下参数只有当 add_key_dic is True 时才有意义，默认为 True
        :param key_point_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为一个包含特殊点的 dict
        :param common_weight: (float) 普通点的权重
        :param extremum_weight:(float) 极值点的权重
        :param inflection_weight: (float) 拐点的权重
        :param max_weight: (float) 最大值点的权重
        :param min_weight: (float) 最小值点的权重
        :param assign_key: (bool) 是否赋予特殊点 key 以权重

        :return balanced_dic: (dict) 包含所有坐标的 DataFrame 组成的 dict，key 为 title，value 为整合后的 DataFrame
        :return weight_dic: (dict) 权重的列表的 dict，key 为 title，value 为权重的 list
        例如：balanced_dic = {'title1': DataFrame1, 'title2': DataFrame2}
             weight_list_dic = {'title1': weight_list1, 'title2': weight_list2}
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 检查是否需要加权
        if add_key_dic:
            # 将需要处理的 key_point 赋给 key_point_dic
            if key_point_dic is not None:
                key_point_dic = copy.deepcopy(key_point_dic)
            else:
                key_point_dic = copy.deepcopy(self.key_adjusted_dic)
        else:
            balanced_dic = data_dic  # 两个属性均不变
            weight_list_dic = self.weight_list_dic
            self.weight_list_dic = weight_list_dic
            self.balanced_dic = balanced_dic
            self.current_dic = 'balanced_dic'

            return balanced_dic, weight_list_dic

        balanced_dic = {}
        weight_list_dic = {}

        # 有特殊点 key 的情况
        if assign_key:

            # 将输入的参数录入类属性
            weight_dic = {'common': common_weight,
                          'extremum': extremum_weight,
                          'inflection': inflection_weight,
                          'max': max_weight,
                          'min': min_weight}
            self.weight_dic = weight_dic

            # 遍历 dict 以完成所有数据的处理
            for title, data_df in data_dic.items():

                # 根据非空条件进行合并
                if key_point_dic[title]['extremum'] is not None:
                    data_df = pd.concat([data_df, key_point_dic[title]['extremum']]).drop_duplicates()
                if key_point_dic[title]['inflection'] is not None:
                    data_df = pd.concat([data_df, key_point_dic[title]['inflection']]).drop_duplicates()
                if key_point_dic[title]['max'] is not None:
                    data_df = pd.concat([data_df, key_point_dic[title]['max']]).drop_duplicates()
                if key_point_dic[title]['min'] is not None:
                    data_df = pd.concat([data_df, key_point_dic[title]['min']]).drop_duplicates()

                # 以行坐标排序并重新设置行索引
                data_df = data_df.sort_values(by=data_df.columns[0])
                data_df = data_df.reset_index(drop=True)
                balanced_dic[title] = data_df

                # 创建一个长度与 DataFrame 相同且全为 common_weight 的列表
                weight_list = [common_weight] * len(data_df)

                # 遍历列表 weight_list 赋值
                for index, value in enumerate(weight_list):
                    # extremum 加权
                    if key_point_dic[title]['extremum'] is not None:
                        if data_df.loc[index, self.x_label] in key_point_dic[title]['extremum'][self.x_label].values:
                            if value == common_weight:
                                weight_list[index] = extremum_weight
                            else:
                                weight_list[index] = max(value, extremum_weight)

                    # inflection 加权
                    if key_point_dic[title]['inflection'] is not None:
                        if data_df.loc[index, self.x_label] in \
                                key_point_dic[title]['inflection'][self.x_label].values:
                            if value == common_weight:
                                weight_list[index] = inflection_weight
                            else:
                                weight_list[index] = max(value, inflection_weight)

                    # max 加权
                    if key_point_dic[title]['max'] is not None:
                        if data_df.loc[index, self.x_label] in key_point_dic[title]['max'][self.x_label].values:
                            if value == common_weight:
                                weight_list[index] = max_weight
                            else:
                                weight_list[index] = max(value, max_weight)

                    # min 加权
                    if key_point_dic[title]['min'] is not None:
                        if data_df.loc[index, self.x_label] in key_point_dic[title]['min'][self.x_label].values:
                            if value == common_weight:
                                weight_list[index] = min_weight
                            else:
                                weight_list[index] = max(value, min_weight)

                weight_list_dic[title] = weight_list

        # 无特殊点 key 的情况
        else:
            # 遍历 dict 以完成所有数据的处理
            for title, data_df in data_dic.items():
                # 创建一个长度与 DataFrame 相同且全为 common_weight 的列表
                weight_list = [common_weight] * len(data_df)
                weight_list_dic[title] = weight_list
                balanced_dic = copy.deepcopy(data_dic)

        self.weight_list_dic = weight_list_dic
        self.balanced_dic = balanced_dic
        self.current_dic = 'balanced_dic'

        return balanced_dic, weight_list_dic

    # 8 拟合曲线
    def fit_curve(self, data_dic: Optional[Dict[str, DataFrame]] = None, degree_fitting: int = 4,
                  smooth_fitting: Optional[float] = 0.05, add_weight: bool = True,
                  weight_dic: Optional[Dict[str, list]] = None) -> Dict[str, any]:
        """
        使用 UnivariateSpline 进行曲线拟合，所用精度为原数据精度
        Perform curve fitting using UnivariateSpline with the accuracy of the original data.

        STEP EIGHT： (Recommended parameter Settings for curve smoothing:)
                    degree = 4, smoothing = 0.05, weight = None

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param degree_fitting: (int) 曲线阶数，越大越平滑，1 <= degree <= 5，推荐为 4
        :param smooth_fitting: (float) 平滑因子，越小越接近原数据， 推荐越小越好:0, 0.05
        :param add_weight: (bool) 是否添加权重，默认为 True
        :param weight_dic: (dict) 权重参数的 dict，若被赋值则会用来加权，只有当 add_weight is True 时才有效

        :return spline_fitting_dic: (dict) 曲线的 dict ，其中 key 为数据的 title，value 为曲线的 scipy
        例如：spline_fitting_dic = {'title1': scipy1, 'title2': scipy2}
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        # 没有外加权重的情况下，用 self.assign_weight_list
        if weight_dic is not None:
            weight_dic = weight_dic
        else:
            weight_dic = copy.deepcopy(self.weight_list_dic)

        spline_fitting_dic = {}

        # 遍历 dict 以完成所有数据的处理
        for title, data_df in data_dic.items():

            # 检查是否添加权重
            if add_weight:
                weight_list = weight_dic[title]
            else:
                weight_list = None

            # 获取 DataFrame 中的 X 和 Y 列数据
            x_section = data_df.iloc[:, 0].values
            y_section = data_df.iloc[:, 1].values

            # 检查 x_section 是否有重复值
            unique_x_section, counts = np.unique(x_section, return_counts=True)
            duplicates = unique_x_section[counts > 1]

            if len(duplicates) > 0:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"x_section contains duplicate values: {duplicates}")

            # 添加检查权重列表的长度是否与数据量一致
            if weight_list is not None and len(weight_list) != len(x_section):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The length of weight_list for '{title}' ({len(weight_list)}) does not match "
                                 f"the data length ({len(x_section)}).")

            # 使用 UnivariateSpline 进行曲线拟合
            spline_fitting = UnivariateSpline(x_section, y_section, k=degree_fitting, s=smooth_fitting, w=weight_list)
            spline_fitting_dic[title] = spline_fitting

        self.spline_fitting_dic = spline_fitting_dic
        self.current_dic = 'spline_fitting_dic'

        return spline_fitting_dic

    # 9 还原精度
    def restore_precision(self, spline_dic=None, precision_fitting: Optional[float] = None) -> Dict[str, DataFrame]:
        """
        将曲线实例化为原本精度的 DataFrame 数据
        Instantiate the curve as DataFrame data with the original precision.

        STEP NINE： (Recommended parameter Settings for this method:)
                   precision_fitting = None

        :param spline_dic: (scipy) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 scipy 曲线
        :param precision_fitting: (float) 曲线的精度，若为 None 则用原曲线的精度，默认为None

        :return fitting_dic: (scipy) 包含原精度曲线表格的 dict，key 为特殊点的类型，value 为对应的高精度 DataFrame
        例如：fitting_dic = {'title1': DataFrame1, 'title2': DataFrame2}
        """

        if spline_dic is not None:
            spline_dic = spline_dic
        else:
            spline_dic = copy.deepcopy(self.spline_fitting_dic)

        fitting_dic = {}
        precision_data_dic = copy.deepcopy(self.precision_data_dic)

        # 遍历 dict 以完成所有数据的处理
        for title, spline in spline_dic.items():

            # 获取插值曲线的节点位置
            knots = spline.get_knots()
            first_knot = float(knots[0])  # 获取第一个节点的数值
            last_knot = float(knots[-1])  # 获取最后一个节点的数值

            # precision_fitting
            if precision_fitting is not None:
                precision = precision_fitting
            else:
                precision = precision_data_dic[title]

            # 创建等间距的 X 轴坐标
            x_fitting = np.linspace(first_knot, last_knot, int((last_knot - first_knot) / precision))
            y_fitting = spline(x_fitting)

            # 创建 DataFrame 表格
            fitting_df = pd.DataFrame({self.x_label: x_fitting, self.y_label: y_fitting})
            # 以行坐标排序并重新设置行索引
            fitting_df = fitting_df.sort_values(by=fitting_df.columns[0])
            fitting_df.reset_index(drop=True, inplace=True)
            fitting_dic[title] = fitting_df

        self.fitting_dic = fitting_dic
        self.current_dic = 'fitting_dic'

        return fitting_dic

    # 10 真实化
    def realize_data(self, data_dic: Optional[Dict[str, DataFrame]] = None, noise_level: float = 0,
                     protected_column: Optional[int] = 0, interval_realize: int = 0) -> Dict[str, DataFrame]:
        """
        给数据添加噪声，使得数据更加真实
        Add noise to the data to make it more realistic.

        STEP TEN： (Recommended parameter Settings for curve smooth_fitting:)
                  noise_level = 0.01, protected_column =0, interval_disperse = 1, weight = None

        :param data_dic: (dict) 若被赋值，则将会对该 dict 进行操作，其中 key 为数据的 title，value 为数据的 DataFrame 表格
        :param noise_level: (float) 噪声等级，越小越接近原数据，此项为正态分布中的标准差，默认为 0
        :param protected_column: (int) 需要保护的列索引，默认为 0，表示保护第一列
        :param interval_realize: (int) 噪声点的间隔（默认为 0，表示每个点都加噪声）

        :return realized_df: (DataFrame) 真实化后的DataFrame数据
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            # 使用 getattr 来动态获取属性
            data_dic = copy.deepcopy(getattr(self, self.current_dic))

        realized_dic = {}
        column_names = [self.x_label, self.y_label]

        # 遍历 dict 以完成所有数据的处理
        for title, data_df in data_dic.items():

            for i, column_name in enumerate(column_names):
                if protected_column is not None and i == protected_column:
                    # 如果指定了保护列，并且当前列是保护列，则不添加噪声
                    continue

                data_column_data = data_df[column_name].values
                noise = np.random.normal(loc=0, scale=noise_level, size=len(data_column_data))  # 均值，标准差，数量
                if interval_realize > 0:
                    for j in range(len(data_column_data)):
                        if (j + 1) % interval_realize == 0:
                            data_column_data[j] += noise[j]
                else:
                    data_column_data += noise
                data_df[column_name] = data_column_data
            realized_dic[title] = data_df

        self.realized_dic = realized_dic
        self.current_dic = 'realized_dic'

        return realized_dic
