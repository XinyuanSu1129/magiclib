"""
magiclib / potter

------------------------------------------------------------------------------------------------------------------------
magiclib / potter is a high-level interface for managing and analyzing cultural heritage datasets stored in PotteryBase.
It provides structured access to sites, methods, and associated data files, supporting a wide range of formats
including TXT, Excel, JSON, and images. With built-in visualization tools, researchers can quickly plot and interpret
experimental results such as spectra or compositional data. The module also enables editing and retrieval of site
information, integrating data access, visualization, and documentation into a unified workflow tailored for
archaeological and material science studies.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""


from . import general

import os
import re
import time
import copy
import json
import shutil
import inspect
import chardet
import numpy as np
import pandas as pd
import seaborn as sns
from io import StringIO
from pandas import DataFrame
from datetime import datetime
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler
from typing import Union, Optional, List, Dict, Callable, Tuple


""" 陶器基因库 """
class PotteryBase(general.Manager):
    """
    陶器属性库

    Manipulation of pottery data. Allows writing data, reading data, and drawing with data, etc.
    """

    Pottery_Database = general.Pottery_Database
    Standard_Template = '0_standard'
    Category_Index = 'site'

    # 初始化
    def __init__(self,

                 # 接收参数 (7)
                 data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None, title: Optional[str] = None,
                 x_list: Optional[list] = None, y_list: Optional[list] = None, x_label: str = None, y_label: str = None,

                 # 关键参数 (7)
                 txt_path: Optional[str] = None, excel_path: Optional[str] = None, json_path: Optional[str] = None,
                 keyword: Optional[str] = None, file_pattern: Optional[str] = None, save_path: Optional[str] = None,
                 magic_database: Optional[str] = None,

                 # 特有参数 (2)
                 site: Optional[str] = None, method: Optional[str] = None, database: Optional[str] = None,
                 file: Optional[str] = None
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

        # 特有参数的初始化
        self.site = site  # 遗址
        self.method = method  # 测试数据的方法
        if database is not None:  # 数据路径的路径 (只有此项为全路径)
            self.database = database
        else:
            self.database = PotteryBase.Pottery_Database
        self.file = file  # 文件的名称
        self.temporary_data = None  # 临时数据

        # site_info.txt
        self.site_site = None
        self.site_time = None
        self.site_information = None

        # method_info.txt
        self.method_site = None
        self.method_method = None
        self.method_time = None
        self.method_keyword = None
        self.method_type = None
        self.method_information = None

        # 包含文件
        self.include_txt = None
        self.include_excel = None
        self.include_json = None
        self.include_image = None

        # 数据初始化分配
        if type(self) == PotteryBase:  # 当 self 为 PotteryBase 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # general.Manager 及其子类需要调用以初始化属性

        # 标准库的路径
        self.standard_template_path = os.path.join(PotteryBase.Pottery_Database, PotteryBase.Standard_Template)

    # 应用默认测试方法模板到其它所有地区中
    def apply_standard_template(self) -> list[str]:
        """
        检查所有库中是否已有标准库的目录，如果有则跳过，如果没有则创建一个新的空目录。
        同时复制标准模板中以 '_info' 结尾的 txt 文件到新创建的目录，并更新其中的 site 和 time 信息
        Check all libraries to see if there is a standard library directory, if there is, skip it,
        if not, create a new empty directory.
        Copy the txt file ending in '_info' from the standard template to the newly created directory,
        and update the site and time information in it.

        :return standard_directories: (list) 标准库的标准方法
        """

        # 1. 检查标准模板路径是否存在
        if not os.path.exists(self.standard_template_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"standard template directory not found: {self.standard_template_path}")

        # 2. 获取标准模板路径下的所有子目录
        standard_directories = [d for d in os.listdir(self.standard_template_path) if  # 记录下标准模板下的所有文件夹名
                                os.path.isdir(os.path.join(self.standard_template_path, d))]

        # 3. 遍历陶器数据库路径下的所有子目录，排除 'standard_template' 目录本身
        for folder_name in os.listdir(self.Pottery_Database):
            site_path = os.path.join(self.Pottery_Database, folder_name)

            # 确保处理的对象是文件夹且不包括 'standard_template' 本身
            if os.path.isdir(site_path) and folder_name != PotteryBase.Standard_Template:
                missing_directories = []  # 存储当前遗址目录缺失的标准目录

                # 检查当前子目录是否包含标准模板中的所有目录
                for directory in standard_directories:
                    target_dir = os.path.join(site_path, directory)

                    # 如果当前遗址目录缺少某个标准目录，则记录并创建该目录
                    if not os.path.exists(target_dir):
                        missing_directories.append(directory)
                        os.makedirs(target_dir)  # 创建缺失的目录

                        # 复制标准模板中以'_info'结尾的txt文件，并更新site和time
                        source_path = os.path.join(self.standard_template_path, directory)
                        for file_name in os.listdir(source_path):
                            if file_name.endswith('_info.txt'):
                                source_file = os.path.join(source_path, file_name)
                                target_file = os.path.join(target_dir, file_name)

                                # 读取源文件内容
                                with open(source_file, 'r') as f:
                                    content = f.readlines()

                                # 处理时间信息
                                time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                updated_content = []
                                for line in content:
                                    if line.startswith("Site: "):
                                        updated_content.append(f"Site: {folder_name}\n")
                                    elif line.startswith("Time:"):
                                        updated_content.append(f"Time: {time_now}\n")
                                    else:
                                        updated_content.append(line)

                                # 将修改后的内容写入目标文件
                                with open(target_file, 'w') as f:
                                    f.writelines(updated_content)

                        print(f"Creating missing method \033[34m{directory}\033[0m in \033[31m{folder_name}\033[0m.")

                # 如果所有标准目录都存在，则打印提示信息
                if not missing_directories:
                    print(f"All standard method(s) are present in \033[31m{folder_name}\033[0m.")
                else:
                    # 如果缺少某些标准目录，则打印哪些目录是缺失的
                    print(f"Missing method(s) have been supplemented in \033[31m{folder_name}\033[0m: "
                          f"\033[34m{', '.join(missing_directories)}\033[0m.\n")

        # 4. 打印标准模板中有哪些属性 (文件夹)
        print(f"\nCurrent standard library \033[95m{PotteryBase.Standard_Template}\033[0m "
              f"contains the following method(s):")
        for directory in standard_directories:
            print(f"- \033[34m{directory}\033[0m")
        print('')

        return standard_directories

    # 检查地区与测试方法与数据是否存在
    def whether_data_exists(self, site: Optional[str] = None, method: Optional[str] = None,
                            database: Optional[str] = None) -> None:
        """
        用于检查所选的地区与测试数据是否存在，不存在则直接报错
        This command is used to check whether the selected region and test data exist.
        If no, an error is reported.

        :param site: (str) 读取测试数据所在地区的名称
        :param method: (str) 读取的测试数据
        :param database: (str) 数据库所在位置

        :return: None
        """

        # 判断 site 是否有输入
        if site is None:
            if self.site is not None:
                site = self.site
            else:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, no available site value.")

        # 判断 method 是否有输入
        if method is None:
            if self.method is not None:
                method = self.method
            else:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, no available method value.")

        # 检查 database 是否被赋值
        if database is None:
            database = self.database

        # 检查 database 目录是否存在
        if not os.path.exists(database):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"database directory '{database}' does not exist.")

        # 判断 site 是否在数据库中
        site_tuple = self.get_subdirectories(folder_path=database)
        if site not in site_tuple:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"site '{site}' not found in database. Available sites: {[s for s in site_tuple]}.")

        # 检查 site 目录下的 method 目录是否存在
        target_site_dir = os.path.join(database, site)
        method_tuple = self.get_subdirectories(folder_path=target_site_dir)
        if method not in method_tuple:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"method '{method}' not found in site '{site}'. "
                             f"Available methods: {[m for m in method_tuple]}.")

        return None

    # 添加新的地区
    def add_site(self, site_name: Optional[str] = None, site_info: Optional[str] = None, apply_standard: bool = True) \
            -> str:
        """
        在库中添加新的遗址地区，同时还会自动创建信息文本
        New site areas are added to the library while informational text is automatically created.

        :param site_name: (str) 新遗址的名称，为必输入项
        :param site_info: (str) 新遗址的相关信息，默认为 None
        :param apply_standard: (str) 是否应用标准库的测试方法，默认为 True

        :return new_site: (str) 新遗址的路径
        """

        # 创建新的目录
        if site_name is not None:
            new_site = os.path.join(self.Pottery_Database, site_name)

            if os.path.isdir(new_site):  # 如果目录已经存在
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"new sites already exist: {site_name}")

            else:
                os.makedirs(new_site)
                self.site = new_site
                print(f"New site has been established in the database: \033[31m{site_name}\033[0m.")

                # 在新目录内创建一个txt文件，记录site信息
                txt_file_path = os.path.join(new_site, f"{site_name}_info.txt")
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(f"Site: {site_name}\n")
                    txt_file.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if site_info:
                        txt_file.write(f"Information: \n{site_info}\n")
                    else:
                        txt_file.write("Information:\n")

        else:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, the site value is required.")

        # 是否复制标准到新的遗址地区中
        if apply_standard:
            # 记录所有复制的文件夹路径
            copied_folders = []

            # 列出 standard_template_path 目录下的所有内容
            for folder_name in os.listdir(self.standard_template_path):
                source_folder = os.path.join(self.standard_template_path, folder_name)  # 源文件夹路径
                destination_folder = os.path.join(new_site, folder_name)  # 目标文件夹路径

                # 检查是否是文件夹，如果是则执行复制操作
                if os.path.isdir(source_folder):

                    # 递归复制文件夹，dirs_exist_ok=True 允许目标文件夹已存在
                    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)

                    # 更新目标文件夹的时间戳为当前时间
                    current_time = datetime.now()
                    os.utime(destination_folder, (current_time.timestamp(), current_time.timestamp()))

                    # 查找目标文件夹内所有以 '_info.txt' 结尾的文件
                    for root, dirs, files in os.walk(destination_folder):
                        for file in files:
                            if file.endswith('_info.txt'):
                                file_path = os.path.join(root, file)

                                # 读取文件内容并替换 Time 和 Site 字段
                                with open(file_path, 'r') as f:
                                    content = f.readlines()

                                # 遍历内容，找到 "Time:" 和 "Site:" 并更新
                                for i, line in enumerate(content):
                                    if line.startswith("Site:"):
                                        content[i] = f"Site: {os.path.basename(new_site)}\n"  # 更新为目标文件夹名称
                                    if line.startswith("Time:"):
                                        # 更新为当前时间
                                        content[i] = f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"

                                # 写回更新后的内容到文件中
                                with open(file_path, 'w') as f:
                                    f.writelines(content)

                    # 记录已复制的文件夹名称
                    copied_folders.append(folder_name)

            # 所有复制操作完成后，打印已复制的文件夹名称
            if copied_folders:
                print(f"The following folders have been copied to \033[31m{site_name}\033[0m:")
                for folder_name in copied_folders:
                    print(f"- \033[34m{folder_name}\033[0m")
                print('')

        return new_site

    # 添加新的测试数据到地区
    def add_method(self, method_name: Union[str, List[str], None] = None, to_site: Union[str, List[str], None] = None,
                   keyword: Optional[str] = None, add_to_standard: bool = False, txt_type: bool = False,
                   excel_type: bool = False, json_type: bool = False, image_type: bool = False,
                   method_info: Optional[str] = None) -> None:
        """
        添加新的测试数据到一个或多个地区，同时也可以选择是否添加到标准中。会自动创建信息文本，但信息文本的 Information 和 Type 相同，
        因此如果创建的测试数据的信息文本不一至时需要逐个创建
        Add new test data to one or more regions and optionally add it to the standard.
        The message text is automatically created, but the Information and Type of the message text are the same.
        Therefore, if the information text of the created test data is different, you need to create one by one.

        :param method_name: (str / list) 添加的测试数据的名称，可以单个或多个
        :param to_site: (str / list) 添加到哪些地区中，可以单个或多个
        :param keyword: (str) 数据保存的关键词，与 general.Manager.keyword 一致
        :param add_to_standard: (bool) 是否添加到标准中，默认为 False
        :param txt_type: (bool) 是否创建 txt 类型文件
        :param excel_type: (bool) 是否创建 excel 类型文件
        :param json_type: (bool) 是否创建 json 类型文件
        :param image_type: (bool) 是否创建 image 类型文件
        :param method_info: (Optional[str]) 关于测试数据的附加信息

        :return: None
        """

        # 检查 method_name 是否为 None，为 None 报错
        if method_name is None:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                             f"'method_name' cannot be None.")

        # 如果为 list，检查是否全为字符串
        if isinstance(method_name, list):
            if not all(isinstance(item, str) for item in method_name):
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                 f"all elements of 'to_site' must be strings.")

        # 检查 to_site 和 add_to_standard 是否同时为 None，如果是，报错
        if to_site is None and not add_to_standard:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                             f"'to_site' and 'add_to_standard' cannot both be None.")

        # 判断 site 是否在数据库中
        site_tuple = self.get_subdirectories(folder_path=self.database)
        if isinstance(to_site, str):
            if to_site not in site_tuple:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"site '{to_site}' not found in database. Available sites: {[s for s in site_tuple]}.")

        if isinstance(to_site, list):
            for site in to_site:
                if site not in site_tuple:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"site '{site}' not found in database. "
                                     f"Available sites: {[s for s in site_tuple]}.")

        # 检查 to_site 为 list 时，是否全为字符串
        if isinstance(to_site, list):
            if not all(isinstance(site, str) for site in to_site):
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                 f"all elements of 'to_site' must be strings.")

        # 如果 to_site 是 str，在 to_site 目录中创建 method_name 的空文件夹
        if isinstance(to_site, str):
            for method in ([method_name] if isinstance(method_name, str) else method_name):
                site_path = os.path.join(self.database, to_site, method)

                # 检查文件夹是否已经存在
                if os.path.exists(site_path):
                    print(f"The method \033[34m{method}\033[0m already exists in \033[31m{to_site}\033[0m.")
                else:
                    os.makedirs(site_path, exist_ok=True)
                    print(f"Created method \033[34m{method}\033[0m in \033[31m{to_site}\033[0m.")

                    # 生成文件内容
                    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    types = []
                    if txt_type:
                        types.append('txt')
                    if excel_type:
                        types.append('excel')
                    if json_type:
                        types.append('json')
                    if image_type:
                        types.append('image')

                    # 修改后的拼接类型字符串逻辑
                    if len(types) > 1:
                        type_str = ', '.join(types[:-1]) + ' & ' + types[-1]
                    else:
                        type_str = types[0] if types else 'None'

                    content = (f"Site: {to_site}\n"
                               f"Method: {method}\n"
                               f"Time: {time_now}\n"
                               f"Keyword: {keyword if keyword else 'Not enter'}\n"
                               f"Type: {type_str if type_str else 'None'}\n"
                               f"Information:\n{method_info or ''}")

                    # 写入文件到新创建的文件夹中
                    record_path = os.path.join(site_path, f'{method}_info.txt')
                    with open(record_path, 'w') as f:
                        f.write(content)

        # 如果 to_site 是 list，在所有 to_site 目录中创建 method_name 的空文件夹
        elif isinstance(to_site, list):
            missing_sites = [site for site in to_site if not os.path.exists(os.path.join(self.database, site))]
            if missing_sites:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"these site(s) do not exist: {', '.join(missing_sites)}")

            for site in to_site:
                for method in ([method_name] if isinstance(method_name, str) else method_name):
                    site_path = os.path.join(self.database, site, method)
                    if os.path.exists(site_path):
                        print(f"The method \033[34m{method}\033[0m already exists in \033[31m{site}\033[0m.")
                    else:
                        os.makedirs(site_path, exist_ok=True)
                        print(f"Created method \033[34m{method}\033[0m in \033[31m{site}\033[0m.")

                        # 生成文件内容
                        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        types = []
                        if txt_type:
                            types.append('txt')
                        if excel_type:
                            types.append('excel')
                        if json_type:
                            types.append('json')
                        if image_type:
                            types.append('image')

                        # 修改后的拼接类型字符串逻辑
                        if len(types) > 1:
                            type_str = ', '.join(types[:-1]) + ' & ' + types[-1]
                        else:
                            type_str = types[0] if types else 'None'

                        content = (f"Site: {site}\n"
                                   f"Method: {method}\n"
                                   f"Time: {time_now}\n"
                                   f"Keyword: {keyword if keyword else 'Not enter'}\n"
                                   f"Type: {type_str if type_str else 'None'}\n"
                                   f"Information:\n{method_info or ''}")

                        # 写入文件到新创建的文件夹中
                        record_path = os.path.join(site_path, f'{method}_info.txt')
                        with open(record_path, 'w') as f:
                            f.write(content)

        # 如果 add_to_standard 为 True，在标准中创建 method_name 的空文件夹
        if add_to_standard:
            for method in ([method_name] if isinstance(method_name, str) else method_name):
                standard_path = os.path.join(self.standard_template_path, method)
                if os.path.exists(standard_path):
                    print(f"The method \033[34m{method}\033[0m already exists "
                          f"in standard library \033[95m{PotteryBase.Standard_Template}\033[0m.")
                else:
                    os.makedirs(standard_path, exist_ok=True)
                    print(f"Created method \033[34m{method}\033[0m "
                          f"in standard library \033[95m{PotteryBase.Standard_Template}\033[0m.")

                    # 生成文件内容
                    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    types = []
                    if txt_type:
                        types.append('txt')
                    if excel_type:
                        types.append('excel')
                    if json_type:
                        types.append('json')
                    if image_type:
                        types.append('image')

                    # 修改后的拼接类型字符串逻辑
                    if len(types) > 1:
                        type_str = ', '.join(types[:-1]) + ' & ' + types[-1]
                    else:
                        type_str = types[0] if types else 'None'

                    content = (f"Site: {PotteryBase.Standard_Template}\n"
                               f"Method: {method}\n"
                               f"Time: {time_now}\n"
                               f"Keyword: {keyword if keyword else 'Not enter'}\n"
                               f"Type: {type_str if type_str else 'None'}\n"
                               f"Information:\n{method_info or ''}")

                    # 写入文件到新创建的文件夹中
                    record_path = os.path.join(standard_path, f'{method}_info.txt')
                    with open(record_path, 'w') as f:
                        f.write(content)

        return None

    # 修改地区的介绍信息
    def update_info_of_site(self, site: Optional[str] = None, new_information: Optional[str] = None) -> str:
        """
        更新指定的信息文件中的 Information: 后的内容，不会更改时间等其它信息。
        Update the Information in the specified information file: after the content,
        does not change the time and other information.

        :param site: (str) 需要修改信息的地区
        :param new_information: (str) 新的信息内容，如果为 None，则不更改现有信息。

        :return new_information: (str) 更新后的信息或保持不变的原信息。
        """

        # 判断 site 是否有输入
        if site is None:
            if self.site is not None:
                site = self.site
            else:
                class_name = self.__class__.__name__
                method_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, no available site value.")

        # 判断 site 是否在数据库中
        site_tuple = self.get_subdirectories(folder_path=self.database)
        if isinstance(site, str):
            if site not in site_tuple:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"site '{site}' not found in database. Available sites: {[s for s in site_tuple]}.")

        # 构建文件路径
        site_path = os.path.join(self.Pottery_Database, site)
        if not os.path.isdir(site_path):  # 检查 site 的路径是否正确
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"the path of site '{site}' is invalid.")

        info_path = os.path.join(site_path, f'{site}_info.txt')  # 构建 info 文件的路径
        updated_lines = []  # 创建一个新列表来存储修改后的内容

        # 读取文件内容
        with open(info_path, 'r') as file:
            lines = file.readlines()
            information_line_found = False

            for line in lines:
                if line.startswith("Site: "):
                    updated_lines.append(line)  # 保留原站点名称
                elif line.startswith("Time: "):
                    updated_lines.append(line)  # 保持时间不变
                elif line.startswith("Information:"):
                    updated_lines.append(line)  # 保留 Information: 行
                    information_line_found = True
                    # 如果有新信息且 new_information 不为 None，则替换信息
                    if new_information is not None:
                        updated_lines.append(new_information + "\n")
                elif information_line_found and new_information is not None:
                    # 如果已经找到 Information: 行，且 new_information 不为 None，则跳过原有信息
                    continue
                else:
                    updated_lines.append(line)

            # 如果 Information: 行未找到，且 new_information 不为 None，则添加新的 Information 部分
            if not information_line_found and new_information is not None:
                updated_lines.append("\nInformation:\n")
                updated_lines.append(new_information + "\n")

        # 写入修改后的内容到文件
        with open(info_path, 'w') as file:
            file.writelines(updated_lines)

        # 根据是否更新了信息返回不同的消息
        if new_information is not None:
            print(f"The \033[31m{site}\033[0m's information has been modified as follows:\n"
                  f"\033[33m{''.join(updated_lines)}\033[0m")  # 使用 ''.join(updated_lines) 打印文件的内容并保持格式
        else:
            print(f"No new information provided, and the \033[31m{site}\033[0m's '_info' file  remains unchanged.")

        return new_information

    # 修改测试数据的介绍信息
    def update_info_of_method(self, site: Optional[str] = None, method: Optional[str] = None,
                              new_information: Optional[str] = None, new_keyword: Optional[str] = None,
                              new_type: Optional[str] = None) -> Tuple[str, str]:
        """
        更新指定方法文件中的 Information: 后的内容，不会更改其他信息
        Update the Information in the specified method file: after the content,
        does not change other information.

        :param site: (str) 需要修改信息的地区
        :param method: (str) 需要修改信息的方法
        :param new_information: (str) 新的信息内容
        :param new_keyword: (str) 新的关键词
        :param new_type: (str) 新的文件类型内容，支持 txt, excel, json & image

        :return all_new: (tuple) 所有新的内容，新类型在前，介绍在后
        """

        # 判断 site 和 method 是否有输入
        if site is None or method is None:
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(
                f"\033[95mIn {method_name} of {class_name}\033[0m, both 'site' and 'method' must be provided.")

        # 判断 site 是否在数据库中
        site_tuple = self.get_subdirectories(folder_path=self.database)
        if isinstance(site, str):
            if site not in site_tuple:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"site '{site}' not found in database. Available sites: {[s for s in site_tuple]}.")

        # 判断 method 是否在数据库中
        target_site_dir = os.path.join(self.database, site)
        method_tuple = self.get_subdirectories(folder_path=target_site_dir)
        if method not in method_tuple:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"method '{method}' not found in site '{site}'. "
                             f"Available methods: {[m for m in method_tuple]}.")

        # 构建文件夹路径
        site_path = os.path.join(self.Pottery_Database, site)
        if not os.path.isdir(site_path):  # 检查 site 的路径是否正确
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, the path of site '{site}' is invalid.")

        # 检查 method 的路径是否正确
        method_path = os.path.join(site_path, method)
        if not os.path.isdir(method_path):
            class_name = self.__class__.__name__
            method_name = inspect.currentframe().f_code.co_name
            raise ValueError(
                f"\033[95mIn {method_name} of {class_name}\033[0m, the path of method '{method}' is invalid.")

        # 检查 type 输入的内容是否包含必要信息
        if new_type is not None:
            # 使用正则表达式分隔字符串，提取单词并转换为小写
            allowed_types = {"txt", "excel", "json", "image"}
            words = re.split(r'\W+', new_type.lower())
            found_words = set(words)

            # 检查找到的词是否完全匹配允许的词
            if not found_words.issubset(allowed_types):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(
                    f"\033[95mIn {method_name} of {class_name}\033[0m, "
                    f"The new_type should only contain one or more of the following types: "
                    f"'txt', 'excel', 'json', 'image'."
                )

        # 信息文件的路径
        info_path = os.path.join(method_path, f"{method}_info.txt")
        updated_lines = []  # 创建一个新列表来存储修改后的内容

        # 读取文件内容
        with open(info_path, 'r') as file:
            lines = file.readlines()
            information_line_found = False

            for line in lines:
                if line.startswith("Site: "):
                    updated_lines.append(line)  # 保留原站点名称
                elif line.startswith("Method: "):
                    updated_lines.append(line)  # 保留原方法名称
                elif line.startswith("Time: "):
                    updated_lines.append(line)  # 保持时间不变
                elif line.startswith("Keyword: "):
                    # 如果有新的类型信息且 new_keyword 不为 None，则替换类型
                    if new_keyword is not None:
                        updated_lines.append(f"Keyword: {new_keyword}\n")
                    else:
                        updated_lines.append(line)  # 保持类型不变
                elif line.startswith("Type: "):
                    # 如果有新的类型信息且 new_type 不为 None，则替换类型
                    if new_type is not None:
                        updated_lines.append(f"Type: {new_type}\n")
                    else:
                        updated_lines.append(line)  # 保持类型不变
                elif line.startswith("Information:"):
                    updated_lines.append(line)  # 保留 Information: 行
                    information_line_found = True
                    # 如果有新信息且 new_information 不为 None，则替换信息
                    if new_information is not None:
                        updated_lines.append(new_information + "\n")
                elif information_line_found and new_information is not None:
                    # 如果已经找到 Information: 行，且 new_information 不为 None，则跳过原有信息
                    continue
                else:
                    updated_lines.append(line)

            # 如果 Information: 行未找到，且 new_information 不为 None，则添加新的 Information 部分
            if not information_line_found and new_information is not None:
                updated_lines.append("\nInformation:\n")
                updated_lines.append(new_information + "\n")

        # 写入修改后的内容到文件
        with open(info_path, 'w') as file:
            file.writelines(updated_lines)

        # 打印更新信息
        if new_information is not None or new_type is not None:
            print(f"The \033[31m{site}\033[0m's method \033[34m{method}\033[0m information "
                  f"has been updated as follows:\n"
                  f"\033[33m{''.join(updated_lines)}\033[0m")  # 使用 ''.join(updated_lines) 打印文件的内容并保持格式

        else:
            print(f"No new information or type provided for \033[31m{site}\033[0m's method \033[34m{method}\033[0m, "
                  f"and the '_info' file remains unchanged.")

        all_new = (new_type, new_information)

        return all_new

    # 以 TXT 的形式保存数据至数据库
    def save_data_to_txt(self, site: Optional[str] = None, method: Optional[str] = None,
                         data_dic: Optional[Dict[str, DataFrame]] = None, database: Optional[str] = None,
                         save_both: bool = False, delimiter: Optional[str] = None,
                         float_precision: Optional[int] = None) -> Dict[str, any]:
        """
        将数据以 TXT 的形式入库存储。自动检查是否已储存过，若已储存，则会跳过该组数据
        The data is stored in the repository as TXT. Automatically checks if it has been saved,
        and if it has, the data is skipped.

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param data_dic: (dict)  key 为 title， value 数据的 DataFrame，默认为 smoothing_dic 数据优先
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param save_both: (bool) 如果发生新存文件与原文件同位置且同名，如果 save_both 为 True 则加后缀保存，否则不保存
        :param delimiter: (str) 列分隔符，默认为三个空格
        :param float_precision: (int) 小数精度，默认为 2

        :return txt_dic: (dict) dict 中 key 为 title， value 为 TXT 格式数据，即储存的数据
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        # 检查 database 是否被赋值
        if database is None:
            database = self.database

        # 到这里表示 site 和 method 都存在，准备保存数据
        target_site_dir = os.path.join(database, site)
        target_method_dir = os.path.join(target_site_dir, method)

        # 将需要处理的数据赋给 data_dic
        if data_dic is None:
            data_dic = copy.deepcopy(self.data_dic)

        # 如果没有输入 delimiter 参数，使用默认的三个空格
        if delimiter is None:
            delimiter = '   '

        # 如果没有输入 float_precision 参数，默认保留两位小数
        if float_precision is None:
            float_precision = 2

        # 用于存储生成的 TXT 文件路径
        txt_dic = {}
        for title, data_df in data_dic.items():
            # 创建 data_df 的副本以防止对原数据进行修改
            data_df = copy.deepcopy(data_df)

            # 获取当前时间并格式化为字符串
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 生成初始文件名，文件名格式为 'title.txt'
            file_name = f"{title}.txt"
            # 生成完整的文件路径
            save_path = os.path.join(target_method_dir, file_name)

            # 检查文件是否已存在
            if os.path.exists(save_path):
                if save_both:
                    # 如果文件已存在且 save_both 为 True，则生成新的文件名
                    count = 1
                    while os.path.exists(save_path):
                        file_name = f"{title}_{count}.txt"
                        save_path = os.path.join(target_method_dir, file_name)
                        count += 1
                else:
                    # 如果文件已存在且 save_both 为 False，打印提示并跳过保存
                    print(f"In site \033[31m{site}\033[0m, method \033[34m{method}\033[0m, "
                          f"file \033[35m{file_name}\033[0m already exists.")
                    continue

            # 打开一个文件用于写入
            with open(save_path, 'w') as file:
                # 写入文件的基本信息，包括站点、方法和当前时间
                file.write(f'Siti: {site}\n')  # 将站点信息写入文件
                file.write(f'Method: {method}\n')  # 将方法信息写入文件
                file.write(f'Time: {current_time}\n\n')  # 写入当前时间并添加换行

                # 写入列标题，列之间用 delimiter 分隔
                header_row = delimiter.join([str(col) for col in data_df.columns])
                file.write(f'Columns: {header_row}\n')  # 写入列标题

                # 遍历 DataFrame 中的每一行数据
                for index, row in data_df.iterrows():
                    row_data = []
                    # 遍历每一行中的每个元素
                    for i, item in enumerate(row):
                        # 如果元素是浮点数，保留指定的小数位数
                        formatted_item = f'{item:.{float_precision}f}' if isinstance(item, float) else str(item)
                        row_data.append(formatted_item)  # 将处理后的元素添加到行数据列表中
                        # 如果不是最后一列，根据列的奇偶性添加分隔符
                        if i < len(row) - 1:
                            row_data.append(delimiter if i % 2 == 0 else delimiter + delimiter)
                    # 将处理后的行数据写入文件
                    file.write(''.join(row_data) + '\n')

            # 保存生成的文件路径到 txt_dic 字典中
            txt_dic[title] = save_path
            # 打印保存路径信息，先提示 site 和 method，然后再输出完整的保存路径
            print(
                f"In site \033[31m{site}\033[0m, method \033[34m{method}\033[0m, "
                f"saved TXT file: \033[36m{file_name}\033[0m.")

        # 返回包含所有生成的 TXT 文件路径的字典
        return txt_dic

    # 以 EXCEL 的形式保存数据至数据库
    def save_data_to_excel(self, site: Optional[str] = None, method: Optional[str] = None,
                           database: Optional[str] = None, data_dic: Optional[Dict[str, DataFrame]] = None) -> None:
        """
        处理目标 site 的 method 中的 Excel 文件，如果不存在则生成一个，如果存在多个则报错
        Process the Excel file in the method of the target site, generate one file if none exists,
        and report an error if multiple files exist.

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param data_dic: (dict)  key 为 title， value 数据的 DataFrame，默认为 smoothing_dic 数据优先
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置

        :return: None
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        # 检查 database 是否被赋值
        if database is None:
            database = self.database

        # 到这里表示 site 和 method 都存在，准备保存数据
        target_site_dir = os.path.join(database, site)
        target_method_dir = os.path.join(target_site_dir, method)

        # 将需要处理的数据赋给 data_dic
        if data_dic is None:
            data_dic = copy.deepcopy(self.data_dic)

        # 找到与 method 同名的 Excel 文件路径
        excel_file_path = os.path.join(target_method_dir, f'{method}.xlsx')

        # 构建新的 DataFrame 的表头 (假设 data_dic 中每个 value 是 DataFrame 类型)
        new_data = pd.concat(data_dic.values(), axis=1)  # 组合所有 DataFrame 列

        if not os.path.exists(excel_file_path):
            # 如果没有 Excel 文件，创建一个新的
            new_data.to_excel(excel_file_path, index=False)
            print(
                f"New data has been saved to \033[36m{method}.xlsx\033[0m "
                f"in site \033[31m{site}\033[0m and method \033[34m{method}\033[0m.")
        else:
            # 如果 Excel 文件已存在，加载并更新数据
            existing_data = pd.read_excel(excel_file_path)

            # 检查并添加缺失的列
            for column in new_data.columns:
                if column not in existing_data.columns:
                    existing_data[column] = None  # 添加新列，初始值为 None

            # 将新数据添加到旧表格
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)

            # 保存更新后的数据
            updated_data.to_excel(excel_file_path, index=False)
            print(f"New data has been saved to \033[32m{method}.xlsx\033[0m "
                  f"in site \033[31m{site}\033[0m and method \033[34m{method}\033[0m.")

        return None

    # 以 JSON 的形式保存数据至数据库
    def save_data_to_json(self, site: Optional[str] = None, method: Optional[str] = None,
                          data_dic: Optional[Dict[str, DataFrame]] = None, database: Optional[str] = None,
                          save_both: bool = False) -> Dict[str, any]:
        """
        将数据以 JSON 的形式入库存储。自动检查是否已储存过，若已储存，则会跳过该组数据
        The data is stored in the repository as JSON. Automatically checks if it has been saved,
        and if it has, the data is skipped.

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param data_dic: (dict)  key 为 title， value 数据的 DataFrame，默认为 smoothing_dic 数据优先
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param save_both: (bool) 如果发生新存文件与原文件同位置且同名，如果 save_both 为 True 则加后缀保存，否则不保存

        :return json_dic: (dict) dict 中 key 为 title， value 为 JSON 格式数据，即储存的数据
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        # 检查 database 是否被赋值
        if database is None:
            database = self.database

        # 到这里表示 site 和 method 都存在，准备保存数据
        target_site_dir = os.path.join(database, site)
        target_method_dir = os.path.join(target_site_dir, method)

        # 将需要处理的数据赋给 data_dic
        if data_dic is None:
            data_dic = copy.deepcopy(self.data_dic)

        # 数据处理
        title_list = []
        sampled_before_interval_dic = {}

        for title, data_df in data_dic.items():
            data_df = copy.deepcopy(data_df)
            title_list.append(title)

            # 以行坐标排序并重新设置行索引
            data_df = data_df.sort_values(by=data_df.columns[0])
            data_df.reset_index(drop=True, inplace=True)
            sampled_before_interval_dic[title] = data_df

        self.title_tuple = tuple(title_list)

        # 将数据储存为 JSON 格式
        json_dic = {}
        for title in title_list:
            saved_sampled_single_dic = {}

            # 获取相应的数据
            saved_data_df = copy.deepcopy(sampled_before_interval_dic[title])

            # 将 title 转化为 JSON 格式
            title_json = title.replace("'", "\"")
            # 将 saved_data_df 转化为 JSON 格式
            saved_data_df_json = saved_data_df.to_json(orient='records')

            # 将数据保存在单独的 dict 中
            saved_sampled_single_dic["title"] = title_json
            saved_sampled_single_dic["data_df"] = saved_data_df_json

            # 将数据汇总在 json_dic 中
            json_dic[title] = json.dumps(saved_sampled_single_dic)

        self.sample_json_dic = json_dic

        # 保存数据至库中
        for title, sample_json in json_dic.items():
            file_name = f"{title}.json"  # 以 title 作为初始文件名
            save_path = os.path.join(target_method_dir, file_name)

            # 检查文件是否已存在
            if os.path.exists(save_path):
                if save_both:
                    count = 1
                    # 如果文件存在，按照 _1, _2 进行命名，从 _1 开始
                    while os.path.exists(save_path):
                        file_name = f"{title}_{count}.json"
                        save_path = os.path.join(target_method_dir, file_name)
                        count += 1
                else:
                    # 如果 save_both 为 False，打印提示并跳过保存
                    print(f"In site \033[31m{site}\033[0m, method \033[34m{method}\033[0m, "
                          f"file \033[35m{file_name}\033[0m already exists.")
                    continue

            # 将 JSON 字符串保存到文件中
            with open(save_path, 'w') as file:
                file.write(sample_json)

            # 打印保存路径信息，先提示 site 和 method，然后再输出完整的保存路径
            print(
                f"In site \033[31m{site}\033[0m, method \033[34m{method}\033[0m, "
                f"saved JSON file: \033[36m{file_name}\033[0m.")

        return json_dic

    # 读取库中 TXT 文件数据，并打印
    def read_data(self, site: Optional[str] = None, method: Optional[str] = None, file: Optional[str] = None,
                  database: Optional[str] = None, delimiter: Optional[str] = None,
                  columns_txt: Union[list, None] = None, file_pattern: Optional[str] = None, show_data: bool = True,
                  print_max_rows: Optional[int] = 10) -> Dict[str, DataFrame]:
        """
        在库中找到指定的地区和测试数据，存入 self.data_dic 中，并打印出来
        Find the specified locale and test data in the library, store it in self.data_dic, and print it out.

        :param site: (str) 读取测试数据所在地区的名称
        :param method: (str) 读取的测试数据
        :param file: (str) 文件的名称，默认为 None，表示其中的所有文件
        :param database: (str) 数据库所在位置
        :param delimiter: (str) 分割符，默认为一个或多个空格
        :param columns_txt: (list) 选取的列，默认为 [0, 1] 表示前两列
        :param file_pattern: (str) 所选文件的正则表达式筛选
        :param show_data: (bool) 是否展示数据，默认为 True
        :param print_max_rows: (bool) 单个 DataFrame 显示的最大行数，最大值为10，默认为 10 行

        :return data_dic: (dict) 返回该 TXT 文件的 dict，key 为文件名的测TXT的名称，value 为 DataFrame (该文件的有效数据)
                 如果初始化时为 TXT 文件路径，则返回的dict中仅有该文件一个数据的 key & value
                 如果初始化时为 TXT 文件的目录路径，则返回的dict中含有该目录下所有 TXT 文件数据的 key & value
        DataFrame: column1:       float64
                   column2:       float64
                   dtype:         object
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        if delimiter is not None:
            delimiter = delimiter
        elif self.delimiter is not None:
            delimiter = self.delimiter
        else:
            delimiter = r'\s+'

        if columns_txt is not None:
            columns_txt = columns_txt
        elif self.delimiter is not None:
            columns_txt = self.columns_txt
        else:
            columns_txt = [0, 1]

        if file_pattern is not None:
            file_pattern = file_pattern
        elif self.file_pattern is not None:
            file_pattern = self.file_pattern
        else:
            file_pattern = r'.*'

        site_path = os.path.join(self.database, site)  # site 的路径
        method_path = os.path.join(site_path, method)  # method 的路径

        # 读取其中一个文件
        data_dic = {}  # 创造一个空的 dict 以存储数据
        if file is not None:
            file_path = os.path.join(method_path, file)  # 文件的路径

            # 检查文件 file 是否存在 (有输入的情况下)
            if not os.path.exists(file_path):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The file \033[35m{file}\033[0m was not found "
                                 f"in the \033[34m{method}\033[0m of \033[31m{site}\033[0m.")

            # 检查是否为 TXT 文件
            if os.path.splitext(file)[1].lower() == ".txt":  # 获取文件扩展名并转换为小写
                # 如果是 txt 文件，执行相关操作
                pass
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"{file} is not a valid txt file.")

            # 检查 TXT 文件的编码类型
            with open(file_path, 'rb') as f:
                # 获取文件的编码类型
                result = chardet.detect(f.read())
                encoding = result['encoding']

            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()  # 读取 TXT 文件

            # 找到有效行
            pattern = re.compile(pattern=r'-?\d+(\.\d+)?\s+-?\d+(\.\d+)?')
            valid_lines = [line for line in lines if pattern.match(line)]  # 运用正则表达式找到以数字开头的行
            # 从有效行创建 DataFrame
            data_df = pd.DataFrame([re.split(delimiter, line.strip()) for line in valid_lines])

            # 只保留需要留下的列
            data_df = data_df.iloc[:, columns_txt]  # 默认为前两列

            # 返回该 TXT 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
            data_dic[re.match(pattern=r'(\.?\/.*\/)(?P<name>[^\/]+)\..*$',
                              string=file_path).group('name')] = data_df

        # 读取文件夹
        else:

            # 创造一个空的 dict 以存储数据
            data_dic = {}
            for file in os.listdir(method_path):

                file_path = os.path.join(method_path, file)  # 目录下所有的文件的路径

                # 忽略以 '_info' 结尾的文件
                if file.endswith('_info.txt') or file.endswith('_info.TXT'):
                    continue

                if re.search(file_pattern, file) and (file_path.endswith('.txt') or file_path.endswith('.TXT')):

                    # 检查 TXT 文件的编码类型
                    with open(file_path, 'rb') as f:
                        # 获取文件的编码类型
                        result = chardet.detect(f.read())
                        encoding = result['encoding']

                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()  # 读取 TXT 文件

                    # 找到有效行
                    pattern = re.compile(pattern=r'-?\d+(\.\d+)?\s+-?\d+(\.\d+)?')
                    valid_lines = [line for line in lines if pattern.match(line)]  # 运用正则表达式找到以数字开头的行
                    # 从有效行创建 DataFrame
                    data_df = pd.DataFrame([re.split(delimiter, line.strip()) for line in valid_lines])

                    # 只保留需要留下的列
                    data_df = data_df.iloc[:, columns_txt]  # 默认为前两列

                    # 返回该 TXT 文件的 dict，key 为文件名，value 为 DataFrame (该文件的有效数据)
                    data_dic[re.match(pattern=r'(\.?\/.*\/)(?P<name>[^\/]+)\..*$',
                                      string=file_path).group('name')] = data_df

        self.data_dic = data_dic  # 将得到的数据的 dict 传给 self.data_dic

        # 数据的初始化分配
        self.data_init()

        # 打印数据
        if show_data:
            self.to_reality(print_max_rows=print_max_rows)

        # 读取文件的信息
        self.read_information(site=site, method=method, database=database)

        return data_dic

    # 读取库中 JSON 文件数据，并打印
    def read_data_from_json(self, site: Optional[str] = None, method: Optional[str] = None,
                            file: Optional[str] = None, database: Optional[str] = None, show_data: bool = True,
                            print_max_rows: Optional[int] = 10) -> Dict[str, DataFrame]:
        """
        在选择的 site 和 method 中绘制图像，并可以选择保存图片与否
        Draws the image in the selected site and method, and can choose whether to save the image or not

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param file: (str) 文件的名称，默认为 None，表示其中的所有文件
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param show_data: (bool) 是否展示数据，默认为 True
        :param print_max_rows: (bool) 单个 DataFrame 显示的最大行数，最大值为10，默认为 10 行

        :return: None
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        site_path = os.path.join(self.database, site)  # site 的路径
        method_path = os.path.join(site_path, method)  # method 的路径

        data_dic = {}
        key_point_dic = {}
        sampled_dic = {}
        sample_single_dic = {}

        # 只读取一个文件
        if file is not None:

            json_path = os.path.join(method_path, file)

            # 是 JSON 文件的情况
            if json_path.endswith('.json') and os.path.isfile(json_path):
                with open(json_path, 'r', encoding='UTF-8') as file:
                    file_content = file.read()

                # 使用 json.loads() 解析文件内容为 dict
                sample = json.loads(file_content)

                # 获取 key 对应的 value 的 python 格式，而不是 JSON 格式
                title = sample.get('title', None)  # 获取 'title'，如果不存在，则为 None

                # 获取 'data_df'，如果不存在，则为 None
                data_df_json = sample.get('data_df', None)
                data_df = pd.read_json(StringIO(data_df_json)) if data_df_json else None

                # 获取 'key_dic'，如果不存在，则为 None
                key_dic_json = sample.get('key_dic', None)
                key_dic = {k: pd.read_json(StringIO(v)) if isinstance(v, str) else
                pd.read_json(StringIO(json.dumps(v))) for k, v in key_dic_json.items()} if key_dic_json else None

                # 获取 'rule_dic'，如果不存在，则为 None
                rule_dic_json = sample.get('rule_dic', None)
                rule_dic = {k: float(v) for k, v in rule_dic_json.items()} if rule_dic_json else None

                # 获取 'interval'，如果不存在，则为 None
                interval_json = sample.get('interval', None)
                interval = float(interval_json) if interval_json else None

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

            # 不是一个 JSON 文件
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"{file} is not a valid JSON file.")

        # 读取 site 中 method 下所有的 json 文件
        else:

            # 是 JSON 文件的情况
            for filename in os.listdir(method_path):

                file_path = os.path.join(method_path, filename)

                if filename.endswith('.json') and os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='UTF-8') as file:
                        file_content = file.read()

                    # 使用 json.loads() 解析文件内容为 dict
                    sample = json.loads(file_content)

                    # 获取 key 对应的 value 的 python 格式，而不是 JSON 格式
                    title = sample.get('title', None)  # 获取 'title'，如果不存在，则为 None

                    # 获取 'data_df'，如果不存在，则为 None
                    data_df_json = sample.get('data_df', None)
                    data_df = pd.read_json(StringIO(data_df_json)) if data_df_json else None

                    # 获取 'key_dic'，如果不存在，则为 None
                    key_dic_json = sample.get('key_dic', None)
                    key_dic = {k: pd.read_json(StringIO(v)) if isinstance(v, str) else
                    pd.read_json(StringIO(json.dumps(v))) for k, v in
                               key_dic_json.items()} if key_dic_json else None

                    # 获取 'rule_dic'，如果不存在，则为 None
                    rule_dic_json = sample.get('rule_dic', None)
                    rule_dic = {k: float(v) for k, v in rule_dic_json.items()} if rule_dic_json else None

                    # 获取 'interval'，如果不存在，则为 None
                    interval_json = sample.get('interval', None)
                    interval = float(interval_json) if interval_json else None

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

        self.data_dic = data_dic
        self.key_point_dic = key_point_dic
        self.sampled_dic = sampled_dic

        # 数据的初始化分配
        self.data_init()

        # 打印数据
        if show_data:
            self.to_reality(print_max_rows=print_max_rows)

        # 读取文件的信息
        self.read_information(site=site, method=method, database=database)

        return data_dic

    # 读取 '_info.txt' 文件信息
    def read_information(self, site: Optional[str] = None, method: Optional[str] = None,
                         database: Optional[str] = None) -> None:
        """
        在指定的地区 site 和 测试数据 method 打开信息文件
        Open the information file in the specified area site and test data method.

        :param site: (str) 读取测试数据所在地区的名称
        :param method: (str) 读取的测试数据
        :param database: (str) 数据库所在位置

        :return: None
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        site_path = os.path.join(self.database, site)  # site 的路径
        method_path = os.path.join(site_path, method)  # method 的路径

        # 读取 '_info.txt'
        file_path = os.path.join(method_path, f"{method}_info.txt")  # 文件的路径
        with open(file_path, 'rb') as f:
            # 获取文件的编码类型
            result = chardet.detect(f.read())
            encoding = result['encoding']

        # 初始化变量
        method_site = method_method = method_time = method_type = ""

        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

            # 标志位用于判断是否在处理 'information' 部分
            in_information = False
            method_information_lines = []

            for line in lines:
                # 去除行首尾的空白字符
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                # 如果已经进入 'information' 部分，继续添加后续行
                if in_information:
                    method_information_lines.append(line)
                    continue

                # 拆分 key 和 value
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    # 赋值给相应的变量
                    if key == 'site':
                        method_site = value
                    elif key == 'method':
                        method_method = value
                    elif key == 'time':
                        method_time = value
                    elif key == 'keyword':
                        method_keyword = value
                    elif key == 'type':
                        method_type = value
                    elif key == 'information':
                        # 开始处理 'information' 部分
                        in_information = True
                        method_information_lines.append(value)

            # 将所有 'information' 部分的内容合并成一个字符串
            method_information = '\n'.join(method_information_lines)

            self.method_site = method_site
            self.method_method = method_method
            self.method_time = method_time
            self.method_keyword = method_keyword
            self.method_type = method_type
            self.method_information = method_information

        return None

    # 添加临时数据
    def add_temporary_data(self, temporary_dict: Optional[dict] = None, data_dic: Optional[dict] = None) \
            -> Dict[str, DataFrame]:
        """
        将第一个 dict 数据转化为 temporary_dict。如果 data_dic 被赋值，那么将会在 data_dic 后拼接整合后的 temporary_dict。
        temporary_dict 可能为多值 dict，但 data_dic 与 result_data_dic 为长度为 1 的 dict
        Convert the first dict data to temporary_dict. If data_dic is assigned,
        the consolidated temporary_dict will be concatenated after data_dic.
        temporary_dict may be a multi-value dict, but data_dic and result_data_dic are dicts with length 1.

        :param temporary_dict: (dict) 需要转化的 dict 数据
        :param data_dic: (dict) 需要整合到的数据

        :return result_data_dic: (dict) 最终的 dict (临时 dict / 拼接后的 dict)
        """

        # 将需要处理的数据赋给 data_dic
        if temporary_dict is not None:
            temporary_dict = copy.deepcopy(temporary_dict)
        else:
            temporary_dict = copy.deepcopy(self.data_dic)

        # 需要拼接数据的情况
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)

            result_data_dic = {}  # 保存最终的数据字典
            all_dataframes = []  # 用于存储合并后的 DataFrame 列表

            # 获取原数据
            title = list(data_dic.keys())[0]
            data_df = list(data_dic.values())[0]

            # 添加 'site' 列到原数据框，值为原数据的标题
            if PotteryBase.Category_Index not in data_df.columns:
                data_df[PotteryBase.Category_Index] = title  # 使用原数据的键作为 site

            # 将原数据添加到合并列表
            all_dataframes.append(data_df)

            # 遍历 temporary_dict 中的所有键值对
            for temporary_title, temporary_data_df in temporary_dict.items():
                # 添加 'site' 列到临时数据框，值为当前的键
                temporary_data_df[PotteryBase.Category_Index] = temporary_title

                # 对齐临时数据框和原数据框的列，缺失的列填充为 0
                temporary_data_df = temporary_data_df.reindex(columns=data_df.columns, fill_value=0)

                # 将临时数据添加到合并列表
                all_dataframes.append(temporary_data_df)

            # 合并所有 DataFrame
            combined_df = pd.concat(all_dataframes, ignore_index=True)

            # 将合并后的 DataFrame 以 title 为 key 存入字典中
            result_data_dic[title] = combined_df

            # 更新 self.data_dic 为合并后的数据
            self.data_dic = result_data_dic

            print("The \033[95mtemporary data\033[0m has been integrated into the \033[35;2mtarget data\033[0m.\n")

        # 不拼接的情况
        else:
            result_data_dic = temporary_dict

        self.temporary_data = temporary_dict

        return result_data_dic

    # 读取所有地区指定的测试数据中的 Excel 文件
    def read_all_from_excel(self, method: Optional[str] = None, database: Optional[str] = None,
                            in_one: bool = True) -> Dict[str, DataFrame]:
        """
        读取 database 路径下所有文件夹中的 method.xlsx 文件，除去 '0_standard' 文件夹
        Read the method.xlsx file in all folders in the database path, except for the '0_standard' folder.

        :param method: (str) 目标 method 文件夹名
        :param database: (str) 数据库路径，默认为 self.pottery_database
        :param in_one: (bool) 如果为 True，合并所有 DataFrame，并加上 'site' 列。否则，分别保存

        :return data_dic: (dict) 读取所有 Excel 表格的字典
            data_dic: 一个字典，in_one 为 True 时 key 为 method，保存合并后的 DataFrame；
                      为 False 时，key 为 site，保存对应的 DataFrame。
        """

        # 检查 database 是否被赋值
        if database is None:
            database = self.database

        # 读取除 PotteryBase.Standard_Template 之外的所有文件夹
        folders = [f for f in os.listdir(database) if
                   os.path.isdir(os.path.join(database, f)) and f != PotteryBase.Standard_Template]

        data_dic = {}  # 保存最终的数据字典
        all_dataframes = []  # 用于存储合并后的 DataFrame 列表

        # 遍历每个文件夹
        for folder in folders:
            site_path = os.path.join(database, folder)
            method_path = os.path.join(site_path, method)

            # 检查 method 文件夹是否存在
            if not os.path.exists(method_path):
                # 如果 method 文件夹不存在，跳过该 site
                continue

            # 找到 method 文件夹中的 method.xlsx 文件
            excel_file_path = os.path.join(method_path, f"{method}.xlsx")
            if os.path.exists(excel_file_path):
                # 读取 Excel 文件
                df = pd.read_excel(excel_file_path)

                if in_one:
                    # 如果需要合并，添加 'site' 列并记录 site 名
                    df[PotteryBase.Category_Index] = folder
                    all_dataframes.append(df)

                    # 合并所有 DataFrame
                    combined_df = pd.concat(all_dataframes, ignore_index=True)
                    data_dic[method] = combined_df  # 将合并后的 DataFrame 以 method 为 key 存入字典中

                else:
                    # 否则将每个 site 对应的 DataFrame 保存在字典中，不添加 'site' 列
                    data_dic[folder] = df

        self.data_dic = data_dic

        return data_dic

    # 读取 txt 文件并绘制图像
    def plot_figure(self, site: Optional[str] = None, method: Optional[str] = None, file: Optional[str] = None,
                    database: Optional[str] = None, save_path: Union[bool, str] = True, **kwargs) -> None:
        """
        在选择的 site 和 method 中绘制图像，并可以选择保存图片与否
        Draws the image in the selected site and method, and can choose whether to save the image or not

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param file: (str) 文件的名称，默认为 None，表示其中的所有文件
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param save_path: (str) 图片的保存路径

        :return: None
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        # 数据读取
        self.read_data(site=site, method=method, file=file, show_data=False)

        # 找到需要绘制图像的类型
        if self.method_keyword is not None:
            self.keyword = self.method_keyword
            self.to_magic()
        elif self.keyword is not None:
            self.to_magic()

        # 图像绘制
        self.plot_line(save_path=save_path, **kwargs)

        # 打印详情
        if file is None:
            print(f"In the \033[34m{method}\033[0m data of \033[31m{site}\033[0m, "
                  f"the figure have been plotted.")
        else:
            print(f"In the \033[34m{method}\033[0m data of \033[31m{site}\033[0m, "
                  f"\033[36m{file}\033[0m have been plotted.")

        return None

    # 读取 txt 文件并绘制图像
    def plot_figure_from_json(self, site: Optional[str] = None, method: Optional[str] = None,
                              file: Optional[str] = None, database: Optional[str] = None,
                              save_path: Union[bool, str] = True, **kwargs) -> None:
        """
        在选择的 site 和 method 中绘制图像，并可以选择保存图片与否
        Draws the image in the selected site and method, and can choose whether to save the image or not

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param file: (str) 文件的名称，默认为 None，表示其中的所有文件
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param save_path: (str) 图片的保存路径

        :return: None
        """

        # 检查所选地区与测试数据是否存在
        self.whether_data_exists(site=site, method=method, database=database)

        # 数据读取
        self.read_data_from_json(site=site, method=method, file=file, show_data=False)

        # 找到需要绘制图像的类型
        if self.method_keyword is not None:
            self.keyword = self.method_keyword
            self.to_magic()
        elif self.keyword is not None:
            self.to_magic()

        # 图像绘制
        self.plot_line(save_path=save_path, **kwargs)

        # 打印详情
        if file is None:
            print(f"In the \033[34m{method}\033[0m data of \033[31m{site}\033[0m, "
                  f"the figure have been plotted.")
        else:
            print(f"In the \033[31m{method}\033[0m data of \033[34m{site}\033[0m, "
                  f"\033[36m{file}\033[0m have been plotted.")

        return None

    # 利用 Excel 数据进行 PCA 分析
    def plot_pca(self, site: Optional[str] = None, method: Optional[str] = None, database: Optional[str] = None,
                 use_temporary_dic: Optional[Dict[str, DataFrame]] = None, save_path: Union[bool, str] = True,
                 draw_ellipse: bool = True, std: float = 2, margin_ratio: float = 0.1, dpi: int = 600,
                 width_height: tuple = (6, 4.5), category: Optional[str] = None, colors: Optional[list] = None,
                 show_result: bool = False, show_legend: bool = True, **kwargs) -> Dict[str, DataFrame]:
        """
        对数据进行 PCA 分析，主要在元素分析使用
        PCA analysis of data, mainly used in elemental analysis.

        :param site: (str) 存储的实验数据的站点，默认为 self.site 中的站点
        :param method: (str) 存储的实验数据的分类方法
        :param database: (str) 数据库的位置，若未赋值则延用类属性中数据库的位置
        :param use_temporary_dic: (bool) 是否使用临时数据，默认为 False
        :param save_path: (str) 图片的保存路径
        :param draw_ellipse: (bool) 是否绘制置信椭圆
        :param std: (float) 置信椭圆的标准差范围，默认为 2，此时置信区间为 90%
        :param margin_ratio: (float) 数据距边界的比例，该值需要介于 0 至 1 之间，默认为 0.1
        :param dpi: (int) 图像保存和展示的精度
        :param width_height: (tuple) 图片的宽度和高度，默认为(6, 4.5)
        :param category: (str) 用于分类的列，默认为 Statistics.Category_Index
        :param colors: (str / list) 置信椭圆的填充颜色
        :param show_result: (bool) 是否打印结果，默认为 False
        :param show_legend: (bool) 是否显示图例，默认为 True
        :param kwargs: Ellipse 方法中的关键字参数

        :return pca_dic: (dict) PCA 分析后的数据 dict ，键为 title，值为 DataFrame 形式的 PCA 结果

        --- **kwargs ---

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值
        """

        # 数据准备工作
        if True:

            # 判断 method 是否有输入
            if method is None:
                if self.method is not None:
                    method = self.method
                else:
                    class_name = self.__class__.__name__
                    method_name = inspect.currentframe().f_code.co_name
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, no available method value.")

            # 检查 database 是否被赋值
            if database is None:
                database = self.database

            # 检查 database 目录是否存在
            if not os.path.exists(database):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"database directory '{database}' does not exist.")

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的py文件的目录下
            else:
                save_path = save_path

            # 用于分类的列
            if category is not None:
                category = category
            else:
                category = PotteryBase.Category_Index

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = colors
            else:
                color_palette = self.color_palette

            # 关键字参数初始化
            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

        # 单一地区绘制
        if site is not None:

            # 判断 site 是否有输入
            if site is None:
                if self.site is not None:
                    site = self.site
                else:
                    class_name = self.__class__.__name__
                    method_name = inspect.currentframe().f_code.co_name
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, no available site value.")

            # 判断 site 是否在数据库中
            site_tuple = self.get_subdirectories(folder_path=database)
            if site not in site_tuple:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"site '{site}' not found in database. Available sites: {[s for s in site_tuple]}.")

            # 检查 site 目录下的 method 目录是否存在
            target_site_dir = os.path.join(database, site)
            method_tuple = self.get_subdirectories(folder_path=target_site_dir)
            if method not in method_tuple:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"method '{method}' not found in site '{site}'. "
                                 f"Available methods: {[m for m in method_tuple]}.")

            site_path = os.path.join(self.database, site)  # site 的路径
            method_path = os.path.join(site_path, method)  # method 的路径
            file_path = os.path.join(method_path, f'{method}.xlsx')  # 文件的路径

            # 读取数据
            data_dic = self.read_excel(excel_path=file_path)
            title, data_df = list(data_dic.items())[0]  # 从 data_dic 中获取标题和数据
            data_df[category] = site

            # 提取数据部分和类别标签部分
            category_index = data_df[category]  # 提取名为 Category_Index 的列作为类别标签
            before_pca_df = data_df.drop(columns=[category])  # 剩下的列作为数据部分

            # 获取数据中的唯一类别值
            unique_category = np.unique(category_index)

            # 如果未提供颜色列表，则默认选择 color_palette 中与类别数量匹配的前几个颜色
            if colors is None:
                if len(unique_category) > len(color_palette):
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"There are {len(unique_category)} categories "
                                     f"but only {len(color_palette)} colors in the default color palette.")
                colors = color_palette[:len(unique_category)]

            # 验证提供的颜色数量是否与类别数量匹配
            elif len(colors) != len(unique_category):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"There are {len(unique_category)} categories but you provided {len(colors)} colors.")

            # 用 0 替换所有缺失值
            before_pca_df = before_pca_df.fillna(0)

            # 标准化处理数据
            x_scaled = StandardScaler().fit_transform(before_pca_df)

            # 使用 PCA 将数据降至 2D
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(x_scaled)

            # 转换 PCA 处理后的结果为 DataFrame，方便后续操作
            pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
            pca_df[category] = category_index

            # 创建绘图对象
            fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

            # 设置刻度限制
            if x_min is not None or x_max is not None:
                plt.xlim((x_min, x_max))
            if y_min is not None or y_max is not None:
                plt.ylim((y_min, y_max))

            # 设置坐标轴字体
            plt.xlabel(xlabel='PC1', fontdict=self.font_title)
            plt.ylabel(ylabel='PC2', fontdict=self.font_title)

            # 设置刻度标签的字体
            plt.xticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'])
            plt.yticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'])
            ax.tick_params(axis='both', which='major', direction='in')

            # 验证提供的颜色数量是否与类别数量匹配
            if len(colors) != len(unique_category):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"the length of the colors list should match the number of categories. "
                                 f"Currently, the length of the colors list is {len(colors)} "
                                 f"while the number of categories is {len(unique_category)}.")

            palette_dict = dict(zip(unique_category, colors))

            # 使用 seaborn 库的 scatterplot 绘制 PCA 结果
            sns.scatterplot(data=pca_df,
                            x='PC1',
                            y='PC2',
                            hue=category,
                            palette=palette_dict,
                            s=40,
                            edgecolor='k',
                            ax=ax)

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            # 如果需要绘制置信椭圆
            if draw_ellipse:
                for target, color in zip(unique_category, colors):
                    subset = pca_df[pca_df[category] == target]

                    # 计算子集的协方差和皮尔逊相关系数
                    cov = np.cov(subset.PC1, subset.PC2)
                    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                    ell_radius_x = np.sqrt(1 + pearson)
                    ell_radius_y = np.sqrt(1 - pearson)

                    # 创建椭圆对象
                    ellipse = Ellipse(xy=(0, 0),
                                      width=ell_radius_x * 2,
                                      height=ell_radius_y * 2,
                                      facecolor=color,
                                      alpha=0.35,
                                      zorder=0,
                                      **kwargs)

                    # 根据子集的数据分布调整椭圆的大小和位置
                    scale_x = np.sqrt(cov[0, 0]) * std
                    mean_x = np.mean(subset.PC1)
                    scale_y = np.sqrt(cov[1, 1]) * std
                    mean_y = np.mean(subset.PC2)

                    # 设置椭圆的变换属性
                    transform = transforms.Affine2D() \
                        .rotate_deg(45) \
                        .scale(scale_x, scale_y) \
                        .translate(mean_x, mean_y)

                    ellipse.set_transform(transform + ax.transData)
                    ax.add_patch(ellipse)

            # 在你设置轴范围之前，设置边距
            ax.margins(margin_ratio)

            plt.tight_layout()

            # 如果提供了保存路径，则保存图像到指定路径
            if save_path is not None:  # 如果 save_path 的值不为 None，则保存
                file_name = title + ".png"  # 初始文件名为 "title.png"
                full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = title + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = title + f"_{count}.png"
                        full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

            # 显示图像
            plt.show()

            if show_result:
                print(pca_df)

            # 使用 join 方法，将列表元素用逗号隔开并打印
            unique_category_str = ', '.join(unique_category)
            print(f"The \033[36mPCA analysis\033[0m images of the \033[34m{title}\033[0m "
                  f"for \033[31m{unique_category_str}\033[0m have been plotted.")

            # 创建 pca_dic 用于返回 PCA 分析后的数据
            pca_dic = {title: pca_df}

            return pca_dic

        # 多地区绘制
        else:

            # 读取数据
            data_dic = self.read_all_from_excel(method=method)

            # 如果有临时数据
            if use_temporary_dic:

                # 检查 self.temporary_data 是否为 None
                if self.temporary_data is None:
                    class_name = self.__class__.__name__
                    method_name = inspect.currentframe().f_code.co_name
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m,"
                                     f"'self.temporary_data' is None.")

                # 取出临时数据
                temporary_dic = self.temporary_data

                # 创建一个空列表来存放所有的DataFrame
                df_list = []

                # 遍历字典并添加 PotteryBase.Category_Index 列
                for key, df in temporary_dic.items():
                    df[PotteryBase.Category_Index] = key
                    df_list.append(df)

                # 使用 concat 合并所有 DataFrame，设置 ignore_index=True 来重置索引
                temporary_df = pd.concat(df_list, ignore_index=True).fillna(0)

                title = list(data_dic.keys())[0]
                data_df = list(data_dic.values())[0]

                # 按行拼接，并将缺失值填充为0
                data_df = pd.concat([data_df, temporary_df], axis=0, ignore_index=True).fillna(0)
                data_dic = {title: data_df}

            if not data_dic:  # 如果输入的 method 不正确，那么会导致为空
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The input method is incorrect, no related method found.")

            title, data_df = list(data_dic.items())[0]  # 从 data_dic 中获取标题和数据

            # 提取数据部分和类别标签部分
            category_index = data_df[category]  # 提取名为 Category_Index 的列作为类别标签
            before_pca_df = data_df.drop(columns=[category])  # 剩下的列作为数据部分

            # 获取数据中的唯一类别值
            unique_category = np.unique(category_index)

            # 如果未提供颜色列表，则默认选择 color_palette 中与类别数量匹配的前几个颜色
            if colors is None:
                if len(unique_category) > len(color_palette):
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"There are {len(unique_category)} categories "
                                     f"but only {len(color_palette)} colors in the default color palette.")
                colors = color_palette[:len(unique_category)]

            # 验证提供的颜色数量是否与类别数量匹配
            elif len(colors) != len(unique_category):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"There are {len(unique_category)} categories but you provided {len(colors)} colors.")

            # 用 0 替换所有缺失值
            before_pca_df = before_pca_df.fillna(0)

            # 标准化处理数据
            x_scaled = StandardScaler().fit_transform(before_pca_df)

            # 使用 PCA 将数据降至 2D
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(x_scaled)

            # 转换 PCA 处理后的结果为 DataFrame，方便后续操作
            pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
            pca_df[category] = category_index

            # 创建绘图对象
            fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

            # 设置刻度限制
            if x_min is not None or x_max is not None:
                plt.xlim((x_min, x_max))
            if y_min is not None or y_max is not None:
                plt.ylim((y_min, y_max))

            # 设置坐标轴字体
            plt.xlabel(xlabel='PC1', fontdict=self.font_title)
            plt.ylabel(ylabel='PC2', fontdict=self.font_title)

            # 设置刻度标签的字体
            plt.xticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'])
            plt.yticks(fontfamily=self.font_ticket['family'],
                       fontweight=self.font_ticket['weight'],
                       fontsize=self.font_ticket['size'])
            ax.tick_params(axis='both', which='major', direction='in')

            # 验证提供的颜色数量是否与类别数量匹配
            if len(colors) != len(unique_category):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"the length of the colors list should match the number of categories. "
                                 f"Currently, the length of the colors list is {len(colors)} "
                                 f"while the number of categories is {len(unique_category)}.")

            palette_dict = dict(zip(unique_category, colors))

            # 使用 seaborn 库的 scatterplot 绘制 PCA 结果
            sns.scatterplot(data=pca_df,
                            x='PC1',
                            y='PC2',
                            hue=category,
                            palette=palette_dict,
                            s=40,
                            edgecolor='k',
                            ax=ax)

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            # 如果需要绘制置信椭圆
            if draw_ellipse:
                for target, color in zip(unique_category, colors):
                    subset = pca_df[pca_df[category] == target]

                    # 计算子集的协方差和皮尔逊相关系数
                    cov = np.cov(subset.PC1, subset.PC2)
                    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                    ell_radius_x = np.sqrt(1 + pearson)
                    # 将 1 - pearson 的值限制在 0 到正无穷之间，确保传递给 np.sqrt() 的值不为负
                    ell_radius_y = np.sqrt(np.clip(1 - pearson, a_min=0, a_max=None))

                    # 创建椭圆对象
                    ellipse = Ellipse(xy=(0, 0),
                                      width=ell_radius_x * 2,
                                      height=ell_radius_y * 2,
                                      facecolor=color,
                                      alpha=0.35,
                                      zorder=0,
                                      **kwargs)

                    # 根据子集的数据分布调整椭圆的大小和位置
                    scale_x = np.sqrt(cov[0, 0]) * std
                    mean_x = np.mean(subset.PC1)
                    scale_y = np.sqrt(cov[1, 1]) * std
                    mean_y = np.mean(subset.PC2)

                    # 设置椭圆的变换属性
                    transform = transforms.Affine2D() \
                        .rotate_deg(45) \
                        .scale(scale_x, scale_y) \
                        .translate(mean_x, mean_y)

                    ellipse.set_transform(transform + ax.transData)
                    ax.add_patch(ellipse)

            # 在你设置轴范围之前，设置边距
            ax.margins(margin_ratio)

            plt.tight_layout()

            # 如果提供了保存路径，则保存图像到指定路径
            if save_path is not None:  # 如果 save_path 的值不为 None，则保存
                file_name = title + ".png"  # 初始文件名为 "title.png"
                full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = title + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = title + f"_{count}.png"
                        full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

            # 显示图像
            plt.show()

            if show_result:
                print(pca_df)

            # 使用 join 方法，将列表元素用逗号隔开并打印
            unique_category_str = ', '.join(unique_category)
            print(f"The \033[36mPCA analysis\033[0m images of the \033[34m{title}\033[0m "
                  f"for \033[31m{unique_category_str}\033[0m have been plotted.")

            # 创建 pca_dic 用于返回 PCA 分析后的数据
            pca_dic = {title: pca_df}

            return pca_dic

    # # 查看图片
    # def open_figure(self, site: Optional[str] = None, method: Optional[str] = None, database: Optional[str] = None):
    #
    #     return None


""" 运行陶器基因库 """
class Pottery(PotteryBase):
    """
    用程序化打开 PotteryBase
    Open PotteryBase programmatically

    """

    # 初始化
    def __init__(self,

                 # 接收参数 (7)
                 data_dic: Optional[dict] = None, data_df: Optional[DataFrame] = None, title: Optional[str] = None,
                 x_list: Optional[list] = None, y_list: Optional[list] = None, x_label: str = None, y_label: str = None,

                 # 关键参数 (7)
                 txt_path: Optional[str] = None, excel_path: Optional[str] = None, json_path: Optional[str] = None,
                 keyword: Optional[str] = None, file_pattern: Optional[str] = None, save_path: Optional[str] = None,
                 magic_database: Optional[str] = None,

                 # 特有参数 (2)
                 site: Optional[str] = None, method: Optional[str] = None, database: Optional[str] = None
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

        # 特有参数的初始化
        self.site = site
        self.method = method
        if database is not None:
            self.database = database
        else:
            self.database = PotteryBase.Pottery_Database

        self.layer = 0  # 所在层数 (第几个表格，以 0 为始)
        self.site_folders = None  # 地区路径 list
        self.selected_site_folder = None  # 选择的地区路径 str
        self.method_folders = None  # 方法路径的 list
        self.selected_method_folder = None  # 选择的方法路径  str
        self.data_folders = None  # 数据路径的 list
        self.selected_data = None  # 选择的数据路径 str
        self.user_input = None  # 用户输入的信息
        self.result_of_input = None  # 用户信息处理的结果

        self.allow_series = []  # 接受的输入序号
        self.not_allow_series = []  # 不接受的输入序号

        self.file_type = None  # 数据文件的类型

        # 数据初始化分配
        if type(self) == Pottery:  # 当 self 为 Pottery 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # general.Manager 及其子类需要调用以初始化属性

        # 标准库的路径
        self.standard_template_path = os.path.join(PotteryBase.Pottery_Database, PotteryBase.Standard_Template)

    # 第一个表格
    def __display_main_folders(self):
        """
        打开第一个表格
        Open the first table.
        """

        # 获取主目录下所有非 '0_standard' 的文件夹
        site_folders = [folder for folder in os.listdir(self.database)
                        if os.path.isdir(os.path.join(self.database, folder)) and folder != '0_standard']

        # 使用 PrettyTable 格式化输出
        table = PrettyTable()
        table.field_names = ["No.", "Folders"]  # 表格的列名称
        table.align = "c"  # 设置表格对齐方式为居中

        # 将过滤后的文件夹逐个添加到表格中，并设置颜色
        for index, folder in enumerate(site_folders, start=1):
            folder_color = "\033[31m"  # 红色字体
            table.add_row([f"\033[32m{index}\033[0m", f"{folder_color}{folder}\033[0m"])

        # 提示信息
        header = f"Current database \033[95m{os.path.basename(self.database)}\033[0m contains the following sites:"

        # 输出表格
        print('-' * 120)  # 在输出表格前进行分割
        print(header)
        print(table)
        print('Enter q to quit.')

        self.site_folders = site_folders

        return None

    # 第二个表格
    def __display_secondary_folders(self):
        """
        打开第二个表格
        Open the secondary table.
        """

        selected_site_folder = self.selected_site_folder

        # 获取文件夹下的所有子文件夹
        method_folders = [folder for folder in os.listdir(selected_site_folder)
                          if os.path.isdir(os.path.join(selected_site_folder, folder))]

        # 使用 PrettyTable 格式化输出
        table = PrettyTable()
        table.field_names = ["No.", "Subfolders"]  # 表格的列名称
        table.align = "c"  # 设置表格对齐方式为居中

        # 将子文件夹逐个添加到表格中，并设置颜色
        for sub_index, subfolder in enumerate(method_folders, start=1):
            subfolder_color = "\033[34m"  # 蓝色字体
            table.add_row([f"\033[32m{sub_index}\033[0m", f"{subfolder_color}{subfolder}\033[0m"])

        # 提示信息
        subheader = f"Site \033[31m{os.path.basename(self.selected_site_folder)}\033[0m contains the following methods:"

        # 输出表格
        print('-' * 120)  # 在输出表格前进行分割
        print(subheader)
        print(table)
        print('Enter b to go back, q to quit, i for information.')

        self.method_folders = method_folders

        return None

    # 第三个表格
    def __display_thirdly_folders(self):
        """
        打开第三个表格
        Open the third table.
        """

        selected_method_folder = self.selected_method_folder

        # 读取信息
        site_base = os.path.basename(self.selected_site_folder)
        method_base = os.path.basename(self.selected_method_folder)

        self.read_information(site=site_base, method=method_base)

        try:  # 存在无 keyword 的情况
            # 遍历 allowed_types 并更新标志变量
            if "txt" in self.method_type.lower():
                self.include_txt = True
            if "excel" in self.method_type.lower():
                self.include_excel = True
            if "json" in self.method_type.lower():
                self.include_json = True
            if "image" in self.method_type.lower():
                self.include_image = True

        except AttributeError:
            self.include_txt = False
            self.include_excel = False
            self.include_json = False
            self.include_image = False

        # 获取子文件夹中的所有文件，排除 _info.txt 文件和 .DS_Store 文件 (MacOs)
        data_folders = [file for file in os.listdir(selected_method_folder) if file != '.DS_Store'
                     and not file.endswith('_info.txt')]

        # 使用 PrettyTable 格式化输出
        file_table = PrettyTable()
        file_table.field_names = ["No.", "Files"]  # 表格的列名称
        file_table.align = "c"  # 设置表格对齐方式为居中

        # 如果文件夹中有文件，按文件类型进行分类并输出
        if data_folders:
            # 按文件类型分类
            txt_files = [file for file in data_folders if file.endswith('.txt') or file.endswith('.TXT')]
            json_files = [file for file in data_folders if file.endswith('.json')]
            excel_files = [file for file in data_folders if file.endswith('.xls') or file.endswith('.xlsx')]
            image_files = [file for file in data_folders if
                           file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg')]
            other_files = [file for file in data_folders if not any(
                file.endswith(ext) for ext in ['.txt', '.TXT', '.json', '.xls', '.xlsx', '.png', '.jpg', '.jpeg'])]

            # 用于记录文件的顺序
            all_files_in_order = []
            allow_series = []
            not_allow_series = []

            # 按类别逐个输出文件
            if txt_files:
                if self.include_txt:  # 如果接受文件类型
                    file_table.add_row(["", "\033[33m-- txt files --\033[0m"])  # 黄色标签
                    for file_index, txt_file in enumerate(txt_files, start=1):
                        file_table.add_row(
                            [f"\033[32m{file_index}\033[0m", f"\033[36m{txt_file}\033[0m"])  # 绿色索引，青色文件名
                        all_files_in_order.append(txt_file)  # 记录文件名
                        allow_series.append(str(file_index))  # 将接受的序号加入 allow_series
                else:
                    file_table.add_row(["", "-- txt files --"])
                    for file_index, txt_file in enumerate(txt_files, start=1):
                        file_table.add_row([f"{file_index}", f"{txt_file}"])
                        all_files_in_order.append(txt_file)  # 记录文件名
                        not_allow_series.append(str(file_index))  # 将不接受的序号加入 not_allow_series

            if json_files:
                if self.include_json:  # 如果接受文件类型
                    file_table.add_row(["", "\033[33m-- json files --\033[0m"])  # 黄色标签
                    for file_index, json_file in enumerate(json_files, start=len(txt_files) + 1):
                        file_table.add_row(
                            [f"\033[32m{file_index}\033[0m", f"\033[36m{json_file}\033[0m"])  # 绿色索引，青色文件名
                        all_files_in_order.append(json_file)  # 记录文件名
                        allow_series.append(str(file_index))  # 将接受的序号加入 allow_series
                else:
                    file_table.add_row(["", "-- json files --"])
                    for file_index, json_file in enumerate(json_files, start=len(txt_files) + 1):
                        file_table.add_row([f"{file_index}", f"{json_file}"])
                        all_files_in_order.append(json_file)  # 记录文件名
                        not_allow_series.append(str(file_index))  # 将不接受的序号加入 not_allow_series

            if excel_files:
                if self.include_excel:  # 如果接受文件类型
                    file_table.add_row(["", "\033[33m-- Excel files --\033[0m"])  # 黄色标签
                    for file_index, excel_file in enumerate(excel_files, start=len(txt_files) + len(json_files) + 1):
                        file_table.add_row(
                            [f"\033[32m{file_index}\033[0m", f"\033[36m{excel_file}\033[0m"])  # 绿色索引，青色文件名
                        all_files_in_order.append(excel_file)  # 记录文件名
                        allow_series.append(str(file_index))  # 将接受的序号加入 allow_series
                else:
                    file_table.add_row(["", "-- Excel files --"])
                    for file_index, excel_file in enumerate(excel_files, start=len(txt_files) + len(json_files) + 1):
                        file_table.add_row([f"{file_index}", f"{excel_file}"])
                        all_files_in_order.append(excel_file)  # 记录文件名
                        not_allow_series.append(str(file_index))  # 将不接受的序号加入 not_allow_series

            if image_files:
                if self.include_image:  # 如果接受文件类型
                    file_table.add_row(["", "\033[33m-- Image files --\033[0m"])  # 黄色标签
                    for file_index, image_file in enumerate(image_files, start=len(txt_files) + len(json_files) + len(
                            excel_files) + 1):
                        file_table.add_row(
                            [f"\033[32m{file_index}\033[0m", f"\033[36m{image_file}\033[0m"])  # 绿色索引，青色文件名
                        all_files_in_order.append(image_file)  # 记录文件名
                        allow_series.append(str(file_index))  # 将接受的序号加入 allow_series
                else:
                    file_table.add_row(["", "-- Image files --"])
                    for file_index, image_file in enumerate(image_files, start=len(txt_files) + len(json_files) + len(
                            excel_files) + 1):
                        file_table.add_row([f"{file_index}", f"{image_file}"])
                        all_files_in_order.append(image_file)  # 记录文件名
                        not_allow_series.append(str(file_index))  # 将不接受的序号加入 not_allow_series

            if other_files:  # 不接受的文件类型，均为白色字体
                file_table.add_row(["", "-- Other files --"])
                for file_index, other_file in enumerate(other_files,
                                                        start=len(txt_files) + len(json_files) + len(excel_files) + len(
                                                                image_files) + 1):
                    file_table.add_row([f"{file_index}", f"{other_file}"])
                    all_files_in_order.append(other_file)  # 记录文件名
                    not_allow_series.append(str(file_index))  # 将不接受的序号加入 not_allow_series

            # 按表格顺序对 selected_method_folder 进行排序后重新赋值
            data_folders = all_files_in_order

            self.allow_series = allow_series
            self.not_allow_series = not_allow_series

        else:
            # 如果没有找到文件，提示文件夹为空
            file_table.add_row(["", "Folder is empty"])

        # 提示信息
        thiheader = (f"Method \033[34m{os.path.basename(self.selected_method_folder)}"
                     f"\033[0m contains the following files:")

        # 输出文件表格和提示信息
        print('-' * 120)  # 在输出表格前进行分割
        print(thiheader)
        print(file_table)
        print('Enter b to go back, q to quit, i for information.')

        self.data_folders = data_folders

        return None

    # 判断绘图的文件类型
    def __decide_figure(self):
        """
        在 self.layer == 2 时，根据输入的数字和文件情况来决定输出图像来源
        At self.layer == 2, the source of the output image is determined based on the number
        and file condition of the input
        """

        index = int(self.user_input) - 1
        selected_data_basename = self.data_folders[index]

        if str(index + 1) in self.allow_series:
            # 用户输入在允许的文件列表中，执行原本的逻辑
            self.selected_data = os.path.join(self.selected_method_folder, selected_data_basename)
            self.title = os.path.basename(self.selected_data)
            self.result_of_input = 'self.__plot_fingure_from_txt()'
            self.layer = 3

            # 获取文件扩展名
            file_extension = os.path.splitext(self.title)[1].lower()  # 提取扩展名并转为小写

            # 根据扩展名判断文件类型
            if file_extension == ".txt":
                self.file_type = "txt"
            elif file_extension == ".json":
                self.file_type = "json"
            elif file_extension in [".xls", ".xlsx"]:
                self.file_type = "excel"
            elif file_extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                self.file_type = "image"
            else:
                self.file_type = "unknown"

        elif str(index + 1) in self.not_allow_series:
            # 用户输入在不允许的文件列表中，检查接受的文件类型并提示
            allowed_types = []

            # 检查支持的文件类型并加入提示中
            if self.include_txt:
                allowed_types.append("txt")
            if self.include_json:
                allowed_types.append("json")
            if self.include_excel:
                allowed_types.append("Excel")
            if self.include_image:
                allowed_types.append("image")

            if allowed_types:
                # 如果有支持的文件类型，打印提示
                allowed_types_str = ", ".join(allowed_types)
                print(
                    f"Please select a valid file type: \033[36m{allowed_types_str}\033[0m")
            else:
                # 如果没有支持的文件类型，可能出现异常情况，这里也可以处理
                print("No supported file types available")
                
        return None

    # 根据 txt 绘制图像
    def __plot_fingure_from_txt(self):
        """
        利用所选数据来绘制图像
        Plot an image using the selected data.
        """

        # 检查文件是否是 txt 或 TXT 格式
        if self.selected_data.endswith('.txt') or self.selected_data.endswith('.TXT'):

            # 清楚当前已储存的数据，以免造成数据覆盖
            self.clear_data()

            # 提取文件名称
            selected_data = os.path.basename(self.selected_data)

            # 尝试打开文件
            self.plot_figure(
                site=self.site,
                method=self.method,
                file=selected_data
            )

        else:
            print('Figure plotting is only applicable to \033[36m.txt files\033[0m.\n')

        self.layer = 2

        return None

    # 根据 Excel 绘制图像
    def __plot_figure_from_excel(self):
        """
        绘制 PCA 分析后的图像，利用 Excel 表格数据
        The PCA images were ploted using Excel spreadsheet data.
        """

        # 检查文件是否是 xlsx 或 xls 格式
        if self.selected_data.endswith('.xlsx') or self.selected_data.endswith('.xls'):

            # 清楚当前已储存的数据，以免造成数据覆盖
            self.clear_data()

            # 尝试打开文件
            self.plot_pca(
                site=self.site,
                method=self.method
            )

        else:
            print('Figure plotting is only applicable to \033[36m.xlsx files\033[0m.\n')

        self.layer = 2

        return None
    
    # 根据 json 绘制图像
    def __plot_figure_from_json(self):
        """
        根据所选的 json 文件来绘制图像
        Plot the image according to the selected json file.
        """

        # 检查文件是否是 json 格式
        if self.selected_data.endswith('.json'):

            # 清楚当前已储存的数据，以免造成数据覆盖
            self.clear_data()

            # 提取文件名称
            selected_data = os.path.basename(self.selected_data)

            # 尝试打开文件
            self.plot_figure_from_json(
                site=self.site,
                method=self.method,
                file=selected_data
            )

        else:
            print('Figure plotting is only applicable to \033[36m.json files\033[0m.\n')

        self.layer = 2

        return None

    # 打开图像
    def __open_figure(self):

        self.layer = 2
        
        pass

    # 用户输入信息
    def __user_input(self):
        """
        根据不同的情况来请求用户输入信息
        Requests for user input information depending on the situation.
        """

        print('')
        user_input = input("Please enter what you want to see:")
        print('')
        self.user_input = user_input

        if user_input.lower() == 'q':
            print("Exiting the program.")
            self.result_of_input = 'quit()'

        elif self.layer == 0 and user_input.lower() == 'i':
            print("Please choose a folder to view the intro.")

        elif (self.layer == 1 or self.layer == 2) and user_input.lower() == 'i':
            self.result_of_input = 'self.__read_information()'

        elif self.layer == 1 and user_input.lower() == 'b':
            self.result_of_input = 'self.__display_main_folders()'
            self.layer = 0

        elif self.layer == 2 and user_input.lower() == 'b':
            self.result_of_input = 'self.__display_secondary_folders()'
            self.layer = 1

        else:  # 输入为数字的情况

            try:
                if self.layer == 0:
                    index = int(user_input) - 1
                    selected_subfolder_basename = self.site_folders[index]
                    # 保存路径至选择的路径
                    self.selected_site_folder = os.path.join(self.database, selected_subfolder_basename)
                    self.site = os.path.basename(self.selected_site_folder)  # 保存地区
                    self.result_of_input = 'self.__display_secondary_folders()'
                    self.layer = 1

                elif self.layer == 1:
                    index = int(user_input) - 1
                    selected_subfolder_basename = self.method_folders[index]
                    # 保存路径至选择的路径
                    self.selected_method_folder = os.path.join(self.selected_site_folder, selected_subfolder_basename)
                    self.method = os.path.basename(self.selected_method_folder)  # 保存测试数据
                    self.result_of_input = 'self.__display_thirdly_folders()'
                    self.layer = 2

                elif self.layer == 2:
                    self.__decide_figure()
                    if self.file_type == "txt":
                        self.result_of_input = 'self.__plot_fingure_from_txt()'
                    elif self.file_type == "json":
                        self.result_of_input = 'self.__plot_figure_from_json()'
                    elif self.file_type == "excel":
                        self.result_of_input = 'self.__plot_figure_from_excel()'
                    elif self.file_type == "image":
                        self.result_of_input = 'self.__open_figure()'
                    else:
                        print("Invalid input. Please enter a valid content.")
                        self.result_of_input = 'self.__wait_input()'

                else:
                    print("Invalid input. Please enter a valid content.")
                    self.result_of_input = 'self.__wait_input()'

            except (IndexError, ValueError):
                print("Invalid input. Please enter a valid content.")
                self.result_of_input = 'self.__wait_input()'

        return None

    # 读取详细信息
    def __read_information(self):
        """
        读取 _info.txt 文件并打印简介信息
        Read the _info.txt file and print the profile information.
        """

        folder_path = self.database

        if self.layer == 1:
            folder_path = self.selected_site_folder
        elif self.layer == 2:
            folder_path = self.selected_method_folder

        # 查找 _info.txt 文件
        info_file = [file for file in os.listdir(folder_path) if file.endswith('_info.txt')]
        if info_file:
            info_file_path = os.path.join(folder_path, info_file[0])
            with open(info_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 解析文件内容为字典格式
            info_dict = {}
            current_key = None
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'Information':
                        current_key = key
                        info_dict[current_key] = []
                    else:
                        info_dict[key] = value
                elif current_key:
                    info_dict[current_key].append(line)

            # 输出简介信息
            if self.layer == 1:
                print(f"Introductory information for site \033[31m{os.path.basename(folder_path)}\033[0m:")
            elif self.layer == 2:
                print(f"Introductory information for method \033[34m{os.path.basename(folder_path)}\033[0m:")
            else:
                pass

            for key, value in info_dict.items():
                if key == 'Information':
                    print(f"\033[33m{key}:\033[0m")
                    for line in value:
                        print(f"\033[33m{line}\033[0m")
                else:
                    print(f"\033[33m{key}: {value}\033[0m")

        else:
            print(f"No _info.txt file found in {os.path.basename(folder_path)}.")

        return None

    # 空白消息组
    @staticmethod
    def __wait_input():
        """
        等待下一步指令
        Wait for next instruction

        """

        pass

        return None

    # 主程序运行
    def run(self):
        """
        主循环，使用时仅调用该程序即可
        The main loop, when used, only the program can be called
        """

        self.__display_main_folders()  # 显示主文件夹并获取列表

        while True:
            self.__user_input()

            if self.result_of_input == 'self.__display_main_folders()':
                self.__display_main_folders()
                continue

            elif self.result_of_input == 'self.__display_secondary_folders()':
                self.__display_secondary_folders()
                continue

            elif self.result_of_input == 'self.__display_thirdly_folders()':
                self.__display_thirdly_folders()
                continue

            elif self.result_of_input == 'self.__plot_fingure_from_txt()':
                self.__plot_fingure_from_txt()
                continue

            elif self.result_of_input == 'self.__plot_figure_from_json()':
                self.__plot_figure_from_json()
                continue

            elif self.result_of_input == 'self.__plot_figure_from_excel()':
                self.__plot_figure_from_excel()
                continue

            elif self.result_of_input == 'self.__open_figure()':
                self.__open_figure()
                continue

            elif self.result_of_input == 'self.__read_information()':
                self.__read_information()
                continue

            elif self.result_of_input == 'self.__wait_input()':
                self.__wait_input()
                continue

            elif self.result_of_input == 'quit()':
                break
