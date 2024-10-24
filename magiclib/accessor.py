"""
magiclib / projector

Attention:
1. Non-core code, do not need too many comments
"""


# 导入顺序不同有可能导致程序异常
from . import general, projector

import pandas as pd
import seaborn as sns
from typing import Optional
from pandas import DataFrame


""" 快速实现 """
class Constructor(general.Magic):
    """
    函数构造区

    Quickly complete a closed-loop functionality to perform large-scale data
    generation and processing within a short period of time, while also having
    the ability to check and update databases.

    关键功能与储存的好用的功能
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
            self.magic_database = general.Magic_Database

        # 数据初始化分配 和 数据类型导入
        if type(self) == Constructor:  # 当 self 为 Constructor 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # Manager 及其子类需要调用以初始化属性

    # 魔法键入
    def magic_access(self):
        self.smooth_curve()
        self.locate_point()
        self.improve_precision()
        self.reduce_precision()
        self.normalize_data()
        self.adjust_data()
        self.assign_weight()
        self.fit_curve()
        self.restore_precision()
        self.realize_data()

    # 用于存储数据为 JSON 类型
    def to_improve_precision(self):
        self.smooth_curve()
        self.locate_point()
        self.improve_precision()

    # 检查生成库
    def examine_library(self):
        pass

    # 生成数据
    def generation_access(self):
        pass

    # 对于 7月16日 汇报，老化力学性能数据生成
    def for_report_tensile(self, excel_path, x_max_limit, y_max_limit):

        self.rename_extension(path=excel_path, old_extension='xls', new_extension='xlsx')
        self.to_magic('tensile')
        self.read_excel(excel_path=excel_path)

        pd.set_option('display.max_rows', 10)  # 只展示10行数据

        self.random_dict(num_pairs=5)
        print('1/14')
        self.smooth_curve(data_dic=self.random_dic)
        print('2/14')
        self.locate_point()
        print('3/14')
        self.improve_precision()
        print('4/14')
        self.reduce_precision(interval_disperse=0.005)
        # self.plot_line()  # 1
        print('5/14')
        self.SF_remove(remove_section=[(0.15, 0.85), 0.15, 2])
        print('6/14')
        self.SF_append(y_shrinking_factor=0.5, y_distribution_limit=1.2)
        # self.plot_line(self.sf_appended_dic)  # 5
        print('7/14')
        # self.plot_line(data_dic=self.sf_appended_dic)  # 2.1
        self.normalize_data(data_dic=self.sf_appended_dic, x_min_limit=0, x_max_limit=x_max_limit, y_min_limit=0,
                            y_max_limit=y_max_limit)
        print('8/14')
        self.adjust_data()
        print('10/14')
        self.assign_weight(data_dic=self.normalized_dic, key_point_dic=self.key_normalized_dic)
        # self.plot_line()  # 4
        print('11/14')
        self.fit_curve(smooth_fitting=0.01)
        print('12/14')
        self.restore_precision(precision_fitting=0.005)
        print('13/14')
        self.realize_data(noise_level=0.01)
        print('14/14')

        print(list(self.random_dic.keys()))  # 将键的视图转换为列表

        self.plot_line(show_in_one=True, background_transparency=0.3)


""" 文章图像 """
class Article:
    """
    历史文章图像

    Wrote about the images used in the article.
    """

    # Banpo XRD
    @staticmethod
    def Banpo_XRD(xrd_path: str, save_path: Optional[str] = None):

        bp_xrd_color = '#640000'
        bp_pdf_color = '#00008B'
        bp_xrd_palette = sns.light_palette(color='#191970', as_cmap=True)
        bp_pdf_palette = sns.light_palette(color='#303030', as_cmap=True)

        banpo_xrd = projector.XRD(read_path=xrd_path, save_path=save_path)
        banpo_xrd.read()
        banpo_xrd.add_xrd_pdf_match(directory=xrd_path)

        banpo_xrd.plot_xrd_pdf(
            show_pdf='material',
            pdf_pattern=r'([a-zA-Z]+)',

            xrd_color=bp_xrd_color,
            pdf_color=bp_pdf_color,

            xrd_background=bp_xrd_palette,
            pdf_background=bp_pdf_palette,

            x_min=5,
            x_max=70,
        )

    # Jiangzhai XRD
    @staticmethod
    def Jiangzhai_XRD(xrd_path: str, save_path: Optional[str] = None):

        jz_xrd_color = '#800020'
        jz_pdf_color = '#2F085E'
        jz_xrd_palette = sns.light_palette(color='#556B2F', as_cmap=True)
        jz_pdf_palette = sns.light_palette(color='#FF7F50', as_cmap=True)

        jiangzhai_xrd = projector.XRD(read_path=xrd_path, save_path=save_path)
        jiangzhai_xrd.read()
        jiangzhai_xrd.add_xrd_pdf_match(directory=xrd_path)

        jiangzhai_xrd.plot_xrd_pdf(
            show_pdf='material',
            pdf_pattern=r'([a-zA-Z]+)',

            xrd_color=jz_xrd_color,
            pdf_color=jz_pdf_color,

            xrd_background=jz_xrd_palette,
            pdf_background=jz_pdf_palette,

            x_min=5,
            x_max=70,
        )

    # Mijiaya XRD
    @staticmethod
    def Mijiaya_XRD(xrd_path: str, save_path: Optional[str] = None):

        mjy_xrd_color = '#000000'
        mjy_pdf_color = '#4B0082'
        mjy_xrd_palette = sns.light_palette(color='#FFC0CB', as_cmap=True)
        mjy_pdf_palette = sns.light_palette(color='#90EE90', as_cmap=True)

        mijiaya_xrd = projector.XRD(read_path=xrd_path, save_path=save_path)
        mijiaya_xrd.read()
        mijiaya_xrd.add_xrd_pdf_match(directory=xrd_path)

        mijiaya_xrd.plot_xrd_pdf(
            show_pdf='material',
            pdf_pattern=r'([a-zA-Z]+)',

            xrd_color=mjy_xrd_color,
            pdf_color=mjy_pdf_color,

            xrd_background=mjy_xrd_palette,
            pdf_background=mjy_pdf_palette,

            x_min=5,
            x_max=70,
        )


""" 属性存储 """
class History:
    """
    历史用到的需要存储的属性

    Properties used in history that need to be stored.
    """

    # 半坡
    xrd_bp_path = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/BP_PP/1129_PB'
    xrd_bp_black = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/BP_PP/Pigment_B'
    xrd_bp_red = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/BP_PP/Pigment_R'
    xrd_bp_wight = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/BP_PP/Pigment_W'

    bp_xrd_color = '#640000'
    bp_pdf_color = '#00008B'
    bp_xrd_palette = sns.light_palette(color='#191970', as_cmap=True)
    bp_pdf_palette = sns.light_palette(color='#303030', as_cmap=True)

    # 姜寨
    xrd_jz_path = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/JZ_PP/1129_JZ'
    xrd_jz_black = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/JZ_PP/Pigment_B'
    xrd_jz_red1 = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/JZ_PP/Pigment_R1'
    xrd_jz_red2 = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/JZ_PP/Pigment_R2'
    xrd_jz_wight = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/JZ_PP/Pigment_(W)'

    jz_xrd_color = '#800020'
    jz_pdf_color = '#2F085E'
    jz_xrd_palette = sns.light_palette(color='#556B2F', as_cmap=True)
    jz_pdf_palette = sns.light_palette(color='#FF7F50', as_cmap=True)

    # 米家崖
    mjy_1_1 = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/MJY_1_P/1'
    mjy_2_1 = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/MJY_2_P/1'
    mjy_3_1 = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/MJY_3_P/1'

    mjy_xrd_color = '#000000'
    mjy_pdf_color = '#4B0082'
    mjy_xrd_palette = sns.light_palette(color='#FFC0CB', as_cmap=True)
    mjy_pdf_palette = sns.light_palette(color='#90EE90', as_cmap=True)

    # 铜锈
    xrd_cu_path = '/Users/sumiaomiao/Downloads/数据库/2-数据分析/XRD/Cu'

    cu_xrd_color = '#A52A2A'
    cu_pdf_color = '#000000'
    cu_xrd_palette = sns.light_palette(color='#D8BFD8', as_cmap=True)
    cu_pdf_palette = sns.light_palette(color='#A9A9A9', as_cmap=True)
