"""
magiclib / grapher

------------------------------------------------------------------------------------------------------------------------
magiclib / grapher is a comprehensive Python visualization and analysis library designed for scientific data processing,
offering advanced statistical methods (PCA, clustering analysis), diverse plotting capabilities (box plots, heatmaps,
ridge plots, distribution charts), and sophisticated curve fitting techniques (polynomial regression, multivariate
fitting, model-based predictions) with professional customization options for research-grade visualizations.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""


from . import general

import os
import copy
import time
import joypy
import inspect
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import scipy.stats as stats
from pandas import DataFrame
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import scipy.cluster.hierarchy as sch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import OptimizeWarning
from typing import Union, Optional, List, Dict, Callable, Tuple
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor


# 忽略警告  Plotter.plot_2d_histogram
warnings.filterwarnings(action="ignore", category=UserWarning, message=r"set_ticklabels\(\) should only be used "
                                                                       r"with a fixed number of ticks, i.e. "
                                                                       r"after set_ticks\(\) or using a FixedLocator\.")
# 忽略 OptimizeWarning
warnings.filterwarnings(action="ignore", category=OptimizeWarning)


""" 统计学分析 """
class Statistics(general.Manager):
    """
    应用统计学方法对数据进行分析，并绘制图像
    Apply statistical methods to analyze data and plot graphs.

    注意： 接入的 data_dic 长度只允许为 1
    """

    Category_Index = general.Category_Index  # 分类索引

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
            self.magic_database = self.Magic_Database

        # 数据初始化分配
        if type(self) == Statistics:  # 当 self 为 Statistics 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # general.Manager 及其子类需要调用以初始化属性

        # pca_analysis()
        self.pca_dic = None
        # pca_loadings()
        self.pca_loadings_dic = None
        # dendrogram_clustering()
        self.tree_dic = None
        # agglomerative_clustering()
        self.agglomerative_dic = None

    # 主成分分析绘图
    def pca_analysis(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True,
                     draw_ellipse: bool = True, std: float = 2,  margin_ratio: float = 0.1, dpi: int = 600,
                     width_height: tuple = (6, 4.5), category: Optional[str] = None, colors: Optional[list] = None,
                     show_result: bool = True, show_legend: bool = True, show_figure: bool = True,
                     loadings_analysis: bool = False, **kwargs) -> Dict[str, DataFrame]:
        """
        此方法用于绘制 PCA 结果图，输入的 data_dic 的长度需为 1，需要有 category 列
        This method is used to plot PCA results. The length of the input data_dic should be 1 and there should
        be a category column.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param draw_ellipse: (bool) 是否绘制置信椭圆
        :param std: (float) 置信椭圆的标准差范围，默认为 2，此时置信区间为 90%
        :param margin_ratio: (float) 数据距边界的比例，该值需要介于 0 至 1 之间，默认为 0.1
        :param dpi: (int) 图像保存的精度
        :param width_height: (tuple) 图片的宽度和高度，默认为(6, 4.5)
        :param category: (str) 用于分类的列，默认为 Statistics.Category_Index
        :param colors: (str / list) 置信椭圆的填充颜色
        :param show_result: (bool) 是否打印结果，默认为 True。结果为 PCA scores (PCA 得分)
                                   与 Explained variance ratio (方差解释率)
        :param show_legend: (bool) 是否显示图例，默认为 True
        :param show_figure: (bool) 是否显示图片，默认为 True，此项是确保被内部方法调用时不会绘制图像
        :param loadings_analysis: (bool) 是否一同绘制载荷图，使用的参数为默认值，默认为 False
        :param kwargs: Ellipse 方法中的关键字参数

        :return pca_dic: (dict) PCA 分析后的数据 dict ，键为 title，值为 DataFrame 形式的 PCA 结果

        --- **kwargs ---

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
        if save_path is True:
            save_path = self.save_path
        # 若 save_path 为 False 时，本图形不保存
        elif save_path is False:
            save_path = None
        # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
        else:
            save_path = save_path

        # 用于分类的列
        if category is not None:
            category = category
        else:
            category = Statistics.Category_Index

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

        # 获取方差解释率（数组形式，长度=2）
        explained_variance_ratio = pca.explained_variance_ratio_

        # 转换 PCA 处理后的结果为 DataFrame，方便后续操作
        pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
        pca_df[category] = category_index

        # 创建绘图对象
        fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

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
            file_name = title + "_PCA.png"  # 初始文件名为 "title_PCA.png"
            full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

            if os.path.exists(full_file_path):  # 查看该文件名是否存在
                count = 1
                file_name = title + "_PCA" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                    count += 1
                    file_name = title + "_PCA" + f"_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

            plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

        # 显示图像
        if show_figure:
            plt.show()
        else:
            plt.clf()  # 清空当前的画板

        if show_result:
            print(f'\033[34mPCA scores\033[0m:\n{pca_df}\n')
            print('\033[32mExplained variance ratio\033[0m:')
            print(f"PC1 explains {explained_variance_ratio[0] * 100:.2f}% of variance")
            print(f"PC2 explains {explained_variance_ratio[1] * 100:.2f}% of variance")
            print()

        # 创建 pca_dic 用于返回 PCA 分析后的数据
        pca_dic = {title: pca_df}
        self.pca_dic = pca_dic

        # 分析载荷图的情况
        if loadings_analysis:
            self.pca_loadings(save_path=save_path)

        return pca_dic

    # PCA 载荷图
    def pca_loadings(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True,
                     margin_ratio: float = 0.1, dpi: int = 600, width_height: tuple = (6, 4.5),
                     category: Optional[str] = None, colors: Optional[list] = None, show_result: bool = True,
                     show_legend: bool = True, **kwargs) -> Dict[str, DataFrame]:
        """
        此方法用于绘制 PCA 结果图中的载荷图，输入 data_dic 的长度需为 1
        This method is used to plot the load plot in the PCA result plot,
        and the length of the input data_dic must be 1.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param margin_ratio: (float) 数据距边界的比例，该值需要介于 0 至 1 之间，默认为 0.1
        :param dpi: (int) 图像保存的精度
        :param width_height: (tuple) 图片的宽度和高度，默认为(6, 4.5)
        :param category: (str) 用于分类的列，默认为 Statistics.Category_Index
        :param colors: (str / list) 第一个颜色为箭头颜色，第二个颜色为文本颜色，默认为红与黑
        :param show_result: (bool) 是否打印结果，默认为 True。打印内容为 Loadings matrix (载荷矩阵)
        :param show_legend: (bool) 是否显示图例，默认为 True
        :param kwargs: Ellipse 方法中的关键字参数

        :return pca_dic: (dict) PCA 分析后的数据 dict ，键为 title，值为 DataFrame 形式的 PCA 结果

        --- **kwargs ---

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
        if save_path is True:
            save_path = self.save_path
        # 若 save_path 为 False 时，本图形不保存
        elif save_path is False:
            save_path = None
        # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
        else:
            save_path = save_path

        # 用于分类的列
        if category is not None:
            category = category
        else:
            category = Statistics.Category_Index

        # 查看是否给出绘图颜色
        if colors is not None:
            color_palette = colors
        else:
            color_palette = ['red', 'black']

        # 关键字参数初始化
        x_min = kwargs.pop('x_min', None)
        x_max = kwargs.pop('x_max', None)
        y_min = kwargs.pop('y_min', None)
        y_max = kwargs.pop('y_max', None)

        title, data_df = list(data_dic.items())[0]  # 从 data_dic 中获取标题和数据

        # 提取数据部分和类别标签部分
        # category_index = data_df[category]  # 提取名为 Category_Index 的列作为类别标签
        before_pca_df = data_df.drop(columns=[category])  # 剩下的列作为数据部分

        if len(color_palette) < 2:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The number of elements in 'colors' must be at least 2.")

        # 用 0 替换所有缺失值
        before_pca_df = before_pca_df.fillna(0)

        # 标准化处理数据
        x_scaled = StandardScaler().fit_transform(before_pca_df)

        # 使用 PCA 将数据降至 2D
        pca = PCA(n_components=2)
        pca.fit(x_scaled)  # 执行 PCA

        # 获取变量载荷
        pca_loadings = pca.components_.T

        pc1_loadings = pca_loadings[:, 0]  # 主成分 1 的载荷
        pc2_loadings = pca_loadings[:, 1]  # 主成分 2 的载荷

        # 绘制变量载荷图
        fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

        for i, variable in enumerate(before_pca_df.columns):
            plt.arrow(0, 0, pc1_loadings[i], pc2_loadings[i],
                      head_width=0.05, head_length=0.05, color=color_palette[0], alpha=0.8)
            if show_legend:
                plt.text(pc1_loadings[i] * 1.2, pc2_loadings[i] * 1.2, variable,
                         color=color_palette[1],
                         ha='center',
                         va='center',
                         fontfamily=self.font_ticket['family'],
                         fontweight=self.font_ticket['weight'],
                         fontsize=self.font_ticket['size'])

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

        # 在你设置轴范围之前，设置边距
        ax.margins(margin_ratio)

        plt.tight_layout()

        # 如果提供了保存路径，则保存图像到指定路径
        if save_path is not None:  # 如果 save_path 的值不为 None，则保存
            file_name = title + "_PCA_loadings.png"  # 初始文件名为 "title_loadings.png"
            full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

            if os.path.exists(full_file_path):  # 查看该文件名是否存在
                count = 1
                file_name = title + "_PCA_loadings" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                    count += 1
                    file_name = title + "_PCA_loadings" + f"_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

            plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

        # 显示图像
        plt.show()

        if show_result:
            print(f'\033[33mLoadings matrix\033[0m:\n{pca_loadings}')
            print()

        # 创建 pca_dic 用于返回 PCA 分析后的数据
        pca_loadings_dic = {title: pca_loadings}
        self.pca_loadings_dic = pca_loadings_dic

        return pca_loadings_dic

    # 树状聚类分析
    def dendrogram_clustering(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True,
                              method: str = 'ward', metric: str = 'euclidean', dpi: int = 600,
                              width_height: tuple = (6, 4.5), threshold: float = 0.7,
                              above_color: Union[str, tuple] = "b", p: int = None, hide_x: bool = True,
                              tree_linewidth: Union[int, float] = 2, axes_linewidth: Union[int, float] = 3,
                              show_result: bool = True, **kwargs) -> Dict[str, np.ndarray]:
        """
        此方法用于绘制树状聚类分析图，输入的 data_dic 的长度需为 1
        This method is used to draw the tree cluster analysis graph, and the length of the input data_dic should be 1.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param method: (str)  参数定义了聚类过程中使用的合并方法，默认为 'ward'。常见的几种方法包括：
                       ['ward', 'single', 'complete', 'average', 'centroid', 'median', 'ward.D', 'ward.D2']
        :param metric: (str) 参数定义了距离度量方式，即在计算样本之间的相似度时使用的度量标准，默认为 'euclidean'。常见的度量方式包括：
                       ['euclidean', 'cityblock', 'cosine', 'minkowski', 'chebyshev', 'hamming', 'jaccard']
        :param dpi: (int) 图像保存的精度
        :param width_height: (tuple) 图片的宽度和高度，默认为(6, 4.5)
        :param threshold: (float) 在 0 至 1 之间，表示上层颜色的值，默认为 0.7
        :param above_color: (str / tuple) 最上方树杈的颜色
        :param p: (int) 横坐标，即叶子节点的个数，默认为 None，表示不限制
        :param hide_x: (bool) 是否隐藏 X 轴刻度，表示样品名，转为为 True
        :param tree_linewidth: (int / float) 树杈的粗线，默认为 2
        :param axes_linewidth: (int / float) 边框的粗线，默认为 3
        :param show_result: (bool) 是否打印结果，默认为 True

        :return tree_dic: (dict) 聚类分析后的数据 dict ，键为 title，值为层次聚类的链接矩阵 (linkage matrix)

        --- **kwargs ---

        - x_label: (str) X 轴的标题
        - y_label: (str) Y 轴的标题
        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值
        - hide_top: (bool) 隐藏上框，默认为 True
        - hide_bottom: (bool) 隐藏下框，默认为 True
        - hide_left: (bool) 隐藏左框，默认为 False
        - hide_right: (bool) 隐藏右框，默认为 True
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
        if save_path is True:
            save_path = self.save_path
        # 若 save_path 为 False 时，本图形不保存
        elif save_path is False:
            save_path = None
        # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
        else:
            save_path = save_path

        # 关键字参数初始化
        x_label = kwargs.pop('x_label', '')
        y_label = kwargs.pop('y_label', '')
        x_min = kwargs.pop('x_min', None)
        x_max = kwargs.pop('x_max', None)
        y_min = kwargs.pop('y_min', None)
        y_max = kwargs.pop('y_max', None)
        hide_top = kwargs.pop('hide_top', True)
        hide_bottom = kwargs.pop('hide_bottom', True)
        hide_left = kwargs.pop('hide_left', False)
        hide_right = kwargs.pop('hide_right', True)

        title, data_df = list(data_dic.items())[0]  # 从 data_dic 中获取标题和数据

        # 检查 data_df 中是否存在 self.Category_Index列
        if self.Category_Index in data_df.columns:
            # 提取数据部分和类别标签部分
            # category_index = data_df[self.Category_Index]  # 提取名为 Category_Index 的列作为类别标签
            data_df = data_df.drop(columns=[self.Category_Index])

        # 用 0 替换所有缺失值
        data_df = data_df.fillna(0)
        # 聚类合并
        z_data = sch.linkage(data_df, method=method, metric=metric)

        # 创建绘图对象
        fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

        # 画树状图，使用条件表达式传递 p
        if p is not None:
            sch.dendrogram(z_data,
                           ax=ax,
                           p=p,
                           truncate_mode="lastp",
                           above_threshold_color=above_color,
                           color_threshold=threshold * max(z_data[:, 2]),
                           **kwargs)
        else:
            sch.dendrogram(z_data,
                           ax=ax,
                           truncate_mode="lastp",
                           above_threshold_color=above_color,
                           color_threshold=threshold * max(z_data[:, 2]),
                           **kwargs)

        # **修改树状图连线的宽度**
        for line in ax.collections:
            line.set_linewidth(tree_linewidth)  # 这里设置线条宽度

        # 设置刻度限制
        if x_min is not None or x_max is not None:
            plt.xlim((x_min, x_max))
        if y_min is not None or y_max is not None:
            plt.ylim((y_min, y_max))

        # **去除横坐标**
        if hide_x:
            ax.set_xticks([])  # 隐藏 X 轴刻度

        # 加粗纵坐标
        ax.spines['top'].set_linewidth(axes_linewidth)  # 加粗上轴
        ax.spines['bottom'].set_linewidth(axes_linewidth)  # 加粗下轴
        ax.spines['left'].set_linewidth(axes_linewidth)  # 加粗左轴
        ax.spines['right'].set_linewidth(axes_linewidth)  # 加粗右轴
        # 是否隐藏顶部、右侧和底部的边框
        if hide_top:
            ax.spines['top'].set_visible(False)
        if hide_bottom:
            ax.spines['bottom'].set_visible(False)
        if hide_left:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])  # 隐藏 Y 轴刻度
        if hide_right:
            ax.spines['right'].set_visible(False)

        # 设置坐标轴字体
        plt.xlabel(xlabel=x_label, fontdict=self.font_title)
        plt.ylabel(ylabel=y_label, fontdict=self.font_title)

        # 设置刻度标签的字体
        plt.xticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])
        plt.yticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])
        ax.tick_params(axis='both', which='major', direction='in')

        # 在你设置轴范围之前，设置边距
        ax.margins(0.1)

        plt.tight_layout()

        # 如果提供了保存路径，则保存图像到指定路径
        if save_path is not None:  # 如果 save_path 的值不为 None，则保存
            file_name = title + "_tree.png"  # 初始文件名为 "title_tree.png"
            full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

            if os.path.exists(full_file_path):  # 查看该文件名是否存在
                count = 1
                file_name = title + "_tree" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                    count += 1
                    file_name = title + "_tree" + f"_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

            plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

        # 显示图像
        plt.show()

        if show_result:
            print(z_data)

        # 创建 tree_dic 用于返回聚类分析后的数据
        tree_dic = {title: z_data}
        self.tree_dic = tree_dic

        return tree_dic

    # 层次聚类分析
    def agglomerative_clustering(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True,
                                 method: str = 'ward', metric: str = 'euclidean', dpi: int = 600,
                                 width_height: tuple = (6, 4.5), n_clusters: int = 2, colors: list = None,
                                 point_style: str = 'o', point_size: float = 3, show_result: bool = True,
                                 **kwargs) -> Dict[str, np.ndarray]:
        """
        此方法用于二维 (两列数据) 的层次聚类分析图，输入的 data_dic 的长度需为 1
        This method is used for a two-dimensional (two-column data) hierarchical cluster analysis graph, and
        the length of the input data_dic should be 1.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame，数据只会用到前两列
        :param save_path: (str) 图片的保存路径
        :param method: (str)  参数定义了聚类过程中使用的合并方法，默认为 'ward'。常见的几种方法包括：
                       ['ward', 'single', 'complete', 'average', 'centroid', 'median', 'ward.D', 'ward.D2']
        :param metric: (str) 参数定义了距离度量方式，即在计算样本之间的相似度时使用的度量标准，默认为 'euclidean'。常见的度量方式包括：
                       ['euclidean', 'cityblock', 'cosine', 'minkowski', 'chebyshev', 'hamming', 'jaccard']
        :param dpi: (int) 图像保存的精度
        :param width_height: (tuple) 图片的宽度和高度，默认为(6, 4.5)
        :param n_clusters: (int) 将数据分为几类，默认为自动分类，建议手动赋值
        :param colors: (list) 散点颜色，有默认颜色
        :param point_style: (str) 最上方散点的风格，即 scatter 图像散点的风格，默认为点状
        :param point_size: (float) 散点的大小，默认为 3
        :param show_result: (bool) 是否打印结果，默认为 True

        :return z_dic: (dict) 聚类分析后的数据 dict ，键为 title，值为层次聚类的链接矩阵 (linkage matrix)

        --- **kwargs ---

        - x_label: (str) X 轴的标题
        - y_label: (str) Y 轴的标题
        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值
        - hide_top: (bool) 隐藏上框，默认为 False
        - hide_bottom: (bool) 隐藏下框，默认为 False
        - hide_left: (bool) 隐藏左框，默认为 False
        - hide_right: (bool) 隐藏右框，默认为 False
        """

        # 将需要处理的数据赋给 data_dic
        if data_dic is not None:
            data_dic = copy.deepcopy(data_dic)
        else:
            data_dic = copy.deepcopy(self.data_dic)

        # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
        if save_path is True:
            save_path = self.save_path
        # 若 save_path 为 False 时，本图形不保存
        elif save_path is False:
            save_path = None
        # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
        else:
            save_path = save_path

        # 检查 colors 是否为 None 且长度是否与 n_clusters 一致
        if colors is not None and len(colors) != n_clusters:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Length of colors ({len(colors)}) list must match n_clusters ({n_clusters}).")

        # 查看是否给出绘图颜色
        if colors is not None:
            color_palette = ListedColormap(colors)
        else:
            color_palette = ListedColormap(self.color_palette)

        # 关键字参数初始化
        x_label = kwargs.pop('x_label', '')
        y_label = kwargs.pop('y_label', '')
        x_min = kwargs.pop('x_min', None)
        x_max = kwargs.pop('x_max', None)
        y_min = kwargs.pop('y_min', None)
        y_max = kwargs.pop('y_max', None)
        hide_top = kwargs.pop('hide_top', False)
        hide_bottom = kwargs.pop('hide_bottom', False)
        hide_left = kwargs.pop('hide_left', False)
        hide_right = kwargs.pop('hide_right', False)

        title, data_df = list(data_dic.items())[0]  # 从 data_dic 中获取标题和数据

        # 检查 data_df 中是否存在 self.Category_Index列
        if self.Category_Index in data_df.columns:
            # 提取数据部分和类别标签部分
            # category_index = data_df[self.Category_Index]  # 提取名为 Category_Index 的列作为类别标签
            data_df = data_df.drop(columns=[self.Category_Index])

        # 用 0 替换所有缺失值
        data_df = data_df.fillna(0)
        # 进行层次聚类
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=method, metric=metric)
        # 进行层次聚类，并返回每个样本的聚类标签
        y_hc = hc.fit_predict(data_df)

        # 进行 PCA 绘制时让数据降维 (不会影响聚类结果，只是用于绘图)
        pca_dic = self.pca_analysis(data_dic=data_dic,
                                    save_path=False,
                                    draw_ellipse=False,
                                    show_result=False,
                                    show_figure=False)
        pca_data_df = list(pca_dic.values())[0]  # 从 pca_dic 中获取数据

        # 创建绘图对象
        fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

        # 画出聚类结果
        plt.scatter(x=pca_data_df.iloc[:, 0],
                    y=pca_data_df.iloc[:, 1],
                    c=y_hc,
                    cmap=color_palette,
                    marker=point_style,
                    s=point_size,
                    **kwargs)

        # 设置刻度限制
        if x_min is not None or x_max is not None:
            plt.xlim((x_min, x_max))
        if y_min is not None or y_max is not None:
            plt.ylim((y_min, y_max))

        # 是否隐藏顶部、右侧和底部的边框
        if hide_top:
            ax.spines['top'].set_visible(False)
        if hide_bottom:
            ax.spines['bottom'].set_visible(False)
        if hide_left:
            ax.spines['left'].set_visible(False)
        if hide_right:
            ax.spines['right'].set_visible(False)

        # 设置坐标轴字体
        plt.xlabel(xlabel=x_label, fontdict=self.font_title)
        plt.ylabel(ylabel=y_label, fontdict=self.font_title)

        # 设置刻度标签的字体
        plt.xticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])
        plt.yticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])
        ax.tick_params(axis='both', which='major', direction='in')

        # 在你设置轴范围之前，设置边距
        ax.margins(0.1)

        plt.tight_layout()

        # 如果提供了保存路径，则保存图像到指定路径
        if save_path is not None:  # 如果 save_path 的值不为 None，则保存
            file_name = title + "_agglomerative.png"  # 初始文件名为 "title_agglomerative.png"
            full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

            if os.path.exists(full_file_path):  # 查看该文件名是否存在
                count = 1
                file_name = title + "_agglomerative" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                    count += 1
                    file_name = title + "_agglomerative" + f"_{count}.png"
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

            plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

        # 显示图像
        plt.show()

        if show_result:
            print(y_hc)

        # 创建 agglomerative_dic 用于返回聚类分析后的数据
        agglomerative_dic = {title: y_hc}
        self.agglomerative_dic = agglomerative_dic

        return agglomerative_dic


""" 绘图 """
class Plotter(general.Manager):
    """
    绘图工具

    1.  有关键字参数时，用 pop 来获取，以确保关键字变量在原曲线方法中仍然有效
    2.  部分函数，对表格处理时需要删除 general.Category_Index 列

    1.  When there are keyword parameters, use pop to obtain them to ensure that the keyword variables remain valid
        in the original curve method.
    2.  For some functions, when processing tables, the general.Category_Index column needs to be deleted.
    """

    Category_Index = general.Category_Index  # 分类索引

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
            self.magic_database = self.Magic_Database

        # 数据初始化分配
        if type(self) == Plotter:  # 当 self 为 Plotter 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # general.Manager 及其子类需要调用以初始化属性

    # 箱线图
    def plot_box(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                 x_label: Optional[str] = None, y_label: Optional[str] = None, show_grid: bool = True,
                 colors: Optional[list] = None, **kwargs) -> None:
        """
        此方法用于绘制箱线图
        This method is used to plot a box plot.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param show_grid: (bool) 是否绘制网格，默认为 True
        :param colors: (list) 绘制的颜色
        :param kwargs: boxplot 方法中的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = colors
            else:
                color_palette = self.color_palette

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 检查并删除 Category_Index 列
            if Plotter.Category_Index in data_df.columns:
                data_df = data_df.drop(columns=Plotter.Category_Index)

            # 绘制箱线图
            plt.figure(figsize=width_height, dpi=200)

            # 执行 boxplot 函数
            bp = data_df.boxplot(patch_artist=True, return_type='dict', **kwargs)

            plt.grid(show_grid)  # 绘制网格

            # 设置箱线图颜色和线条粗细
            for patch, color in zip(bp['boxes'], color_palette):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            for whisker in bp['whiskers']:
                whisker.set_linewidth(2)

            for cap in bp['caps']:
                cap.set_linewidth(2)

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 坐标轴标题字体
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

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

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 小提琴图
    def plot_violin(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                    x_label: Optional[str] = None, y_label: Optional[str] = None, show_grid: bool = True,
                    colors: Optional[list] = None, show_line: bool = False, width: float = 0.6, **kwargs) -> None:
        """
        此方法用于绘制一个组小提琴图
        This method is used to plot a vollin plot.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param dpi: 保存图片的精度，默认为 600
        :param x_label: X 轴的标题
        :param y_label: Y 轴的标题
        :param show_grid: 是否绘制网格，默认为 True
        :param colors: 绘制的颜色
        :param show_line: (bool) 显示连接线
        :param width: (float) 小提琴部分的宽度，默认为 0.6
        :param kwargs: 小提琴方法中的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = colors
            else:
                color_palette = self.color_palette

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 检查并删除 Category_Index 列
            if Plotter.Category_Index in data_df.columns:
                data_df = data_df.drop(columns=Plotter.Category_Index)

            melted_data = pd.melt(data_df)

            # 创建画布
            fig, ax = plt.subplots(figsize=width_height, dpi=200, facecolor="w")

            # 小提琴图 (显示分布)
            sns.violinplot(x="variable", y="value", data=melted_data,
                           hue="variable", palette=color_palette, width=width,
                           linewidth=0.5, saturation=1, legend=False, ax=ax, **kwargs)

            # 透明箱线图 (仅显示框线，隐藏箱体)
            sns.boxplot(x="variable", y="value", data=melted_data,
                        width=0.2, showcaps=False, boxprops={"facecolor": "none"},
                        whiskerprops={'linewidth': 0}, medianprops={'color': 'black'},
                        ax=ax)

            # 散点图 (显示数据点)
            sns.stripplot(x="variable", y="value", data=melted_data,
                          size=4, color="black", alpha=0.6, ax=ax)

            # 折线图 (替代 pointplot=show_line)
            if show_line:  # 只有 show_line=True 时才绘制折线
                sns.pointplot(x="variable", y="value", data=melted_data,
                              color="black", linestyles="-", markers="o",
                              markersize=5, linewidth=1, ax=ax)

            plt.grid(show_grid)  # 绘制网格

            ax.set_xlim(-1, len(data_df.columns))  # 扩大一列以显示所有的数据 (第三方库的 BUG 导致)

            ax.set_xlabel(x_label)  # X 轴标题
            ax.set_ylabel(y_label)  # Y 轴标题

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # X 轴标题字体
            plt.xlabel(x_label, fontdict=self.font_title)
            # Y 轴标题字体
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 热图
    def plot_heatmap(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                     x_label: Optional[str] = None, y_label: Optional[str] = None, display_digit: bool = True,
                     colors: all = None, **kwargs) -> None:
        """
        此方法用于绘制热图图像
        This method is used to draw a heatmap.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'X'
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Y'
        :param display_digit: (bool) 是否显示数字，若为 True 则在图中会显示数字，默认为 True
        :param colors: (str) 颜色色系的名称，默认为红蓝色系，建议用渐变色调色板
        :param kwargs: 绘制热图时的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - custom_xticks: (list) 自定义 X 轴刻度
        - custom_yticks: (list) 自定义 Y 轴刻度
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度

        - cbar: (bool) 是否显示色条
        - cbar_interval: (float) 色条数值的显示间隔
        - vmin: (float) 色条的最小值
        - vmax: (float) 色条的最大值

        - text_mapping: (dict) 数字与其它文本的映射，比 display_digit 优先度更高，默认为无映射
        - txt_format: (str) 方框中的内容的显示格式，'data_dic' 表示整数，'.2f' 表示小数点后两位有效数字，默认为不改变
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 获取调色板，建议用渐变色调色板
            if colors is not None:
                if isinstance(colors, str):
                    color_palette = sns.color_palette(colors)
                else:
                    color_palette = colors
            else:
                color_palette = sns.color_palette('vlag')

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            custom_xticks = kwargs.pop('custom_xticks', True)
            custom_yticks = kwargs.pop('custom_yticks', True)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            show_cbar = kwargs.get('cbar', True)  # 是否显示色条，该项在绘制时需要用到，故用 get 属性
            cbar_interval = kwargs.pop('cbar_interval', None)
            cbar_min = kwargs.pop('cbar_min', None)
            cbar_max = kwargs.pop('cbar_max', None)

            text_mapping = kwargs.pop('text_mapping', None)
            txt_format = kwargs.pop('txt_format', '')

        for title, data_df in data_dic.items():

            # 检查并删除 Category_Index 列
            if Plotter.Category_Index in data_df.columns:
                data_df = data_df.drop(columns=Plotter.Category_Index)

            # 将数字数据转换为字母数据
            if text_mapping is not None:
                data_df = data_df.round().astype(int)  # 将数据四舍五入到整数
                letter_data_df = data_df.applymap(text_mapping.get)

            else:
                letter_data_df = None

            annot_value = letter_data_df if letter_data_df is not None else display_digit  # 使用动态选择的注释数据

            # 绘制热图
            plt.figure(figsize=width_height, dpi=200)
            ax = sns.heatmap(data_df,
                             cmap=color_palette,
                             annot=annot_value,
                             annot_kws=self.font_legend,
                             fmt=txt_format,
                             xticklabels=custom_xticks,
                             yticklabels=custom_yticks,
                             vmin=cbar_min,
                             vmax=cbar_max,
                             **kwargs)

            # 更改色条显示
            if show_cbar:
                # 获取原始颜色条对象
                cbar = ax.collections[0].colorbar

                # 设置色条刻度的间隔
                if cbar_interval is not None:
                    locator = MultipleLocator(base=cbar_interval)
                    cbar.ax.yaxis.set_major_locator(locator)
                    plt.draw()  # 这里需要重新绘制，确保色条刻度更新

                # 使用映射创建自定义的色条标签
                if text_mapping is not None:
                    labels = [text_mapping.get(int(float(item.get_text())), '') for item in cbar.ax.get_yticklabels()]
                    locations = cbar.ax.get_yticks()
                    cbar.ax.yaxis.set_major_locator(plt.FixedLocator(locations))  # 添加回 FixedLocator
                    cbar.ax.set_yticklabels(labels, minor=False)

                # 获取刻度标签对象并设置字体样式
                yticklabels = cbar.ax.get_yticklabels()
                for label in yticklabels:
                    label.set_size(self.font_ticket['size'])
                    label.set_weight(self.font_ticket['weight'])
                    label.set_family(self.font_ticket['family'])

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 设置坐标轴标签和标题
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 条形统计图
    def plot_bars(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                  x_label: Optional[str] = None, y_label: Optional[str] = None, show_line: Union[bool, float] = False,
                  show_label: bool = False, colors: all = None, bar_width: float = 0.15, **kwargs) -> None:
        """
        绘制带有平均线的条形图
        Plotting bar charts with average lines.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'X'
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Y'
        :param show_line: (bool/float) 是否显示水平线，为 True 时显示均值，为 float 时为该值，默认为 False
        :param show_label: (bool) 是否显示图例，默认为 False
        :param colors: (str) 颜色色系的名称，建议用渐变色调色板
        :param bar_width: (float) 条的宽度，默认为 0.15
        :param kwargs: 绘制条形图时的关键字参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值

        - custom_xticks: (list) 自定义 X 轴刻度
        - custom_yticks: (list) 自定义 Y 轴刻度
        - custom_legend: (list) 自定义注释，只有 show_label == True 时才有意义
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = colors
            else:
                color_palette = self.color_palette

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            custom_legend = kwargs.pop('custom_legend', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 检查并删除 Category_Index 列
            if Plotter.Category_Index in data_df.columns:
                data_df = data_df.drop(columns=Plotter.Category_Index)

            # 从 DataFrame 中提取数据
            data = data_df.values
            num_x_values = len(data_df.index.tolist())
            num_y_values = len(data_df.columns.tolist())
            x = np.arange(num_y_values)

            if show_line is True:
                mean_values = data.mean(axis=0)
            else:
                mean_values = None

            # 创建图形
            plt.figure(figsize=width_height, dpi=200)

            # custom_legend 被赋值的情况
            if custom_legend is not None:

                # 长度一致时
                if len(custom_legend) == num_x_values:

                    for i in range(num_x_values):
                        bars = plt.bar(x + i * bar_width, data[i, :],
                                       width=bar_width,
                                       label=custom_legend[i],
                                       color=color_palette[i % len(color_palette)]
                                       )

                        # 添加自定义线
                        if isinstance(show_line, float) or isinstance(show_line, int):
                            plt.axhline(y=show_line,
                                        color='red',
                                        linestyle='--',
                                        alpha=0.7,
                                        )

                        # 添加均值线
                        elif show_line is True:
                            plt.axhline(y=mean_values[i],
                                        color=bars[i].get_facecolor(),
                                        linestyle='--',
                                        alpha=0.7,
                                        )

                # 长度不一致时
                else:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"the length of the colors list should match the number of categories. "
                                     f"The DataFrame line number should be the same as the custom_legend length. "
                                     f"The DataFrame has {num_x_values} lines, "
                                     f"but the custom_legend is {len(custom_legend)}.")

            # custom_legend 未被赋值的情况
            else:

                for i in range(num_x_values):
                    bars = plt.bar(x + i * bar_width, data[i, :],
                                   width=bar_width,
                                   color=color_palette[i % len(color_palette)]
                                   )

                    # 添加自定义线
                    if isinstance(show_line, float) or isinstance(show_line, int):
                        plt.axhline(y=show_line,
                                    color='red',
                                    linestyle='--',
                                    alpha=0.7,
                                    )

                    # 添加均值线
                    elif show_line is True:
                        plt.axhline(y=mean_values[i],
                                    color=bars[i].get_facecolor(),
                                    linestyle='--',
                                    alpha=0.7,
                                    )

            # 设置刻度限制
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            # X 轴刻度
            if custom_xticks is not None:
                locations, _ = plt.xticks()  # 获取当前图表的 X 轴刻度位置
                valid_locations = locations[1:-1]  # 取用中间有效区间，只有 X 轴需要此操作
                plt.xticks(ticks=valid_locations, labels=custom_xticks)

            # Y 轴刻度
            locations, _ = plt.yticks()  # 获取当前图表的 Y 轴刻度位置
            if custom_yticks is not None:
                locations, _ = plt.yticks()  # 获取当前图表的 Y 轴刻度位置
                plt.yticks(ticks=locations, labels=custom_yticks)

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 设置坐标轴标签和标题
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            if show_label:
                plt.legend(prop=self.font_legend)

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 2D 直方图
    def plot_2d_histogram(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                          x_label: Optional[str] = None, y_label: Optional[str] = None, bar_label: Optional[str] = None,
                          bins: Union[int, list, tuple] = 20, colors: all = None, limit_value: float = 0.5, **kwargs) \
            -> None:
        """
        绘制二维直方图，并尽量确保左侧主图是正方形的。右侧的颜色条与主图之间保留一定的距离
        Draw a two-dimensional histogram and try to make sure that the left main graph is square.
        Leave some distance between the color bar on the right and the main image.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'X'
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Y'
        :param bar_label: (str) 色条的标题，若无赋值则显示 'Number of points per pixel'
        :param bins: (int / list / tuple) 每行 (X) 和每列 (Y) 中小方格的数量，此值越大则精度越高，但需要的数据也大；
                    也可以接收 list 或 tuple，此时第一项为 X 轴有被分成的小方格数量，第二项为 Y 轴被分成的小方格数量，默认为 20
        :param colors: (str) 颜色色系的名称，建议用渐变色调色板
        :param limit_value: (float) 数值低于此值时会显示白色，默认为 0.5 表示无值时
        :param kwargs: 绘制 2D 直方图时的关键字参数

        --- **kwargs ---

        - width_height: (tuple) 图片的宽度和高度，默认为 (6.5, 5)

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值

        - custom_xticks: (list) 自定义 X 轴刻度
        - custom_yticks: (list) 自定义 Y 轴刻度
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (6) 多了个 bar_label
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 获取调色板，建议用渐变色调色板
            if colors is not None:
                if isinstance(colors, str):
                    color_palette = sns.color_palette(colors)
                else:
                    color_palette = colors
            else:
                color_palette = 'plasma'

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

            if bar_label is not None:
                bar_label = bar_label
            else:
                bar_label = 'Number of points per pixel'

            # 关键字参数初始化
            width_height = kwargs.pop('width_height', (6.5, 5))  # 只有 2D 直方图为 (6.5, 5)

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 提取 X 和 Y 值
            x = data_df.iloc[:, 0]
            y = data_df.iloc[:, 1]

            # 创建直方图的图形和轴，调整大小以适应正方形主图和颜色条
            fig = plt.figure(figsize=width_height, dpi=200)

            # 创建具有正方形外观的主轴
            ax = fig.add_subplot(111)

            # 创建二维直方图
            hist2d_params = {'bins': bins, 'cmap': color_palette}
            if limit_value is not None:
                hist2d_params['cmin'] = limit_value

            # 绘制二维直方图
            counts, xedges, yedges, image = ax.hist2d(x, y, **hist2d_params, **kwargs)

            # 设置刻度限制
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            # 添加带刻度的边框
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            # 创建与主图对齐并带有间隙的颜色条
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.2)
            colorbar = plt.colorbar(image, cax=cax)
            colorbar.set_label(bar_label)
            colorbar.ax.tick_params(direction='in', length=3, width=1.5)

            # 设置颜色条刻度标签的格式为一位小数
            colorbar.formatter = FormatStrFormatter('%d')
            colorbar.update_ticks()

            # 设置颜色条两侧的刻度标记
            colorbar.ax.yaxis.set_ticks_position('both')

            # 为主图添加次要刻度
            ax.minorticks_on()
            ax.tick_params(which='minor', size=5, direction='in', colors='black')

            # 延长主刻度以提高可见性
            ax.tick_params(which='major', length=10, width=2)

            # 设置 X 和 Y 轴的标签
            ax.set_xlabel(x_label, fontdict=self.font_title)
            ax.set_ylabel(y_label, fontdict=self.font_title)

            # 在所有边启用刻度，但仅在左轴和下轴显示标签
            ax.tick_params(labeltop=False, labelright=False, top=True, right=True)
            ax.tick_params(which='major', direction='in', length=6, width=2)
            ax.tick_params(which='minor', direction='in', length=3, width=1)

            # X 轴刻度
            if custom_xticks is not None:
                ax.set_xticks(custom_xticks)

            # Y 轴刻度
            if custom_yticks is not None:
                ax.set_yticks(custom_yticks)

            # 设置 X 轴刻度标签的字体、大小和粗细，并旋转
            ax.set_xticklabels(ax.get_xticks(),
                               fontproperties=self.font_ticket,
                               rotation=x_rotation)

            # 设置 Y 轴刻度标签的字体、大小和粗细，并旋转
            ax.set_yticklabels(ax.get_yticks(),
                               fontproperties=self.font_ticket,
                               rotation=y_rotation)

            # 设置 bar 标题及字体、大小并加粗
            plt.ylabel(bar_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'])
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'])

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 山脊图
    def plot_ridge(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                   x_label: Optional[str] = None, y_label: Optional[str] = None, bar_label: Optional[str] = None,
                   colors: all = None, overlap: float = 1, show_bar: bool = True, custom_order: Optional[list] = None,
                   bar_ticks: Union[list, tuple, None] = None, x_range: Union[list, tuple, None] = None,
                   **kwargs) -> None:
        """
        绘制山脊图，并可以调节横坐标，自定义纵坐标，还可以添加色条。数据要求只有两列，其中第一列为数据值，第二列为标识 category
        Draw the ridge map, adjust the horizontal coordinate, customize the vertical coordinate,
        and also add color bars. The data requires the first column to identify the column

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'X'
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Y'
        :param bar_label: (str) 色条的标题，若无赋值则显示 'Intensity'，只有 show_bar == True 时才有意义
        :param colors: (str) 颜色色系的名称，建议用渐变色调色板
        :param overlap: (float) 峰面积的大小，默认为 1
        :param show_bar: (bool) 是否显示色条，默认为 True
        :param custom_order: (list) 自定义显示山脊的顺序，从上至下
        :param bar_ticks: (list / tuple) 自定义色条刻度标签，只有 show_bar == True 时才有意义
        :param x_range: (list / tuple) X 轴的最大最小值，需要两项同时输入
        :param kwargs: 绘制山脊图时的关键字参数

        --- **kwargs ---

        - width_height: (tuple) 图片的宽度和高度，默认为 (6.5, 5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴和色条刻度的旋转角度
        """

        # 检查赋值 (6) 多了个 bar_label
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 获取调色板，建议用渐变色调色板
            if colors is not None:
                if isinstance(colors, str):
                    color_palette = sns.color_palette(colors)
                else:
                    color_palette = colors
            else:
                color_palette = sns.color_palette('plasma')

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

            if bar_label is not None:
                bar_label = bar_label
            else:
                bar_label = 'Intensity'

            # 关键字参数初始化
            width_height = kwargs.pop('width_height', (6, 4))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            # 反转颜色序列
            if isinstance(color_palette, mcolors.ListedColormap):
                color_sequence = color_palette.colors
            else:
                color_sequence = color_palette
            reversed_color_sequence = color_sequence[::-1]

            # 创建一个新的 ListedColormap 对象
            color_palette_reversed = mcolors.ListedColormap(reversed_color_sequence)

        for title, data_df in data_dic.items():

            print(data_df)

            column_first, column_second = data_df.columns[:2]  # 第二列为 category

            # 计算 column_first 中不同数据的数量
            unique_values_count = data_df[column_second].nunique()
            if bar_ticks is not None:
                # 计算 bar_ticks 的长度
                number_bar_ticks = len(bar_ticks)
                if unique_values_count != number_bar_ticks:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"The length of bar_ticks ({number_bar_ticks}) must be the same as the number of "
                                     f"different categories in the original data ({unique_values_count})")

            data_df[Statistics.Category_Index] = pd.Categorical(data_df[Statistics.Category_Index],
                                                                categories=custom_order, ordered=True)  # 自定义顺序

            # 创建 joyplot
            fig, axes = joypy.joyplot(
                data_df,
                by=column_second,
                column=column_first,

                figsize=width_height,
                colormap=color_palette_reversed,  # 使用反转后的 color_palette，因为原图绘制是反的
                x_range=x_range,
                ylabels=bar_ticks,
                overlap=overlap,

                grid="x",
                linewidth=1,

                **kwargs
            )

            # 初始化颜色列表
            joyplot_colors = []

            # 对于每个轴，检查是否有 collections 对象，并提取颜色
            for ax in axes:
                if ax.collections:  # 如果 collections 不为空
                    # collections 可能包含多个对象，我们只关心填充的部分，通常是第一个
                    facecolor = ax.collections[0].get_facecolor()
                    if facecolor.size:  # 检查 facecolor 是否非空
                        joyplot_colors.append(facecolor[0])  # 取第一个颜色

            # 添加 X 轴 标签
            fig.text(0.5, 0.05,
                     x_label,
                     ha='center',
                     va='center',
                     fontdict=self.font_title)

            # 显示色条时，隐藏原 Y 轴标签
            if not show_bar:
                # 添加 Y 轴标签
                fig.text(0.04, 0.5,
                         y_label,
                         ha='center',
                         va='center',
                         fontdict=self.font_title,
                         rotation='vertical')

            # 调整子图之间的间距和边缘间距
            fig.set_dpi(200)

            # 对坐标内详细内容进行调整
            for ax in axes:
                # 为每个轴添加边框
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                    spine.set_color('black')

                # 设置 X 轴标签的大小和字体
                for label in ax.get_xticklabels():
                    label.set_fontname(self.font_ticket['family'])    # 设置字体类型
                    label.set_fontsize(self.font_ticket['size'])      # 设置字体大小
                    label.set_fontweight(self.font_ticket['weight'])  # 设置字体为粗体
                    label.set_rotation(x_rotation)                    # 旋转刻度

                # 设置 Y 轴标签的大小和字体
                for label in ax.get_yticklabels():
                    label.set_fontname(self.font_ticket['family'])    # 设置字体类型
                    label.set_fontsize(self.font_ticket['size'])      # 设置字体大小
                    label.set_fontweight(self.font_ticket['weight'])  # 设置字体为粗体
                    label.set_rotation(y_rotation)                    # 旋转刻度

            # 显示色条时，隐藏原 Y 轴刻度标签
            if show_bar:
                # 隐藏原有的 Y 轴刻度标签
                for ax in axes:
                    ax.set_yticklabels([])

            # 显示色条
            if show_bar:

                if bar_ticks is not None:
                    plt.subplots_adjust(left=0.03, right=0.87, top=0.95, bottom=0.13)
                else:
                    plt.subplots_adjust(left=0.03, right=0.87, top=0.95, bottom=0.13)

                # 假设 joyplot_colors_tuples 包含从 joyplot 提取的颜色元组
                joyplot_colors_tuples = [tuple(color_array) for color_array in joyplot_colors]
                # 将颜色元组转换为十六进制字符串
                hex_colors = [mcolors.to_hex(color) for color in joyplot_colors_tuples]
                # 反转颜色列表
                hex_colors_reversed = hex_colors[::-1]

                # 使用反转后的颜色创建 ListedColormap
                cmap = ListedColormap(colors=hex_colors_reversed)

                # 创建色条
                sm = ScalarMappable(cmap=cmap,
                                    norm=Normalize(vmin=data_df[column_first].min(),
                                                   vmax=data_df[column_first].max()))
                sm.set_array([])  # 只是为了通过 matplotlib 的检查
                cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05, pad=0.03)
                cbar.set_label(bar_label,
                               fontname=self.font_title['family'],  # 设置刻度的字体类型
                               fontsize=self.font_title['size'],    # 设置刻度的字体大小
                               weight=self.font_title['weight'])    # 设置刻度的字体加粗

                # 有 Y 轴标签自定义输入时
                if bar_ticks is not None:

                    # 获取色条的规范化对象
                    norm = sm.norm

                    # 获取色条的上限和下限
                    vmin, vmax = norm.vmin, norm.vmax

                    # 根据标签数量计算刻度位置
                    if len(bar_ticks) > 1:
                        # 计算色条中颜色块的边界
                        boundaries = np.linspace(start=vmin, stop=vmax, num=len(bar_ticks) + 1, endpoint=True)
                        # 计算颜色块的中点位置
                        tick_positions = (boundaries[:-1] + boundaries[1:]) / 2
                    else:
                        # 如果只有一个标签，则放在中间
                        tick_positions = [(vmax + vmin) / 2]

                    # 设置色条的刻度位置
                    cbar.set_ticks(tick_positions)

                    # 设置色条的刻度标签
                    cbar.set_ticklabels(bar_ticks)

                # 设置色条刻度标签的字体大小和加粗
                for label in cbar.ax.get_yticklabels():
                    label.set_family(self.font_ticket['family'])  # 设置刻度的字体类型
                    label.set_fontsize(self.font_ticket['size'])  # 设置刻度的字体大小
                    label.set_weight(self.font_ticket['weight'])  # 设置刻度的字体加粗
                    label.set_rotation(y_rotation)       # 设置色条刻度的旋转

            # 不显示色条
            else:

                plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.13)

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # QQ图
    def plot_qq(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                x_label: Optional[str] = None, y_label: Optional[str] = None, column_index: int = 1,
                color_dot: Union[str, tuple, None] = None, color_line: Union[str, tuple, None] = None,
                markersize: float = 8, linewidth: float = 2, show_label: bool = True, **kwargs) -> None:
        """
        用于绘制Q-Q图，来检验样品是否符合正态分布
        It is used to draw Q-Q plots to test whether the sample conforms to the normal distribution.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'Theoretical quantiles'
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Sample quantiles'
        :param column_index: (int) 用于检验哪一列的数据，默认为 1
        :param color_dot: (str / tuple) 散点的颜色，默认为蓝色，'#0000FF'
        :param color_line: (str / tuple) 散点的颜色，默认为红色，'#FF0000'
        :param markersize: (float) 散点的大小，默认为8
        :param linewidth: (float) 线条的宽度，默认为 2
        :param show_label: (bool) 是否显示图例，默认为 True
        :param kwargs: 绘制 Q-Q 图时的关键字参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值

        - custom_xticks: (list) 自定义 X 轴刻度
        - custom_yticks: (list) 自定义 Y 轴刻度
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (6)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            if x_label is not None:
                x_label = x_label
            else:
                x_label = 'Theoretical quantiles'

            if y_label is not None:
                y_label = y_label
            else:
                y_label = 'Sample quantiles'

            # 原数据的颜色
            if color_dot is not None:
                color_dot = color_dot
            else:
                color_dot = '#0000FF'  # 蓝色

            # 正态分布曲线的颜色
            if color_line is not None:
                color_line = color_line
            else:
                color_line = '#FF0000'  # 红色

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 检查索引是否在 DataFrame 的列数范围内
            if column_index < 0 or column_index >= len(data_df.columns):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The index '{column_index}' is outside the scope of the DataFrame column")

            # 提取列数据
            data = data_df.iloc[:, column_index]

            # 生成 QQ 图
            fig = plt.figure(figsize=width_height, dpi=200)
            ax = fig.add_subplot(111)
            (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)

            # 绘制 QQ 图的点，黑色外圈和内圈
            ax.plot(osm, osr, 'o',
                    color='black',
                    markerfacecolor=color_dot,
                    markersize=markersize,
                    label='sample quantiles')

            # 绘制拟合线，暗红色
            line = slope * osm + intercept
            ax.plot(osm, line,
                    color=color_line,
                    linewidth=linewidth,
                    label='line of best fit')

            # 设置刻度限制
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            # X 轴刻度
            if custom_xticks is not None:
                ax.set_xticks(custom_xticks)

            # Y 轴刻度
            if custom_yticks is not None:
                ax.set_yticks(custom_yticks)

            # 设置 X 和 Y 轴的标签
            ax.set_xlabel(x_label, fontdict=self.font_title)
            ax.set_ylabel(y_label, fontdict=self.font_title)

            # 设置刻度限制
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            # X 轴刻度
            if custom_xticks is not None:
                locations, _ = plt.xticks()  # 获取当前图表的 X 轴刻度位置
                valid_locations = locations[1:-1]  # 取用中间有效区间，只有 X 轴需要此操作
                plt.xticks(ticks=valid_locations, labels=custom_xticks)

            # Y 轴刻度
            locations, _ = plt.yticks()  # 获取当前图表的 Y 轴刻度位置
            if custom_yticks is not None:
                locations, _ = plt.yticks()  # 获取当前图表的 Y 轴刻度位置
                plt.yticks(ticks=locations, labels=custom_yticks)

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                column_name = data_df.columns[column_index]
                plt.title(column_name, fontdict=self.font_title)

            # 设置坐标轴标签和标题
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            if show_label:
                plt.legend(prop=self.font_legend)

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 概率分布图
    def plot_distribution(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                          x_label: Optional[str] = None, y_label: Optional[str] = None, density: bool = True,
                          bins: Union[str, int] = 'auto', color_data: Union[str, tuple, None] = None,
                          color_kde: Union[str, tuple, None] = None, color_normal: Union[str, tuple, None] = None,
                          show_label: bool = True, **kwargs) -> None:
        """
        绘制 DataFrame 中单列数据的概率分布直方图，并添加正态分布曲线。注意该方法只接收只有一列的 DataFrame 表格
        Plot the probability distribution histogram for a single column of data in the DataFrame
        and add a normal distribution curve. Note that this method accepts only a DataFrame table with only one column.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'Density' or 'Count' (依据情况)
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Data'
        :param density: (bool) 是否为概率密度图，否为计数图，默认为 True
        :param bins: (int) 直方图的分格数量，默认为自动调节
        :param color_data: (str / tuple) 原数据绘制成的线的颜色，默认为天蓝色 ('#87CEEB')
        :param color_kde: (str / tuple) 核密度曲线的颜色，默认为深蓝色 ('#00008B')，只有 density == True 时才有意义
        :param color_normal: (str / tuple) 正太分布曲线的颜色，默认为红色 ('#FF0000')，只有 density == True 时才有意义
        :param show_label: (bool) 是否显示图例，默认为 True
        :param kwargs: 绘制概率分布图时的关键字参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为 True
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴和色条刻度的旋转角度
        """

        # 检查赋值 (7)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            if x_label is not None:
                x_label = x_label
            else:
                x_label = 'Data'

            if y_label is not None:
                y_label = y_label
            else:
                y_label = 'Density' if density else 'Count'

            # 原数据的颜色
            if color_data is not None:
                color_data = color_data
            else:
                color_data = '#6FB9D6'  # 天蓝色

            # 正态分布曲线的颜色
            if color_kde is not None:
                color_kde = color_kde
            else:
                color_kde = '#00008B'  # 暗蓝色

            # 正态分布曲线的颜色
            if color_normal is not None:
                color_normal = color_normal
            else:
                color_normal = '#FF0000'  # 红色

            # 关键字参数初始化
            image_title = kwargs.pop('title', True)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 检查并删除 Category_Index 列
            if Plotter.Category_Index in data_df.columns:
                data_df = data_df.drop(columns=Plotter.Category_Index)

            # 检查 Dataframe 是否只有一列
            if data_df.shape[1] != 1:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The DataFrame should contain only one column of data.")

            plt.figure(figsize=width_height)

            # 绘制直方图
            sns.histplot(data_df.iloc[:, 0],  # 提取数据列，否则会导致 bug：颜色无法正常更换
                         kde=False,  # 不自动绘制，因为会有 bug，在下面代码中已手动绘制
                         stat="density" if density else "count",
                         linewidth=1,
                         bins=bins,
                         color=color_data,
                         label='Original data',
                         **kwargs
                         )

            # 计算概率时，手动计算 KDE，KDE 是 Kernel Density Estimation（核密度估计）
            if density:

                data = data_df.iloc[:, 0]  # 提取第一列作为一维数组
                kde = gaussian_kde(data)  # 提取第一列作为一维数组
                x_range = np.linspace(start=data.min(), stop=data.max(), num=1000)
                kde_values = kde(x_range)

                # 手动绘制 KDE 曲线
                plt.plot(x_range, kde_values,
                         color=color_kde,
                         lw=2,
                         label='Fitting curve')

            mean = data_df.mean().item()  # 提取单个数值
            std = data_df.std().item()  # 提取单个数值

            # 如果 density 为 True，则添加正态分布曲线
            if density:
                xmin, xmax = plt.xlim()
                x = np.linspace(start=xmin, stop=xmax, num=100)
                p = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
                p = p / (std * np.sqrt(2 * np.pi))
                plt.plot(x, p,
                         color=color_normal,
                         linewidth=2,
                         label='Normal distribution')

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title +
                          ': mean = {:.2f}, std = {:.2f}'.format(mean, std), fontdict=self.font_title)
            elif image_title is True:
                plt.title(title +
                          ': mean = {:.2f}, std = {:.2f}'.format(mean, std), fontdict=self.font_title)

            # 设置坐标轴标签和标题
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            if show_label:
                plt.legend(prop=self.font_legend)

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 联合分布图
    def plot_jointdistribution(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True,
                               dpi: int = 600, x_label: Optional[str] = None, y_label: Optional[str] = None,
                               x_input: int = 0, y_input: int = 1, category: Optional[str] = None,
                               kind: str = 'scatter', colors: Union[list, tuple, None] = None, show_legend: bool = True,
                               **kwargs) -> None:
        """
        绘制联合分布的散点图
        Plot a scatter plot of the joint distribution.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) 横坐标的标题，若无赋值则会显示 self.x_label，若仍无赋值则显示 'X'
        :param y_label: (str) 纵坐标的标题，若无赋值则会显示 self.y_label，若仍无赋值则显示 'Y'
        :param x_input: (int) 使用的 X 轴数据在第几列，默认为第一列，0
        :param y_input: (int) 使用的 Y 轴数据在第几列，默认为第二列，1
        :param category: (str) 数据分类所用的列的名称，默认为 Plotter.Category_Index
        :param kind: (str) 联合分布的数据类型，默认为 'scatter'，还有 'reg', 'resid', 'kde', 'hex'
        :param colors: (list / tuple) 用于绘制的颜色，默认为 sns 配色，此项长度必须与数据各类一致
        :param show_legend: (bool) 是否显示图例，默认为 True
        :param kwargs: 绘制条形图时的关键字参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) X 轴最小值
        - x_max: (float) X 轴最大值
        - y_min: (float) Y 轴最小值
        - y_max: (float) Y 轴最大值

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (6)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            if x_label is not None:
                x_label = x_label
            else:
                x_label = 'X'

            if y_label is not None:
                y_label = y_label
            else:
                y_label = 'Y'

            # 数据的颜色
            if colors is not None:
                colors = colors
            else:
                colors = [
                            "#1f77b4",  # 蓝色
                            "#ff7f0e",  # 橙色
                            "#2ca02c",  # 绿色
                            "#d62728",  # 红色
                            "#9467bd",  # 紫色
                            "#8c564b",  # 棕色
                            "#e377c2",  # 粉红色
                            "#7f7f7f",  # 灰色
                            "#bcbd22",  # 黄绿色
                            "#17becf"   # 青色
                        ]

            # 分类的列
            if category is not None:
                category = category
            else:
                category = Plotter.Category_Index

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        for title, data_df in data_dic.items():

            # 取出 X 和 Y 的数据
            first_column = data_df.columns[x_input]
            second_column = data_df.columns[y_input]

            # 获取前部分颜色，否则颜色与数量各类不同引发错误，sns.jointplot() 的问题
            num_unique_categories = data_df[category].nunique()
            colors = colors[:num_unique_categories]

            # 创建联合分布图，该函数返回一个 JointGrid 对象
            g = sns.jointplot(data=data_df,
                              x=first_column,
                              y=second_column,
                              hue=category,
                              kind=kind,
                              palette=colors,
                              xlim=(x_min, x_max) if x_min is not None or x_max is not None else None,
                              ylim=(y_min, y_max) if y_min is not None or y_max is not None else None,
                              legend=show_legend,
                              space=0,
                              **kwargs)

            # 调整图形大小
            g.fig.set_size_inches(width_height)
            g.fig.set_dpi(200)

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            g.ax_joint.set_xlabel(x_label, fontdict=self.font_title)
            g.ax_joint.set_ylabel(y_label, fontdict=self.font_title)

            # 设置 X 轴和 Y 轴的刻度线向内
            g.ax_joint.tick_params(axis='x', direction='in')
            g.ax_joint.tick_params(axis='y', direction='in')

            # 设置 X 轴刻度标签的字体、大小和加粗
            for label in g.ax_joint.get_xticklabels():
                label.set_family(self.font_ticket['family'])  # 设置刻度的字体类型
                label.set_fontsize(self.font_ticket['size'])  # 设置刻度的字体大小
                label.set_weight(self.font_ticket['weight'])  # 设置刻度的字体加粗
                label.set_rotation(x_rotation)  # 设置色条刻度的旋转

            # 设置 Y 轴刻度标签的字体、大小和加粗
            for label in g.ax_joint.get_yticklabels():
                label.set_family(self.font_ticket['family'])  # 设置刻度的字体类型
                label.set_fontsize(self.font_ticket['size'])  # 设置刻度的字体大小
                label.set_weight(self.font_ticket['weight'])  # 设置刻度的字体加粗
                label.set_rotation(y_rotation)  # 设置色条刻度的旋转

            # 设置图例标题和标签的字体、大小和加粗
            if show_legend:
                # 获取图例对象
                leg = g.ax_joint.get_legend()
                # 设置图例标题的样式
                leg.set_title(None)

                # 设置每个图例标签的样式
                for text in leg.get_texts():
                    text.set_fontfamily(self.font_legend['family'])  # 设置字体类型
                    text.set_fontsize(self.font_legend['size'])      # 设置字体大小
                    text.set_fontweight(self.font_legend['weight'])  # 设置字体加粗

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 反卷积
    def peak_deconvolution(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                           x_label: Optional[str] = None, y_label: Optional[str] = None, show_legend: bool = False,
                           colors: Union[tuple, list, None] = None, standard_dic: Optional[dict] = None,
                           deconvolution: Union[tuple, List[int], None] = None,
                           area: Union[tuple, List[int], None] = None, high: Union[tuple, List[int], None] = None,
                           **kwargs) -> None:
        """
        此方法用于绘制反卷积图像
        This method is used to plot a drawing deconvolution graph.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param show_legend: (bool) 是否绘制图例，默认为 False
        :param colors: (tuple / list) 绘制线条的颜色
        :param standard_dic: (dict) 标准曲线，如被赋值则在此曲线基础上进行卷积曲线绘制
        :param deconvolution: (tuple / list) 选择出峰的位置，为 None 时不进行反卷积
        :param area: (tuple / list) 反卷积峰的面积，长度需要与 deconvolution 相同
        :param high: (tuple / list) 反卷积峰的高度，长度需要与 deconvolution 相同
        :param kwargs: 绘制原始数据时的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = colors
            else:
                color_palette = self.color_palette

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

            # 判断 deconvolution, area, high 是否长度相同，不同则抛出错误
            if deconvolution is not None and not (len(deconvolution) == len(area) == len(high)):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"deconvolution, area, and high must have the same length.")

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        # 遍历 dict 以获取数据
        for title, data_df in data_dic.items():

            # 提取数据的 X 和 Y 值
            x = data_df.iloc[:, 0]
            data = data_df.iloc[:, 1]

            # 初始化重构数据数组
            reconstructed_data = np.zeros_like(data, dtype=float)

            # 设置图像大小和分辨率
            plt.figure(figsize=width_height, dpi=200)
            # 绘制原始数据
            plt.plot(x, data, label='Original Line', color='blue', **kwargs)

            # 初始化标准数据和标准范围
            standard_data = None
            in_standard_range = np.ones_like(data, dtype=bool)

            # 判断是否存在标准数据
            if standard_dic and title in standard_dic:
                standard_data = standard_dic[title].iloc[:, 1]
                # 获取标准数据大于 0 的范围
                in_standard_range = standard_data > 0

            # 判断是否进行反卷积
            if deconvolution is not None:
                # 循环处理每个反卷积的峰值
                for i, (peak_idx, peak_area, peak_high) in enumerate(zip(deconvolution, area, high)):
                    color = color_palette[i % len(color_palette)]
                    width = peak_area / peak_high

                    # 计算峰值数据
                    peak_data = peak_high * np.exp(-0.5 * ((x - peak_idx) / width) ** 2)

                    # 有标准数据的情况
                    if standard_data is not None:
                        # 根据标准范围更新重构数据
                        reconstructed_data[in_standard_range] += peak_data[in_standard_range]
                        # 绘制填充区域
                        plt.fill_between(x, peak_data + standard_data, standard_data, color=color, alpha=0.3,
                                         where=in_standard_range)

                    # 无标准数据的情况
                    else:
                        reconstructed_data += peak_data
                        plt.fill_between(x, peak_data, color=color, alpha=0.3)

                # 绘制重构的峰值数据
                plt.plot(x[in_standard_range], reconstructed_data[in_standard_range] + (
                    standard_data[in_standard_range] if standard_data is not None else 0), label='Reconstructed Line',
                         linestyle='--', color='red')

            # 若存在标准数据，则绘制标准线
            if standard_data is not None:
                plt.plot(x[in_standard_range], standard_data[in_standard_range],
                         label='Standard Line', linestyle='--', color='grey')

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 设置坐标轴标签和标题
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            plt.xlim(np.min(x), np.max(x))  # 设置最大和最小值

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            plt.tight_layout()  # 调整布局

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 绘制函数图像
    def plot_function(self, function: Optional[Callable] = None,
                      x_range: Union[List[float], Tuple[float, float], None] = None,
                      save_path: Union[bool, str] = True, **kwargs) -> None:
        """
        在指定范围内绘制给定函数的图形
        Plots the graph of a given function over a specified range.

        :param function: (Callable) 需要绘制的函数，此项为必输入项
        :param x_range: (list / tuple) 绘制函数的 X 的定义域，此项为必输入项
        :param save_path: (str) 图片的保存路径

        :return None

         --- **kwargs ---

        - dpi: (int) 保存图片的精度，默认为 600
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)
        - show_grid: (bool) 是否绘制网格，默认为 True
        - show_legend: (bool) 是否绘制图例，默认为 True

        - x_min: (float) 横坐标的最小值
        - x_max: (float) 横坐标的最大值
        - y_min: (float) 纵坐标的最小值
        - y_max: (float) 纵坐标的最大值

        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度

        - background_color: (str) 背景色，默认无背景
        - background_transparency: (float) 背景色的透明度，只有存在背景色时才有意义，默认为 0.15
        """

        # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
        if save_path is True:
            save_path = self.save_path
        # 若 save_path 为 False 时，本图形不保存
        elif save_path is False:
            save_path = None
        # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
        else:
            save_path = save_path

        # 关键字参数初始化
        width_height = kwargs.pop('width_height', (6, 4.5))
        dpi = kwargs.pop('dpi', 600)
        show_grid = kwargs.pop('show_grid', True)
        show_legend = kwargs.pop('show_legend', True)

        x_min = kwargs.pop('x_min', None)
        x_max = kwargs.pop('x_max', None)
        y_min = kwargs.pop('y_min', None)
        y_max = kwargs.pop('y_max', None)

        custom_xticks = kwargs.pop('custom_xticks', None)
        custom_yticks = kwargs.pop('custom_yticks', None)
        x_rotation = kwargs.pop('x_rotation', None)
        y_rotation = kwargs.pop('y_rotation', None)

        background_color = kwargs.pop('background_color', None)
        background_transparency = kwargs.pop('background_transparency', 0.15)

        if function is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'function'.")

        if x_range is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"Please enter 'x_range'.")

        x = np.linspace(x_range[0], x_range[1], 1000)
        y = function(x)

        plt.figure(figsize=width_height, dpi=200)
        plt.plot(x, y,
                 color='red',
                 linestyle='-',
                 linewidth=3,
                 label=f'f(x) = {function.__name__}(x)')

        # 背景
        if background_color is not None:
            # 调用函数加背景，防止刻度被锁住
            self.change_imshow(background_color=background_color,
                               background_transparency=background_transparency, show_in_one=False)

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))

        plt.grid(show_grid)

        # 只有当 show_legend 为 True 时才会有图例
        if show_legend:
            plt.legend(prop=self.font_legend)
        else:
            plt.legend().remove()

        if custom_xticks is not None:
            plt.xticks(custom_xticks)
        if custom_yticks is not None:
            plt.yticks(custom_yticks)

        # 坐标轴标题字体
        plt.xlabel(xlabel='x', fontdict=self.font_title)
        plt.ylabel(ylabel='f(x)', fontdict=self.font_title)

        # 刻度轴字体
        plt.xticks(fontsize=self.font_ticket['size'],
                   fontweight=self.font_ticket['weight'],
                   fontfamily=self.font_ticket['family'],
                   rotation=x_rotation)
        plt.yticks(fontsize=self.font_ticket['size'],
                   fontweight=self.font_ticket['weight'],
                   fontfamily=self.font_ticket['family'],
                   rotation=y_rotation)

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

        plt.show()
        time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 饼图
    def plot_pie(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True, dpi: int = 600,
                 labels: Optional[list] = None, colors: [list] = None, explode: [list] = None,
                 show_number: Union[str, bool, None] = True, startangle: Optional[int] = None, **kwargs) -> None:
        """
        此方法用于绘制热图图像
        This method is used to draw a heatmap.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: 保存路径，若无赋值则用初始化中的 self.save_path
        :param dpi: (int) 保存图片的精度，默认为 600
        :param labels: (list) 饼图外围的标签，长度需要写数值个数一致
        :param colors: (list) 颜色色系的名称，有默认色板
        :param explode: (list) 第几部分突出，突出多少，如 (0.2, 0, 0, 0) 表示第一部分突出 0.2，长度需要写数值个数一致
        :param show_number: (str) 是否显示数值，默认为显示两位小数 '%1.2f%%'
        :param startangle: (int) 是否有初始旋转角度，默认为 140
        :param kwargs: 绘制热图时的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - pctdistance: (float) 调整数值（autopct）的位置，默认 0.7 表示更靠近饼图中心
        - labeldistance: (float) 调整标签（labels）位置，默认 1.1

        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 颜色
            if colors is None:
                colors = ['#FF6F61', '#6B8E23', '#FFD700', '#4B8B3B', '#D2691E',
                          '#1E90FF', '#9932CC', '#FF8C00', '#00CED1', '#8A2BE2']

            # 显示数值
            if isinstance(show_number, str):
                show_number = show_number  # 保持原样
            elif show_number is True:
                show_number = '%1.2f%%'  # 为 True 时，设为默认格式
            else:  # 为 False 或 None 时
                show_number = None

            # 初始旋转角度
            if startangle is None:
                startangle = 140

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            pctdistance = kwargs.pop('pctdistance', 0.7)
            labeldistance = kwargs.pop('labeldistance', 1.1)

        for title, data_df in data_dic.items():

            # 检查并删除 Category_Index 列
            if Plotter.Category_Index in data_df.columns:
                data_df = data_df.drop(columns=Plotter.Category_Index)

            # 数据的长度需要为 1
            if len(data_df) != 1:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"data_df must have exactly one row, but it has {len(data_df)} rows.")

            # data_df 是长度为 1 的 DataFrame
            data_list = data_df.iloc[0].tolist()
            # 使用 list comprehension 将每个元素转换为 float
            data_list = [float(i) for i in data_list]

            # 检查 explode 的长度是否和 data_list 匹配
            if explode is not None and len(explode) != len(data_list):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"'explode' length ({len(explode)}) must be equal to "
                                 f"'data_list' length ({len(data_list)})")

            # 绘制饼图
            plt.figure(figsize=width_height, dpi=200)
            plt.pie(data_list,
                    labels=labels,
                    colors=colors,
                    explode=explode,
                    autopct=show_number,
                    startangle=startangle,
                    textprops=self.font_legend,  # 这里设置 label 和数值字体
                    pctdistance=pctdistance,  # 调整数值（autopct）的位置，0.7 表示更靠近饼图中心
                    labeldistance=labeldistance,  # 调整标签（labels）位置，默认 1.1
                    **kwargs)

            # 设置标题
            if isinstance(image_title, str):
                plt.title(label=image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(label=title, fontdict=self.font_title)

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

            plt.show()  # 显示图形
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None


""" 拟合 """
class Fitter(general.Manager):
    """
    注意：
    1.  部分方法要求 data_dic 的长度为 1
    2.  以列名为横坐标

    Note:
    1.  Some methods require the length of data_dic to be 1
    2.  The column name is the horizontal coordinate
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
            self.magic_database = self.Magic_Database

        # 数据初始化分配
        if type(self) == Fitter:  # 当 self 为 Fitter 的直接实例时为真
            self.data_init()
            # 如果有接入的 keyword
            if keyword is not None:
                self.to_magic()  # general.Manager 及其子类需要调用以初始化属性

    # 多项式拟合 (仅接受长度为 1 的 dict 进行拟合)
    def polynomial_fitting(self, data_dic: Optional[dict] = None, degree: int = 3,
                           save_path: Union[bool, str] = True, dpi: int = 600,
                           x_label: Optional[str] = None, y_label: Optional[str] = None, show_legend: bool = True,
                           show_grid: bool = False, show_formula: bool = True, colors: all = None, **kwargs) -> str:
        """
        此方法用于绘制单自变量单因变量拟合图像，并给出拟合公式
        This method is used to draw the fitting image of single independent variable and single dependent variable,
        and give the fitting formula.

        注意：输入 dict 应当至少有两列，第一列为 X，第二列为 Y
        Attention: The dict should have at least two columns, with the first column X and the second column Y.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param degree: (int) 多项式拟合的最高阶数，默认为 3
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param show_legend: (bool) 是否绘制图例，默认为 True
        :param show_grid: (bool) 是否绘制网格，默认为 False
        :param show_formula: (bool) 是否打印拟合公式，默认为 True
        :param colors: (list) 绘制的颜色
        :param kwargs: plot 方法中的关键字参数

        :return fitted_formula: (str) 拟合公式，带有具体的拟合参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = iter(colors)
            else:
                color_palette = iter(self.color_palette)

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            if len(data_dic) != 1:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The dictionary must contain exactly one key-value pair.")

        # 解包并检查数据，然后进行拟合
        if True:

            # 获取数据
            title = list(data_dic.keys())[0]
            data_df = list(data_dic.values())[0]

            # 确保 DataFrame 有至少两列，第一列为 X，第二列为 Y
            if data_df.shape[1] < 2:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The DataFrame of input dict should have at least two columns.")

            # 从 DataFrame 提取数据
            x_values = data_df.iloc[:, 0].values
            y_values = data_df.iloc[:, 1].values

            # 多项式拟合，计算参数
            poly_coeffs = np.polyfit(x_values, y_values, degree)
            poly_func = np.poly1d(poly_coeffs)  # 绘制拟合曲线

        # 绘制图像
        if True:

            # 创建画布
            plt.figure(figsize=width_height, dpi=200)

            # 绘制原始数据
            current_color = next(color_palette)
            plt.scatter(x_values, y_values, label='Original Data', color=current_color)

            # 绘制拟合后的数据
            current_color = next(color_palette)
            plt.plot(x_values, poly_func(x_values), label='Fitting Data', color=current_color, **kwargs)

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            plt.grid(show_grid)  # 绘制网格

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 坐标轴标题字体
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

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

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        # 打印公式
        if True:

            # 构建拟合后的公式
            fitted_formula = 'Y='
            for i, coeff in enumerate(poly_coeffs):
                if i != len(poly_coeffs) - 1:
                    fitted_formula += f'{coeff:.4f}*X^{degree - i} + '
                else:
                    fitted_formula += f'{coeff:.4f}'

            if show_formula:

                # 构建原始公式
                original_formula = 'Y=' + ' + '.join([f'a{i}*X^{degree - i}' for i in range(degree)] + ['a0'])

                print(f"Original formula: \033[34m{original_formula}\033[0m")
                print(f"The fitting formula of \033[33m{title}\033[0m is: \033[95m{fitted_formula}\033[0m")

        return fitted_formula

    # 单自变量单因变量拟合 (仅接受长度为 1 的 dict 进行拟合)
    def univariate_fitting(self, data_dic: Optional[dict] = None, function_model: Optional[Callable] = None,
                           lower: Optional[list] = None, upper: Optional[list] = None,
                           p0: Optional[list] = None, evaluation: Optional[int] = 1000,
                           save_path: Union[bool, str] = True, dpi: int = 600,
                           x_label: Optional[str] = None, y_label: Optional[str] = None, show_legend: bool = True,
                           show_grid: bool = False, show_parameter: bool = True, colors: all = None, **kwargs) -> str:
        """
        此方法用于绘制单自变量单因变量拟合图像，并给出拟合公式
        This method is used to draw the fitting image of single independent variable and single dependent variable,
        and give the fitting formula.

        注意：
        1.  输入 DataFrame 应当至少有两列，第一列为 X，第二列为 Y
        2.  输入的 function_model 必需有且只有一个变量参数，且只能在第一个位置

        Note:
        1.  The DataFrame should have at least two columns, with the first column X and the second column Y;
        2.  The input function_model must have only one variable parameter, and it can only be in the first position.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param function_model: (Callable) 进行拟合的函数，此项必需输入
        :param lower: (list) 对应参数拟合的最小值，如为负无穷大则输入 None，默认均为负无穷大
        :param upper: (list) 对应参数拟合的最大值，如为正无穷大则输入 None，默认均为正无穷大
        :param p0: (list) 对应参数开始拟合的点
        :param evaluation: (int) 进行拟合的次数，数值越大则进行拟合次数越多，时间也越长，默认为 1000 次
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param show_legend: (bool) 是否绘制图例，默认为 True
        :param show_grid: (bool) 是否绘制网格，默认为 False
        :param show_parameter: (bool) 是否打印拟合的参数，默认为 True
        :param colors: (list) 绘制的颜色
        :param kwargs: curve_fit 方法中的关键字参数

        :return fitted_formula: (str) 拟合公式，带有具体的拟合参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度

        - ftol: (float) 是相对误差容忍度，用于确定函数值（即拟合的目标函数）的变化量
                如果连续迭代中目标函数的变化量小于 ftol 设定的阈值，算法会认为已经达到了足够的精度，从而停止迭代
        - xtol: (float) 是相对于自变量的容忍度。它决定了算法在自变量（即拟合参数）的变化上的敏感度
                如果连续迭代中参数的变化小于 xtol 指定的阈值，那么算法会认为参数已经足够接近最优解，进而停止迭代
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = iter(colors)
            else:
                color_palette = iter(self.color_palette)

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            if len(data_dic) != 1:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The dictionary must contain exactly one key-value pair.")

        # 解包并检查数据，然后进行拟合
        if True:

            # 获取数据
            title = list(data_dic.keys())[0]
            data_df = list(data_dic.values())[0]

            # 确保 DataFrame 有至少两列，第一列为 X，第二列为 Y
            if data_df.shape[1] < 2:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The DataFrame of input dict should have at least two columns.")

            if not callable(function_model):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The 'function_model' argument must be a callable (function).")

            # 从 DataFrame 提取数据
            x_values = data_df.iloc[:, 0].values
            y_values = data_df.iloc[:, 1].values

            # 获取函数的签名
            signature = inspect.signature(function_model)
            # 计算参数数量 (减去第一个自变量参数)
            num_parameters = len(signature.parameters) - 1

            # 创建边界
            if lower is None:
                lower_bound = [-np.inf] * num_parameters
            else:
                lower_bound = [-np.inf if x is None else x for x in lower]  # 将列表中的 None 替换为 np.inf

            if upper is None:
                upper_bound = [np.inf] * num_parameters
            else:
                upper_bound = [np.inf if x is None else x for x in upper]  # 将列表中的 None 替换为 np.inf

            # 使用 curve_fit 进行模型拟合
            fit_results = curve_fit(f=function_model,
                                    xdata=x_values,
                                    ydata=y_values,
                                    p0=p0,
                                    bounds=(lower_bound, upper_bound),
                                    maxfev=evaluation,
                                    **kwargs
                                    )

            popt = fit_results[0]  # 进行拟合的关键参数
            # pcov = fit_results[1]  # 进行评估的关键参数

            # 绘制拟合曲线
            x_smooth = np.linspace(x_values.min(), x_values.max(), 1000)
            y_fit = function_model(x_smooth, *popt)

        # 绘制图像
        if True:
            # 创建画布
            plt.figure(figsize=width_height, dpi=200)

            # 绘制原始数据
            current_color = next(color_palette)
            plt.scatter(x_values, y_values, label='Original Data', color=current_color)

            # 绘制拟合后的数据
            current_color = next(color_palette)
            plt.plot(x_smooth, y_fit, label='Fitted Curve', color=current_color)

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            plt.grid(show_grid)  # 绘制网格

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 坐标轴标题字体
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

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

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        # 打印优化的参数值
        param_values = ", ".join(  # 警告为 IDE 误解
            [f"\033[34m{name}\033[0m=\033[32m{value:.6f}\033[0m" for name, value in
             zip(function_model.__code__.co_varnames[1:], popt)])

        # 使用拟合参数生成预测值
        y_pred = function_model(x_values, *popt)

        # 计算 R² 值
        rss = np.sum((y_values - y_pred) ** 2)
        tss = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = 1 - (rss / tss)

        # 打印参数
        if show_parameter:
            print(f"For \033[33m{title}\033[0m, Optimized parameters: {param_values}", end=', ')
            print(f"\033[31mR_squared\033[0m=\033[31m{r_squared:.6f}\033[0m")

        return param_values

    # 根据取值进行函数拟合 (仅接受长度为 1 的 dict 进行拟合，可添加背景色)
    def fitting_functional_after_fetching(self, data_dic: Optional[dict] = None,
                                          function_model: Optional[Callable] = None, fetching_method: str = 'mean',
                                          box_pattern: Optional[str] = None,
                                          lower: Optional[list] = None, upper: Optional[list] = None,
                                          p0: Optional[list] = None, evaluation: Optional[int] = 1000,
                                          save_path: Union[bool, str] = True, dpi: int = 600,
                                          x_label: Optional[str] = None, y_label: Optional[str] = None,
                                          show_legend: bool = True, show_grid: bool = False,
                                          show_parameter: bool = True, colors: all = None, **kwargs) -> Optional[str]:
        """
        对数据以要求的方式进行处理，然后进行曲线拟合，并可以在拟合后绘制统计学图片
        Process the data as required, then perform curve fitting, and be able to draw statistical graphs after fitting.

        注意：
        1.  输入的 DataFrame 应当为多列，按每列的数据进行要求取值
        2.  输入的 function_model 必需有且只有一个变量参数，且只能在第一个位置

        Note:
        1.  The input DataFrame should consist of multiple columns,
            with values fetched according to requirements for each column.;
        2.  The input function_model must have only one variable parameter, and it can only be in the first position.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param function_model: (Callable) 进行拟合的函数，默认不拟合
        :param fetching_method: (str) 在数据表格中取值的方法，此值用于拟合曲线，有 mean 和 median，默认为均值
        :param box_pattern: (str) 绘制箱类图，默认为不绘制箱类图
        :param lower: (list) 对应参数拟合的最小值，如为负无穷大则输入 None，默认均为负无穷大
        :param upper: (list) 对应参数拟合的最大值，如为正无穷大则输入 None，默认均为正无穷大
        :param p0: (list) 对应参数开始拟合的点
        :param evaluation: (int) 进行拟合的次数，数值越大则进行拟合次数越多，时间也越长，默认为 1000 次
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param show_legend: (bool) 是否绘制图例，默认为 True
        :param show_grid: (bool) 是否绘制网格，默认为 False
        :param show_parameter: (bool) 是否打印拟合的参数，默认为 True
        :param colors: (list) 绘制的颜色
        :param kwargs: curve_fit 方法中的关键字参数

        :return fitted_formula: (str) 拟合公式，带有具体的拟合参数

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) 横坐标的最小值
        - x_max: (float) 横坐标的最大值
        - y_min: (float) 纵坐标的最小值
        - y_max: (float) 纵坐标的最大值

        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度

        - background_color: (str) 背景色，默认无背景
        - background_transparency: (float) 背景色的透明度，只有存在背景色时才有意义，默认为 0.15

        - ftol: (float) 是相对误差容忍度，用于确定函数值（即拟合的目标函数）的变化量
                如果连续迭代中目标函数的变化量小于 ftol 设定的阈值，算法会认为已经达到了足够的精度，从而停止迭代
        - xtol: (float) 是相对于自变量的容忍度。它决定了算法在自变量（即拟合参数）的变化上的敏感度
                如果连续迭代中参数的变化小于 xtol 指定的阈值，那么算法会认为参数已经足够接近最优解，进而停止迭代
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = iter(colors)
            else:
                color_palette = iter(self.color_palette)

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            background_color = kwargs.pop('background_color', None)
            background_transparency = kwargs.pop('background_transparency', 0.15)

            if len(data_dic) != 1:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The dictionary must contain exactly one key-value pair.")

        # 解包并检查数据，然后进行拟合
        if True:

            # 获取数据
            title = list(data_dic.keys())[0]
            data_df = list(data_dic.values())[0]

            if function_model is not None and not callable(function_model):
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"The 'function_model' argument must be a callable (function).")

            # 检查取值方式
            if fetching_method == 'mean':   # 计算每列的均值
                data_df_values = data_df.mean()

            elif fetching_method == 'median':  # 计算每列的中值
                data_df_values = data_df.median()

            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Please enter a correct fetching method: mean, median")

            # 创建一个包含均值的新 DataFrame，并将第一列设置为原数据的列名
            data_optimized_df = pd.DataFrame(data_df_values).reset_index()
            data_optimized_df.columns = ['Column', 'Mean']

            # 从 DataFrame 提取数据
            x_values = data_optimized_df.iloc[:, 0].values
            y_values = data_optimized_df.iloc[:, 1].values

            # 找到每组数据的最大和最小值
            x_min_global = min(list(x_values))
            x_max_global = max(list(x_values))
            y_min_global = data_df.min().min()
            y_max_global = data_df.max().max()

            if function_model is not None:

                # 获取函数的签名
                signature = inspect.signature(function_model)
                # 计算参数数量 (减去第一个自变量参数)
                num_parameters = len(signature.parameters) - 1

                # 创建边界
                if lower is None:
                    lower_bound = [-np.inf] * num_parameters
                else:
                    lower_bound = [-np.inf if x is None else x for x in lower]  # 将列表中的 None 替换为 np.inf

                if upper is None:
                    upper_bound = [np.inf] * num_parameters
                else:
                    upper_bound = [np.inf if x is None else x for x in upper]  # 将列表中的 None 替换为 np.inf

                # 使用 curve_fit 进行模型拟合
                fit_results = curve_fit(f=function_model,
                                        xdata=x_values,
                                        ydata=y_values,
                                        p0=p0,
                                        bounds=(lower_bound, upper_bound),
                                        maxfev=evaluation,
                                        **kwargs
                                        )

                popt = fit_results[0]  # 进行拟合的关键参数
                # pcov = fit_results[1]  # 进行评估的关键参数

                # 拟合曲线数据
                x_smooth = np.linspace(x_values.min(), x_values.max(), 1000)
                y_fit = function_model(x_smooth, *popt)

            else:
                x_smooth = None
                y_fit = None
                popt = None
                # pcov = None

        # 绘制图像
        if True:
            # 创建画布
            plt.figure(figsize=width_height, dpi=200)

            # 将 DataFrame 转换为长格式，适用于 seaborn 绘图函数
            df_long = data_df.melt(var_name=x_label, value_name=y_label)
            # 绘制散点图，使用列名作为横坐标，对应值作为纵坐标
            current_color = next(color_palette)
            sns.scatterplot(data=df_long,
                            x=x_label, y=y_label,
                            color=current_color,
                            s=50,
                            edgecolor='black',
                            label='Original Data')

            # 交换 DataFrame 的第一列和第二列，并赋值给一个新的 DataFrame
            df_swapped = df_long.copy()  # 绘制箱体部分
            df_swapped[df_swapped.columns[1]], df_swapped[df_swapped.columns[0]] = \
                df_long[df_long.columns[0]], df_long[df_long.columns[1]]
            df_swapped = df_swapped.astype(float)  # 转换为浮点型

            # 绘制箱类图
            if box_pattern is None:
                pass

            elif box_pattern == 'boxplot':  # 绘制制箱形图，显示分布的中位数、四分位数和异常值
                current_color = next(color_palette)
                sns.boxplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'violinplot':  # 绘制小提琴图，结合了箱形图的特点和核密度估计
                current_color = next(color_palette)
                sns.violinplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'stripplot':  # 绘制散点图，显示所有单个数据点
                current_color = next(color_palette)
                sns.stripplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'swarmplot':  # 绘制不重叠的散点图，显示所有单个数据点
                current_color = next(color_palette)
                sns.swarmplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'barplot':  # 绘制条形图，显示数值变量的中心趋势估计
                current_color = next(color_palette)
                sns.barplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'countplot':  # 绘制条形图，显示类别变量中每个类别的观测数量
                current_color = next(color_palette)
                sns.countplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'pointplot':  # 绘制点图，显示点估计和置信区间
                current_color = next(color_palette)
                sns.pointplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'lineplot':  # 绘制线形图，适合显示数据随时间变化的趋势
                current_color = next(color_palette)
                sns.lineplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'regplot':  # 绘制回归模型的拟合线和散点图
                current_color = next(color_palette)
                sns.regplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'scatterplot':  # 绘制散点图，适合查看两个数值变量之间的关系
                current_color = next(color_palette)
                sns.scatterplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'histplot':  # 绘制直方图，显示数值变量的分布
                current_color = next(color_palette)
                sns.histplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'kdeplot':  # 绘制核密度估计图，显示数值变量的分布趋势
                current_color = next(color_palette)
                sns.kdeplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'ecdfplot':  # 绘制经验累积分布函数图
                current_color = next(color_palette)
                sns.ecdfplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            elif box_pattern == 'boxenplot':  # 绘制增强箱形图，适合大型数据集
                current_color = next(color_palette)
                sns.boxenplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

            else:
                box_patterns = [
                    'boxplot',
                    'violinplot',
                    'stripplot',
                    'swarmplot',
                    'barplot',
                    'countplot',
                    'pointplot',
                    'lineplot',
                    'regplot',
                    'scatterplot',
                    'histplot',
                    'kdeplot',
                    'ecdfplot',
                    'boxenplot'
                ]
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"Please enter the correct box_pattern. "
                                 f"Allowed keyword arguments are: {', '.join(box_patterns)}")

            # 绘制取值后的数据
            current_color = next(color_palette)
            plt.scatter(x_values, y_values,
                        label=fetching_method.capitalize() + ' Data',
                        color=current_color,
                        s=100,
                        edgecolor='black')

            # 绘制拟合后的数据
            if function_model is not None:
                current_color = next(color_palette)
                plt.plot(x_smooth, y_fit, label='Fitted Curve', color=current_color)

            # 使用条件表达式来设置边界
            x_lower_limit = x_min if (x_min is not None) else x_min_global
            x_upper_limit = x_max if (x_max is not None) else x_max_global
            y_lower_limit = y_min if (y_min is not None) else y_min_global
            y_upper_limit = y_max if (y_max is not None) else y_max_global

            # 背景
            if background_color is not None:
                # 调用函数加背景，防止刻度被锁住
                self.change_imshow(background_color=background_color,
                                   background_transparency=background_transparency, show_in_one=True,
                                   x_min=np.float64(x_lower_limit), x_max=np.float64(x_upper_limit),
                                   y_min=np.float64(y_lower_limit), y_max=np.float64(y_upper_limit))

            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            if custom_xticks is not None:
                plt.xticks(custom_xticks)
            if custom_yticks is not None:
                plt.yticks(custom_yticks)

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            plt.grid(show_grid)  # 绘制网格

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)
            elif image_title is True:
                plt.title(title, fontdict=self.font_title)

            # 坐标轴标题字体
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

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

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        # 打印优化的参数值
        if function_model is not None:
            param_values = ", ".join(  # 警告为 IDE 误解
                [f"\033[34m{name}\033[0m=\033[32m{value:.6f}\033[0m" for name, value in
                 zip(function_model.__code__.co_varnames[1:], popt)])

            # 使用拟合参数生成预测值
            y_pred = function_model(x_values, *popt)

            # 计算 R² 值
            rss = np.sum((y_values - y_pred) ** 2)
            tss = np.sum((y_values - np.mean(y_values)) ** 2)
            r_squared = 1 - (rss / tss)

            # 打印参数
            if show_parameter:
                print(f"For \033[33m{title}\033[0m", end=': ')
                print(f"\033[34mfetching method\033[0m: \033[32m{fetching_method}\033[0m", end=', ')
                print(f"{param_values}", end=', ')
                print(f"\033[31mR²\033[0m=\033[31m{r_squared:.6f}\033[0m")

            return param_values

        else:
            return None

    # 根据取值进行函数拟合 (可接受长度大于 1 的 dict 进行拟合，可添加背景色)
    def fitting_multiple_functional_after_fetching(self, data_dic: Optional[dict] = None,
                                                   function_model: Optional[Callable] = None,
                                                   fetching_method: str = 'mean',
                                                   box_pattern: Optional[str] = None,
                                                   lower: Optional[list] = None, upper: Optional[list] = None,
                                                   p0: Optional[list] = None, evaluation: Optional[int] = 1000,
                                                   save_path: Union[bool, str] = True, dpi: int = 600,
                                                   x_label: Optional[str] = None, y_label: Optional[str] = None,
                                                   show_legend: bool = True, show_grid: bool = False,
                                                   show_parameter: bool = True, colors: all = None, **kwargs) -> None:
        """
        对数据以要求的方式进行处理，然后进行曲线拟合，并可以在拟合后绘制统计学图片
        Process the data as required, then perform curve fitting, and be able to draw statistical graphs after fitting.

        注意：
        1.  输入的 DataFrame 应当为多列，按每列的数据进行要求取值
        2.  输入的 function_model 必需有且只有一个变量参数，且只能在第一个位置

        Note:
        1.  The input DataFrame should consist of multiple columns,
            with values fetched according to requirements for each column.;
        2.  The input function_model must have only one variable parameter, and it can only be in the first position.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param function_model: (Callable) 进行拟合的函数，默认不拟合
        :param fetching_method: (str) 在数据表格中取值的方法，此值用于拟合曲线，有 mean 和 median，默认为均值
        :param box_pattern: (str) 绘制箱类图，默认为不绘制箱类图
        :param lower: (list) 对应参数拟合的最小值，如为负无穷大则输入 None，默认均为负无穷大
        :param upper: (list) 对应参数拟合的最大值，如为正无穷大则输入 None，默认均为正无穷大
        :param p0: (list) 对应参数开始拟合的点
        :param evaluation: (int) 进行拟合的次数，数值越大则进行拟合次数越多，时间也越长，默认为 1000 次
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param show_legend: (bool) 是否绘制图例，默认为 True
        :param show_grid: (bool) 是否绘制网格，默认为 False
        :param show_parameter: (bool) 是否打印拟合的参数，默认为 True
        :param colors: (list) 绘制的颜色，需要输入所有线条的颜色，每次循环时不刷新，为了突出差别
        :param kwargs: curve_fit 方法中的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) 横坐标的最小值
        - x_max: (float) 横坐标的最大值
        - y_min: (float) 纵坐标的最小值
        - y_max: (float) 纵坐标的最大值

        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度

        - background_color: (str) 背景色，默认无背景
        - background_transparency: (float) 背景色的透明度，只有存在背景色时才有意义，默认为 0.15

        - ftol: (float) 是相对误差容忍度，用于确定函数值（即拟合的目标函数）的变化量
                如果连续迭代中目标函数的变化量小于 ftol 设定的阈值，算法会认为已经达到了足够的精度，从而停止迭代
        - xtol: (float) 是相对于自变量的容忍度。它决定了算法在自变量（即拟合参数）的变化上的敏感度
                如果连续迭代中参数的变化小于 xtol 指定的阈值，那么算法会认为参数已经足够接近最优解，进而停止迭代
        """

        # 检查赋值 (5)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

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

            # 查看是否给出绘图颜色
            if colors is not None:
                color_palette = iter(colors)
            else:
                color_palette = iter(self.color_palette)

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            background_color = kwargs.pop('background_color', None)
            background_transparency = kwargs.pop('background_transparency', 0.15)

        # 解包并检查数据，然后进行拟合
        if True:

            # 创建画布
            plt.figure(figsize=width_height, dpi=200)

            # 找到每组数据的最大和最小值
            x_min_global = None
            x_max_global = None
            y_min_global = None
            y_max_global = None

            for title, data_df in data_dic.items():

                if function_model is not None and not callable(function_model):
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"The 'function_model' argument must be a callable (function).")

                # 检查取值方式
                if fetching_method == 'mean':  # 计算每列的均值
                    data_df_values = data_df.mean()

                elif fetching_method == 'median':  # 计算每列的中值
                    data_df_values = data_df.median()

                else:
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"Please enter a correct fetching method: mean, median")

                # 创建一个包含均值的新 DataFrame，并将第一列设置为原数据的列名
                data_optimized_df = pd.DataFrame(data_df_values).reset_index()
                data_optimized_df.columns = ['Column', 'Mean']

                # 从 DataFrame 提取数据
                x_values = data_optimized_df.iloc[:, 0].values
                y_values = data_optimized_df.iloc[:, 1].values

                # 检查并更新全局最大最小值
                if x_min_global is None or min(list(x_values)) < x_min_global:
                    x_min_global = min(list(x_values))
                if x_max_global is None or max(list(x_values)) > x_max_global:
                    x_max_global = max(list(x_values))
                if y_min_global is None or data_df.min().min() < y_min_global:
                    y_min_global = data_df.min().min()
                if y_max_global is None or data_df.max().max() > y_max_global:
                    y_max_global = data_df.max().max()

                if function_model is not None:

                    # 获取函数的签名
                    signature = inspect.signature(function_model)
                    # 计算参数数量 (减去第一个自变量参数)
                    num_parameters = len(signature.parameters) - 1

                    # 创建边界
                    if lower is None:
                        lower_bound = [-np.inf] * num_parameters
                    else:
                        lower_bound = [-np.inf if x is None else x for x in lower]  # 将列表中的 None 替换为 np.inf

                    if upper is None:
                        upper_bound = [np.inf] * num_parameters
                    else:
                        upper_bound = [np.inf if x is None else x for x in upper]  # 将列表中的 None 替换为 np.inf

                    # 使用 curve_fit 进行模型拟合
                    fit_results = curve_fit(f=function_model,
                                            xdata=x_values,
                                            ydata=y_values,
                                            p0=p0,
                                            bounds=(lower_bound, upper_bound),
                                            maxfev=evaluation,
                                            **kwargs
                                            )

                    popt = fit_results[0]  # 进行拟合的关键参数
                    # pcov = fit_results[1]  # 进行评估的关键参数

                    # 拟合曲线数据
                    x_smooth = np.linspace(x_values.min(), x_values.max(), 1000)
                    y_fit = function_model(x_smooth, *popt)

                else:
                    x_smooth = None
                    y_fit = None
                    popt = None
                    # pcov = None

                # 将 DataFrame 转换为长格式，适用于 seaborn 绘图函数
                df_long = data_df.melt(var_name=x_label, value_name=y_label)
                # 绘制散点图，使用列名作为横坐标，对应值作为纵坐标
                current_color = next(color_palette)
                sns.scatterplot(data=df_long,
                                x=x_label, y=y_label,
                                color=current_color,
                                s=50,
                                edgecolor='black',
                                label='Original Data')

                # 交换 DataFrame 的第一列和第二列，并赋值给一个新的 DataFrame
                df_swapped = df_long.copy()  # 绘制箱体部分
                df_swapped[df_swapped.columns[1]], df_swapped[df_swapped.columns[0]] = \
                    df_long[df_long.columns[0]], df_long[df_long.columns[1]]
                df_swapped = df_swapped.astype(float)  # 转换为浮点型

                # 绘制箱类图
                if box_pattern is None:
                    pass

                elif box_pattern == 'boxplot':  # 绘制制箱形图，显示分布的中位数、四分位数和异常值
                    current_color = next(color_palette)
                    sns.boxplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'violinplot':  # 绘制小提琴图，结合了箱形图的特点和核密度估计
                    current_color = next(color_palette)
                    sns.violinplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'stripplot':  # 绘制散点图，显示所有单个数据点
                    current_color = next(color_palette)
                    sns.stripplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'swarmplot':  # 绘制不重叠的散点图，显示所有单个数据点
                    current_color = next(color_palette)
                    sns.swarmplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'barplot':  # 绘制条形图，显示数值变量的中心趋势估计
                    current_color = next(color_palette)
                    sns.barplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'countplot':  # 绘制条形图，显示类别变量中每个类别的观测数量
                    current_color = next(color_palette)
                    sns.countplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'pointplot':  # 绘制点图，显示点估计和置信区间
                    current_color = next(color_palette)
                    sns.pointplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'lineplot':  # 绘制线形图，适合显示数据随时间变化的趋势
                    current_color = next(color_palette)
                    sns.lineplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'regplot':  # 绘制回归模型的拟合线和散点图
                    current_color = next(color_palette)
                    sns.regplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'scatterplot':  # 绘制散点图，适合查看两个数值变量之间的关系
                    current_color = next(color_palette)
                    sns.scatterplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'histplot':  # 绘制直方图，显示数值变量的分布
                    current_color = next(color_palette)
                    sns.histplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'kdeplot':  # 绘制核密度估计图，显示数值变量的分布趋势
                    current_color = next(color_palette)
                    sns.kdeplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'ecdfplot':  # 绘制经验累积分布函数图
                    current_color = next(color_palette)
                    sns.ecdfplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                elif box_pattern == 'boxenplot':  # 绘制增强箱形图，适合大型数据集
                    current_color = next(color_palette)
                    sns.boxenplot(data=df_swapped, x=x_label, y=y_label, color=current_color)

                else:
                    box_patterns = [
                        'boxplot',
                        'violinplot',
                        'stripplot',
                        'swarmplot',
                        'barplot',
                        'countplot',
                        'pointplot',
                        'lineplot',
                        'regplot',
                        'scatterplot',
                        'histplot',
                        'kdeplot',
                        'ecdfplot',
                        'boxenplot'
                    ]
                    class_name = self.__class__.__name__  # 获取类名
                    method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"Please enter the correct box_pattern. "
                                     f"Allowed keyword arguments are: {', '.join(box_patterns)}")

                # 绘制取值后的数据
                current_color = next(color_palette)
                plt.scatter(x_values, y_values,
                            label=fetching_method.capitalize() + ' Data',
                            color=current_color,
                            s=100,
                            edgecolor='black')

                # 绘制拟合后的数据
                if function_model is not None:
                    current_color = next(color_palette)
                    plt.plot(x_smooth, y_fit, label='Fitted Curve', color=current_color)

                # 打印优化的参数值
                if function_model is not None:
                    param_values = ", ".join(  # 警告为 IDE 误解
                        [f"\033[34m{name}\033[0m=\033[32m{value:.6f}\033[0m" for name, value in
                         zip(function_model.__code__.co_varnames[1:], popt)])

                    # 使用拟合参数生成预测值
                    y_pred = function_model(x_values, *popt)

                    # 计算 R² 值
                    rss = np.sum((y_values - y_pred) ** 2)
                    tss = np.sum((y_values - np.mean(y_values)) ** 2)
                    r_squared = 1 - (rss / tss)

                    # 打印参数
                    if show_parameter:
                        print(f"For \033[33mFitting\033[0m", end=': ')
                        print(f"\033[34m{title}\033[0m: \033[32m{fetching_method}\033[0m", end=', ')
                        print(f"{param_values}", end=', ')
                        print(f"\033[31mR²\033[0m=\033[31m{r_squared:.6f}\033[0m")

        # 图片格式调整
        if True:

            # 将最大最小值调整为原来的 1.15 倍
            x_range = x_max_global - x_min_global
            x_min_global -= 0.15 * x_range
            x_max_global += 0.15 * x_range

            y_range = y_max_global - y_min_global
            y_min_global -= 0.15 * y_range
            y_max_global += 0.15 * y_range

            # 使用条件表达式来设置边界
            x_lower_limit = x_min if (x_min is not None) else x_min_global
            x_upper_limit = x_max if (x_max is not None) else x_max_global
            y_lower_limit = y_min if (y_min is not None) else y_min_global
            y_upper_limit = y_max if (y_max is not None) else y_max_global

            # 背景
            if background_color is not None:
                # 调用函数加背景，防止刻度被锁住
                self.change_imshow(background_color=background_color,
                                   background_transparency=background_transparency, show_in_one=True,
                                   x_min=np.float64(x_lower_limit), x_max=np.float64(x_upper_limit),
                                   y_min=np.float64(y_lower_limit), y_max=np.float64(y_upper_limit))

            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))

            if custom_xticks is not None:
                plt.xticks(custom_xticks)
            if custom_yticks is not None:
                plt.yticks(custom_yticks)

            # 只有当 show_legend 为 True 时才会有图例
            if show_legend:
                plt.legend(prop=self.font_legend)
            else:
                plt.legend().remove()

            plt.grid(show_grid)  # 绘制网格

            # 设置标题
            if isinstance(image_title, str):
                plt.title(image_title, fontdict=self.font_title)

            # 坐标轴标题字体
            plt.xlabel(x_label, fontdict=self.font_title)
            plt.ylabel(y_label, fontdict=self.font_title)

            # 刻度轴字体
            plt.xticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=x_rotation)
            plt.yticks(fontsize=self.font_ticket['size'],
                       fontweight=self.font_ticket['weight'],
                       fontfamily=self.font_ticket['family'],
                       rotation=y_rotation)

            plt.tight_layout()

        # 保存图片
        if True:

            # 如果提供了保存路径，则保存图像到指定路径
            if save_path is not None:  # 如果 save_path 的值不为 None，则保存
                file_name = "Fitting.png"  # 初始文件名为 "Fitting.png"
                full_file_path = os.path.join(save_path, file_name)  # 创建完整的文件路径

                if os.path.exists(full_file_path):  # 查看该文件名是否存在
                    count = 1
                    file_name = "Fitting.png" + f"_{count}.png"  # 若该文件名存在则在后面加 '_1'
                    full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                    while os.path.exists(full_file_path):  # 找是否存在，并不断 +1，直到不重复
                        count += 1
                        file_name = "Fitting.png" + f"_{count}.png"
                        full_file_path = os.path.join(save_path, file_name)  # 更新完整的文件路径

                plt.savefig(fname=full_file_path, dpi=dpi)  # 使用完整路径将散点图保存到指定的路径

            plt.show()  # 显示图像
            time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None

    # 多自变量单因变量拟合 (仅接受长度为 1 的 dict 进行拟合)
    def multivariable_fitting(self):
        pass

    # 根据函数进行拟合  (可对多组数据进行拟合，可添加背景色)
    def fitting_multiple_functional(self, data_dic: Optional[dict] = None, function_model: Optional[Callable] = None,
                                    lower: Optional[list] = None, upper: Optional[list] = None,
                                    p0: Optional[list] = None, evaluation: Optional[int] = 1000,
                                    save_path: Union[bool, str] = True, dpi: int = 600,
                                    x_label: Optional[str] = None, y_label: Optional[str] = None,
                                    content_list: Optional[List[str]] = None, x_list: Optional[list] = None,
                                    add_zero: bool = False, show_legend: bool = True, show_grid: bool = False,
                                    show_parameter: bool = True, colors: all = None, **kwargs) -> None:
        """
        对数据以要求的方式进行处理，然后进行曲线拟合，并可以在拟合后绘制统计学图片
        Process the data as required, then perform curve fitting, and be able to draw statistical graphs after fitting.

        注意：
        输入的 function_model 必需有且只有一个变量参数，且只能在第一个位置
        Attention:
        The input function_model must have only one variable parameter, and it can only be in the first position.

        :param data_dic: (dict) 可包含多个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param function_model: (Callable) 进行拟合的函数，默认不拟合
        :param lower: (list) 对应参数拟合的最小值，如为负无穷大则输入 None，默认均为负无穷大
        :param upper: (list) 对应参数拟合的最大值，如为正无穷大则输入 None，默认均为正无穷大
        :param p0: (list) 对应参数开始拟合的点
        :param evaluation: (int) 进行拟合的次数，数值越大则进行拟合次数越多，时间也越长，默认为 1000 次
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param content_list: (list) legend() 显示的名称，如果不想显示某条曲线，则将其命名为 None，默认为原数据的 row 索引
        :param x_list: (list) 曲线的公共横坐标，默认为原数据的列名
        :param add_zero: (bool) 在第一个数据前加 (0, 0)，默认为 False
        :param show_legend: (bool) 是否绘制图例，默认为 True
        :param show_grid: (bool) 是否绘制网格，默认为 False
        :param show_parameter: (bool) 是否打印拟合的参数，默认为 True
        :param colors: (list) 绘制的颜色
        :param kwargs: curve_fit 方法中的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) 横坐标的最小值
        - x_max: (float) 横坐标的最大值
        - y_min: (float) 纵坐标的最小值
        - y_max: (float) 纵坐标的最大值

        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度

        - background_color: (str) 背景色，默认无背景
        - background_transparency: (float) 背景色的透明度，只有存在背景色时才有意义，默认为 0.15

        - ftol: (float) 是相对误差容忍度，用于确定函数值（即拟合的目标函数）的变化量
                如果连续迭代中目标函数的变化量小于 ftol 设定的阈值，算法会认为已经达到了足够的精度，从而停止迭代
        - xtol: (float) 是相对于自变量的容忍度。它决定了算法在自变量（即拟合参数）的变化上的敏感度
                如果连续迭代中参数的变化小于 xtol 指定的阈值，那么算法会认为参数已经足够接近最优解，进而停止迭代
        """

        # 检查赋值 (5，算上调色板)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

            background_color = kwargs.pop('background_color', None)
            background_transparency = kwargs.pop('background_transparency', 0.15)

            # 数据初始化
            result_dict = {}
            x_min_global_dic = {}
            x_max_global_dic = {}
            y_min_global_dic = {}
            y_max_global_dic = {}
            for key, df in data_dic.items():

                # 使用 shape 获取 DataFrame 的行数和列数
                num_rows, num_columns = df.shape

                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名

                # 如果 content_list 被赋值，检查 content_list 的长度是否与原数据的行数一致
                if content_list is not None and not num_rows == len(content_list):
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"content_list is not consistent with the original data, "
                                     f"the original data has {num_rows} rows, "
                                     f"but the length of 'content_list' is {len(content_list)}.")

                elif content_list is not None and num_rows == len(content_list):
                    content_list = content_list

                else:  # content_list 为 None 的情况
                    content_list = df.index.tolist()

                # 如果 x_list 被赋值，检查 x_list 的长度是否与原数据的列数一致
                if x_list is not None and not num_columns == len(x_list):
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"x_list is not consistent with the original data, "
                                     f"the original data has {num_columns} columns, "
                                     f"but the length of 'x_list' is {len(x_list)}.")

                elif x_list is not None and num_columns == len(x_list):
                    x_list = x_list

                else:  # x_list 为 None 的情况
                    # 获取DataFrame的列名
                    column_names = df.columns.tolist()

                    # 将列名转换为float类型，如果无法转换则报错
                    x_list = []
                    for col_name in column_names:
                        try:
                            # 尝试将列名转换为float类型
                            converted_name = float(col_name)
                        except ValueError:
                            # 转换失败，报错
                            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                             f"Unable to convert column name '{col_name}' to type float.")
                        x_list.append(converted_name)

                # 创建一个新的 dict ，用于存储数据
                key_dict = {}
                for idx, row in df.iterrows():

                    # 不添加 (0, 0)
                    if not add_zero:
                        data_df = pd.DataFrame({
                            x_label: x_list,
                            y_label: row.values.tolist()
                        })

                    # 在每行的开头加一行 (0, 0)
                    else:
                        data_df = pd.DataFrame({
                            x_label: [0] + x_list,
                            y_label: [0] + row.values.tolist()
                        })

                    if content_list is not None:
                        # 将数据 DataFrame 存入 dict
                        key_dict[content_list[int(idx)]] = data_df

                # 将当前标题的数据 dict 存入最终结果 dict
                result_dict[key] = key_dict

                # 找到每组数据的最大和最小值
                x_min_global = min(x_list)
                x_max_global = max(x_list)
                y_min_global = df.min().min()
                y_max_global = df.max().max()

                # 将提取的数据存入字典
                x_min_global_dic[key] = x_min_global
                x_max_global_dic[key] = x_max_global
                y_min_global_dic[key] = y_min_global
                y_max_global_dic[key] = y_max_global

        # 数据拟合
        if True:

            # 数据初始化
            x_smooth_dic = {}
            y_fit_dic = {}
            popt_dic = {}
            # pcov_dic = {}

            for title, single_dic in result_dict.items():

                # 数据初始化
                x_smooth_single_dic = {}
                y_fit_single_dic = {}
                popt_single_dic = {}
                # pcov_single_dic = {}

                for data_df_title, data_df in single_dic.items():

                    # 从 DataFrame 提取数据
                    x_values = data_df.iloc[:, 0].values
                    y_values = data_df.iloc[:, 1].values

                    if function_model is not None:

                        # 获取函数的签名
                        signature = inspect.signature(function_model)
                        # 计算参数数量 (减去第一个自变量参数)
                        num_parameters = len(signature.parameters) - 1

                        # 创建边界
                        if lower is None:
                            lower_bound = [-np.inf] * num_parameters
                        else:
                            lower_bound = [-np.inf if x is None else x for x in lower]  # 将列表中的 None 替换为 np.inf

                        if upper is None:
                            upper_bound = [np.inf] * num_parameters
                        else:
                            upper_bound = [np.inf if x is None else x for x in upper]  # 将列表中的 None 替换为 np.inf

                        # 使用 curve_fit 进行模型拟合
                        fit_results = curve_fit(f=function_model,
                                                xdata=x_values,
                                                ydata=y_values,
                                                p0=p0,
                                                bounds=(lower_bound, upper_bound),
                                                maxfev=evaluation,
                                                **kwargs
                                                )

                        popt = fit_results[0]  # 进行拟合的关键参数
                        # pcov = fit_results[1]  # 进行评估的关键参数

                        # 拟合曲线数据
                        x_smooth = np.linspace(x_values.min(), x_values.max(), 1000)
                        y_fit = function_model(x_smooth, *popt)

                        x_smooth_single_dic[data_df_title] = x_smooth
                        y_fit_single_dic[data_df_title] = y_fit
                        popt_single_dic[data_df_title] = popt
                        # pcov_single_dic[data_df_title] = pcov

                x_smooth_dic[title] = x_smooth_single_dic
                y_fit_dic[title] = y_fit_single_dic
                popt_dic[title] = popt_single_dic
                # pcov_dic[title] = pcov_single_dic

        # 绘制图像
        if True:

            for title, single_dic in result_dict.items():

                # 创建画布
                plt.figure(figsize=width_height, dpi=200)

                # 查看是否给出绘图颜色
                if colors is not None:
                    color_palette = iter(colors)
                else:
                    color_palette = iter(self.color_palette)

                x_smooth_single_dic = x_smooth_dic[title]
                y_fit_single_dic = y_fit_dic[title]

                for data_df_title, data_df in single_dic.items():

                    # 绘制散点图，使用列名作为横坐标，对应值作为纵坐标
                    current_color = next(color_palette)
                    sns.scatterplot(data=data_df,
                                    x=x_label,
                                    y=y_label,
                                    color=current_color)

                    # 绘制拟合后的数据
                    if function_model is not None:
                        x_smooth = x_smooth_single_dic[data_df_title]
                        y_fit = y_fit_single_dic[data_df_title]
                        plt.plot(x_smooth, y_fit,
                                 label=data_df_title,
                                 color=current_color)

                # 将提取的数据存入字典
                x_min_global = x_min_global_dic[title]
                x_max_global = x_max_global_dic[title]
                y_min_global = y_min_global_dic[title]
                y_max_global = y_max_global_dic[title]

                # 使用条件表达式来设置边界
                x_lower_limit = x_min if (x_min is not None) else x_min_global
                x_upper_limit = x_max if (x_max is not None) else x_max_global
                y_lower_limit = y_min if (y_min is not None) else y_min_global
                y_upper_limit = y_max if (y_max is not None) else y_max_global

                # 背景
                if background_color is not None:
                    # 调用函数加背景，防止刻度被锁住
                    self.change_imshow(background_color=background_color,
                                       background_transparency=background_transparency, show_in_one=True,
                                       x_min=np.float64(x_lower_limit), x_max=np.float64(x_upper_limit),
                                       y_min=np.float64(y_lower_limit), y_max=np.float64(y_upper_limit))

                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))

                if custom_xticks is not None:
                    plt.xticks(custom_xticks)
                if custom_yticks is not None:
                    plt.yticks(custom_yticks)

                # 设置标题
                if isinstance(image_title, str):
                    plt.title(image_title, fontdict=self.font_title)
                elif image_title is True:
                    plt.title(title, fontdict=self.font_title)

                # 设置坐标轴标签和标题
                plt.xlabel(x_label, fontdict=self.font_title)
                plt.ylabel(y_label, fontdict=self.font_title)

                # 刻度轴字体
                plt.xticks(fontsize=self.font_ticket['size'],
                           fontweight=self.font_ticket['weight'],
                           fontfamily=self.font_ticket['family'],
                           rotation=x_rotation)
                plt.yticks(fontsize=self.font_ticket['size'],
                           fontweight=self.font_ticket['weight'],
                           fontfamily=self.font_ticket['family'],
                           rotation=y_rotation)

                if show_grid:
                    plt.grid(True)

                # 只有当 show_legend 为 True 时才会有图例
                if show_legend:
                    plt.legend(prop=self.font_legend)
                else:
                    plt.legend().remove()

                plt.tight_layout()  # 调整布局

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

                plt.show()  # 显示图像
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

            # 打印优化的参数值
            if function_model is not None:

                for title, single_dic in result_dict.items():

                    popt_single_dic = popt_dic[title]
                    # pcov_single_dic = pcov_dic[title]

                    for data_df_title, data_df in single_dic.items():

                        popt = popt_single_dic[data_df_title]
                        # pcov = pcov_single_dic[data_df_title]

                        # 从 DataFrame 提取数据
                        x_values = data_df.iloc[:, 0].values
                        y_values = data_df.iloc[:, 1].values

                        param_values = ", ".join(  # 警告为 IDE 误解
                            [f"\033[34m{name}\033[0m=\033[32m{value:.6f}\033[0m" for name, value in
                             zip(function_model.__code__.co_varnames[1:], popt)])

                        # 使用拟合参数生成预测值
                        y_pred = function_model(x_values, *popt)

                        # 计算 R² 值
                        rss = np.sum((y_values - y_pred) ** 2)
                        tss = np.sum((y_values - np.mean(y_values)) ** 2)
                        r_squared = 1 - (rss / tss)

                        # 打印参数
                        if show_parameter:
                            print(f'For \033[33m{data_df_title}\033[0m of \033[35m{title}\033[0m', end=': ')
                            print(f"{param_values}", end=', ')
                            print(f"\033[31mR_squared\033[0m=\033[31m{r_squared:.6f}\033[0m")

        return None

    # 根据模型进行拟合  (可对多组数据进行拟合)
    def fitting_multiple_according_model(self, data_dic: Optional[dict] = None, save_path: Union[bool, str] = True,
                                         dpi: int = 600, x_label: Optional[str] = None, y_label: Optional[str] = None,
                                         fitting_method: str = 'polynomial', content_list: Optional[List[str]] = None,
                                         x_list: Optional[list] = None, add_zero: bool = False,
                                         x_start: Optional[float] = None, x_end: Optional[float] = None,
                                         show_legend: bool = False, show_grid: bool = True, show_parameter: bool = True,
                                         colors: Union[tuple, list, None] = None, **kwargs) -> None:
        """
        根据模型绘制多条拟合曲线
        Multiple fit curves are drawn according to the model.

        :param data_dic: (dict) 包含一个键值对，键为 title，值为包含多个指标及类别序号的 DataFrame
        :param save_path: (str) 图片的保存路径
        :param dpi: (int) 保存图片的精度，默认为 600
        :param x_label: (str) X 轴的标题
        :param y_label: (str) Y 轴的标题
        :param fitting_method: (str) 拟合方式，默认为多项式拟合
        :param content_list: (list) legend() 显示的名称，如果不想显示某条曲线，则将其命名为 None，默认为原数据的 row 索引
        :param x_list: (list) 曲线的公共横坐标，默认为原数据的列名
        :param add_zero: (bool) 在第一个数据前加 (0, 0)，默认为不添加
        :param x_start: (float) 预测曲线的起始点
        :param x_end: (float) 预测曲线的终点
        :param show_legend: (bool) 是否绘制图例，默认为 False
        :param show_grid: (bool) 是否显示背景网格，默认为 True
        :param show_parameter: (bool) 是否打印拟合的参数，只有多项式回归和线性回归可以打印公式，默认为 True
        :param colors: (tuple / list) 绘制图片线条和散点的颜色
        :param kwargs: 模型中的关键字参数

        :return: None

        --- **kwargs ---

        - title: (str) 图片的标题，为 True 时为 title，为 str 类型时为其值，默认为无标题
        - width_height: (tuple) 图片的宽度和高度，默认为 (6, 4.5)

        - x_min: (float) 横坐标的最小值
        - x_max: (float) 横坐标的最大值
        - y_min: (float) 纵坐标的最小值
        - y_max: (float) 纵坐标的最大值

        - custom_xticks: (list) 横坐标的标点，如想不显示，可以 custom_xticks = []
        - custom_yticks: (list) 纵坐标的标点，如想不显示，可以 custom_yticks = []
        - x_rotation: (float) X 轴刻度的旋转角度
        - y_rotation: (float) Y 轴刻度的旋转角度
        """

        # 检查赋值 (5，算上调色板)
        if True:

            # 将需要处理的数据赋给 data_dic
            if data_dic is not None:
                data_dic = copy.deepcopy(data_dic)
            else:
                data_dic = copy.deepcopy(self.data_dic)

            # 当 save_path == True 时，沿用 self.save_path 的设置，此项为默认项
            if save_path is True:
                save_path = self.save_path
            # 若 save_path 为 False 时，本图形不保存
            elif save_path is False:
                save_path = None
            # 当有指定的 save_path 时，save_path 将会被其赋值，若 save_path == '' 则保存在运行的 py 文件的目录下
            else:
                save_path = save_path

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

            # 关键字参数初始化
            image_title = kwargs.pop('title', None)
            width_height = kwargs.pop('width_height', (6, 4.5))

            x_min = kwargs.pop('x_min', None)
            x_max = kwargs.pop('x_max', None)
            y_min = kwargs.pop('y_min', None)
            y_max = kwargs.pop('y_max', None)

            custom_xticks = kwargs.pop('custom_xticks', None)
            custom_yticks = kwargs.pop('custom_yticks', None)
            x_rotation = kwargs.pop('x_rotation', None)
            y_rotation = kwargs.pop('y_rotation', None)

        # 数据拟合
        if True:

            # 数据表格整理
            result_dict = {}

            for key, df in data_dic.items():

                # 使用 shape 获取 DataFrame 的行数和列数
                num_rows, num_columns = df.shape

                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名

                # 如果 content_list 被赋值，检查 content_list 的长度是否与原数据的行数一致
                if content_list is not None and not num_rows == len(content_list):
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"content_list is not consistent with the original data, "
                                     f"the original data has {num_rows} rows, "
                                     f"but the length of 'content_list' is {len(content_list)}.")

                elif content_list is not None and num_rows == len(content_list):
                    content_list = content_list

                elif content_list is None:  # content_list 为 None 的情况
                    content_list = df.index.tolist()

                # 如果 x_list 被赋值，检查 x_list 的长度是否与原数据的列数一致
                if x_list is not None and not num_columns == len(x_list):
                    raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                     f"x_list is not consistent with the original data, "
                                     f"the original data has {num_columns} columns, "
                                     f"but the length of 'x_list' is {len(x_list)}.")

                elif x_list is not None and num_columns == len(x_list):
                    x_list = x_list

                elif x_list is None:
                    # 获取DataFrame的列名
                    column_names = df.columns.tolist()

                    # 将列名转换为float类型，如果无法转换则报错
                    x_list = []
                    for col_name in column_names:
                        try:
                            # 尝试将列名转换为float类型
                            converted_name = float(col_name)
                        except ValueError:
                            # 转换失败，报错
                            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                             f"Unable to convert column name '{col_name}' to type float.")
                        x_list.append(converted_name)

                # 创建一个新的 dict ，用于存储数据
                key_dict = {}

                # 遍历每一行
                for idx, row in df.iterrows():

                    # 不添加 (0, 0)
                    if not add_zero:
                        data_df = pd.DataFrame({
                            x_label: x_list,
                            y_label: row.values.tolist()
                        })

                    # 在每行的开头加一行 (0, 0)
                    else:
                        data_df = pd.DataFrame({
                            x_label: [0] + x_list,
                            y_label: [0] + row.values.tolist()
                        })

                    if content_list is not None:
                        # 将数据 DataFrame 存入 dict
                        key_dict[content_list[idx]] = data_df

                # 将当前标题的数据 dict 存入最终结果 dict
                result_dict[key] = key_dict

        # 绘制图像
        if True:

            for title, single_dic in result_dict.items():

                plt.figure(figsize=width_height, dpi=200)

                # 查看是否给出绘图颜色
                if colors is not None:
                    color_palette = iter(colors)
                else:
                    color_palette = iter(self.color_palette)

                for data_df_title, data_df in single_dic.items():
                    x = data_df.iloc[:, 0].values.reshape(-1, 1)
                    y = data_df.iloc[:, 1].values

                    # 多项式回归
                    if fitting_method == 'polynomial':
                        # 可以输入 degree (**kwargs)，来改变拟合阶数，默认为 2
                        model = make_pipeline(PolynomialFeatures(**kwargs), LinearRegression())

                    # 线性回归
                    elif fitting_method == 'line':
                        model = LinearRegression(**kwargs)

                    # 岭回归
                    elif fitting_method == 'ridge':
                        model = Ridge(alpha=1.0, **kwargs)  # alpha 参数控制正则化的强度

                    # 套索回归
                    elif fitting_method == 'lasso':
                        model = Lasso(alpha=1.0, **kwargs)  # alpha 参数控制正则化的强度

                    # 弹性回归
                    elif fitting_method == 'elastic':
                        model = ElasticNet(alpha=1.0, l1_ratio=0.5, **kwargs)  # alpha 控制正则化强度，l1_ratio 控制混合率

                    # 支持向量回归
                    elif fitting_method == 'vector':
                        model = SVR(kernel='rbf', **kwargs)  # 还可以选择其他核函数，'linear', 'poly', 'rbf', 'sigmoid'

                    # 决策树回归
                    elif fitting_method == 'tree':
                        model = DecisionTreeRegressor(max_depth=5, **kwargs)  # max_depth 控制决策树的深度

                    # 随机森林回归
                    elif fitting_method == 'forest':
                        model = RandomForestRegressor(n_estimators=100, **kwargs)

                    else:
                        # 判断是否可行
                        class_name = self.__class__.__name__  # 获取类名
                        method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                        available_methods = ['polynomial', 'line', 'ridge', 'lasso', 'elastic', 'vector', 'tree',
                                             'forest']
                        raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                         f"the inputted fitting_method is incorrect. "
                                         f"Available methods are: {available_methods}")

                    # 数据拟合
                    model.fit(x, y)

                    # 预测曲线的范围
                    if x_start is None:
                        x_start = min(data_df.iloc[:, 0])
                    if x_end is None:
                        x_end = max(data_df.iloc[:, 0])

                    # 为了得到平滑的曲线，预测一系列的值
                    x_range = np.linspace(start=x_start, stop=x_end, num=100).reshape(-1, 1)
                    y_pred = model.predict(x_range)

                    current_color = next(color_palette)

                    # 绘制实际的数据点
                    plt.scatter(x, y,
                                color=current_color,
                                label=data_df_title)

                    # 绘制预测的曲线
                    plt.plot(x_range, y_pred,
                             linestyle='--',
                             color=current_color)  # 使用颜色为线条上色

                    # 打印参数
                    if show_parameter:

                        print(f'For \033[33m{data_df_title}\033[0m of \033[35m{title}\033[0m', end=': ')

                        # 打印公式
                        if fitting_method == 'polynomial':
                            coefs = model.named_steps['linearregression'].coef_
                            intercept = model.named_steps['linearregression'].intercept_
                            formula = f'\033[34my\033[0m = \033[32m{intercept:.2f}\033[0m + ' + ' + '.join(
                                [f'\033[32m{coef:.3f}*x^{i}\033[0m' for i, coef in enumerate(coefs[1:], start=1)])

                        elif fitting_method == 'line':
                            # 直接提取系数和截距
                            coefs = model.coef_
                            intercept = model.intercept_
                            formula = f'y = {intercept:.2f} + ' + ' + '.join(
                                [f'{coef:.2f}*x^{i}' for i, coef in enumerate(coefs, start=1)])

                        else:
                            formula = None

                        print(formula, end=', ')

                        # 计算 R²
                        r_squared = model.score(x, y)
                        print(f"\033[31mR_squared\033[0m of \033[94m{fitting_method}\033[0m:"
                              f" \033[31m{r_squared:.6f}\033[0m")

                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))

                if custom_xticks is not None:
                    plt.xticks(custom_xticks)
                if custom_yticks is not None:
                    plt.yticks(custom_yticks)

                # 设置标题
                if isinstance(image_title, str):
                    plt.title(image_title, fontdict=self.font_title)
                elif image_title is True:
                    plt.title(title, fontdict=self.font_title)

                # 设置坐标轴标签和标题
                plt.xlabel(x_label, fontdict=self.font_title)
                plt.ylabel(y_label, fontdict=self.font_title)

                # 刻度轴字体
                plt.xticks(fontsize=self.font_ticket['size'],
                           fontweight=self.font_ticket['weight'],
                           fontfamily=self.font_ticket['family'],
                           rotation=x_rotation)
                plt.yticks(fontsize=self.font_ticket['size'],
                           fontweight=self.font_ticket['weight'],
                           fontfamily=self.font_ticket['family'],
                           rotation=y_rotation)

                if show_grid:
                    plt.grid(True)

                # 只有当 show_legend 为 True 时才会有图例
                if show_legend:
                    plt.legend(prop=self.font_legend)
                else:
                    plt.legend().remove()

                plt.tight_layout()  # 调整布局

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

                plt.show()
                time.sleep(self.interval_time)  # 让程序休息一段时间，防止绘图过快导致程序崩溃

        return None
