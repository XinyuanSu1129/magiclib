"""
magiclib / inspector

------------------------------------------------------------------------------------------------------------------------
magiclib/inspector is a versatile module within the magiclib library designed to assist researchers and data scientists
with data observation, visualization, and code inspection. It provides a collection of tools for generating color
palettes, exploring plotting parameters, displaying encoding options, and summarizing mathematical operations in
Python. By consolidating these functionalities, the module allows users to quickly inspect, customize, and optimize
their data analysis workflows, making the research process more efficient and visually informative.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""


from typing import Union
import seaborn as sns
from prettytable import PrettyTable


"""常用参数"""
class Tool:

    # 常用科研配色 (不连续)
    if True:
        # 蓝色调系列
        blue_colors = [
            "#003f5c", "#2f4b7c", "#665191", "#a05195",
            "#d45087", "#f95d6a", "#ff7c43", "#ffa600"
        ]

        # 绿色调系列
        green_colors = [
            "#004d40", "#00796b", "#26a69a", "#4db6ac",
            "#80cbc4", "#b2dfdb", "#e0f2f1", "#ffffff"
        ]

        # 灰色和中性色
        neutral_colors = [
            "#f5f5f5", "#eeeeee", "#e0e0e0", "#bdbdbd",
            "#9e9e9e", "#757575", "#616161", "#424242"
        ]

        # 暖色调（红色、橙色）
        warm_colors = [
            "#b71c1c", "#e53935", "#ef5350", "#fb8c00",
            "#ffa726", "#ffb74d", "#ffcc80", "#ffe0b2"
        ]

        # 彩虹色系
        rainbow_colors = [
            "#ff0000", "#ffa500", "#ffff00", "#008000",
            "#0000ff", "#4b0082", "#ee82ee", "#ff00ff"
        ]

        # 双色梯度
        gradient_colors = [
            "#6a1b9a", "#7b1fa2", "#8e24aa", "#9c27b0",
            "#ab47bc", "#ba68c8", "#ce93d8", "#e1bee7"
        ]

        # 对比色
        contrast_colors = [
            "#0d47a1", "#1976d2", "#42a5f5", "#64b5f6",
            "#ffa726", "#fb8c00", "#f57c00", "#ef6c00"
        ]

        color_schemes = [blue_colors, green_colors, neutral_colors, warm_colors,
                         rainbow_colors, gradient_colors, contrast_colors]

    # 亮色
    @staticmethod
    def light_palette(color: Union[str, tuple, None] = None):
        light_palette = sns.light_palette('seagreen' if color is None else color, as_cmap=True)
        return light_palette

    # 暗色
    @staticmethod
    def dark_palette(color: Union[str, tuple, None] = None):
        dark_palette = sns.dark_palette('#69d' if color is None else color, reverse=True, as_cmap=True)
        return dark_palette


""" 显示可改的参数设置 """
class Observer:

    @staticmethod
    def print_width_height():
        print("-" * 70 + "width_height" + "-" * 70)
        print("图片尺寸推荐 width_height:(6, 4.5), (9, 6)")
        print('\n')

    @staticmethod
    def print_markers():
        print("-" * 70 + "markers" + "-" * 70)
        markers = "标记点 markers: . , o, v, ^, <, >, 1, 2, 3, 4, 8, s, p, *, h, H, +, x, x_threshold, data_dic, |, _"
        print(markers)
        print('\n')

    @staticmethod
    def print_encoding():
        print("-" * 70 + "encoding" + "-" * 70)
        print("常见的编码形式 encoding:UTF-8，UTF-16, iso-8859-1")
        print('\n')

    @staticmethod
    def print_color_palette():
        print("-" * 70 + "color_palette" + "-" * 70)
        print("可选的背景颜色 background_color: 'Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'deep', "
              "'deep6', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', "
              "'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', "
              "'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', "
              "'binary', 'bone', 'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', "
              "'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', "
              "'gnuplot', 'gnuplot2', 'gray', 'hls', 'hot', 'hsv', 'husl', 'inferno', 'jet', 'magma', 'muted', "
              "'muted6', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spring', "
              "'summer', 'tab10', 'tab20', 'tab20b', 'tab20c', 'terrain', 'turbo', 'twilight', 'twilight_shifted', "
              "'viridis', 'winter', 'pastel', 'pastel6', 'bright', 'bright6', 'dark', 'dark6', 'colorblind', "
              "'colorblind6', 'cubehelix'")
        print("推荐的背景色 Recommended background_color: 'YlOrRd', 'GnBu', 'BuPu', 'Blues', 'Purples', 'OrRd', 'RdBu', "
              "'Oranges', 'coolwarm', 'autumn', 'BrBG', 'summer', 'pink', 'GnBu', 'afmhot', 'RdGy', "
              "'cubehelix', 'spring'")
        print("渐变调色板 color_palette: 'rocket', 'mako', 'flare', 'crest', 'magma', 'viridis'")
        print("冷暖色: 'coolwarm', 'vlag'")
        print("多彩色: 'Spectral', 'icefire'")
        print("所有颜色均可反转: + '_r'")
        print('\n')

    @staticmethod
    def print_custom_color_palette():
        print("-" * 70 + "custom_color_palette" + "-" * 70)
        print("自定义调色板: color_palette = sns.color_palette('ch:start=.2,rot=-.3', as_cmap=True)")
        print("自定义发散调色板: color_palette = sns.diverging_palette(220, 20, as_cmap=True)")
        print("亮色调色板: color_palette = sns.light_palette('seagreen', as_cmap=True)")
        print("暗色调色板: color_palette = sns.dark_palette('#69d', reverse=True, as_cmap=True)")
        print("指定颜色调色板: color_palette = sns.color_palette('light:b', as_cmap=True)")
        print("相关网站: https://seaborn.pydata.org/tutorial/color_palettes.html")
        print("配色方法: https://en.wikipedia.org/wiki/Color_blindness")
        print('\n')

    @staticmethod
    def print_parameter_about_math():
        # 创建一个 PrettyTable 对象
        table = PrettyTable()
        # 添加列名
        table.field_names = ["Mathematical Operation", "Description", "Python Code Example"]
        # 向表格中添加行
        math_operations = [
            ("Addition", "Sum of two numbers", "a + b"),
            ("Subtraction", "Difference of two numbers", "a - b"),
            ("Multiplication", "Product of two numbers", "a * b"),
            ("Division", "Division of two numbers", "a / b"),
            ("Exponentiation", "Raising a number to the power of another", "a ** b or pow(a, b)"),
            ("Square Root", "Square root of a number", "math.sqrt(x) or np.sqrt(x)"),
            ("Logarithm", "Logarithm of a number with a specified base",
             "math.log(x, base) or np.log(x) / np.log(base)"),
            ("Sine", "Sine of an angle (in radians)", "math.sin(x) or np.sin(x)"),
            ("Cosine", "Cosine of an angle (in radians)", "math.cos(x) or np.cos(x)"),
            ("Tangent", "Tangent of an angle (in radians)", "math.tan(x) or np.tan(x)"),
            ("Pi", "Ratio of the circumference of a circle to its diameter", "math.pi or np.pi"),
            ("Euler's Number (e)", "Base of the natural logarithm", "math.e or np.e"),
            ("Absolute Value", "Absolute value of a number", "abs(x) or np.abs(x)"),
            ("Summation", "Sum of a sequence of numbers", "sum([a, b, c, ...]) or np.sum([a, b, c, ...])"),
            ("Maximum", "Maximum value in a sequence of numbers", "max([a, b, c, ...]) or np.max([a, b, c, ...])"),
            ("Minimum", "Minimum value in a sequence of numbers", "min([a, b, c, ...]) or np.min([a, b, c, ...])"),
            ("Mean", "Average of a sequence of numbers",
             "sum([a, b, c, ...]) / len([a, b, c, ...]) or np.mean([a, b, c, ...])"),
            ("Standard Deviation", "Standard deviation of a sequence of numbers",
             "math.sqrt(sum([(x - mean) ** 2 for x in [a, b, c, ...]]) /"
             " len([a, b, c, ...])) or np.std([a, b, c, ...])"),
            ("Trigonometric Functions (Radians)", "Trigonometric functions of an angle (in radians)",
             "math.sin(radians), math.cos(radians), math.tan(radians)"),
            ("Trigonometric Functions (Degrees)", "Trigonometric functions of an angle (in degrees)",
             "np.sin(np.radians(degrees)), np.cos(np.radians(degrees)), np.tan(np.radians(degrees))"),
            ("Radians to Degrees", "Convert an angle from radians to degrees",
             "math.degrees(radians) or np.degrees(radians)"),
            ("Degrees to Radians", "Convert an angle from degrees to radians",
             "math.radians(degrees) or np.radians(degrees)")
        ]

        for operation in math_operations:
            table.add_row(operation)
        table.align = "l"  # 左对齐
        print(table)


""" 检查代码"""
class Monitor:
    pass


"""数据观测"""
if __name__ == '__main__':
    print('♥ 测数据各各有用， 发文章篇篇都中 ♥', end='\n')
    observer = Observer()
    observer.print_width_height()
    observer.print_markers()
    observer.print_encoding()
    observer.print_color_palette()
    observer.print_custom_color_palette()
    observer.print_parameter_about_math()
    