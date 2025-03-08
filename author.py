"""
magiclib / author

Attention:
1. statements should be abbreviated
2. This module is a standalone module and does not depend on other parts of the Magic library
"""

# 导入顺序不同有可能导致程序异常
''' from . import general'''

import os
import re
import inspect
import difflib
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT
from typing import Union, Tuple, Optional, List, Dict, Text

Style = {
    "main_title": {
        "font": {
            "font_name": "Times New Roman",
            "font_size": 16,
            "bold": True
        },
        "paragraph": {
            "alignment": "center"  # 居中对齐
        }
    },
    "secondary_title": {
        "font": {
            "font_name": "Times New Roman",
            "font_size": 12,
            "bold": True
        },
        "paragraph": {
            "alignment": "justify"  # 两端对齐
        }
    },
    "tertiary_title": {
        "font": {
            "font_name": "Times New Roman",
            "font_size": 12,
            "bold": False
        },
        "paragraph": {
            "alignment": "justify"  # 两端对齐
        }
    },
    "body_text": {
        "font": {
            "font_name": "Times New Roman",
            "font_size": 12,
            "bold": False
        },
        "paragraph": {
            "alignment": "justify",  # 两端对齐
            "first_line_indent": 0.63  # 首行缩进 0.63 cm
        }
    },
    "figure_caption": {
        "font": {
            "font_name": "Times New Roman",
            "font_size": 12,
            "bold": False
        },
        "paragraph": {
            "alignment": "left",  # 左对齐
            "space_after": True  # 段后空一行
        }
    },
    "table_caption": {
        "font": {
            "font_name": "Times New Roman",
            "font_size": 12,
            "bold": False
        },
        "paragraph": {
            "alignment": "left",  # 左对齐
            "space_after": True  # 段后空一行
        }
    }
}


""" 检查或修改文本内容 """
class Helper:
    """
    主要用于英文文本内容。用于检查文本 (支持段落) 的内容，也可以用于修改指定内容
    Used to check the content of text (supporting paragraphs), and can also be used to modify specified content.
    Mainly used for English text content

    注意： 接入的 data_dic 长度只允许为 1
    """

    # 初始化
    def __init__(self, text: Optional[Text] = None, text2: Optional[Text] = None):
        """
        # 接收参数 (1)
        :param text: (str) 需要处理的文本内容
        :param text2: (str) 需要处理的第二个文本内容
        """

        self.text = text
        self.text2 = text2

    # 检查文本内容的词数 / 字数
    def count_words(self, show_print: bool = True) -> Tuple[int, int]:
        """
        用于检查文本内容的词数 / 字数，中文则是为字数，英文则为词数
        The number of words/words used to check the content of the text is the number of words in Chinese
        and the number of words in English.

        注意：1. 英文会按照在一起的字母来检索，不会检查拼写是否正确
             2. 不会检查标点或者其它 emoji 符号
        Attention: 1. English is searched by the letters together, without checking for correct spelling.
                   2. Don't check punctuation or other emojis.

        :param show_print: (bool) 是否打印结果，默认为 True

        :return num_english_words: (int) 英文本文内容的词数
        :return num_chinese_characters: (int) 中文本文内容的字数
        """

        # 检查赋值
        if self.text is not None:
            input_text = self.text
        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The variable text should be assigned a value and be a string.")

        # 统计中文字符数
        chinese_characters = re.findall(pattern='[\u4e00-\u9fff]', string=input_text)
        num_chinese_characters = len(chinese_characters)

        # 统计英文单词数
        english_words = re.findall(pattern='[a-zA-Z]+', string=input_text)
        num_english_words = len(english_words)

        # 打印结果
        if show_print:
            if num_chinese_characters > 0 and num_english_words > 0:
                print(
                    f"The input string contains \033[94m{num_english_words}\033[0m English words and "
                    f"\033[91m{num_chinese_characters}\033[0m Chinese characters.")
            elif num_chinese_characters > 0:
                print(f"The input string contains \033[91m{num_chinese_characters}\033[0m Chinese characters.")
            else:
                print(f"The input string contains \033[94m{num_english_words}\033[0m English words.")

        return num_english_words, num_chinese_characters

    # 比较文本内容
    def compare_text(self, show_print: bool = True) -> str:
        """
        比较两个文本内容，输出差异并高亮显示，并返回所有打印的内容。
        Compare the two text contents, print the difference and highlight it, and return all the printed content.

        注意：1. 根据中文和英文的句号来拆分句子，然后来判断每一句话是否一致
             2. 每个句号后会包容任意数量的与之相连的空白字符，即会忽略这些的存在
        Attention: 1. Split sentences according to the Chinese and English periods, and then determine whether
                      each sentence is consistent.
                   2. Any number of associated whitespace characters are ignored after each period.

        :param show_print: (bool) 是否打印结果，默认为 True

        :return output_text: (str) 包含所有需要打印内容的字符串
        """

        # 检查赋值
        if self.text is not None and self.text2 is not None:
            input_text1 = self.text
            input_text2 = self.text2
        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The variables text and text2 should be assigned values and both be a string.")

        # 使用正则表达式来根据中文和英文的句号拆分文本为句子
        sentences1 = re.split(pattern=r'(?<=[。.])\s*', string=input_text1)
        sentences1 = [s for s in sentences1 if s]

        sentences2 = re.split(pattern=r'(?<=[。.])\s*', string=input_text2)
        sentences2 = [s for s in sentences2 if s]

        # 使用difflib.Differ来比较两个句子列表
        d = difflib.Differ()
        diff = list(d.compare(sentences1, sentences2))

        result1 = []
        for line in diff:
            if line.startswith("- "):
                # 使用红色高亮显示第一个文本中独有的内容
                result1.append("\033[38;5;161m" + line[2:] + "\033[0m")
            elif line.startswith("  "):
                # 无差异的内容
                result1.append(line[2:])

        # 生成第二个文本的差异，这涉及到反转 + 和 - 符号
        reversed_diff = [line.replace('-', '+') if line.startswith('-')
                         else line.replace('+', '-') for line in diff]

        result2 = []
        for line in reversed_diff:
            if line.startswith("- "):
                # 使用蓝色高亮显示第二个文本中独有的内容
                result2.append("\033[34m" + line[2:] + "\033[0m")
            elif line.startswith("  "):
                # 无差异的内容
                result2.append(line[2:])

        # 打印第一个文本和差异
        output = ["-" * 50 + " Text 1 " + "-" * 50, "\n".join(result1), "-" * 50 + " Text 2 " + "-" * 50,
                  "\n".join(result2)]

        output_text = "\n".join(output)

        # 打印结果
        if show_print:
            print(output_text)

        # 返回所有打印的内容
        return output_text

    # 修正文本中空格的数量
    def normalize_spaces(self, show_print: bool = True) -> str:
        """
        根据中英文的要求来修正文本中空格的数量
        Normalize the number of Spaces in the text according to the requirements of Chinese and English.

        注意：1. 中文标点 '。！？，、；：' 后会直接接文本，而英文 '.!?,;:' 后会加一个空格再接文本
             2. 中文和英文之间会加上空格，无论哪个在前哪个在后
             3. 会删除开头和结尾的换行符，且删除多余一个的换行符使其保留一个
        Attention: 1. Chinese punctuation '。！？，、；：' After will be directly followed by text,
                      while English '.!?,;:' is followed by a space before the text.
                   2. A space is added between Chinese and English, whichever comes first and whichever comes last.
                   3. The beginning and ending newlines are deleted, and the extra newline character is deleted
                      so that it remains one.

        :param show_print: (bool) 是否打印结果，默认为 True

        :return output_text: (str) 包含所有需要打印内容的字符串
        """

        # 检查赋值
        if self.text is not None:
            input_text = self.text
        else:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The variable text should be assigned a value and be a string.")

        # 移除文章开头的换行符
        text = re.sub(pattern=r'^\s+', repl='', string=input_text)

        # 连续应用正则表达式直到中文字符间的空格完全移除
        while re.search(pattern=r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', string=text):
            text = re.sub(pattern=r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', repl=r'\1\2',    string=text)

        # 在中文和英文之间添加必要的空格
        text = re.sub(pattern=r'([\u4e00-\u9fff])([A-Za-z])', repl=r'\1 \2', string=text)  # 中文后跟英文
        text = re.sub(pattern=r'([A-Za-z])([\u4e00-\u9fff])', repl=r'\1 \2', string=text)  # 英文后跟中文

        # 替换英文文本中多余的空格为单个空格
        text = re.sub(pattern=r'[ \t]+', repl=' ', string=text)

        # 处理所有中文标点，确保后面没有空格
        chinese_punctuation = "。！？，、；："
        for punct in chinese_punctuation:
            text = re.sub(pattern=r'\{}[\t ]*'.format(punct), repl=punct, string=text)

        # 处理所有英文标点，确保后面有一个空格
        english_punctuation = ".!?,;:"
        for punct in english_punctuation:
            text = re.sub(pattern=r'\{}[\t ]*'.format(punct), repl=punct + ' ', string=text)

        # 删除文本末尾不必要的空格及标点后的多余空格
        text = re.sub(pattern=r'\s+$', repl='', string=text)
        text = re.sub(pattern=r'\s+([.,;!?])', repl=r'\1', string=text)

        # 处理多余的换行符，只保留一个
        text_without_spaces = re.sub(pattern=r'\n+', repl='\n', string=text)

        # 打印结果
        if show_print:
            print(text_without_spaces)

        return text_without_spaces


""" 对 word 文档进行修改 """
class Word:
    """
    用于对 word 文档的操作，查找，修改等

    """

    Style = Style

    def __init__(self, read_path: Optional[str] = None, save_path: Optional[str] = None):
        """
        接受参数

        :param read_path: (str) word 文档的读取路径
        :param save_path:  (str) word 文档的保存路径
        """

        self.read_path = read_path
        self.save_path = save_path

    # 对文本格式的修改
    def format_word_document(self, inspection_mode: bool = False) -> None:
        """
        修改 Word 文档中的内容格式，使用预定义的 self.Font_Style 字典
        Modify the format of content in a Word document, using the predefined self.Font_Style dictionary

        文章需要以下格式才能正常被识别：
            1. 开头第一段直接为文章的标题内容
            2. 图名必需以 'Fig.' 开头
            3. 表标题必需第一行以 'Table' 开头，且第二行为其名
            4. 正文在首个二级标题到参考文献之间

        The article must follow the format below to be properly recognized:
            1.	The first paragraph at the beginning should directly contain the title of the article.
            2.	Figure captions must start with “Fig.”
            3.	Table captions must begin with “Table” on the first line, followed by the name on the second line.
            4.	The main text should be located between the first second-level heading and the references section.

        :param inspection_mode: (bool) 是否打开检查模式，默认为 False

        :return: None
        """

        # 文件路径
        read_path = self.read_path
        save_path = self.save_path

        # 检查文件路径有效性
        if not os.path.isfile(self.read_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"The specified read_path does not exist or is not a valid file: {self.read_path}")

        # 打开 Word 文档
        doc = Document(read_path)

        # 标志变量
        found_main_title = False
        in_body_text = False
        table_next_paragraph = False

        # 正则表达式模式
        second_level_heading_pattern = r"^\s*\d+\.\s*[^0-9]+$"
        third_level_heading_pattern = r"^\s*\d+\.\d+\.\s*[^0-9]+$"
        fig_title_pattern = r"^Fig\.\s*\d+\.\s*.*$"
        table_title_pattern = r"^Table\s*\d+\s*.*$"
        references_pattern = r"^References$"

        # 遍历文档中的每个段落
        for idx, para in enumerate(doc.paragraphs):
            text = para.text.strip()  # 获取段落文本并去掉首尾空格
            if not text:  # 如果段落为空，则跳过
                continue

            # 打印原文段落
            if inspection_mode:
                print(f'{text}')

            # 如果还未找到主标题，则将当前段落处理为主标题
            if not found_main_title:
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["main_title"]["font"]["font_name"]
                run.font.size = Pt(self.Style["main_title"]["font"]["font_size"])
                run.bold = self.Style["main_title"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER  # 使用正确的对齐方式
                if "first_line_indent" in self.Style["main_title"]["font"]:
                    para.paragraph_format.first_line_indent = Cm(self.Style["main_title"]["font"]["first_line_indent"])
                if ("space_after" in self.Style["main_title"]["font"]
                        and self.Style["main_title"]["font"]["space_after"]):
                    para.paragraph_format.space_after = Pt(12)
                found_main_title = True  # 标记已找到主标题
                print(f'\033[1m\033[3m{text}\033[0m')  # 打印主标题
                continue

            # 如果匹配到二级标题格式
            if re.match(second_level_heading_pattern, text):
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["secondary_title"]["font"]["font_name"]
                run.font.size = Pt(self.Style["secondary_title"]["font"]["font_size"])
                run.bold = self.Style["secondary_title"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                in_body_text = True  # 标记进入正文部分
                print(f'\033[91m{text}\033[0m')  # 打印二级标题
                continue

            # 如果匹配到三级标题格式
            if re.match(third_level_heading_pattern, text):
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["tertiary_title"]["font"]["font_name"]
                run.font.size = Pt(self.Style["tertiary_title"]["font"]["font_size"])
                run.bold = self.Style["tertiary_title"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                if inspection_mode:
                    print(f'\033[95m{text}\033[0m')  # 打印三级标题
                else:
                    print('', end='  ')
                    print(f'\033[95m{text}\033[0m')  # 打印三级标题
                continue

            # 如果匹配到图标题格式
            if re.match(fig_title_pattern, text):
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["figure_caption"]["font"]["font_name"]
                run.font.size = Pt(self.Style["figure_caption"]["font"]["font_size"])
                run.bold = self.Style["figure_caption"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                if inspection_mode:
                    print(f'\033[34;2m{text}\033[0m')  # 打印图标题
                continue

            # 如果匹配到表格标题格式
            if re.match(table_title_pattern, text):
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["table_caption"]["font"]["font_name"]
                run.font.size = Pt(self.Style["table_caption"]["font"]["font_size"])
                run.bold = self.Style["table_caption"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                table_next_paragraph = "(continued)" not in para.text  # 标记下一段也是表格标题
                if inspection_mode:
                    print(f'\033[32;2m{text}\033[0m')  # 打印表格标题
                continue

            # 如果上一段标记为表格标题的后续段落
            if table_next_paragraph:
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["table_caption"]["font"]["font_name"]
                run.font.size = Pt(self.Style["table_caption"]["font"]["font_size"])
                run.bold = self.Style["table_caption"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                table_next_paragraph = False  # 重置标记
                if inspection_mode:
                    print(f'\033[32;2m{text}\033[0m')  # 打印表格标题续
                continue

            # 如果匹配到参考文献部分
            if re.match(references_pattern, text):
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["body_text"]["font"]["font_name"]
                run.font.size = Pt(self.Style["body_text"]["font"]["font_size"])
                run.bold = self.Style["body_text"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                in_body_text = False  # 标记退出正文部分
                if inspection_mode:
                    print(f'\033[37;2m{text}\033[0m')  # 打印参考文献
                continue

            # 处理正文段落
            if (found_main_title and not re.match(second_level_heading_pattern, text) and in_body_text
                    and not re.match(fig_title_pattern, text) and not re.match(table_title_pattern, text)):
                run = para.runs[0] if para.runs else para.add_run()
                run.font.name = self.Style["body_text"]["font"]["font_name"]
                run.font.size = Pt(self.Style["body_text"]["font"]["font_size"])
                run.bold = self.Style["body_text"]["font"]["bold"]
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT  # 使用正确的对齐方式
                if inspection_mode:
                    print(f'\033[4m{text}\033[0m')  # 打印正文段落）

        # 保存修改后的文档
        if save_path is not None:
            doc.save(save_path)

        return None
