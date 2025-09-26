"""
magiclib / performer

------------------------------------------------------------------------------------------------------------------------
performer is the GUI module of magiclib providing a streamlined interface for lab analysis. It manages multiple windows
for XRD, Raman, and XRF workflows. Users can input data_path and save_path, execute analysis via RUN buttons, and view
results in a read-only log. The main window organizes buttons in a scrollable grid, each opening a specialized
subwindow. Designed with PyQt5, it integrates plotting, reading, and processing functionality from projector, enabling
interactive visualization and analysis while keeping operations concise and accessible.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""


from . import projector

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QSizePolicy, \
    QGridLayout, QMessageBox, QDialog, QMainWindow
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit,
                             QPushButton, QHBoxLayout, QTextEdit)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


""" 主窗口 """
class Main_Window(QWidget):
    """
    Main window of software Library and Laboratory
    """

    def __init__(self):
        super().__init__()
        self.sub_window = None
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('Library and Laboratory')
        self.setGeometry(400, 300, 900, 600)

        # 创建主布局
        main_layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("Library and Laboratory")
        title_label.setFont(QFont("Times new roman", 40))  # 设置标题字体和大小
        title_label.setFixedSize(800, 150)  # 设置标题大小
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        main_layout.addWidget(title_label, alignment=Qt.AlignHCenter)
        title_label.setStyleSheet("background-color: yellow")  # 设置标题的背景色为黄色

        # 创建滚动区域和容器窗口
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # 创建一个容器窗口，用于放置按钮
        container = QWidget()

        # 创建网格布局来放置按钮
        grid_layout = QGridLayout()

        # 创建8个彩色按钮
        for i in range(8):
            row = i // 2
            col = i % 2

            # 第 1 个按钮 XRD
            if i == 0:
                button = QPushButton('XRD')
                button.setStyleSheet('background-color: #5C0120; font-size: 30px; color: #FFB6C1')  # 设置按钮的背景颜色和文字大小
                button.setFixedSize(400, 200)  # 调整按钮的大小
                button.setFont(QFont("Times New Roman"))  # 设置按钮的字体为"Times New Roman"
                grid_layout.addWidget(button, row, col)

                # 添加按钮间的间隙
                vertical_spacer = QWidget()
                vertical_spacer.setFixedSize(1, 230)
                grid_layout.addWidget(vertical_spacer, row, col)

                # 小窗口
                button.clicked.connect(self.open_XRD_Window)

            # 第 2 个按钮 Raman
            elif i == 1:
                button = QPushButton('Raman')
                button.setStyleSheet('background-color: #00008B; font-size: 30px; color: #ADD8E6')  # 设置按钮的背景颜色和文字大小
                button.setFixedSize(400, 200)  # 调整按钮的大小
                button.setFont(QFont("Times New Roman"))  # 设置按钮的字体为"Times New Roman"
                grid_layout.addWidget(button, row, col)

                # 添加按钮间的间隙
                vertical_spacer = QWidget()
                vertical_spacer.setFixedSize(1, 230)
                grid_layout.addWidget(vertical_spacer, row, col)

                # 小窗口
                button.clicked.connect(self.open_Raman_Window)

            # 第 3 个按钮 XRF
            elif i == 2:
                button = QPushButton('XRF')
                button.setStyleSheet('background-color: #006400; font-size: 30px; color: #98FB98')  # 设置按钮的背景颜色和文字大小
                button.setFixedSize(400, 200)  # 调整按钮的大小
                button.setFont(QFont("Times New Roman"))  # 设置按钮的字体为"Times New Roman"
                grid_layout.addWidget(button, row, col)

                # 添加按钮间的间隙
                vertical_spacer = QWidget()
                vertical_spacer.setFixedSize(1, 230)
                grid_layout.addWidget(vertical_spacer, row, col)

                # 小窗口
                button.clicked.connect(self.open_XRF_Window)  # 点击时执行该方法

            # # 第 4 个按钮 XPS
            # elif i == 3:
            #     button = QPushButton('XPS')
            #     button.setStyleSheet('background-color: #8B4513; font-size: 30px; color: #FAFAD2')  # 设置按钮的背景颜色和文字大小
            #     button.setFixedSize(400, 200)  # 调整按钮的大小
            #     button.setFont(QFont("Times New Roman"))  # 设置按钮的字体为"Times New Roman"
            #     grid_layout.addWidget(button, row, col)
            #
            #     # 添加按钮间的间隙
            #     vertical_spacer = QWidget()
            #     vertical_spacer.setFixedSize(1, 230)
            #     grid_layout.addWidget(vertical_spacer, row, col)
            #
            #     # 小窗口
            #     button.clicked.connect(self.open_XRD_Window)

            else:

                button = QPushButton(f'Button {i + 1}')
                button.setStyleSheet('background-color: #1f77b4; font-size: 30px; color: white;')  # 设置按钮的背景颜色、文字大小和字体颜色
                button.setFixedSize(400, 200)  # 调整按钮的大小
                button.setFont(QFont("Times New Roman"))  # 设置按钮的字体为"Times New Roman"
                grid_layout.addWidget(button, row, col)

                # 添加按钮间的间隙
                vertical_spacer = QWidget()
                vertical_spacer.setFixedSize(1, 230)
                grid_layout.addWidget(vertical_spacer, row, col)

        # 设置容器窗口的布局
        container.setLayout(grid_layout)

        # 将容器窗口设置为滚动区域的窗口部件
        scroll_area.setWidget(container)

        # 设置滚动区域的大小策略
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)

        # 设置主窗口的布局
        self.setLayout(main_layout)

        # 显示窗口
        self.show()

    # 1 XRD
    def open_XRD_Window(self):

        self.sub_window = XRD_Window()
        self.sub_window.show()

    # 2 Raman
    def open_Raman_Window(self):

        self.sub_window = Raman_Window()
        self.sub_window.show()

    # 3 XRF
    def open_XRF_Window(self):

        self.sub_window = XRF_Window()
        self.sub_window.show()

    # 4 XPS
    # def open_XPS_Window(self):
    #
    #     self.sub_window = XPS_Window()
    #     self.sub_window.show()

    # SubWindow (暂时无用)
    def openSubWindow(self):

        self.sub_window = SubWindow()
        self.sub_window.show()


""" XRD """
class XRD_Window(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("XRD")
        self.resize(600, 400)  # 设置窗口的初始大小

        # 初始化变量以保存路径
        self.data_path = None
        self.save_path = None

        # 创建布局
        layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("XRD")
        title_label.setFont(QFont("Times new roman", 40))  # 设置标题字体和大小
        title_label.setFixedHeight(150)  # 固定标题的高度为 150
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        title_label.setStyleSheet('background-color: #5C0120; font-size: 30px; color: #FFB6C1')
        layout.addWidget(title_label)

        # 添加两个带标签的文本输入框  第一个输入框
        self.data_path_edit = QLineEdit()  # 保持对文本框的引用
        data_path_layout = QHBoxLayout()
        data_path_label = QLabel("data_path:")
        data_path_layout.addWidget(data_path_label)
        data_path_layout.addWidget(self.data_path_edit)
        layout.addLayout(data_path_layout)

        # 第二个输入框
        self.save_path_edit = QLineEdit()  # 保持对文本框的引用
        save_path_layout = QHBoxLayout()
        save_path_label = QLabel("save_path:")
        save_path_layout.addWidget(save_path_label)
        save_path_layout.addWidget(self.save_path_edit)
        layout.addLayout(save_path_layout)

        # 添加RUN按钮
        run_button = QPushButton("RUN")
        run_button.clicked.connect(self.XRD_run)
        layout.addWidget(run_button)

        # 添加只读文本框显示Python返回的信息
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFixedHeight(200)  # 设置为 200 像素高
        # 设置文本框的字体为等宽字体
        font = QFont("Courier New", 14)  # 或者其他你喜欢的等宽字体
        self.log_text_edit.setFont(font)

        layout.addWidget(self.log_text_edit)

        # 创建中心部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # 设置中心部件
        self.setCentralWidget(central_widget)

    def XRD_run(self):

        # 从文本框中获取路径并保存到实例变量中
        self.data_path = self.data_path_edit.text()
        self.save_path = self.save_path_edit.text()

        # 打印或以其他方式使用路径
        xrd = projector.XRD(read_path=self.data_path,
                            save_path=self.save_path)
        xrd.read()
        xrd.add_xrd_pdf_match()
        details = xrd.plot_xrd_pdf()

        # 更新日志文本框内容
        self.log_text_edit.append(str(details))  # 示意文本，可以根据需要更改


""" Raman """
class Raman_Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raman")
        self.resize(600, 400)  # 设置窗口的初始大小

        # 初始化变量以保存路径
        self.data_path = None
        self.save_path = None

        # 创建布局
        layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("Raman")
        title_label.setFont(QFont("Times new roman", 40))  # 设置标题字体和大小
        title_label.setFixedHeight(150)  # 固定标题的高度为 150
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        title_label.setStyleSheet('background-color: #00008B; font-size: 30px; color: #ADD8E6')
        layout.addWidget(title_label)

        # 添加两个带标签的文本输入框
        # 第一个输入框
        self.data_path_edit = QLineEdit()  # 保持对文本框的引用
        data_path_layout = QHBoxLayout()
        data_path_label = QLabel("data_path:")
        data_path_layout.addWidget(data_path_label)
        data_path_layout.addWidget(self.data_path_edit)
        layout.addLayout(data_path_layout)

        # 第二个输入框
        self.save_path_edit = QLineEdit()  # 保持对文本框的引用
        save_path_layout = QHBoxLayout()
        save_path_label = QLabel("save_path:")
        save_path_layout.addWidget(save_path_label)
        save_path_layout.addWidget(self.save_path_edit)
        layout.addLayout(save_path_layout)

        # 添加RUN按钮
        run_button = QPushButton("RUN")
        run_button.clicked.connect(self.Raman_run)
        layout.addWidget(run_button)

        # 添加只读文本框显示Python返回的信息
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFixedHeight(200)  # 设置为 200 像素高
        # 设置文本框的字体为等宽字体
        font = QFont("Courier New", 14)  # 或者其他你喜欢的等宽字体
        self.log_text_edit.setFont(font)

        layout.addWidget(self.log_text_edit)

        # 创建中心部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # 设置中心部件
        self.setCentralWidget(central_widget)

    def Raman_run(self):

        # 从文本框中获取路径并保存到实例变量中
        self.data_path = self.data_path_edit.text()
        self.save_path = self.save_path_edit.text()

        # 打印或以其他方式使用路径
        raman = projector.Raman(read_path=self.data_path,
                            save_path=self.save_path)
        raman.read()
        details = raman.plot()

        # 更新日志文本框内容
        self.log_text_edit.append(details)  # 示意文本，可以根据需要更改


""" XRF """
class XRF_Window(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("XRF")
        self.resize(600, 400)  # 设置窗口的初始大小

        # 初始化变量以保存路径
        self.data_path = None
        self.save_path = None

        # 创建布局
        layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("XRF")
        title_label.setFont(QFont("Times new roman", 40))  # 设置标题字体和大小
        title_label.setFixedHeight(150)  # 固定标题的高度为 150
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        title_label.setStyleSheet('background-color: #006400; font-size: 30px; color: #98FB98')
        layout.addWidget(title_label)

        # 添加两个带标签的文本输入框  第一个输入框
        self.data_path_edit = QLineEdit()  # 保持对文本框的引用
        data_path_layout = QHBoxLayout()
        data_path_label = QLabel("data_path:")
        data_path_layout.addWidget(data_path_label)
        data_path_layout.addWidget(self.data_path_edit)
        layout.addLayout(data_path_layout)

        # 第二个输入框
        self.save_path_edit = QLineEdit()  # 保持对文本框的引用
        save_path_layout = QHBoxLayout()
        save_path_label = QLabel("save_path:")
        save_path_layout.addWidget(save_path_label)
        save_path_layout.addWidget(self.save_path_edit)
        layout.addLayout(save_path_layout)

        # 添加RUN按钮
        run_button = QPushButton("RUN")
        run_button.clicked.connect(self.XRF_run)
        layout.addWidget(run_button)

        # 添加只读文本框显示Python返回的信息
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setFixedHeight(200)  # 设置为 200 像素高
        # 设置文本框的字体为等宽字体
        font = QFont("Courier New", 14)  # 或者其他你喜欢的等宽字体
        self.log_text_edit.setFont(font)

        layout.addWidget(self.log_text_edit)

        # 创建中心部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # 设置中心部件
        self.setCentralWidget(central_widget)

    def XRF_run(self):

        # 从文本框中获取路径并保存到实例变量中
        self.data_path = self.data_path_edit.text()
        self.save_path = self.save_path_edit.text()

        # 打印或以其他方式使用路径
        xrf = projector.XRF(read_path=self.data_path,
                            save_path=self.save_path)
        xrf.read()
        details = xrf.print_element()
        xrf.plot()

        # 更新日志文本框内容
        self.log_text_edit.append(str(details))  # 示意文本，可以根据需要更改


# """ Raman """
# class Raman_Window(QMainWindow):
#
#     def __init__(self):
#
#         super().__init__()
#         self.setWindowTitle("Raman")
#         self.resize(600, 400)  # 设置窗口的初始大小
#
#         # 初始化变量以保存路径
#         self.data_path = None
#         self.save_path = None
#
#         # 创建布局
#         layout = QVBoxLayout()
#
#         # 添加标题
#         title_label = QLabel("Raman")
#         title_label.setFont(QFont("Times new roman", 40))  # 设置标题字体和大小
#         title_label.setFixedHeight(150)  # 固定标题的高度为 150
#         title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
#         title_label.setStyleSheet('background-color: #00008B; font-size: 30px; color: #ADD8E6')
#         layout.addWidget(title_label)
#
#         # 添加两个带标签的文本输入框
#         # 第一个输入框
#         self.data_path_edit = QLineEdit()  # 保持对文本框的引用
#         data_path_layout = QHBoxLayout()
#         data_path_label = QLabel("data_path:")
#         data_path_layout.addWidget(data_path_label)
#         data_path_layout.addWidget(self.data_path_edit)
#         layout.addLayout(data_path_layout)
#
#         # 第二个输入框
#         self.save_path_edit = QLineEdit()  # 保持对文本框的引用
#         save_path_layout = QHBoxLayout()
#         save_path_label = QLabel("save_path:")
#         save_path_layout.addWidget(save_path_label)
#         save_path_layout.addWidget(self.save_path_edit)
#         layout.addLayout(save_path_layout)
#
#         # 添加RUN按钮
#         run_button = QPushButton("RUN")
#         run_button.clicked.connect(self.Raman_run)
#         layout.addWidget(run_button)
#
#         # 添加只读文本框显示Python返回的信息
#         self.log_text_edit = QTextEdit()
#         self.log_text_edit.setReadOnly(True)
#         self.log_text_edit.setFixedHeight(200)  # 设置为 200 像素高
#         # 设置文本框的字体为等宽字体
#         font = QFont("Courier New", 14)  # 或者其他你喜欢的等宽字体
#         self.log_text_edit.setFont(font)
#
#         layout.addWidget(self.log_text_edit)
#
#         # 创建中心部件并设置布局
#         central_widget = QWidget()
#         central_widget.setLayout(layout)
#
#         # 设置中心部件
#         self.setCentralWidget(central_widget)
#
#     def Raman_run(self):
#
#         # 从文本框中获取路径并保存到实例变量中
#         self.data_path = self.data_path_edit.text()
#         self.save_path = self.save_path_edit.text()
#
#         # 打印或以其他方式使用路径
#         raman = projector.Raman(read_path=self.data_path,
#                             save_path=self.save_path)
#         raman.read()
#         details = raman.plot()
#
#         # 更新日志文本框内容
#         self.log_text_edit.append(details)  # 示意文本，可以根据需要更改


""" 小窗口 """
class SubWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("XRD")
        self.resize(400, 300)  # 设置窗口的初始大小

        # 创建布局
        layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("Library and Laboratory")
        title_label.setFont(QFont("Times New Roman", 40))  # 设置标题字体和大小
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        title_label.setStyleSheet("background-color: yellow")  # 设置标题的背景色为黄色
        layout.addWidget(title_label)

        # 创建中心部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # 设置中心部件
        self.setCentralWidget(central_widget)


""" 调用 """
def run_app():

    app = QApplication(sys.argv)  # 创建 QApplication 实例
    window = Main_Window()        # 创建窗口实例
    window.show()                 # 显示窗口
    sys.exit(app.exec_())         # 启动事件循环
