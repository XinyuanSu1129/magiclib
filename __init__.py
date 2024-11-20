
__version__ = '6.1.4'

__all__ = ['general', 'grapher', 'projector', 'performer', 'inspector', 'accessor', 'author', 'potter']


# Import warning
import warnings

# Ignore warnings (unavoidable warnings due to version issues)
warnings.filterwarnings(action="ignore", message=".*is_categorical_dtype is deprecated.*")
warnings.filterwarnings(action="ignore", message="The behavior of DataFrame concatenation with empty or "
                                                 "all-NA entries is deprecated.*")

# Function
r"""
MAGIC
========================================================================================================================

-----   /   -----   /   -----   /   -----   *  *  *  Magic  Library  *  *  *   -----   \   -----   \   -----   \   -----

========================================================================================================================

---------------------------------------------------- general -----------------------------------------------------------

 -Function-      函数插件区： 完成小功能
 -Optimizer-     数据重组区 (Function)： 对数据进行处理和优化
 -Manager-       文件管理区 (Optimizer)： 对文件进行操作和绘图
 -Module-        模组控制区 (Optimizer)： 对 DataFrame 进行操作，在中心函数前后布置
 -Magic-         中心函数区 (Manager, Module)： 数据处理的中心部分

---------------------------------------------------- grapher -----------------------------------------------------------

 -Statistics-    统计学类 (Manager)： 用于统计学分析和绘图
 -Plotter-       绘图类 (Manager)：用于绘制各种图像
 -Fitter-        拟合类 (Manager)：用于拟合数据，绘制拟合后的图像

--------------------------------------------------- projector ----------------------------------------------------------

 -Keyword-       标准函数组 (ABC)： 对特定作业进行的特定函数组合的准备，为抽象类
 -Tensile-       拉伸曲线类 (Keyword)： 绘制拉伸曲线 (应力-应变)
 -Compression-   压缩曲线类 (Keyword)： 绘制压缩曲线 (应力-应变)
 -Torsion-       扭转曲线类 (Keyword)： 绘制扭转曲线 (应力-应变)
 -XRD-           XRD 类 (Keyword)： 绘制 XRD 曲线
 -Raman-         Raman 类 (Keyword)： 绘制 Raman 曲线
 -TEA-           热膨胀类 (Keyword)： 绘制热膨胀曲线
 -XPS-           XPS 类 (Keyword)： 绘制 XPS 曲线
 
 --------------------------------------------------- performer ---------------------------------------------------------
 
 -Main_Window-   主窗口 (QWidget)： 软件的主窗口
 -XRD_Window-    XRD 窗口 (QMainWindow)： 进行 XRD 计算的小窗口
 -Raman_Window-  Raman 窗口 (QMainWindow)： 进行 Raman 计算的小窗口
 -XRF_Window-    XRF 窗口 (QMainWindow)： 进行 XRF 计算的小窗口
 -SubWindow-     附加小窗口 (QMainWindow)： 附加的小窗口，可进行其它功能设置

---------------------------------------------------- inspector ---------------------------------------------------------

 -Tool-          工具类： 用于直接调用的属性和方法
 -Observer-      观察类： 用于观测可用数据
 -Monitor-       检查类： 用于检查代码

 ---------------------------------------------------- accessor ---------------------------------------------------------

 -Constructor-   函数构造区： 用于构造函数来完成用户的请求
 -Article-       文章图像： 用于保留绘制文章中函数的图像
 -History-       属性存储： 用于存储用到过以后可能还会用的属性
 
 ----------------------------------------------------- author ----------------------------------------------------------
 
 -Helper-        文本编辑区：用于检查、修改和比较文本内容

 ----------------------------------------------------- potter ----------------------------------------------------------
 
 -PotteryBase-   基因库操作区：用于写入、读取和比较陶器数据
 -Pottery-       基因库读取区：用于简单，明快地查找和读取数据和浏览图片

========================================================================================================================

---------------------------------------------------- general -----------------------------------------------------------

 -Function-      Function Plugin Area: Completes small functions
 -Optimizer-     Data Reorganization Area (Function): Processes and optimizes data
 -Manager-       File Management Area (Optimizer): Operates on files and plots
 -Module-        Module Control Area (Optimizer): Operates on DataFrame, arranges around the central function
 -Magic-         Central Function Area (Manager, Module): The central part of data processing

---------------------------------------------------- grapher -----------------------------------------------------------

 -Statistics-    Statistics Class (Manager): For statistical analysis and plotting
 -Plotter-       Plotting Class (Manager): For drawing various images
 -Fitter-        Fitting Class (Manager): For fitting data and plotting the fitted images

--------------------------------------------------- projector ----------------------------------------------------------

 -Keyword-       Standard Function Group (ABC): Preparation for specific function combinations for specific tasks
 -Tensile-       Tensile Curve Class (Keyword): Draws tensile curves (stress-strain)
 -Compression-   Compression Curve Class (Keyword): Draws compression curves (stress-strain)
 -Torsion-       Torsion Curve Class (Keyword): Draws torsion curves (stress-strain)
 -XRD-           XRD Class (Keyword): Draws XRD curves
 -Raman-         Raman Class (Keyword): Draws Raman curves
 -TEA-           Thermal Expansion Class (Keyword): Draws thermal expansion curves
 -XPS-           XPS Class (Keyword): Draws XPS curves

 --------------------------------------------------- performer ---------------------------------------------------------

 -Main_Window-   Main Window (QWidget): The main window of the software
 -XRD_Window-    XRD Window (QMainWindow): A small window for performing XRD calculations
 -Raman_Window-  Raman Window (QMainWindow): A small window for performing Raman calculations
 -XRF_Window-    XRF Window (QMainWindow): A small window for performing XRF calculations
 -SubWindow-     Additional Small Window (QMainWindow): An additional small window for setting other functions

---------------------------------------------------- inspector ---------------------------------------------------------

 -Tool-          Utility Class: Attributes and methods used for direct calls
 -Observer-      Observation Class: For observing available data
 -Monitor-       Inspection Class: For checking code

---------------------------------------------------- accessor ---------------------------------------------------------

 -Constructor-   Function Construction Area: Used for constructing functions to fulfill user requests
 -Article-       Article Image: Used to store images drawn in the article
 -History-       Attribute Storage: Used for storing attributes that have been used and might be needed again later

----------------------------------------------------- author ----------------------------------------------------------

 -Helper-        Text Editing: Used to examine, modify, and compare text content
 
 ----------------------------------------------------- potter ----------------------------------------------------------

 -PotteryBase-   Gene Bank operation area: used to write, read and compare pottery data
 -Pottery-       Gene Library Reading area: for easy, crisp finding and reading data and viewing images

========================================================================================================================

"""

# Version
"""
 ------ Version ------

--> 0.0.0 Alpha 23-09-22 21:15 Manager upgrade.
--> 0.0.0 Beta  23-09-25 11:29 XPS upgrade.
--> 1.0.0       23-11-07 Add a new module access.
--> 1.1.0       23-11-08 save_txt() & save_excel()
--> 2.0.0 Beta  23-11-10 Add a new class Fitting.
--> 2.0.0       23-11-11 17:30 Fitting, optimize the general class code.
--> 2.0.1       23-11-14 00:05 The function of XRD multi-plot is added.
                Fixed the BUG of PDF card self-positioning in XRD.
--> 2.1.0       23-11-14 Add the fitting_functional_after_fetching function to the class Fitting.
--> 2.2.0       23-11-15 1. Update method: __data_init;
                2. Add new class: Optimizer.
--> 2.3.0       23-11-15 Add the method fitting_multiple_functional() and
                fitting_multiple_according_model() to the class Fitting,
                these two methods allow multiple images to be generated at once.
                Add the method split_df_by_row() to the class Optimizer.
--> 2.3.1       23-11-17 15:11
                Add the method merge_dfs_to_single_df_wide(), insert_data_to_df() and sort_df() to the class Optimizer.
                Optimize the method merge_dfs_to_single_df_long() and split_df_by_row() of class Optimizer.
                And optimize some methods of class Fitting.
--> 2.3.2       23-11-18 Optimize the method fitting_functional_after_fetching().
                and fitting_multiple_functional() of class Fitting.
                Add the method plot_function() to the class Plotter.
                14:30 Fix the bug of save_txt() and plot_heatmap().
--> 2.3.3       23-11-24 20:49 Changed character description: "R_square", "PC1" & "PC2".
                Fixed an issue where width_height could not be changed in plot_box.
                Fixed an issue with realisation error in XPS drawing: self.curve_noising._Function__data_init().
--> 2.4.0       23-11-28 20:32 Add the method add_xrd_pdf_match() to the class XRD.
--> 2.4.1       23-11-29 00:53 Improved font modification method when drawing.
--> 3.0.0       23-11-29 11:52 Add the class Tool to the module inspector.
                Add filling area function to plot_line() of Manager.
                Updated some descriptions in the __init__.py file.
                Change the attribute name 'character' in class Manager to the attribute name 'delimiter'.
                Updated JSON-related methods in class Manager. Add class Mapping into model projector.
--> 3.0.1       23-11-30 Improved the XRD class in the module projector.
                Improved class Mapping in the module projector.
--> 3.0.2       23-12-01 Fixed an issue where Al was identified as A when Mapping was used to identify images.
                Add the method inspect_pdf() to the class XRD.
--> 3.0.3       Changed the color of key content in the printed text.
                Improved method plt_xrd_pdf() of the class XRD.
--> 3.1.0       New classes Article & class History are created in the module access.
--> 3.1.1       23-12-06 Add the class XRF into the model projector.
                Improved the description part and partial comment description of regular expressions in module general.
--> 3.1.2       23-12-07 Improved method plot() of the class XRF.
--> 3.2.0       23-12-08 Adapt the project to python3.12
                Add an interval to the drawing program to prevent drawing too fast from causing the program to crash.
--> 3.2.1       23-12-09 Add the method calculate_statistics() to the class Optimizer.
--> 3.2.2       23-12-11 Add the method to_reality() to the class Manager.
                Improved class Optimizer.
--> 3.2.3       23-12-22 Optimized method to_reality() in class Manager.
--> 4.0.0       23-12-26 The new module performer has been added.
--> 4.0.1       24-01-10 Improved class Torsion of the model projector.
                Fixed a bug where the parameter 'swap_column' could not exchange column names correctly in general.
--> 4.0.2       24-01-11 Improved the to_reality() method in class Manager.
--> 4.0.3       24-01-13 Add the method plot_2d_histogram() & plot_ridge() to the class Plotter.
                The method __data_init() in class Function has been optimized so that precision_data_dic & 
                point_scale_dic is now computed only when instantiating class Magic and its subclasses.
                Changed a special character into a translator.
--> 4.0.4       24-01-16 Add the method split_df_by_column() to the class Optimizer.
                Adjustment class Torsion in the module projector.
                Improved the generate_rows_to_df() method in class Manager.
--> 4.1.0       24-01-17 The custom_legend function in Plotter method plot_bars() in the module grapher is optimized.
                Add the method plot_distribution() to the class Plotter.
                Changed update policy, updates will be categorized by degree, not by time.
                Changed the policy for getting x_list & y_list in __data_init(). 
                Changed some comments, especially symbols.
                Changed the default tolerance in generate_rows_to_df() to 0.05 in class Optimizer.
                Improved the normalize_columns() method in class Manager.
--> 4.1.1       24-01-18 Add the method plot_qq() to the class Plotter.
                Deprivatize the method self.data_init() in class Function.
                Changed the policy for getting x_list & y_list in data_init(). 
                Improved the fitting_functional_after_fetching() method in class Fitter.
--> 4.1.2       24-01-19 Add the method plot_jointdistribution() to the class Plotter. 
                The ability to float part of the data after reading it in read_txt() and read_excel() has been removed, 
                as it already exists and is more refined in data_init().
                Add the method find_most_color_pixel() to the class Function. 
--> 5.0.0       24-07-24 A new standalone module, author, has been added.
                Modified the comment description of the section in module general.
                Changes were made to the __init__() section.
                Add module seaborn as sns in module general.
--> 5.1.0       24-08-29 Make the output image name to the title of the data.
                update some of the annotation content. 
                update the path is now relative path.
                The module general can receive relative paths when the file is read now.
                Improved part of the annotation content.
--> 5.1.1       24-08-30 11:27          
                The add_xrd_pdf_match() method in class PDF allows you to read XRD data files with TXT or txt extension.
                Optimize all classes in module protector: Support for relative paths.
--> 6.0.0       24-09-14 13:28 A new standalone module, potter, has been added.
                Extends the color in the module general class Manager method to_reality().
--> 6.1.0       24-09-25 17:20 Optimized the class Manger in the module general. 
                The method pca_analysis() in class Statistics of the module grapher is optimized.
                Add method clear_data() in class Function.
                Converts data to float format when reading txt and Excel files, except for the category column.
--> 6.1.1       24-10-16 20:29 Fixed an issue with PDF card reading path error in class XRD.
--> 6.1.2       24-10-24 16:30 Fixed a bug that found a valid line when reading a file, in read_txt() of class Manager.
                In the '__init__.py' file, correctly rename locate_key() to locate_point().
                Updated functionality for method to_reality() in class Manager.
                Add comment "When adding a method in the classes Optimizer, Module, and Magic, 
                it is necessary to include self.current_dic."
                Optimized partial comments in module general.
                Add a new method move_points() to class Module.
--> 6.1.3       24-10-25 13:31 Add a new method handle_duplicate_x() to class Module.
                Changed how the classes Manager, Module, and Magic dictionary are read.
                Added the option to choose whether to be weighted or not, in the class Magic.
--> 6.1.4       24-11-20 11:00 The method sort_df () is optimized, and the plot_ridge () is also optimized.
                    
                    
 ------ Needing ------

--> Added the ability to map in the current directory in module potter, and added the ability to read images.
--> Supplement the author content of the module.
--> Supplement the pottery content of the module.

========================================================================================================================
                                     The  Best  Way  Out  Is  Always  Through .
╔══╗
║██║
║ o║
╚══╝
・・・★
★・・・・・・★
・・★・・★
・・・・★・・・・★
・・・・・・★・・・・・・★
・・・・・・・・★・・・・・・・・★
・・・・・・・・・・★・・・・・・・・・・★
・・・・・・・・・・・・★・・・・・・・・・・・・★
★・・・・・・・・・・・・・・★
・・★・・・・・・・・・・★
・・・・★・・・・・・★
・・・・・・★・・★                                          Start time: May 7, 2023
・・・・・・・・★・・・・★
・・・・・・・・・・★・・・・・・★                              
・・・・・・・・・・・・★・・・・・・・・★
・・・・・・・・・・・・・・★・・・・・・・・・・★                Sole developer: Xinyuan SU          
========================================================================================================================

"""

# Content
"""
-----   *  *  *  *  *  Magic  Library  *  *  *  *  *   -----


/ / / / / * general * / / / / /

database_path
Magic_Database
Standard_Database
Pottery_Database
Category_Index
interval_time

------ Function ------
data_init()
clear_data()
change_imshow()
detect_encodings()
covert_encoding()
rename_extension()
get_point_normal()
calculate_std()
calculate_probability()
get_subdirectories()
sort_data()
find_most_color_pixel()

------ Optimizer ------
generate_rows_to_df()
add_noise_to_dataframe()
change_to_random()
normalize_columns()
scale_df()
separate_df_by_category()
split_df_by_row()
split_df_by_column()
merge_dfs_to_single_df_wide()
merge_dfs_to_single_df_long()
convert_one_length_dict()
insert_data_to_df()
sort_df()
calculate_statistics()

------ Manager ------
Magic_Database
Category_Index
Font_title
Font_ticket
Font_legend
Font_mark
interval_time

read_txt()
read_excel()
read_json()
save_txt()
save_excel()
save_json()
to_magic()
to_reality()
plot_line()
plot_scatter()

------ Module ------
custom_point()
remove_point()
remove_section()
append_points()
append_section()
get_peak()
find_data_peaks()
control_peak()
get_plateau()
control_plateau()
SF_remove()
SF_append()
random_dict()
move_points()
handle_duplicate_x()

------ Magic ------
smooth_curve()
locate_point()
improve_precision()
reduce_precision()
normalize_data()
adjust_data()
assign_weight()
fit_curve()
restore_precision()
realize_data()


/ / / / / * grapher * / / / / /

------ Statistics ------
Category_Index

pca_analysis()

------ Plotter ------
plot_box()
plot_raincloud()
plot_heatmap()
plot_bars()
plot_2d_histogram()
plot_ridge()
plot_qq()
plot_distribution()
plot_jointdistribution()
peak_deconvolution()
plot_function()

------ Fitter ------
polynomial_fitting()
univariate_fitting()
fitting_functional_after_fetching()
multivariable_fitting()
fitting_multiple_functional()
fitting_multiple_according_model()


/ / / / / * projector * / / / / /

plot()

------ Keyword ------
Magic_Database
Standard_Database

read()
plot()
magic_plot()
save_json()

------ Tensile ------
read()
plot()
magic_plot()
save_json()

------ Compression ------
read()
plot()
magic_plot()
save_json()

------ Torsion ------
read()
plot()
magic_plot()
save_json()
print_peaks()

------ XRD ------
read()
plot()
magic_plot()
save_json()
plot_xrd_pdf()
plot_xrd_xrd()
add_xrd_pdf_match()
inspect_pdf()
__read_pdf()
__match_xrd_pdf()
__print_pdf_details()

------ Raman ------
read()
plot()
magic_plot()
save_json()
__curve_fitting()

------ TEA ------
read()
plot()
magic_plot()
save_json()

------ XPS ------
read()
plot()
magic_plot()
save_json()
__curve_fitting()

------ Mapping ------
read()
plot()
magic_plot()
save_json()
analyze_Mapping()
__composite_image()
__print_element_details()

------ XRF ------
read()
plot()
magic_plot()
save_json()
print_element()
save_to_excel()
__extract_data()


/ / / / / * performer * / / / / /

run_app()

------ Tool ------
Main_Window()
initUI()
open_XRD_Window()
open_Raman_Window()
open_XRF_Window()
openSubWindow()

------ XRD_Window ------
XRD_run()

------ Raman_Window ------
Raman_run()

------ XRF_Window ------
XRF_run()

------ SubWindow ------


/ / / / / * inspector * / / / / /

------ Tool ------
blue_colors
green_colors
neutral_colors
warm_colors
rainbow_colors
gradient_colors
contrast_colors
color_schemes

light_palette()
dark_palette()

------ Observer ------
print_background_color()
print_width_height()
print_markers()
print_encoding()
print_custom_color_palette()

------ Monitor ------


/ / / / / * accessor * / / / / /

------ Constructor ------
magic_access()
to_improve_precision()
examine_library()
generation_access()
for_report_tensile()

------ Article ------
Banpo_XRD()
Jiangzhai_XRD()

------ History ------
(The class properties of History are not shown here)


/ / / / / * author * / / / / /

------ Helper ------
count_words()
compare_text()
normalize_spaces()


/ / / / / * potter * / / / / /

------ PotteryBase ------
Pottery_Database
Standard_Template
Category_Index

apply_standard_template()
whether_data_exists()
add_site()
add_method()
update_info_of_site()
update_info_of_method()
save_data_to_txt()
save_data_to_json()
save_data_to_excel()
read_data()
read_data_from_json()
read_information()
add_temporary_data()
read_all_from_excel()
plot_figure()
plot_figure_from_json()
plot_pca()
open_figure()

------ Pottery ------
__display_main_folders()
__display_secondary_folders()
__display_thirdly_folders()
__decide_figure()
__plot_fingure_from_txt()
__plot_figure_from_json()
__plot_figure_from_excel()
__open_figure()
__user_input()
__read_information()
__wait_input()
run()


========================================================================================================================

"""
