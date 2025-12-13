
__version__ = '6.3.6'

__all__ = ['general', 'grapher', 'projector', 'performer', 'potter', 'author', 'generator', 'learny', 'inspector']


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

 --------------------------------------------------- general -----------------------------------------------------------

 -Function-        函数插件区： 完成小功能
 -Optimizer-       数据重组区 (Function)： 对数据进行处理和优化
 -Manager-         文件管理区 (Optimizer)： 对文件进行操作和绘图
 -Module-          模组控制区 (Optimizer)： 对 DataFrame 进行操作，在中心函数前后布置
 -Magic-           中心函数区 (Manager, Module)： 数据处理的中心部分

 --------------------------------------------------- grapher -----------------------------------------------------------

 -Statistics-      统计学类 (Manager)： 用于统计学分析和绘图
 -Plotter-         绘图类 (Manager)：用于绘制各种图像
 -Fitter-          拟合类 (Manager)：用于拟合数据，绘制拟合后的图像

 -------------------------------------------------- projector ----------------------------------------------------------

 -Keyword-         标准函数组 (ABC)： 对特定作业进行的特定函数组合的准备，为抽象类
 -Tensile-         拉伸曲线类 (Keyword)： 绘制拉伸曲线 (应力-应变)
 -Compression-     压缩曲线类 (Keyword)： 绘制压缩曲线 (应力-应变)
 -Torsion-         扭转曲线类 (Keyword)： 绘制扭转曲线 (应力-应变)
 -XRD-             XRD 类 (Keyword)： 绘制 XRD 曲线
 -Raman-           Raman 类 (Keyword)： 绘制 Raman 曲线
 -DIL-             热膨胀类 (Keyword)： 绘制热膨胀曲线
 -XPS-             XPS 类 (Keyword)： 绘制 XPS 曲线
 
 --------------------------------------------------- performer ---------------------------------------------------------
 
 -Main_Window-     主窗口 (QWidget)： 软件的主窗口
 -XRD_Window-      XRD 窗口 (QMainWindow)： 进行 XRD 计算的小窗口
 -Raman_Window-    Raman 窗口 (QMainWindow)： 进行 Raman 计算的小窗口
 -XRF_Window-      XRF 窗口 (QMainWindow)： 进行 XRF 计算的小窗口
 -SubWindow-       附加小窗口 (QMainWindow)： 附加的小窗口，可进行其它功能设置
 
 ----------------------------------------------------- potter ----------------------------------------------------------
 
 -PotteryBase-     基因库操作区：用于写入、读取和比较陶器数据
 -Pottery-         基因库读取区：用于简单，明快地查找和读取数据和浏览图片
 
 ----------------------------------------------------- author ----------------------------------------------------------
 
 -TextEditing-     文本编辑区：用于检查、修改和比较文本内容
 -Word-            Word 文档修改区：用于检查、修改和编辑 Word 文档中的内容
 -PDF-             PDF 操作区：用于对 PDF 文件进行拼接、抽取、转换及压缩的功能
 -ArticleFetcher-  数据采集区：获取文章、研究热点和前沿技术

 --------------------------------------------------- generator ---------------------------------------------------------
 
 -Tools-           AI 工具区：用于存放 AI 可以调用的工具
 -AI-              AI 参数公有区：用于管理公有参数与 OpenAI 实例化
 -Human-           用户交互区：用于单或多用户与 AI 交互
 -DeepSeek-        DeepSeek 区：利用 DeepSeek 进行对话与分析操作
 -Gemini-          Gemini 区：利用 Gemini 进行对话与分析操作，还可以处理图像、音频与视频数据
 -Jimeng_video-    即梦 AI 视频生成区：利用 即梦 AI 生成视频
 -Jimeng_image-    即梦 AI 图片生成区：利用 即梦 AI 生成图片
 -Assist-          AI 生产力区：AI 大模型协助用户进行生产力工作
 -Muse-            AI 灵感区：AI 大模型与用户休闲交互，娱乐
 -ChatBoat-        多 AI 对话区：多个 AI 大模型对话，可进行娱乐与生产力工作
 
 ---------------------------------------------------- learny -----------------------------------------------------------
 
 -Preprocessor-    数据预处理区： 用于对需要机器学习的数据进行预处理
 -MLBase-          机器学习模型区： 用于存放各种机器学习的模型
 -ML_C_Operate-    分类机器学习操作区： 用于分类模型的训练、预测、反馈机器学习的结果
 
 --------------------------------------------------- inspector ---------------------------------------------------------

 -Tool-            工具区： 用于直接调用的属性和方法
 -Observer-        观察区： 用于观测可用数据
 -Monitor-         检查区： 用于检查代码
 
========================================================================================================================

 --------------------------------------------------- general -----------------------------------------------------------

 -Function-        Function Plugin Area: Completes small functions
 -Optimizer-       Data Reorganization Area (Function): Processes and optimizes data
 -Manager-         File Management Area (Optimizer): Operates on files and plots
 -Module-          Module Control Area (Optimizer): Operates on DataFrame, arranges around the central function
 -Magic-           Central Function Area (Manager, Module): The central part of data processing

 --------------------------------------------------- grapher -----------------------------------------------------------

 -Statistics-      Statistics Class (Manager): For statistical analysis and plotting
 -Plotter-         Plotting Class (Manager): For drawing various images
 -Fitter-          Fitting Class (Manager): For fitting data and plotting the fitted images

 -------------------------------------------------- projector ----------------------------------------------------------

 -Keyword-         Standard Function Group (ABC): Preparation for specific function combinations for specific tasks
 -Tensile-         Tensile Curve Class (Keyword): Draws tensile curves (stress-strain)
 -Compression-     Compression Curve Class (Keyword): Draws compression curves (stress-strain)
 -Torsion-         Torsion Curve Class (Keyword): Draws torsion curves (stress-strain)
 -XRD-             XRD Class (Keyword): Draws XRD curves
 -Raman-           Raman Class (Keyword): Draws Raman curves
 -DIL-             Thermal Expansion Class (Keyword): Draws thermal expansion curves
 -XPS-             XPS Class (Keyword): Draws XPS curves

 --------------------------------------------------- performer ---------------------------------------------------------

 -Main_Window-     Main Window (QWidget): The main window of the software
 -XRD_Window-      XRD Window (QMainWindow): A small window for performing XRD calculations
 -Raman_Window-    Raman Window (QMainWindow): A small window for performing Raman calculations
 -XRF_Window-      XRF Window (QMainWindow): A small window for performing XRF calculations
 -SubWindow-       Additional Small Window (QMainWindow): An additional small window for setting other functions
 
 ----------------------------------------------------- potter ----------------------------------------------------------

 -PotteryBase-     Gene Bank operation area: used to write, read and compare pottery data
 -Pottery-         Gene Library Reading area: for easy, crisp finding and reading data and viewing images
 
 ----------------------------------------------------- author ----------------------------------------------------------

 -TextEditing-     Text Editing: Used to examine, modify, and compare text content
 -Word-            Word area: Used for checking, modifying and editing the content in Word documents
 -PDF-             PDF area：The function for splicing, extracting, converting and compressing PDF files
 -ArticleFetcher-  Data Acquisition Area: It is used to obtain articles, research hotspots and cutting-edge technologies
 
 --------------------------------------------------- generator ---------------------------------------------------------
  
 -Tools-           AI Tool area: It is used to store tools that AI can invoke
 -AI-              AI Parameter Public Zone: Used for managing public parameters and OpenAI instantiations
 -Human-           User Interaction Area: Used for single or multiple users to interact with AI
 -DeepSeek-        DeepSeek area: Utilize DeepSeek for dialogue and analysis operations
 -Gemini-          Gemini Zone: Utilize Gemini for dialogue and analysis operations, and also handle audio or video data
 -Jimeng_video-    Jimeng video AI Zone: Create videos with Jimeng AI
 -Jimeng_image-    Jimeng Image AI Zone: Create Image with Jimeng AI
 -Assist-          AI Productivity Zone: AI large models assist users in productivity work
 -Muse-            AI Inspiration Zone: AI large models interact with users in a casual and entertaining way
 -ChatBoat-        Multi-AI Dialogue Area: Multiple large AI models have conversations

 ---------------------------------------------------- learny -----------------------------------------------------------
 
 -Preprocessor-    Data Preprocessing area: It is used for preprocessing data that requires machine learning
 -MLBase-          Machine Learning Model Area: It is used to store various machine learning models
 -ML_C_Operate-    Classification Machine Learning Operation Area: Used for classification training, predicting, ect
 
 --------------------------------------------------- inspector ---------------------------------------------------------

 -Tool-            Utility Class: Attributes and methods used for direct calls
 -Observer-        Observation Class: For observing available data
 -Monitor-         Inspection Class: For checking code

========================================================================================================================
"""

# Log
"""
======================================================= Version ========================================================

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
--> 6.1.5       24-12-12 16:00 All class attributes are optimized 
                so that they can be changed normally when instantiated.
                Add a new method copy_files() to class Function.
--> 6.1.6       24-12-27 Add a new method fitting_multiple_functional_after_fetching() to class Fitting.
                Fixed a bug in method fitting_multiple_functional().
                Added a new class Word in the module Author and added the method format_word_document () to it.
--> 6.1.7       25-01-03 Some values are adjusted in general and projector.
                New methods have been added to accessor.
--> 6.1.8       25-01-15 17:14 Add a new method pca_loadings() to class Statistics.
                Optimize the class Statistics.
--> 6.1.9       25-02-24 15:31 Add a new method plot_pie() to class Plotter.
--> 6.1.10      25-03-04 11:51 Optimize the method plot_pie() of class Plotter.
--> 6.1.11      25-03-05 12:47 Add two new method remove_category() and add_category() to class Optimizer.
                Made improvements to the to_reality() method in the Manager class, adding retrieval for the category 
                self.Category_Index and adapting it to the data_init() initialization method in the Function class.
                Optimized some of the comments.
--> 6.2.0       25-03-07 13:17 Data can be directly updated when calling read_txt(), read_excel(), and read_json().
                Updated to_reality().
                Optimized the description and data update logic of data_init().
                Add two methods split_df_by_category() and merge_df_by_category() to class Function.
                Added # Notice in the __init__.py file, which prohibits the use of the ptitprince library due 
                to library conflicts.
                Replace the raincloud diagram with the violin diagram.
                Some of the content in class author and class potter has been modified, 
                but the functionality has not been modified.
                Move the accessor class out of the library and replace it in the Database.
                The order and layout of the code are optimized.
                Optimize the XPS, Mapping and other classes in the class projector.
-> 6.2.1        25-03-20 11:55 Adjusted the order of some of the code in the pca_analysis() method in class Statistics.
                Add a new method dendrogram_clustering() to class Statistics.
                Add a new method agglomerative_clustering() to class Statistics.
                Changed the self.color_palette 20 color scheme in class Manager to remove white.
-> 6.3.0 Beta   25-07-29 17:26 Add a new method get_files() & sort_file_list() to class Function.
                Add a new class PDF in module author, and add two method merge_files_to_pdf() & extract_pages() in it.
-> 6.3.0        25-08-07 20:49 In the principal component analysis of the Statistic class, the Explained variance ratio 
                calculation has been added, and the explanation of the printed content is clearer.
                A new module, generator, has been added. This module enables the integration of large AI models 
                for data analysis and dialogue.
                The naming of serial number '1. 'has been standardized.
-> 6.3.1 Alpha  25-08-09 17:51 The content in the generator module has been improved, and classes such as AI, Human, 
                DeepSeek, OtherAI, Assist, and Muse have been added. It is currently still in the development and 
                testing stage. There are still many functions that need to be adjusted.
                The layout format of the serial numbers has been changed. Now it is the same as the /t indentation.
-> 6.3.1 Beta   25-08-14 00:25 A major overhaul of the module generator has been made, replacing OpenAI 
                with tequests () requests. 
                Added the function that AI can invoke tools.
-> 6.3.1 Beta   25-08-19 19:39 A new class, Gemini, has been added.
-> 6.3.1 Beta   25-09-10 14:24 Fixed the bug where the tool tool failed to assign a value during assignment. 
                Add a new method show_messages() to class AI.
                Allow chat() to call the tool, but it will not be returned to the AI for secondary processing, 
                and the result of the tool call will be saved to messages in the form of system.
                The invocation of tools has been optimized, and now custom tools are supported.
                A new class ArticleFetcher has been added to the module author.
                Add a new method seek_doi() to class ArticleFetcher.
                Add a new method seek_doi() to class Tools.
-> 6.3.1 Beta   25-09-15 18:37 The modules author and generator have been optimized, but there are still the 
                following functions to be updated: using cosine to determine similarity, rerank function, 
                adding new embedding content, and adding functions to the class Assistant.
-> 6.3.1 Beta   25-09-18 15:14 Some models cannot use tools. The logic has been modified: when tools=[], 
                it will not be added to the model request_body.
                The return contents of chat() and continue_chat() in class AI & class Gemini have been modified, 
                and the parameter return_all_messages has been added to the chat() of both.
                Add a new method stream_yield_chat() to class AI.
-> 6.3.1        25-09-25 22:10 The descriptions of all modules have been modified.
                A new module, learny, and its classes, Preprocessor, MLBase, and ML_C_Operate, have been added.
                Add a new class Jemeng_vedio to model generator.
-> 6.3.2        25-09-26 22:40 The entire magiclib has been updated, modifying the logic of comments and some code, 
                as well as importing libraries.
                The chat() method in the Human class has been adjusted.
-> 6.3.3        25-10-10 15:40 In the module generator, a Chatbo-like feature has been added, which enables simultaneous 
                conversations with multiple ais.
                In the module learny, the manipulation of the model is divided into ML_C_Operate 
                and ML_R_Operate (not added yet).
-> 6.3.4        25-10-16 11:47 The Human class has been optimized, and the setup_environment_dict() method 
                in the Muse class has been improved. Now you can directly add a user within it.
                Add method replace_similar_color() to class Function.
                Update method read_pdf() of class PDF.
-> 6.3.5        25-12-09 11:47 The generator class has been updated.
                Add a new class Jimeng_image  to model generator.
-> 6.3.6        25-12-13 18:00 ChatBoat has been optimized. Now, more dialogue Settings such as show_response, 
                stream, show_reasoning, tools, etc. can be made.
                
       
 ------ Attention ------
For general:
1.  statements should be written in detail
2.  When adding a method in the classes Optimizer, Module, and Magic, it is necessary to include self.current_dic.
3.  Please place the package (magiclib) in the same directory as the database (Database).

For generator:
1.  statements should be abbreviated
2.  Pay attention to the usage traffic of API keys
3.  The character is added each time, and the price is calculated last
             
 ------ Needing ------ 
--> method format_word_document() cannot format the text correctlyin class Author.
--> Supplement the pottery content of the module.
--> Supplement the performer content of the module.
--> Supplement the inspector content of the module.


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

# Notice
"""
======================================================== Mptice ========================================================

 --1--  seaborn conflicts with ptitprince in version 0.13.2 or later, and seaborn has heatmap required methods, 
        so importing ptitprince is prohibited.

========================================================================================================================
"""

# Content
"""
======================================================== Content =======================================================


-----   *  *  *  *  *  Magic  Library  *  *  *  *  *   -----


/ / / / / * general * / / / / /

current_dir
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
get_files()
sort_file_list()
sort_data()
find_most_color_pixel()
replace_similar_color()
copy_files()

------ Optimizer ------
Magic_Database
Category_Index

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
remove_category()
add_category()
split_df_by_category()
merge_df_by_category()

------ Manager ------
Magic_Database
Category_Index
font_title
font_ticket
font_legend
font_mark
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
pca_loadings()
dendrogram_clustering()
agglomerative_clustering()

------ Plotter ------
Category_Index

plot_box()
plot_violin()
plot_heatmap()
plot_bars()
plot_2d_histogram()
plot_ridge()
plot_qq()
plot_distribution()
plot_jointdistribution()
peak_deconvolution()
plot_function()
plot_pie()

------ Fitter ------
polynomial_fitting()
univariate_fitting()
fitting_functional_after_fetching()
fitting_multiple_functional_after_fetching()
multivariable_fitting()
fitting_multiple_functional()
fitting_multiple_according_model()


/ / / / / * projector * / / / / /

plot()

------ Keyword ------
Magic_Database
Standard_Database
interval_time

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

------ DIL ------
read()
plot()
magic_plot()
save_json()
plot_derivative()

------ XPS ------
read()
plot()
magic_plot()
save_json()
__curve_fitting()
__add_noise()

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

------ Main_Window ------
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
save_data_to_excel()
save_data_to_json()
read_data()
read_data_from_json()
read_information()
add_temporary_data()
read_all_from_excel()
plot_figure()
plot_figure_from_json()
plot_pca()
# open_figure()

------ Pottery ------
__display_main_folders()
__display_secondary_folders()
__display_thirdly_folders()
__decide_figure()
__plot_fingure_from_txt()
__plot_figure_from_excel()
__plot_figure_from_json()
__open_figure()
__user_input()
__read_information()
__wait_input()
run()


/ / / / / * author * / / / / /

Style

------ TextEditing ------
count_words()
compare_text()
normalize_spaces()
replace_text()

------ Word ------
Style

format_word_document()

------ PDF ------
merge_files_to_pdf()
extract_pages()
read_pdf()
print_pdf()

------ ArticleFetcher ------
seek_doi()



/ / / / / * generator * / / / / /

current_time_zone_location
api_key_1
base_url_1
success_requests_per_minute
avaliable_model
DeepSeek_api_key
DeepSeek_base_url
DeepSeek_avaliable_model
Gemini_api_key_1
Gemini_base_url
Jimeng_api_key
Jimeng_api_secret
messages_save_path

set_api_config()

------ Tools ------
read_txt()
read_excel()
read_json()
plot_line()
plot_scatter()
seek_doi()
read_pdf()
print_pdf()
generate_image()
save_image()
text_to_speech()
speech_to_text()
embedding()
rerank()

------ AI ------
toolkit
status_code_messages
ai_thinking_parameters

chat()
continue_chat()
stream_yield_chat()
show_messages()
calculate_cost()
summarize_conversation()
reset_conversation()
get_api_call_rate()
save_messages_to_txt()
load_messages_from_txt()
list_historical_conversations()

------ Human ------
chat()

------ DeepSeek ------

------ Gemini ------
__gemini_client()
__get_files_from_kwargs()
__convert_openai_messages_to_gemini()
__convert_openai_tools_to_gemini()
chat()
continue_chat()

------ Jimeng_video ------
__generate_signature()
__send_signed_request()
submit_task()
__get_task_result()
__download_video()
__display_waiting()
__poll_task_result()
run_video_generation()

------ Jimeng_image ------
__detect_mode()
__generate_image_name()
__generate_signature()
__send_signed_request()
__images_to_base64_list()
submit_task()
__get_task_result()
__poll_task_result()
__save_base64_image()
run_image_generation()

------ Assist ------
revise_manuscript()

------ Muse ------
setup_environment()
setup_environment_models()
setup_environment_dict()

------ ChatBoat ------
__inint_chating()
turns_to_speak()
__convert_history_to_user_content()
__player_output_process()
run()


/ / / / / * learny * / / / / /

------ Preprocessor ------
read_csv()
save_to_excel()
sg_smooth()
snv()
msc()
preprocess_all()

------ MLBase ------
sklearn_model_names

train_svm_c()
train_rf_c()
train_knn_c()
train_lr_c()
train_dt_c()
train_gb_c()
train_mlp_c()
train_svm_r()
train_rf_r()
train_knn_r()
train_lr_r()
train_dt_r()
train_gb_r()
train_mlp_r()

------ ML_C_Operate ------
init_data_dic()
save_model()
load_model()
predict()
plot_confusion()
classification_report_save()
plot_roc_curve()
evaluate_all()

------ RandomForest ------


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
print_width_height()
print_markers()
print_encoding()
print_color_palette()
print_custom_color_palette()
print_parameter_about_math()

------ Monitor ------


========================================================================================================================
"""
