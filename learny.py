"""
magiclib / learny

------------------------------------------------------------------------------------------------------------------------
magiclib/learny is a comprehensive Python machine learning library specifically designed for spectroscopic data
analysis, providing integrated workflows for data preprocessing (including SG filtering, SNV, and MSC transformations),
classification modeling with SVM and Random Forest algorithms, and comprehensive evaluation with confusion matrices,
ROC curves, and classification reports, all wrapped in an object-oriented architecture with automatic file management
and visualization capabilities.
------------------------------------------------------------------------------------------------------------------------

Xinyuan Su
"""

from . import general

import os
import joblib
import pickle
import inspect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# 模型
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
# 数据集拆分
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score


""" 数据的前处理 """
class Preprocessor:
    """
    数据的前处理类

    1. 数据接口需要用 self.data_dic
    """

    # 初始化
    def __init__(self, csv_path: str = None, data_dic: dict = None, save_path: str = None):
        """
        CSV 文件数据格式要求:
        1. 文件必须存在且为 .csv 格式。
        2. 第一列为 标签列，存放样本编号或类别。
        3. 第二列及以后为 光谱数据列，必须是数值型。
        4. CSV 至少包含 2 列：1 列标签 + 1 列光谱数据。
        5. 每一行表示 一个样本。
        6. 列名第一列为标签名称，后续列为光谱点或波长编号。

        self.data_dic =
        {
            'original': df_original,                  # 原始数据
            'sg_smooth': df_sg,                       # SG平滑
            'sg_smooth_1_derivative': df_sg1d,       # SG一阶导
            'sg_smooth_2_derivative': df_sg2d,       # SG二阶导
            'snv': df_snv,                            # SNV处理
            'msc': df_msc                             # MSC处理
        }

        :param csv_path: (str) 输入的 CSV 文件路径
        :param data_dic: (dict) 需要分析的数据，如果 csv_path 与 data_dic 均有值，则 data_dic 优先
        :param save_path: (str) 保存预处理结果的输出目录
        """

        # 检查 csv_path
        if csv_path is not None and not csv_path.lower().endswith(".csv"):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"input file is not a CSV: {csv_path}")

        if save_path is not None and not os.path.exists(save_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise FileNotFoundError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                    f"output directory does not exist: {save_path}")

        # 参数初始化
        self.csv_path = csv_path
        self.data_dic = data_dic
        self.save_path = save_path

        self.title = None
        self.data_df = None
        self.labels = None
        self.X = None
        self.feature_names = None

        # 只有在 self.data_dic 为 None 且 self.csv_path 有值的情况下会读取 CSV 文件
        if self.data_dic is None and self.csv_path:
            self.read_csv()

    # 读取 CSV 文件
    def read_csv(self):
        """
        私有方法，初始化时读取 CSV 文件
        """

        # 保存文件名（不带路径）
        self.title = os.path.splitext(os.path.basename(self.csv_path))[0]

        # 读取 CSV 文件
        self.data_df = pd.read_csv(self.csv_path)
        self.labels = self.data_df.iloc[:, 0]  # 第一列作为标签
        self.X = self.data_df.iloc[:, 1:].values  # 后续列作为光谱数据
        self.feature_names = self.data_df.columns[1:]  # 特征名称

        # 初始化 data_dic，原始数据 key 为 self.title， value 为 DataFrame
        self.data_dic = {self.title: self.data_df}

        return None

    # 保存至 Excel
    def save_to_excel(self, data_df: pd.DataFrame, suffix: str = ""):
        """
        保存 DataFrame 到 Excel 文件，避免文件覆盖

        :param data_df: 要保存的 DataFrame
        :param suffix: 保存文件后缀，如 "_SNV"
        """

        if self.save_path is None:
            return  # 不保存

        base_name = os.path.splitext(self.title)[0]  # 去掉 .csv
        file_name = base_name + suffix
        full_file_path = os.path.join(self.save_path, file_name + ".xlsx")

        count = 1
        while os.path.exists(full_file_path):
            full_file_path = os.path.join(self.save_path, f"{file_name}_{count}.xlsx")
            count += 1

        data_df.to_excel(full_file_path, index=False)
        print(f"File saved: \033[36m{full_file_path}\033[0m")

        return None

    # SG 及其导致分析
    def sg_smooth(self, window_length: int = 15, polyorder: int = 2, deriv: int = 0, target_key: str = None,
                  save_key: str = None):
        """
        Savitzky-Golay 滤波及导数

        :param window_length: (int) 滑动窗口的长度（必须为正奇数）。窗口大小越大，平滑效果越强，但可能导致细节丢失
        :param polyorder: (int) 拟合多项式的阶数。阶数越高，拟合曲线越灵活，但过高可能引入噪声。必须小于 window_length
        :param deriv: (int) 导数阶数。0 表示返回平滑后的信号；1 表示返回一阶导数；2 表示返回二阶导数，以此类推
        :param target_key: (str) self.data_dic 中的目标 key - value 的 key，默认为 None，表示用第 1 个 key - value
        :param save_key: (str) 保存至 self.data_dic 时的 key，默认为 'sg_smooth' 或 'sg_smooth_{deriv}_derivative'，
                         也会影响保存的文件后缀

        :return: result (dict) key = 方法名, value = 处理后的 DataFrame
        """

        if target_key is None:
            key = next(iter(self.data_dic))  # 取第一个 key
        else:
            key = target_key

        X = self.data_dic[key].iloc[:, 1:].values
        labels = self.data_dic[key].iloc[:, 0]
        feature_names = self.data_dic[key].columns[1:]

        X_sg = savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        df_sg = pd.DataFrame(X_sg, columns=feature_names)
        df_sg.insert(0, self.data_df.columns[0], labels)

        if not save_key:  # 生成 key
            key = "sg_smooth" if deriv == 0 else f"sg_smooth_{deriv}_derivative"
        else:
            key = save_key

        # 保存
        self.data_dic[key] = df_sg

        suffix = f"_{key}"
        self.save_to_excel(df_sg, suffix)

        return {key: df_sg}

    # SNV 分析
    def snv(self, target_key: str = None, save_key: str = None):
        """
        标准正态变量变换 (SNV)

        :param target_key: (str) self.data_dic 中的目标 key - value 的 key，默认为 None，表示用第 1 个 key - value
        :param save_key: (str) 保存至 self.data_dic 时的 key，默认为 'snv'，也会影响保存的文件后缀

        :return: result (dict) key = 方法名, value = 处理后的 DataFrame
        """

        if target_key is None:
            key = next(iter(self.data_dic))  # 取第一个 key
        else:
            key = target_key

        X = self.data_dic[key].iloc[:, 1:].values
        labels = self.data_dic[key].iloc[:, 0]
        feature_names = self.data_dic[key].columns[1:]

        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, ddof=1, keepdims=True)
        X_snv = (X - mean) / std
        df_snv = pd.DataFrame(X_snv, columns=feature_names)
        df_snv.insert(0, self.data_df.columns[0], labels)

        if not save_key:  # 生成 key
            key = "snv"
        else:
            key = save_key

        # 保存
        self.data_dic[key] = df_snv

        suffix = f"_{key}"
        self.save_to_excel(df_snv, suffix)

        return {key: df_snv}

    # MSC 分析
    def msc(self, target_key: str = None, save_key: str = None):
        """
        多元散射校正 (MSC)

        :param target_key: (str) self.data_dic 中的目标 key - value 的 key，默认为 None，表示用第 1 个 key - value
        :param save_key: (str) 保存至 self.data_dic 时的 key，默认为 'msc'，也会影响保存的文件后缀

        :return: result (dict) key = 方法名, value = 处理后的 DataFrame
        """

        if target_key is None:
            key = next(iter(self.data_dic))  # 取第一个 key
        else:
            key = target_key

        X = self.data_dic[key].iloc[:, 1:].values
        labels = self.data_dic[key].iloc[:, 0]
        feature_names = self.data_dic[key].columns[1:]

        ref = np.mean(X, axis=0)
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            fit = np.polyfit(ref, X[i, :], 1, full=True)
            slope, intercept = fit[0][0], fit[0][1]
            X_msc[i, :] = (X[i, :] - intercept) / slope

        df_msc = pd.DataFrame(X_msc, columns=feature_names)
        df_msc.insert(0, self.data_df.columns[0], labels)

        if not save_key:  # 生成 key
            key = 'msc'
        else:
            key = save_key

        # 保存
        self.data_dic[key] = df_msc

        suffix = f"_{key}"
        self.save_to_excel(df_msc, suffix)

        return {key: df_msc}

    # 进行所有分析
    def preprocess_all(self, target_key: str = None):
        """
        一次性运行所有预处理方法并保存结果，注意，后面的 key 会覆盖掉前面的
        在此方法中，key 只会以如下值进行保存:
        {
            'original': df_original,                  # 原始数据
            'sg_smooth': df_sg,                       # SG平滑
            'sg_smooth_1_derivative': df_sg1d,       # SG一阶导
            'sg_smooth_2_derivative': df_sg2d,       # SG二阶导
            'snv': df_snv,                            # SNV处理
            'msc': df_msc                             # MSC处理
        }

        :param target_key: (str) self.data_dic 中的目标 key - value 的 key，默认为 None，表示用第 1 个 key - value

        :return: result (dict) key=方法名, value=对应的 result dict
        """

        result = {}
        result.update(self.sg_smooth(window_length=15, polyorder=2, deriv=0, target_key=target_key))
        result.update(self.sg_smooth(window_length=15, polyorder=2, deriv=1, target_key=target_key))
        result.update(self.sg_smooth(window_length=15, polyorder=2, deriv=2, target_key=target_key))
        result.update(self.snv(target_key=target_key))
        result.update(self.msc(target_key=target_key))
        print("\033[35mAll preprocessing done.\033[0m")

        return result


""" 机器学习模型 """
class MLBase:
    """
    机器学习模型

    1.  SVM 不支持直接增量训练，扩充训练集必须重新训练。
    2.  如果只是预测新样本，直接加载原模型即可。
    3.  c 为分类模型，r 为回归模型

    A classification model is used when the output is a category or label. It predicts which group an input belongs
    to, such as “spam” or “not spam,” or “cat” versus “dog.” The results are discrete, meaning the model chooses
    one of several possible classes rather than giving a number.

    A regression model is used when the output is a continuous value. Instead of predicting categories, it estimates
    numerical outcomes like house prices, temperatures, or growth rates. The results are real numbers, which can
    take on a wide range of values.
    """

    sklearn_model_map = {
        "RandomForestClassifier": "RandomForest",
        "RandomForestRegressor": "RandomForest",
        "DecisionTreeClassifier": "DecisionTree",
        "DecisionTreeRegressor": "DecisionTree",
        "GradientBoostingClassifier": "GradientBoosting",
        "GradientBoostingRegressor": "GradientBoosting",
        "AdaBoostClassifier": "AdaBoost",
        "AdaBoostRegressor": "AdaBoost",
        "BaggingClassifier": "Bagging",
        "BaggingRegressor": "Bagging",
        "ExtraTreesClassifier": "ExtraTrees",
        "ExtraTreesRegressor": "ExtraTrees",
        "KNeighborsClassifier": "KNN",
        "KNeighborsRegressor": "KNN",
        "SVC": "SVM",
        "SVR": "SVM",
        "LinearSVC": "SVM",
        "LinearSVR": "SVM",
        "LogisticRegression": "LogisticRegression",
        "LinearRegression": "LinearRegression",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "ElasticNet",
        "MLPClassifier": "MLP",
        "MLPRegressor": "MLP",
        "GaussianNB": "NaiveBayes",
        "MultinomialNB": "NaiveBayes",
        "ComplementNB": "NaiveBayes",
        "BernoulliNB": "NaiveBayes",
        "QuadraticDiscriminantAnalysis": "QDA",
        "LinearDiscriminantAnalysis": "LDA",
        "VotingClassifier": "Voting",
        "VotingRegressor": "Voting",
        "StackingClassifier": "Stacking",
        "StackingRegressor": "Stacking",
    }

    # 初始化
    def __init__(self):
        """
        专门用于存放机器学习的模型
        """

        if type(self) is MLBase:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise TypeError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                            f"the class MLBase cannot be instantiated.")

        # 当前模型
        self.current_model = None

        # Classifier 分类模型
        self.svm_c_model = None
        self.rf_c_model = None
        self.knn_c_model = None
        self.lr_c_model = None
        self.dt_c_model = None
        self.gb_c_model = None
        self.mlp_c_model = None

        # Regressor 回归模型
        self.svm_r_model = None
        self.rf_r_model = None
        self.knn_r_model = None
        self.lr_r_model = None
        self.dt_r_model = None
        self.gb_r_model = None
        self.mlp_r_model = None

        # 用到的参数
        self.random_state = None
        self.X_train = None
        self.y_train = None

    # SVM Classifier 建模
    def train_svm_c(self, C: int = 10, gamma: float = 0.001, kernel: str = 'poly', cv: int = 5):
        """
        训练 SVM 模型，并进行交叉验证

        :param C: (int) SVM C 参数
        :param gamma: (float) 核函数 gamma
        :param kernel: (str) 核函数类型，可选值：'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        :param cv: (int) 交叉验证折数
        """

        self.svm_c_model = SVC(C=C,
                               gamma=gamma,
                               kernel=kernel,
                               probability=True,
                               random_state=self.random_state)
        self.current_model = self.svm_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.svm_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)  # 如果没有映射，就返回原名

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: \033[94m{scores.mean():.4f} "
              f"\033[34m± {scores.std():.4f}\033[0m")

        self.svm_c_model.fit(self.X_train, self.y_train)

        return None

    # RF Classifier 建模
    def train_rf_c(self, n_estimators: int = 144, max_depth: int = 40, min_samples_split: int = 3,
                   min_samples_leaf: int = 1, max_features: str = 'sqrt', cv: int = 5):
        """
        训练 Random Forest 模型，并进行交叉验证

        :param n_estimators: (int) 树的数量
        :param max_depth: (int) 树的最大深度
        :param min_samples_split: (int) 内部节点再划分所需最小样本数
        :param min_samples_leaf: (int) 叶子节点最少样本数
        :param max_features: (str) 寻找最佳分割时考虑的特征数量
        :param cv: (int) 交叉验证折数
        """

        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

        self.rf_c_model = RandomForestClassifier(**best_params,
                                                 random_state=self.random_state,
                                                 n_jobs=-1)
        self.current_model = self.rf_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.rf_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)  # 如果没有映射，就返回原名

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: \033[94m{scores.mean():.4f} "
              f"\033[34m± {scores.std():.4f}\033[0m")

        self.rf_c_model.fit(self.X_train, self.y_train)

        return None

    # KNN Classifier 建模
    def train_knn_c(self, n_neighbors: int = 5, weights: str = 'uniform', algorithm: str = 'auto',
                    leaf_size: int = 30, p: int = 2, metric: str = 'minkowski', cv: int = 5):
        """
        训练 KNN 分类模型，并进行交叉验证

        :param n_neighbors: (int) 邻居个数
        :param weights: (str) 权重函数 {‘uniform’, ‘distance’}
        :param algorithm: (str) 最近邻搜索算法 {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
        :param leaf_size: (int) 叶子大小（kd_tree 或 ball_tree 用到）
        :param p: (int) 距离度量参数 (p=1 曼哈顿距离, p=2 欧式距离)
        :param metric: (str) 距离度量方法
        :param cv: (int) 交叉验证折数
        """

        self.knn_c_model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                weights=weights,
                                                algorithm=algorithm,
                                                leaf_size=leaf_size,
                                                p=p,
                                                metric=metric,
                                                n_jobs=-1)
        self.current_model = self.knn_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.knn_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.knn_c_model.fit(self.X_train, self.y_train)

        return None

    # Logistic Classifier 回归建模
    def train_lr_c(self, penalty: str = 'l2', C: float = 1.0, solver: str = 'lbfgs',
                   max_iter: int = 200, class_weight=None, cv: int = 5):
        """
        训练 Logistic 回归分类模型，并进行交叉验证

        :param penalty: (str) 正则化类型 {'l1','l2','elasticnet','none'}
        :param C: (float) 正则化强度的倒数
        :param solver: (str) 优化算法
        :param max_iter: (int) 最大迭代次数
        :param class_weight: (dict or 'balanced') 类别权重
        :param cv: (int) 交叉验证折数
        """

        self.lr_c_model = LogisticRegression(penalty=penalty,
                                             C=C,
                                             solver=solver,
                                             max_iter=max_iter,
                                             class_weight=class_weight,
                                             random_state=self.random_state,
                                             n_jobs=-1)
        self.current_model = self.lr_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.lr_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.lr_c_model.fit(self.X_train, self.y_train)

        return None

    # 决策树 Classifier 建模
    def train_dt_c(self, criterion: str = 'gini', splitter: str = 'best', max_depth: int = None,
                   min_samples_split: int = 2, min_samples_leaf: int = 1, max_features=None, cv: int = 5):
        """
        训练 Decision Tree 分类模型，并进行交叉验证

        :param criterion: (str) 划分质量指标 {‘gini’, ‘entropy’, ‘log_loss’}
        :param splitter: (str) 划分策略 {‘best’, ‘random’}
        :param max_depth: (int) 最大深度
        :param min_samples_split: (int) 内部节点再划分所需最小样本数
        :param min_samples_leaf: (int) 叶子节点最少样本数
        :param max_features: (str/int/float) 每次划分考虑的最大特征数
        :param cv: (int) 交叉验证折数
        """

        self.dt_c_model = DecisionTreeClassifier(criterion=criterion,
                                                 splitter=splitter,
                                                 max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf,
                                                 max_features=max_features,
                                                 random_state=self.random_state)
        self.current_model = self.dt_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.dt_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.dt_c_model.fit(self.X_train, self.y_train)

        return None

    # Gradient Boosting Classifier 分类建模
    def train_gb_c(self, n_estimators: int = 100, learning_rate: float = 0.1,
                   max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1,
                   subsample: float = 1.0, max_features=None, cv: int = 5):
        """
        训练 Gradient Boosting 分类模型，并进行交叉验证

        :param n_estimators: (int) 基学习器数量
        :param learning_rate: (float) 学习率
        :param max_depth: (int) 基学习器的最大深度
        :param min_samples_split: (int) 内部节点再划分所需最小样本数
        :param min_samples_leaf: (int) 叶子节点最少样本数
        :param subsample: (float) 每棵树使用的样本比例
        :param max_features: (str/int/float) 每次划分考虑的最大特征数
        :param cv: (int) 交叉验证折数
        """

        self.gb_c_model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                     learning_rate=learning_rate,
                                                     max_depth=max_depth,
                                                     min_samples_split=min_samples_split,
                                                     min_samples_leaf=min_samples_leaf,
                                                     subsample=subsample,
                                                     max_features=max_features,
                                                     random_state=self.random_state)
        self.current_model = self.gb_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.gb_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.gb_c_model.fit(self.X_train, self.y_train)

        return None

    # MLP Classifier 分类建模
    def train_mlp_c(self, hidden_layer_sizes=(100,), activation: str = 'relu',
                    solver: str = 'adam', alpha: float = 0.0001, batch_size: str = 'auto',
                    learning_rate: str = 'constant', learning_rate_init: float = 0.001,
                    max_iter: int = 200, cv: int = 5):
        """
        训练 MLP 神经网络分类模型，并进行交叉验证

        :param hidden_layer_sizes: (tuple) 隐藏层结构
        :param activation: (str) 激活函数 {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        :param solver: (str) 权重优化器 {‘lbfgs’, ‘sgd’, ‘adam’}
        :param alpha: (float) L2 正则化系数
        :param batch_size: (int/‘auto’) 每批次大小
        :param learning_rate: (str) 学习率调度 {‘constant’, ‘invscaling’, ‘adaptive’}
        :param learning_rate_init: (float) 初始学习率
        :param max_iter: (int) 最大迭代次数
        :param cv: (int) 交叉验证折数
        """

        self.mlp_c_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                         activation=activation,
                                         solver=solver,
                                         alpha=alpha,
                                         batch_size=batch_size,
                                         learning_rate=learning_rate,
                                         learning_rate_init=learning_rate_init,
                                         max_iter=max_iter,
                                         random_state=self.random_state)
        self.current_model = self.mlp_c_model

        cv_split = StratifiedKFold(n_splits=cv,
                                   shuffle=True,
                                   random_state=self.random_state)
        scores = cross_val_score(self.mlp_c_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='accuracy',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV accuracy: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.mlp_c_model.fit(self.X_train, self.y_train)

        return None

    # SVM Regressor 建模
    def train_svm_r(self, C: float = 10.0, epsilon: float = 0.1, gamma: float = 0.001, kernel: str = 'poly',
                    cv: int = 5):
        """
        训练 SVM 回归模型，并进行交叉验证

        :param C: (float) SVM C 参数
        :param epsilon: (float) epsilon-不敏感损失函数参数
        :param gamma: (float) 核函数 gamma
        :param kernel: (str) 核函数类型，可选值：'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        :param cv: (int) 交叉验证折数
        """

        self.svm_r_model = SVR(C=C,
                               epsilon=epsilon,
                               gamma=gamma,
                               kernel=kernel)
        self.current_model = self.svm_r_model

        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.svm_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: \033[94m{scores.mean():.4f} "
              f"\033[34m± {scores.std():.4f}\033[0m")

        self.svm_r_model.fit(self.X_train, self.y_train)

        return None

    # RF Regressor 建模
    def train_rf_r(self, n_estimators: int = 144, max_depth: int = 40, min_samples_split: int = 3,
                   min_samples_leaf: int = 1, max_features: str = 'sqrt', cv: int = 5):
        """
        训练 Random Forest 回归模型，并进行交叉验证

        :param n_estimators: (int) 树的数量
        :param max_depth: (int) 树的最大深度
        :param min_samples_split: (int) 内部节点再划分所需最小样本数
        :param min_samples_leaf: (int) 叶子节点最少样本数
        :param max_features: (str) 寻找最佳分割时考虑的特征数量
        :param cv: (int) 交叉验证折数
        """

        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

        self.rf_r_model = RandomForestRegressor(**best_params,
                                               random_state=self.random_state,
                                               n_jobs=-1)
        self.current_model = self.rf_r_model

        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.rf_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: \033[94m{scores.mean():.4f} "
              f"\033[34m± {scores.std():.4f}\033[0m")

        self.rf_r_model.fit(self.X_train, self.y_train)

        return None

    # KNN Regressor 建模
    def train_knn_r(self, n_neighbors: int = 5, weights: str = 'uniform', algorithm: str = 'auto',
                    leaf_size: int = 30, p: int = 2, metric: str = 'minkowski', cv: int = 5):
        """
        训练 KNN 回归模型，并进行交叉验证

        :param n_neighbors: (int) 邻居个数
        :param weights: (str) 权重函数 {‘uniform’, ‘distance’}
        :param algorithm: (str) 最近邻搜索算法 {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
        :param leaf_size: (int) 叶子大小（kd_tree 或 ball_tree 用到）
        :param p: (int) 距离度量参数 (p=1 曼哈顿距离, p=2 欧式距离)
        :param metric: (str) 距离度量方法
        :param cv: (int) 交叉验证折数
        """

        self.knn_r_model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                              weights=weights,
                                              algorithm=algorithm,
                                              leaf_size=leaf_size,
                                              p=p,
                                              metric=metric,
                                              n_jobs=-1)
        self.current_model = self.knn_r_model

        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.knn_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.knn_r_model.fit(self.X_train, self.y_train)

        return None

    # Linear Regression 建模
    def train_lr_r(self, fit_intercept: bool = True, normalize: bool = False, cv: int = 5):
        """
        训练 Linear 回归模型，并进行交叉验证

        :param fit_intercept: (bool) 是否计算截距
        :param normalize: (bool) 是否在回归之前对特征进行标准化
        :param cv: (int) 交叉验证折数
        """

        # 如果需要归一化，用 pipeline 连接 StandardScaler 和 LinearRegression
        if normalize:
            self.lr_r_model = LinearRegression(
                fit_intercept=fit_intercept,
                copy_X=True,
                n_jobs=None,
                positive=False
            )
        else:
            self.lr_r_model = LinearRegression(fit_intercept=fit_intercept,
                                               copy_X=True,
                                               n_jobs=None,
                                               positive=False)

        self.current_model = self.lr_r_model

        # KFold 交叉验证
        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.lr_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        # 拟合模型
        self.lr_r_model.fit(self.X_train, self.y_train)

        return None

    # 决策树 Regressor 建模
    def train_dt_r(self, criterion: str = 'squared_error', splitter: str = 'best', max_depth: int = None,
                   min_samples_split: int = 2, min_samples_leaf: int = 1, max_features=None, cv: int = 5):
        """
        训练 Decision Tree 回归模型，并进行交叉验证

        :param criterion: (str) 划分质量指标 {‘squared_error’, ‘friedman_mse’, ‘absolute_error’, ‘poisson’}
        :param splitter: (str) 划分策略 {‘best’, ‘random’}
        :param max_depth: (int) 最大深度
        :param min_samples_split: (int) 内部节点再划分所需最小样本数
        :param min_samples_leaf: (int) 叶子节点最少样本数
        :param max_features: (str/int/float) 每次划分考虑的最大特征数
        :param cv: (int) 交叉验证折数
        """

        self.dt_r_model = DecisionTreeRegressor(criterion=criterion,
                                                splitter=splitter,
                                                max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                max_features=max_features,
                                                random_state=self.random_state)
        self.current_model = self.dt_r_model

        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.dt_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.dt_r_model.fit(self.X_train, self.y_train)

        return None

    # Gradient Boosting Regressor 建模
    def train_gb_r(self, n_estimators: int = 100, learning_rate: float = 0.1,
                   max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1,
                   subsample: float = 1.0, max_features=None, cv: int = 5):
        """
        训练 Gradient Boosting 回归模型，并进行交叉验证

        :param n_estimators: (int) 基学习器数量
        :param learning_rate: (float) 学习率
        :param max_depth: (int) 基学习器的最大深度
        :param min_samples_split: (int) 内部节点再划分所需最小样本数
        :param min_samples_leaf: (int) 叶子节点最少样本数
        :param subsample: (float) 每棵树使用的样本比例
        :param max_features: (str/int/float) 每次划分考虑的最大特征数
        :param cv: (int) 交叉验证折数
        """

        self.gb_r_model = GradientBoostingRegressor(n_estimators=n_estimators,
                                                   learning_rate=learning_rate,
                                                   max_depth=max_depth,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf,
                                                   subsample=subsample,
                                                   max_features=max_features,
                                                   random_state=self.random_state)
        self.current_model = self.gb_r_model

        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.gb_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.gb_r_model.fit(self.X_train, self.y_train)

        return None

    # MLP Regressor 建模
    def train_mlp_r(self, hidden_layer_sizes=(100,), activation: str = 'relu',
                    solver: str = 'adam', alpha: float = 0.0001, batch_size: str = 'auto',
                    learning_rate: str = 'constant', learning_rate_init: float = 0.001,
                    max_iter: int = 200, cv: int = 5):
        """
        训练 MLP 神经网络回归模型，并进行交叉验证

        :param hidden_layer_sizes: (tuple) 隐藏层结构
        :param activation: (str) 激活函数 {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        :param solver: (str) 权重优化器 {‘lbfgs’, ‘sgd’, ‘adam’}
        :param alpha: (float) L2 正则化系数
        :param batch_size: (int/‘auto’) 每批次大小
        :param learning_rate: (str) 学习率调度 {‘constant’, ‘invscaling’, ‘adaptive’}
        :param learning_rate_init: (float) 初始学习率
        :param max_iter: (int) 最大迭代次数
        :param cv: (int) 交叉验证折数
        """

        self.mlp_r_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                       activation=activation,
                                       solver=solver,
                                       alpha=alpha,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       learning_rate_init=learning_rate_init,
                                       max_iter=max_iter,
                                       random_state=self.random_state)
        self.current_model = self.mlp_r_model

        cv_split = KFold(n_splits=cv,
                         shuffle=True,
                         random_state=self.random_state)
        scores = cross_val_score(self.mlp_r_model, self.X_train, self.y_train,
                                 cv=cv_split,
                                 scoring='r2',
                                 n_jobs=-1)

        model_sklearn_name = self.current_model.__class__.__name__
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)

        print(f"\033[31m{model_name}\033[0m \033[37;2m{cv}\033[0m-fold CV R2: "
              f"\033[94m{scores.mean():.4f} \033[34m± {scores.std():.4f}\033[0m")

        self.mlp_r_model.fit(self.X_train, self.y_train)

        return None


""" 机器学习模型操作 """
class MLOperate(general.Manager, Preprocessor, MLBase):
    """
    对 MLBase 中的模型进行操作

    进行步骤:
    从建模: 1. train, 2. predict, 3. evaluate_all (or plot_confusion / classification_report_save / plot_roc_curve)
    从保存的模型: 1. load_model, 2. predict, 3. evaluate_all
    """

    # 初始化
    def __init__(self, csv_path: str = None, data_dic: dict = None, target_key: str = None, save_path: str = None,
                 test_size: float = 0.2, val_size: float = 0.25, random_state: int = None):
        """
        初始化参数，读取数据，并划分训练、测试、盲测集

        在数据集划分中，test_size 和 val_size 用于不同层次的划分。首先，test_size 用来从整个数据集中划出一部分样本作为
        盲测集（blind test set）。盲测集是完全独立的，它在模型训练和参数调优阶段不会被使用，仅用于最终评估模型的性能。
        例如，如果数据集共有 100 个样本，test_size = 0.2 表示会划出 20 个样本作为盲测集，其余 80 个样本用于训练和测试集的进一步划分。

        在剩余的训练和测试数据中，val_size 决定了其中多少比例作为 测试集（test set），用于模型训练过程中的验证和调参，
        剩余的则作为训练集。例如，对于盲测集划分之后剩下的 80 个样本，如果 val_size=0.25，则会划出 25%（即 20 个样本）
        作为测试集用于验证模型效果，剩下 60 个样本用于实际训练。

        :param csv_path: (str) CSV 文件路径
        :param data_dic: (dict) 需要分析的数据，如果 csv_path 与 data_dic 均有值，则 data_dic 优先
        :param target_key: (str) 需要分析 data_dic 中的目标 key，仅在 data_dic 有值且长度不为 1 时用到
        :param save_path: (str) 输出文件夹
        :param test_size: (float) 盲测集占比
        :param val_size: (float) 测试集占比 (在训练 + 测试集划分中)
        :param random_state: (int) 随机种子，None 表示完全随机
        """

        # 超类初始化 (仅类 Preprocessor)
        Preprocessor.__init__(self, csv_path=csv_path, data_dic=data_dic, save_path=save_path)

        # 确保 csv_path 与 data_dic 至少有一个被输入
        if csv_path is None and data_dic is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"at least one parameter in csv_path or data_dic must be provided!")

        # 检查 csv_path
        if csv_path is not None and not csv_path.lower().endswith(".csv"):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"input file is not a CSV: {csv_path}")

        # 检查保存路径
        if save_path is not None and not os.path.exists(save_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise FileNotFoundError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                    f"output directory does not exist: {save_path}")

        # 参数初始化
        self.csv_path = csv_path
        self.data_dic = data_dic
        self.target_key = target_key
        self.save_path = save_path

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.ml_data_dic = {}  # 根据 self.data_dic & self.target_key or self.csv_path 得到

        # 如果指定了随机种子，设置 np.random.seed
        if random_state is not None:
            np.random.seed(random_state)

        # 模型
        self.current_model = None

        # 标签
        self.le = None
        self.y_encoded = None
        self.n_classes = None

        # X & y
        self.X = None
        self.X_train = None
        self.X_test = None
        self.X_blind = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.y_blind = None

        # 预测
        self.y_train_pred = None
        self.y_test_pred = None
        self.y_blind_pred = None

        # 处理 self.data_dic 得到 self.ml_data_dic
        self.init_data_dic()

    # 初始化 self.data 以及 self.X_train & self.y_train
    def init_data_dic(self, target_key: str = None):
        """
        初始化时处理 self.data_dic，优先处理 self.data_dic，没有时再处理 scv_path
        在指定需要分析的数据时，需要先调用该方法，再训练模型

        :param target_key: (str) 需要分析 data_dic 中的目标 key，仅在 data_dic 有值且长度不为 1 时用到
        """

        # 手动修改目标训练集时覆盖原参数
        if target_key is not None:
            self.target_key = target_key

        # data_dic 被输入的情况
        if self.data_dic:
            if len(self.data_dic) == 1:
                self.ml_data_dic = self.data_dic

            elif len(self.data_dic) != 1 and self.target_key:
                self.ml_data_dic = {self.target_key: self.data_dic[self.target_key]}

            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"the length of data_dic needs to be 1, currently {len(self.data_dic)}")

        else:
            self.read_csv()
            self.ml_data_dic = self.data_dic

        # 数据分配
        self.title = list(self.ml_data_dic.keys())[0]
        self.data_df = list(self.ml_data_dic.values())[0]
        self.X = list(self.ml_data_dic.values())[0].iloc[:, 1:].values
        self.y = list(self.ml_data_dic.values())[0].iloc[:, 0].values

        # 标签编码
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)
        self.n_classes = len(self.le.classes_)

        # 数据划分，X_temp 与 y_temp 为划分后剩余的数据
        X_temp, self.X_blind, y_temp, self.y_blind = train_test_split(
            self.X, self.y_encoded,
            test_size=self.test_size,
            stratify=self.y_encoded,
            random_state=self.random_state
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            stratify=y_temp,
            random_state=self.random_state
        )

        return None

    # 保存模型 (非必要)
    def save_model(self, model_name: str = None, le_name: str = "label_encoder.pkl"):
        """
        保存训练好的模型及 LabelEncoder

        模型文件（例如 SVM.pkl）保存了训练好的机器学习模型。这个文件中包含了模型学习到的参数和规律，
        使得在以后直接加载它，对新的数据进行预测，而无需重新训练模型。简单来说，它就像模型的大脑，掌握了如何根据输入数据做出判断。

        LabelEncoder 文件（例如 label_encoder.pkl 或 classes.npy）则保存了类别标签的编码信息。在训练模型时，
        类别数据（如 "cat"、"dog"）需要被转换成数字（如 [0,1]）才能让模型处理。LabelEncoder 的作用就是在数字和原始类别之间进行互相转换。
        在预测阶段，它可以把模型输出的数字结果还原成人类可读的类别名称，从而保证预测结果可理解。

        :param model_name: (str) 模型文件名
        :param le_name: (str) 标签编码器文件名
        """

        # 检查 model_name 赋值
        if model_name is None:
            model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
            model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)  # 如果没有映射，就返回原名
            model_name += ".pkl"  # 如果没有映射，就返回原名

        # 检查 self.current_model & self.save_path
        if self.current_model is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"no trained model to save!")
        if self.save_path is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"no output directory specified!")

        # 保存模型
        model_path = os.path.join(self.save_path, model_name)
        joblib.dump(self.current_model, model_path)
        print(f"The model saved to: \033[33m{model_path}\033[0m")

        # 保存 LabelEncoder
        le_path = os.path.join(self.save_path, le_name)
        joblib.dump(self.le, le_path)
        print(f"LabelEncoder saved to: \033[33m{le_path}\033[0m")

        return None

    # 读取模型 (非必要)
    def load_model(self, model_path: str):
        """
        加载已保存的模型和 LabelEncoder

        :param model_path: (str) 保存的 .pkl 模型文件路径
        """

        if not os.path.isfile(model_path):
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise FileNotFoundError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                    f"model file not found: {model_path}")

        # 加载模型
        self.current_model = joblib.load(model_path)
        print(f"Loaded model from \033[33m{model_path}\033[0m")

        # 同目录加载 label_encoder.pkl
        le_path = os.path.join(os.path.dirname(model_path), "label_encoder.pkl")
        if os.path.exists(le_path):
            le_loaded = joblib.load(le_path)
            if isinstance(le_loaded, LabelEncoder):
                self.le = le_loaded
                self.n_classes = len(self.le.classes_)
            else:
                # 如果是 numpy 数组，手动构建 LabelEncoder
                self.le = LabelEncoder()
                self.le.classes_ = le_loaded
                self.n_classes = len(self.le.classes_)
            print(f"Loaded LabelEncoder from \033[33m{le_path}\033[0m")
        else:
            print(f"No LabelEncoder found in \033[33m{os.path.dirname(le_path)}\033[0m")

        return None

    # 预测
    def predict(self):
        """
        对训练集、测试集和盲测集进行预测
        """

        if self.current_model is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"model not trained yet!")

        self.y_train_pred = self.current_model.predict(self.X_train)
        self.y_test_pred = self.current_model.predict(self.X_test)
        self.y_blind_pred = self.current_model.predict(self.X_blind)

        return None

    # 绘制混淆矩阵 (Classifier) (单独)
    def plot_confusion(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, dataset_name: str = None,
                       normalize: bool = False, unify_samples: bool = False, save_path: str = None, cmap="Blues"):
        """
        绘制混淆矩阵热力图，仅 Classifier 分类模型可用

        :param y_true: (np.ndarray) 样本的真实标签
        :param y_pred: (np.ndarray) 模型预测标签
        :param dataset_name: (str) 数据集名称，如 "Train"、"Test"、"Blind Test"
        :param normalize: (bool) 是否对混淆矩阵归一化，True 为每行归一化
        :param unify_samples: (bool) 是否统一每类样本数量，按最少样本数随机抽样
        :param save_path: (str) 保存图像路径，为 None 则查看 self.save_path 的情况
        :param cmap: (any) 颜色，默认为蓝色
        """

        # 如果未提供 y_true/y_pred，则自动使用类内保存的预测结果
        if y_true is None or y_pred is None:
            if dataset_name.lower() == "train":
                y_true = self.y_train
                y_pred = self.y_train_pred
            elif dataset_name.lower() == "test":
                y_true = self.y_test
                y_pred = self.y_test_pred
            elif dataset_name.lower() in ["blind", "blind test"]:
                y_true = self.y_blind
                y_pred = self.y_blind_pred
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"unknown dataset_name: {dataset_name}")

        cm = confusion_matrix(y_true, y_pred)

        # 样本统一
        if unify_samples:
            min_count = np.min(np.bincount(y_true))
            indices = []
            for cls in np.unique(y_true):
                idx_cls = np.where(y_true == cls)[0]
                idx_selected = np.random.choice(idx_cls, min_count, replace=False)
                indices.extend(idx_selected)
            cm = confusion_matrix(y_true[indices], y_pred[indices])

        # 归一化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6), dpi=200)
        ax = sns.heatmap(cm,
                         annot=True,
                         fmt=".2f" if normalize else "d",
                         xticklabels=self.le.classes_,
                         yticklabels=self.le.classes_,
                         cmap=cmap,
                         annot_kws={"fontdict": self.font_ticket})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)

        # 获取刻度标签对象并设置字体样式
        yticklabels = cbar.ax.get_yticklabels()
        for label in yticklabels:
            label.set_size(self.font_ticket['size'])
            label.set_weight(self.font_ticket['weight'])
            label.set_family(self.font_ticket['family'])

        # 设置标题和坐标轴字体
        plt.title(label=f'{dataset_name} Confusion Matrix', fontdict=self.font_title)
        plt.xlabel(xlabel='Predicted label', fontdict=self.font_title)
        plt.ylabel(ylabel='True label', fontdict=self.font_title)

        # 设置刻度标签的字体
        plt.xticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])
        plt.yticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])

        # 保存
        if save_path is None and self.save_path is None:
            # 没给路径，不保存
            plt.show()
            return None

        else:
            # 使用 save_path 或 self.save_path
            base_dir = save_path if save_path else self.save_path

            # 初始文件名
            file_name = f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png"
            file_path = os.path.join(base_dir, file_name)

            # 确保文件名唯一
            counter = 1
            while os.path.exists(file_path):
                file_name = f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix_{counter}.png"
                file_path = os.path.join(base_dir, file_name)
                counter += 1

            # 保存文件
            plt.savefig(file_path, dpi=600, bbox_inches='tight')
            plt.show()
            print(f"\033[35m{dataset_name}\033[0m Confusion Matrix saved to \033[36m{file_path}\033[0m")

            return None

    # 输出分类报告 (Classifier) (单独)
    def classification_report_save(self, y_true: np.ndarray = None, y_pred: np.ndarray = None,
                                   dataset_name: str = "Dataset", save_path: str = None):
        """
        输出并保存分类报告，仅 Classifier 分类模型可用

        :param y_true: (np.ndarray) 样本的真实标签，None 时使用训练/测试/盲测预测结果
        :param y_pred: (np.ndarray) 模型预测标签，None 时使用训练/测试/盲测预测结果
        :param dataset_name: (str) 数据集名称，如 "Train"、"Test"、"Blind Test"，用于文件命名和打印
        :param save_path: (str) 保存图像路径，为 None 则查看 self.save_path 的情况
        """
        # 根据 dataset_name 自动选择 y_true 和 y_pred
        if y_true is None or y_pred is None:
            if dataset_name.lower() == "train":
                y_true = self.y_train
                y_pred = self.y_train_pred
            elif dataset_name.lower() == "test":
                y_true = self.y_test
                y_pred = self.y_test_pred
            elif dataset_name.lower() in ["blind test", "blind"]:
                y_true = self.y_blind
                y_pred = self.y_blind_pred
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"no data available for dataset '{dataset_name}'")

        report = classification_report(y_true, y_pred, target_names=self.le.classes_)
        model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)  # 如果没有映射，就返回原名
        print(f"\033[31m{model_name}\033[0m \033[35m{dataset_name}\033[0m Classification Report:")
        print(report)

        # 保存分类报告
        if save_path is None and self.save_path is None:
            # 都没给路径，不保存
            return None

        else:
            # 使用 save_path 或 self.save_path
            base_dir = save_path if save_path else self.save_path

            # 初始文件名
            file_name = f"{dataset_name.lower().replace(' ', '_')}_classification_report.txt"
            report_path = os.path.join(base_dir, file_name)

            # 确保文件名唯一
            counter = 1
            while os.path.exists(report_path):
                file_name = f"{dataset_name.lower().replace(' ', '_')}_classification_report_{counter}.txt"
                report_path = os.path.join(base_dir, file_name)
                counter += 1

            # 保存文件
            with open(report_path, 'w') as f:
                f.write(report)

            print(f"\033[35m{dataset_name}\033[0m Classification Report saved to \033[36m{report_path}\033[0m")

            return None

    # 绘制多分类 ROC (Classifier) (单独)
    def plot_roc_curve(self, X_data: np.ndarray = None, y_data: np.ndarray = None, dataset_name: str = "Dataset",
                       save_path: str = None, background=None):
        """
        绘制多分类 ROC 曲线，并保存图像，仅 Classifier 分类模型可用

        :param X_data: (np.ndarray) 待预测样本特征矩阵，None 时根据 dataset_name 使用对应训练/测试/盲测集
        :param y_data: (np.ndarray) 待预测样本真实标签，None 时根据 dataset_name 使用对应训练/测试/盲测集
        :param dataset_name: (str) 数据集名称，用于图表标题和保存文件命名
        :param save_path: (str) 保存图像路径，为 None 则查看 self.save_path 的情况
        :param save_path: (str) 保存图像路径，为 None 则查看 self.save_path 的情况
        :param background: (any) 背景色，默认为白色
        """

        if self.current_model is None:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f"model not trained yet!")

        # 根据 dataset_name 自动选择 X_data 和 y_data
        if X_data is None or y_data is None:
            if dataset_name.lower() == "train":
                X_data = self.X_train
                y_data = self.y_train
            elif dataset_name.lower() == "test":
                X_data = self.X_test
                y_data = self.y_test
            elif dataset_name.lower() in ["blind test", "blind"]:
                X_data = self.X_blind
                y_data = self.y_blind
            else:
                class_name = self.__class__.__name__  # 获取类名
                method_name = inspect.currentframe().f_code.co_name  # 获取方法名
                raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                                 f"no data available for dataset '{dataset_name}'")

        y_binarized = label_binarize(y_data, classes=np.arange(self.n_classes))
        y_score = self.current_model.predict_proba(X_data)

        plt.figure(figsize=(8, 6), dpi=200)
        if self.n_classes == 2:
            fpr, tpr, _ = roc_curve(y_binarized[:, 0], y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label='{} vs {} (AUC = {:.2f})'.format(self.le.classes_[0], self.le.classes_[1], roc_auc))
        else:
            for i in range(self.n_classes):
                fpr, tpr, _ = roc_curve(y_binarized[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         lw=2,
                         label='{} (AUC = {:.2f})'.format(self.le.classes_[i], roc_auc))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # 设置标题和坐标轴字体
        plt.title(label=f'ROC Curve ({dataset_name})', fontdict=self.font_title)
        plt.xlabel(xlabel='False Positive Rate', fontdict=self.font_title)
        plt.ylabel(ylabel='True Positive Rate', fontdict=self.font_title)

        # 设置刻度标签的字体
        plt.xticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])
        plt.yticks(fontfamily=self.font_ticket['family'],
                   fontweight=self.font_ticket['weight'],
                   fontsize=self.font_ticket['size'])

        plt.legend(loc="lower right", prop=self.font_legend)

        background_color = sns.dark_palette(color=background, reverse=True, as_cmap=True)
        general.Function.change_imshow(background_color=background_color,
                                       background_transparency=0.15)

        # 保存 ROC 图
        if save_path is None and self.save_path is None:
            # 没有路径，直接显示图像
            plt.show()
            return None

        else:
            # 使用 save_path 或 self.save_path
            base_dir = save_path if save_path else self.save_path

            # 初始文件名
            file_name = f"roc_{dataset_name.lower().replace(' ', '_')}.png"
            roc_path = os.path.join(base_dir, file_name)

            # 确保文件名唯一
            counter = 1
            while os.path.exists(roc_path):
                file_name = f"roc_{dataset_name.lower().replace(' ', '_')}_{counter}.png"
                roc_path = os.path.join(base_dir, file_name)
                counter += 1

            # 保存并显示
            plt.savefig(roc_path, dpi=600, bbox_inches='tight')
            plt.show()

            print(f"\033[35m{dataset_name}\033[0m ROC curve saved to \033[36m{roc_path}\033[0m")

            return None

    # 批量绘制所有评估 (Classifier)
    def evaluate_all(self, color_list: list = None):
        """
        对训练、测试、盲测集绘制混淆矩阵、分类报告和 ROC 曲线。即调用了 plot_confusion(), classification_report_save()
        与 plot_roc_curve()，需要先 train(), predict()，仅 Classifier 分类模型可用

        如果输入了保存路径 save_path，那么会在目标目录下创建文件夹进行保存，文件夹名为 'SVM' or 'SVM_1', 'SVM_2'... 此类

        :param color_list: (list) 绘图的颜色集，需要长度为 3，颜色分别对应 "Train"，"Test"，"Blind Test"，需要十六进制颜色代码
        """

        model_sklearn_name = self.current_model.__class__.__name__  # 获取模型名
        model_name = self.sklearn_model_map.get(model_sklearn_name, model_sklearn_name)  # 如果没有映射，就返回原名

        # 创建文件夹
        full_path = None
        if self.save_path:
            full_path = os.path.join(self.save_path, model_name)
            counter = 1
            while os.path.exists(full_path):
                full_path = os.path.join(self.save_path, f"{model_name}_{counter}")
                counter += 1
            os.makedirs(full_path)

        # 数据集
        datasets = [
            ("Train", self.X_train, self.y_train, self.y_train_pred),
            ("Test", self.X_test, self.y_test, self.y_test_pred),
            ("Blind Test", self.X_blind, self.y_blind, self.y_blind_pred)
        ]

        if color_list is not None and len(color_list) != 3:
            class_name = self.__class__.__name__  # 获取类名
            method_name = inspect.currentframe().f_code.co_name  # 获取方法名
            raise ValueError(f"\033[95mIn {method_name} of {class_name}\033[0m, "
                             f" The length of color_list must be 3. Now the length is {len(color_list)}")
        # 给三个数据集指定淡色十六进制颜色
        if color_list is None:
            dataset_colors = {
                "Train": "#8fd9a8",  # 淡绿色
                "Test": "#a2c8f0",  # 淡蓝色
                "Blind Test": "#c49bd7"  # 紫色
            }
        else:
            dataset_colors = {
                "Train": color_list[0],
                "Test": color_list[1],
                "Blind Test": color_list[2]
            }

        # 遍历数据集 datasets 并保存
        for name, X_data, y_true, y_pred in datasets:

            print(f"\n---- \033[31m{model_name}\033[0m \033[95m{name}\033[0m Confusion Matrices ----")

            color = dataset_colors.get(name, "Blues")  # 默认蓝色
            cmap = sns.light_palette(color=color, reverse=False, as_cmap=True)

            self.plot_confusion(y_true=y_true,
                                y_pred=y_pred,
                                dataset_name=f"{name} Original Samples",
                                save_path=full_path,
                                cmap=cmap)
            self.plot_confusion(y_true=y_true,
                                y_pred=y_pred,
                                dataset_name=f"{name} Unified Samples",
                                unify_samples=True,
                                save_path=full_path,
                                cmap=cmap)
            self.plot_confusion(y_true=y_true,
                                y_pred=y_pred,
                                dataset_name=f"{name} Normalized",
                                normalize=True,
                                save_path=full_path,
                                cmap=cmap)

            # 保存分类报告
            self.classification_report_save(y_true=y_true,
                                            y_pred=y_pred,
                                            dataset_name=name,
                                            save_path=full_path)

            # 保存 ROC 图
            self.plot_roc_curve(X_data=X_data,
                                y_data=y_true,
                                dataset_name=name,
                                save_path=full_path,
                                background=color)

        return None
