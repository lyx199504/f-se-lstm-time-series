# f-se-lstm-time-series
本项目是论文《F-SE-LSTM: A Time Series Anomaly Detection Method Based on Frequency Domain Information》的实验代码，实现了多种时间序列异常检测模型，并构建一个异常检测方法。<br>
This project is the experimental code of the paper "*F-SE-LSTM: A Time Series Anomaly Detection Method Based on Frequency Domain Information*", which implements a variety of time series anomaly detection models and create an anomaly detection method.

## 目录 Table of Contents

- [项目目录 Project Directory](#项目目录-project-directory)
- [使用方法 Getting Started](#使用方法-getting-started)
- [项目声明 Project Statement](#项目声明-project-statement)

<h2 id="project">项目目录 Project Directory</h2>

├─ datasets (数据集目录 Dataset directory)<br>
&emsp;├─ Numenta Anomaly Benchmark (NAB数据集目录 NAB dataset directory)<br>
&emsp;├─ Yahoo! Webscope S5 (雅虎数据集目录 Yahoo dataset directory)<br>
├─ dl_models (模型目录 Model directory) <br>
&emsp;├─ \_\_init\_\_.py (异常检测模型父类，包含挑选验证模型 Anomaly detection model parent class, including selection validation model)<br>
&emsp;├─ c_lstm.py (C-LSTM方法 C-LSTM method)<br>
&emsp;├─ c_lstm_ae.py (C-LSTM-AE方法 C-LSTM-AE method)<br>
&emsp;├─ dnn.py (DNN相关模型 Various models of DNN)<br>
&emsp;├─ cnn.py (CNN相关模型 Various models of CNN)<br>
&emsp;├─ lstm.py (LSTM相关模型 Various models of LSTM)<br>
&emsp;├─ lstm_dnn.py (LSTM+DNN模型 LSTM+DNN model)<br>
&emsp;├─ cnn_lstm_dnn.py (CNN+LSTM+DNN模型 CNN+LSTM+DNN model)<br>
&emsp;├─ se_lstm_dnn.py (提出方法的SENet+LSTM+DNN模型 SENet+LSTM+DNN model of the proposed method)<br>
├─ dataPreprocessing.py (数据预处理 Data preprocessing)<br>
├─ train_ablation.py (消融实验训练代码 Ablation experiment training code)<br>
├─ train_dl.py (深度学习验证频率矩阵的训练代码 Training code for validating frequency matrices using deep learning)<br>
├─ train_ml.py (机器学习验证频率特征的代码 Training code to validate frequency feature using machine learning)<br>
├─ train_paper.py (与其他论文方法的对比实验代码 Training code to compare experiments with other paper methods)<br>
├─ train_win.py (不同滑动窗口大小的训练代码 Training code for different sliding window sizes)<br>
├─ requirements.txt (项目依赖 Project dependencies)<br>

> 以上列出了模型文件及主要的训练代码文件，其余未列出的文件均为项目基础文件，无需重点关注。<br>
> The model files and main training code files are listed above, and the rest of the unlisted files are the basic files of the project and do not need to be paid attention to.<br>
> 本项目使用的数据集是网上公开的数据集，并非私有。因此，为了维护数据集的版权，我们并未将数据集一并上传。数据集的原链接如下：<br>
> The datasets used in this project are publicly available online, not private. Therefore, in order to maintain the copyright of the dataset, we did not upload the dataset together. The original link to the dataset is as follows:<br>
> NAB: https://github.com/numenta/NAB<br>
> Yahoo: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70

<h2 id="get-start">使用方法 Getting Started</h2>

首先，拉取本项目到本地。<br>
First, pull the project to the local.

    $ git clone git@github.com:lyx199504/f-se-lstm-time-series.git

接着，进入到项目中并安装本项目的依赖。但要注意，pytorch可能需要采取其他方式安装，安装完毕pytorch后可直接用如下代码安装其他依赖。<br>
Next, enter the project and install the dependencies of the project. However, it should be noted that pytorch may need to be installed in other ways. After installing pytorch, you can directly install other dependencies with the following code.

    $ cd f-se-lstm-time-series/
    $ pip install -r requirements.txt

然后，分别将NAB和雅虎数据集下载到项目的NAB数据集目录和雅虎数据集目录中。<br>
Then, download the NAB and Yahoo datasets to the project's NAB dataset directory and Yahoo dataset directory, respectively.

最后，执行train\_\*.py即可训练模型。<br>
Finally, execute train\_\*.py to train the model.

<h2 id="statement">项目声明 Project Statement</h2>

本项目的作者及单位：<br>
The author and affiliation of this project:

    项目名称（Project Name）：f-se-lstm-time-series
    项目作者（Author）：Yixiang Lu, Ziquan Huang, Mingjin He, Rui Guo
    作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security, Jinan University）

本实验代码基于param-opt训练工具，原项目作者及出处如下：<br>
The experimental code is based on the param-opt training tool. The author and source of the original project are as follows:<br>
**Author: Yixiang Lu**<br>
**Project: [param-opt](https://github.com/lyx199504/param-opt)**

若要引用本论文，可按照如下latex引用格式：<br>
If you want to cite this paper, you could use the following latex citation format:

    To be determined...

