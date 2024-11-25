0.版本
python==3.8.19
pgmpy==2.5.2
networkx==3.1
---------步骤如下------------
---------run Part1   SIND数据集交互片段提取----------
见文件夹《SinD无保护左转提取代码》
---------run Part2   交互片段数据格式转换----------
见《ReforcementLearningEHMI/handle_data/dataStructChange》

前两部分结束后可以得到./data中的数据

---------run Part3   DBN模型----------
该部分包括载入训练数据、模型构建和训练等
主函数为intentionUncertainty.py
模型构建见dynamicBayesianNetworkModel.py
载入训练数据见data_preprocess.py
全局可调参数见config.py


