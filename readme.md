##特征工程    
  ###1、特征提取
    字典数据提取：
       使用sklearn.feature_extraction的DictVectorizer,实例化一个对象，然后将数据(迭代器)传入进行提取。
       代码：feature_engine_demo0    --在feature_extract_demo.py中
    文本数据提取：（有根据单词、句子、字母提取等方法。下边的方法都是根据单词提取）
      方法一: 使用CountVectorizer
        统计每个词在文档中出现的次数
        使用sklearn.feature_extraction.text的CountVectorizer,实例化一个对象，然后将数据(迭代器)传入进行提取。
        如果是中文文本，需要进行分词。借助一些库(比如jieba)
        代码：feature_engine_demo1, feature_engine_demo2   --在feature_extract_demo.py中
      方法二：使用TfidfVectorizer
        统计每个词在文档中出现的频率
        使用sklearn.feature_extraction.使用TfidfVectorizer,实例化一个对象，然后将数据(迭代器)传入进行提取。
        如果是中文文本，需要进行分词。借助一些库(比如jieba)
        代码：feature_engine_demo3    --在feature_extract_demo.py中
    图片数据提取：
  ###2、特征预处理
    通过一些转换函数将特征数据转换成更加适合算法模型的特征数据的过程。
    包含内容：
      数值型数据的无量纲化（归一化和标准化）。
    归一化：
      通过对原始数据进行变换把数据映射到(默认[0,1])之间。
      计算公式：
        X' = (x-min)/(max-min), X''=X'*(mx-mi)+mi
        max和min为列的最大值和最小值，mx和mi为指定区间的值(默认mx为1，mi为0)。X''为最终结果。
      方法：
        使用sklearn.preprocessing的MinMaxScaler,实例化一个对象，然后将数据(ndarray类型)传入进行提取。
        代码：demo1    --在feature_prepeocess_demo.py中
      缺点：
        受异常值影响较大(如数据缺失)
    标准化：
      通过对原始数据进行变换把数据变换到均值为0，标准差为1的范围中。
      计算公式：
        X‘ = (x-mean)/std
      方法：
        使用sklearn.preprocessing的StandardScaler,实例化一个对象，然后将数据(ndarray类型)传入进行提取。
        代码：demo2    --在feature_prepeocess_demo.py中