# 特征提取demo
from base import wrap


@wrap
def dataset_demo():
    """
    数据集> sklearn 数据集初体验
    :return:
    """
    from sklearn.datasets import load_iris
    bunch = load_iris()
    print('iris数据集: \n', bunch)
    print(bunch['DESCR'])  # 既可以用字典方法取值
    print('=================================================')
    print(bunch.data.shape)  # 也可以用点取值


@wrap
def feature_engine_demo0():
    """
    特征工程> 字典数据提取
    :return:
    """
    from sklearn.feature_extraction import DictVectorizer
    data = [{'书名': '天龙八部', '角色': '乔峰'},
            {'书名': '笑傲江湖', '角色': '林胡从'},
            {'书名': '笑傲江湖', '角色': '人阴影'},
            {'书名': '倚天屠龙记', '角色': '张三丰'}]
    dv = DictVectorizer()
    data_trans = dv.fit_transform(data)
    print("特性名字:\n", dv.get_feature_names())
    print("提取结果-sparse矩阵形式:\n", data_trans, type(data_trans))
    print("提取结果-二维数组形式:\n", data_trans.toarray(), type(data_trans.toarray()))


@wrap
def feature_engine_demo1():
    """
    特征工程> 文本数据提取-英文文本
    :return:
    """
    from sklearn.feature_extraction.text import CountVectorizer
    data = ['this is python', 'this is holiday', 'i want to learn machine learn']
    cv = CountVectorizer()
    data_trans = cv.fit_transform(data)
    print("特性名字:\n", cv.get_feature_names())
    print("提取结果-sparse矩阵形式:\n", data_trans, type(data_trans))
    print("提取结果-二维数组形式:\n", data_trans.toarray(), type(data_trans.toarray()))


@wrap
def feature_engine_demo2():
    """
    特征工程> 文本数据提取-中文文本
    :return:
    """
    import jieba
    from sklearn.feature_extraction.text import CountVectorizer
    def cut_word(text):
        return ' '.join(list(jieba.cut(text)))

    data = ['这世界有那么多人', '多幸运我有个我们', '我迷茫的眼睛里长存', '初见你蓝色清晨', '我们好幸运']

    # 分词
    word_list = [cut_word(item) for item in data]

    cv = CountVectorizer()
    data_trans = cv.fit_transform(word_list)
    print("特性名字:\n", cv.get_feature_names())
    print("提取结果-sparse矩阵形式:\n", data_trans, type(data_trans))
    print("提取结果-二维数组形式:\n", data_trans.toarray(), type(data_trans.toarray()))


@wrap
def feature_engine_demo3():
    """
    特征工程> 文本数据提取-tfidf
    :return:
    """
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    def cut_word(text):
        return ' '.join(list(jieba.cut(text)))

    data = ['这世界有那么多人', '多幸运我有个我们', '我迷茫的眼睛里长存', '初见你蓝色清晨', '我们好幸运']

    # 分词
    word_list = [cut_word(item) for item in data]

    cv = TfidfVectorizer()
    data_trans = cv.fit_transform(word_list)
    print("特性名字:\n", cv.get_feature_names())
    print("提取结果-sparse矩阵形式:\n", data_trans, type(data_trans))
    print("提取结果-二维数组形式:\n", data_trans.toarray(), type(data_trans.toarray()))


if __name__ == '__main__':
    dataset_demo()
    feature_engine_demo0()
    feature_engine_demo1()
    feature_engine_demo2()
    feature_engine_demo3()
