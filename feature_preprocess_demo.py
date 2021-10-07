# 特征预处理demo
from base import wrap
import pandas as pd

@wrap
def demo1():
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    data = [[180, 80], [175, 75], [166, 55], [173, 60]]  # 身高与体重
    data = pd.DataFrame(data, columns=['height', 'weight'])
    print('原始数据:\n', data)
    mm = MinMaxScaler()  # 默认归一化到[0,1]
    data_trans = mm.fit_transform(data)
    print('归一化后的数据\n', data_trans)


@wrap
def demo2():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    data = [[180, 80], [175, 75], [166, 55], [173, 60]]  # 身高与体重
    data = pd.DataFrame(data, columns=['height', 'weight'])
    print('原始数据:\n', data)
    mm = StandardScaler()  # 默认归一化到[0,1]
    data_trans = mm.fit_transform(data)
    print('归一化后的数据\n', data_trans)


if __name__ == '__main__':
    demo1()
    demo2()
