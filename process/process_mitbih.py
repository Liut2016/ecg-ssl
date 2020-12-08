import wfdb
import pywt
import numpy as np
import scipy.io

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 读取心电数据记录
    print("正在读取 " + str(number) + " 号心电数据...")
    record = wfdb.rdrecord('../data/MIT-BIH/' + str(number), channel_names=['MLII'])
    if record.p_signal is None:
        return [], []
    data = record.p_signal.flatten()
    # 小波去噪
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('../data/MIT-BIH/' + str(number), 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    X_data = []
    Y_data = []
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return X_data, Y_data

def loadData():
    dataSet = []
    labelSet = []
    # 测试集比例
    RATIO = 0.3

    # 102和104缺少MLII, 故去除
    for number in [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 208, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]:
        x, y = getDataSet(number)
        dataSet.extend(x)
        labelSet.extend(y)


    dataSet = np.array(dataSet).reshape(-1, 300)
    labelSet = np.array(labelSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, labelSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))  # RATIO = 0.3
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test


def saveData():
    X_train, Y_train, X_test, Y_test = loadData()
    scipy.io.savemat('../data/MIT-BIH/processedData.mat', {'X_train':X_train,
                                                           'Y_train':Y_train,
                                                           'X_test':X_test,
                                                           'Y_test':Y_test})
    print('processedData saved success')

saveData()



#record = wfdb.rdrecord('../data/MIT-BIH/' + str(100), channel_names=['MLII'])
#annotation = wfdb.rdann('../data/MIT-BIH/' + str(100), 'atr')
#print(annotation.symbol)
#wfdb.plot_wfdb(record=record, title='100')
#print(record.__dict__)
#wfdb.io.get_record_list('../data/MIT-BIH')