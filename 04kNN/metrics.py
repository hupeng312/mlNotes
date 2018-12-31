import numpy as np

def accuracy_score(y_test, y_predict):
    assert y_test.shape[0] == y_predict.shape[0], '测试集和预测结果数量不一致'
    return np.sum(y_test == y_predict) / len(y_predict)