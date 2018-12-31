import numpy as np

def train_test_split(X,y,test_ratio=0.2,seed=None):
    assert X.shape[0] == y.shape[0], '样本和标签个数不一致'
    assert 0<=test_ratio<1, '无效的测试比例'
    if seed:
        np.random.seed(seed)
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    return X[train_index], X[test_index], y[train_index], y[test_index]