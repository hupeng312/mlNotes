import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:

    def __init__(self, k):
        '''k 为knn的个数'''
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X, y):
        '''X 为样本 因为分类结果'''
        assert X.shape[0] == y.shape[0], '样本和标签数量不一致'
        assert self.k <= X.shape[0], '样本数量级必须大于指定的k'
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X_predict):
        '''x 为预测结果'''
        assert self._X_train is not None and self._y_train is not None, 'must fit before predict!'
        assert X_predict.shape[1] == self._X_train.shape[1], '预测数据特征数必须等于训练数据特征数'

        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        '''实际预测算法'''
        # 欧拉距离
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearSet = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearSet[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        '''根基测试数据集X_test 和 y_test确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return np.sum(y_predict == y_test) / len(y_test)

    def __repr__(self):
        return "KNN(k=%d)" % self.k