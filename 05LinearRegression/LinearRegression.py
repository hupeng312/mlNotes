import numpy as np

class LinearRegression:

    def __init__(self):
        '''初始化LinearRegression模型'''
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        '''训练模型'''
        assert X_train.shape[0] == y_train.shape[0], 'the size of X_train must equal y_train'
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None, "must fit before predict"
        assert X_predict.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        '''根据测试数据集X_test 和 y_test 确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        y_mean = np.mean(y_test)
        return 1 - np.sum((y_predict - y_test) ** 2) / np.sum((y_test - y_mean) ** 2)

    def __repr__(self):
        return "LinearRegression()"