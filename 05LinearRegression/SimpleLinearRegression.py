import numpy as np

class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, 'Simple Linear Regresor can only solve single feature traning data.'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x_i,y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x):
        assert x.ndim == 1, 'this model can onle predict the data which ndim == 1'
        assert self.a_ is not None and self.b_ is not None, 'must fit before predict'
        return np.array([self._predict(x_single) for x_single in x])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, 'Simple Linear Regresor can only solve single feature traning data.'
        assert len(x_train) == len(y_train), 'the size of x_train must be equal to the size of y_train'
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x):
        assert x.ndim == 1, 'this model can onle predict the data which ndim == 1'
        assert self.a_ is not None and self.b_ is not None, 'must fit before predict'
        return np.array([self._predict(x_single) for x_single in x])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"