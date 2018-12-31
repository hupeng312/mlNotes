import numpy as np

def r2_score(y_test, y_predict):
    y_mean = np.mean(y_test)
    return 1 - np.sum((y_predict - y_test) ** 2) / np.sum((y_test - y_mean) ** 2)