import numpy as np

# 实现方法1
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

# 是想方法2
def softmax2(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    x = np.exp(x)

    if x.ndim == 1:
        sumcol = sum(x)
        for i in range(x.size):
            x[i] = x[i]/float(sumcol)
    if x.ndim > 1:
        sumcol = x.sum(axis = 0)
        for row in x:
            for i in range(row.size):
                row[i] = row[i]/float(sumcol[i])
    return x


data1 = [1]
data2 = [0]
data3 = [1,2,3,4]
data4 = [[1,5,7,3],[2,3,4,5]]
data5 = [[[1,2],[5,3]],[[9,18],[4,22]]]

print(softmax(data1) ,"\n")
print(softmax(data2) ,"\n")
print(softmax(data3) ,"\n")
print(softmax(data4) ,"\n")
print(softmax(data5) ,"\n")

print(softmax2(data1) ,"\n")
print(softmax2(data2) ,"\n")
print(softmax2(data3) ,"\n")
print(softmax2(data4) ,"\n")
#print(softmax2(data5) ,"\n")



import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 200)
y = softmax(x)
print(x,y)
plt.plot(x,y)
plt.show()