#Numpy函数库基础

from numpy import *
# 产生一个随机数组
print(random.rand(5,5))
# 产生一个随机矩阵
print(mat(random.rand(4,4)))
# 求矩阵的逆
invRandMat = mat(random.rand(4,4)).I
# 创建单位矩阵
print(eye(4))

