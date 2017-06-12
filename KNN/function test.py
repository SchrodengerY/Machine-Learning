# -------------函数的测试程序------------

# .min(0)和.min(1)的函数测试
from numpy import *
import operator
dataset = array([[0,1,2],[1,5,3],[-2,0,4]])
minValuecolumn = dataset.min(0)
minValuerow = dataset.min(1)
print("矩阵：\n",dataset)
print("列最小值：",minValuecolumn)
print("行最小值：",minValuerow)