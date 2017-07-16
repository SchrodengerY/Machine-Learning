from numpy import *
import numpy as np

# ===============参数训练================
def loadDataSet():
	import re
	# 定义数据集和标签集
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		# 去掉空格，并将每一行的数据提取出来
		# 写成了fr.readline()，只能读取一个字符，为此debug了很久。被自己蠢哭。
		lineArr = line.strip().split()
		# print(lineArr)
		# 下行代码需要详细解释，见备注1-A
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		# 使用正则表达式匹配字符串，此处代码有误
		'''tempList1 = []; tempList2 = []
		tempList1 = lineArr[0]
		tempList2 = lineArr[1]
		print(tempList1)
		print(tempList2)
		if(tempList1[0] == '-'):
			print(re.match(r'\d*',tempList1).group(0))
			# num1 = float(0.0 - float(re.match(r'\d*',lineArr[0]).group(0)))
		else:
			num1 = float(lineArr[0])
		if(tempList2[0] == '-'):
			print(re.match(r'\d*', tempList2).group(0))
			# num2 = float(0.0 - float(re.match(r'\d*',lineArr[1]).group(0)))
		else:
			num2 = float(lineArr[1])
		dataMat.append([1.0, num1, num2])'''
		# 提取标签信息
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	# 将结果值代入，计算sigmoid函数的值
	return 1.0/(1+exp(-inX))

# 梯度上升算法函数（重要指数：*****）
def gradAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	# 将标签矩阵转置(原：1xN的行向量；现：Nx1的列向量)
	labelMat = mat(classLabels).transpose()
	# 计算矩阵的行(m)和列(n)
	m, n = shape(dataMatrix)
	# 设定步长，alpha
	alpha = 0.001
	# 最大迭代次数为500
	maxCycles = 500
	# 初始化系数向量为元素为1的列向量
	weights = ones((n,1))
	for k in range(maxCycles):
		# h为nx1的列向量，用来存放分类得到的标签值
		h = sigmoid(dataMatrix*weights)
		# 正确标签和计算得到的标签之间的误差
		error = (labelMat - h)
		# 梯度上升，注意此处需要矩阵转置，是因为矩阵乘法需要行列对应。
		weights = weights + alpha * dataMatrix.transpose() * error
		# 以上三行公式的证明，见备注1-B

	return weights


# =============图像绘制============
def plotBestFit(weights):
	# 使用python3画图时需要添加以下两行语句，否则报错。
	import matplotlib
	matplotlib.use('Qt5Agg')

	import matplotlib.pyplot as plt

	dataMat, labelMat = loadDataSet()
	# 将dataMat转化为数组
	dataArr = array(dataMat)
	# 取数组的行数作为数据个数
	n = shape(dataArr)[0]
	# 标签为1的数据的x、y坐标和标签为0的数据的x、y坐标
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

	# 开始画图
	fig = plt.figure()
	# 111表示画一个大图
	ax = fig.add_subplot(111)
	# 将标签为1的点绘制为数量为30(s=30)，颜色为红色(c=red)，形状为正方形(marker='s')
	# 更多scatter用法：http://blog.csdn.net/u013634684/article/details/49646311
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green', marker = 'o')
	x = arange(-3.0,3.0,0.1)
	# 纵坐标的取值，具体解释见备注1-C
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()
# =========测试程序========
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
# 注：此处使用getA()是为了返回一个n维数组，因为在计算y时用到了weights[1]*x，如果weights[1]不是一个n维数组，而是一个数时，维度不匹配。
plotBestFit(weights.getA())


# =========算法升级=========
# 原算法在每次更新回归系数时需要遍历整个数据集，当样本量很大或者特征量很大时，计算复杂度过高！
# 改进方法是一次仅用一个样本点来更新回归系数，该方法称为随机梯度上升算法。
# 采用随机梯度上升算法对分类器进行增量式更新。
# 该方法是一个在线学习算法，与之对应的原梯度上升算法一次性处理全部数据，为批处理方式

def stocGradAscent0(dataMatrix, classLabels):
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	# 循环部分和原梯度上升算法不同：
	# 1. 循环次数不再是设定的迭代次数-maxCycles，而是dataMatrix矩阵的行数，即样本个数
	for i in range(m):
		# 2. 利用回归方程计算结果值时，所求得结果为【数值】而非【向量】！
		# 注意：假定dataMatrix[i]包含两个特征x,y（即n=2），则梯度上升算法计算得到【（x*w1，y*w2）】
		#	  而随机梯度上升算法计算得到【x*w1+y*w2】
		h = sigmoid(sum(dataMatrix[i]*weights))
		# 3. error计算也是数值，而非向量。理由同上。
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights

# =========测试算法==========
# 测试算法得到的图可以说明，该随机梯度上升算法分类准确性下降明显，分类器错了接近1/3的样本。
dataArr, labelMat = loadDataSet()
weights = stocGradAscent0(array(dataArr), labelMat)
plotBestFit(weights)


# =========算法再次升级=========
# 1. 改进后算法考虑了迭代次数的影响
# 2. 改进后的算法考虑了alpha值的调整，随迭代次数减小，降低了高次迭代时数据的波动
# 3. 改进后的算法在降低alpha的函数中，alpha每次减少1/(i+j)，其中j是迭代次数，i是样本点下标。
#    当j<<max(i)时，alpha就不是严格下降的。
# 其中第3条的优点需要补充：
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	m, n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i) + 0.01
			# 随机选取样本点
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights

# ============测试算法============
weights = stocGradAscent1(array(dataArr), labelMat, 200)
plotBestFit(weights)
