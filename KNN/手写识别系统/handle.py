# ---------手写识别系统----------

from numpy import *
import operator
from os import listdir

# 构造分类器
def classify0(inX, dataSet, labels, k):
	# 读取矩阵第一维的长度，参考网站：http://www.cnblogs.com/zuizui1204/p/6511050.html
	dataSetSize = dataSet.shape[0]
	# tile:重复维度，进行扩展。参考网站：http://blog.csdn.net/wy250229163/article/details/52453201
	# diffMat：表示输入向量和训练样本集之间的距离
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	# 分别计算平方:a^2,b^2（d=sqrt(a^2+b^2)）
	sqDiffMat = diffMat**2
	# 将矩阵的每一行元素相加：a^2+b^2。若axis=0则为普通求和
	sqDistances = sqDiffMat.sum(axis = 1)
	# 计算距离开根号
	distances = sqDistances**0.5
	# 按照距离排序，并返回索引
	sortedDistIndices = distances.argsort()
	classCount = {}
	# 选择距离最小的k个点
	for i in range(k):
		# 获取label值
		voteIlabel = labels[sortedDistIndices[i]]
		# 计算label出现次数
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 按照label出现的多少排序，降序排列。注：python3中使用items替代iteritems
	sortedClassCount = sorted(classCount.items(),
							  key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

# -------图像格式化处理--------
# 将一个32x32的二进制图像矩阵转换成1x1024的向量，以便分类器处理
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

# 手写数字识别系统的测试代码
def handwrtingClassTest():
	hwLabels = []
	# 获取目录内容
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		# 获取文件名，如0_14.txt
		fileNameStr = trainingFileList[i]
		# 从文件名中提取.txt前的部分，即数字部分
		fileStr = fileNameStr.split('.')[0]
		# 提取出_前的部分，即0,1,2，...，8,9这些数字
		classNumStr = int(fileStr.split('_')[0])
		# 将提取出的数字添加进标签
		hwLabels.append(classNumStr)
		# 得到训练矩阵
		trainingMat[i,:] = img2vector("trainingDigits/%s" %(fileNameStr))
	# 开始测试，基本方法同上
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,
									 trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is: %d" %
			  (classifierResult, classNumStr))
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print("\nthe total number of errors is: %d" % errorCount)
	print("\nthe total error rate is: %f" %(errorCount/float(mTest)))

# 执行测试
handwrtingClassTest()

"""小结：改变变量k的值，修改函数handwritingClassTest随机选取的训练样本，改变训练样本数目，都会对kNN算法的错误率产生影响。
		算法执行效率低。因为算法需要对每个测试向量做2000次距离计算，每个距离计算包括1024个维度的浮点运算，总共执行900次。
		可以使用k决策树优化
		KNN算法的另一个缺点时无法给出任何数据的基础结构信息，因此无法知晓平均实例样本和典型实例样本具有什么特征。"""