# 准备数据
def loadDataSet():
	# 创建词汇表
	postingList = [['my', 'dog', 'has', 'flea',
					'problems', 'help', 'please'],
				   ['maybe', 'not', 'take', 'him',
					'to', 'dog', 'park', 'stupid'],
				   ['my', 'dalmation', 'is', 'so', 'cute',
					'I', 'love', 'him'],
				   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
					'to', 'stop', 'him'],
				   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	# 是否含有侮辱性词汇: 0-未含有 1-含有
	classVec = [0, 1, 0, 1, 0, 1]
	return postingList, classVec

# 构造不含有重复词汇的集合vocabSet
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		# 取并集
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

# 输入参数为：词汇表，输入文档
def setOfWords2Vec(vocabList, inputSet):
	# 输出文档向量，表示词汇表中的单词是否出现：1-出现 0-未出现
	returnVec = [0] * len(vocabList)
	# 对于输入文档中的每个词汇
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print("the word: %s is not in my Vocabulary!" % word)
	return returnVec

# ============测试文档===========
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print("VocabSet:", myVocabList, "\n")
print(setOfWords2Vec(myVocabList, listOPosts[0]))


# 训练算法
from numpy import *

# 二类分类问题
# 该函数未考虑相乘概率值为0（即p0Num, p1Num, p0Denom, p1Denom的初始化）和下溢出(p1Vect和p0Vect)的问题
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)

	'''p0Num = zeros(numWords)
	p1Num = zeros(numWords)
	p0Denom = 0.0
	p1Denom = 0.0'''

	# 以上四行代码修改为：
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0


	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	'''p1Vect = p1Num/p1Denom
	p0Vect = p0Num/p0Denom'''

	# 以上两行代码修改为：
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)

	return p0Vect, p1Vect, pAbusive

# 多类分类问题（以3类为例，其在trainCategory中的分类为0,1,2）
def trainNB4(trainMatrix, trainCategory):
	# 取矩阵行数，作为文档数
	numTrainDocs = len(trainMatrix)
	# 取矩阵第一行的列数，作为词汇数
	numWords = len(trainMatrix[0])
	# 提取训练类别中的非重复元素（0,1,2）
	mySet = set(trainCategory)
	# 计算不同类别的发生概率，并存储在类别p中
	p = []
	for item in mySet:
		p.append((trainCategory.count(item))/float(numWords))

	# 构造向量用来存放对应三种类别的词汇出现的频次
	# 不使用zeros的原因是：
	# 当使用公式 p(w0,w1,w2,...,wN|Ci)=p(w0|Ci)*p(w1|Ci)*...*p(wN|Ci) 计算条件概率时，避免出现计算结果为0的情况
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p2Num = ones(numWords)

	# 以下三个变量用来存储三种类别下总词汇数，作为分母，为避免出现0值，因此初始化为2
	# 不能初始化为1，因为分子初始化为1
	p0Denom = 2.0
	p1Denom = 2.0
	p2Denom = 2.0

	# 取每一个训练文本
	for i in range(numTrainDocs):
		if trainCategory[i] == 0:
			# p0Num是一个长度为numWords的向量，p0Num += trainMatrix[i]表示对应词汇个数+1
			p0Num += trainMatrix[i]
			# 计算总的词汇个数，计算概率时作为分母
			p0Denom += sum(trainMatrix[i])
		elif trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p2Num += trainMatrix[i]
			p2Denom = sum(trainMatrix[i])

	# 计算概率，取对数的原因是避免下溢出
	# 下溢出：大量很小的浮点数相乘，四舍五入后得到0
	p0Vect = log(p0Num/p0Denom)
	p1Vect = log(p1Num/p1Denom)
	p2Vect = log(p2Num/p2Denom)

	return p, p0Vect, p1Vect, p2Vect

#=======测试代码=======


'''randMat = mat(random.random_integers(0,1,[6,32]))
用该代码生成一个6X32的伯努利矩阵，但不符合trainNB的规范，因此使用原书中的矩阵合并方式
# print(randMat)
Classes = [0,1,2,0,1,1]
pp, p0V, p1V, p2V = trainNB4(randMat, Classes)'''

# 测试二类分类器
trainMat = []
for postinDoc in listOPosts:
	trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# print(trainMat)
p0Vec, p1Vec, pAb = trainNB0(trainMat, listClasses)

# 测试三类分类器
Classes = [0,1,2,0,1,1]
pp, p0V, p1V, p2V = trainNB4(trainMat, Classes)
print(pp,"\n",p0V,"\n",p1V,"\n",p2V,"\n")


# 朴素贝叶斯分类函数
# 输入参数：测试文本列表，文档分类为0的概率向量（对数），文档分类为1的概率向量，全部文档分类为1的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	# 概率相乘，因为是对数形式，所以为求和形式
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	# 贝叶斯准则
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()

# bag-of-words model:使用词带替换set
# 该模型考虑了词出现多次的情况
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

#=======测试样例1：垃圾邮件过滤=======
import re

def textParse(bigString):
	# 使用正则表达式去除文本中的空格和标点符号
	listOfTokens = re.split(r'\\W*',bigString)
	# 词汇全部小写，并且仅保留长度大于2的词汇
	return[tok.lower() for tok in listOfTokens if len(tok) > 2]

# 交叉验证
def spamTest():
	docList = []; classList = []; fullText = []
	for i in range(1,26):
		data = open('email/spam/%d.txt' % i,'r+',encoding = 'iso-8859-15').read()
		wordList = textParse(data)
		# wordList = textParse(open('email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		data = open('email/ham/%d.txt' % i,'r+',encoding = 'iso-8859-15').read()
		wordList = textParse(data)
		# wordList = textParse(open('email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = list(range(50)); testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pAb = trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pAb) != classList[docIndex]:
			errorCount += 1
	print("The error rate is: ", float(errorCount)/len(testSet))

spamTest()


