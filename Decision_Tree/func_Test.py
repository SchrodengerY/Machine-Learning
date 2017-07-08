# 这是一个函数测试文件

# 测试-1
def createDataSet():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	# dataset意义：不浮出水面是否可以生存（1-是；0-否）；是否有脚蹼（1-是；0-否）；属于鱼类
	labels = ['no surfacing', 'flippers']
	# 两类标签：不浮出水面是否可以生存，有脚蹼
	return dataSet, labels

myDat, labels = createDataSet()
print(myDat[0])