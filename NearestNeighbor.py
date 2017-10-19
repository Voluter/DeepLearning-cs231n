import numpy as np

#近邻算法代码，简单实现
class NearestNeighbor:
    def __init__(self):
        pass
    def train(self, X, y):
        self.strX = X
        self.stry = y
    def predict(self, X):
        num_test = X.shape[0]
        preY = np.zeros(num_test, dtype = self.stry.dtype)
        for i in range(num_test):
            distance = np.sum(np.abs(self.strX-X[i,:]), axis = 1)
            minIndex = np.argmin(distance)
            preY[i] = self.stry[minIndex]
        return preY
    def score(self, prey, y):
        num_test = y.shape[0]
        return int(sum(prey==y))/num_test
#加载数据
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict
batch1 = unpickle('batch_data1')
test = unpickle('test_batch')
train_X = np.array(batch1[b'data'])
y = np.array(batch1[b'labels'])

test_X = np.array(test[b'data'])
test_y = np.array(test[b'labels'])
#测试
nn = NearestNeighbor()
nn.train(train_X,y)
pre_Y = nn.predict(test_X)
nn.score(pre_Y, test_y)#0.2235
        
