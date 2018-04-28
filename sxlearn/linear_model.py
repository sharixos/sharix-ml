import numpy as np

class LinearRegression(object):
    """train_x must be a n*k list
    
    Args:
        train_x, train_y, need_normalized (optional)
    Attributes:
        n: the input_size, 
        k: the dimension of input x
        theta:
        train_x: the dimension here is k+1
        train_y:
        need_normalzied: 1 xdata need to be normalized, 0 no need
    """
    def __init__(self, train_x, train_y, need_normalized = 0):
        self.n,self.k = np.shape(train_x)
        self.need_normalized = need_normalized
        self.train_x,self.train_y = train_x,train_y
        if self.need_normalized == 1:
            self.normalized_value = []
            for i in range(self.k):
                minxi = np.min(self.train_x[:,i])
                maxxi = np.max(self.train_x[:,i])
                self.train_x[:,i] = (self.train_x[:,i] - minxi)/(maxxi-minxi)
                self.normalized_value.append([minxi, maxxi])
        self.train_x = np.concatenate((np.ones([self.n,1]), train_x), axis=1) # there is a theta0
        self.theta = np.array(np.ones(self.k+1))

    def hypothesis(self):
        return np.dot(self.train_x, self.theta.T)

    def loss(self):
        return np.sum((self.hypothesis() - self.train_y) ** 2) / 2*self.n

    def gratitude(self):
        delta = np.array(self.hypothesis() - self.train_y)
        return np.dot(delta.T, self.train_x) / self.n

    def train(self,iteration, alpha):
        """use gratitude to move steps
        
        Args:
            iteration
            alpha
        
        Returans:
            a list of loss
        """
        cost_list = []
        for i in range(iteration):
            self.theta -= alpha * self.gratitude()
            cost_list.append(self.loss())
        return cost_list
        
    def predict(self, x):
        """
        Args:
            x:2-dim array to express x, [[x1,x2,...]]
        Returns:
            the predicted y
        """
        x = np.array(x, dtype = float)
        if self.need_normalized == 1:
            for i in range(self.k):
                minxi,maxxi = self.normalized_value[i]
                x[:,i] = (x[:,i] - minxi)/(maxxi-minxi)
        x = np.concatenate((np.ones([1,1]), x), axis=1)
        return np.dot(x, self.theta.T)


class Sensor(object):
    """train_x must be a n*k list
    
    Args:
        train_x, train_y, need_normalized (optional)
    Attributes:
        n: the input_size, 
        k: the dimension of input x
        theta:
        train_x: the dimension here is k+1
        train_y: +1,-1
        need_normalzied: 1 xdata need to be normalized, 0 no need
    """

    def __init__(self, train_x, train_y, need_normalized = 0):
        self.n,self.k = np.shape(train_x)
        self.need_normalized = need_normalized
        self.train_x,self.train_y = train_x,train_y
        if self.need_normalized == 1:
            self.normalized_value = []
            for i in range(self.k):
                minxi = np.min(self.train_x[:,i])
                maxxi = np.max(self.train_x[:,i])
                self.train_x[:,i] = (self.train_x[:,i] - minxi)/(maxxi-minxi)
                self.normalized_value.append([minxi, maxxi])
        self.train_x = np.concatenate((np.ones([self.n,1]), train_x), axis=1) # there is a theta0
        self.theta = np.array(np.ones(self.k+1))

    def hypothesis(self):
        return np.dot(self.train_x, self.theta.T)

    def loss(self):
        cost = 0
        h = self.hypothesis()
        for i in range(self.n):
            if self.train_y[i] * h[i] < 0:
                cost += -1 * self.train_y[i] * h[i]
        return cost

    def gratitude(self):
        delta = np.zeros([1,self.k+1])
        h = self.hypothesis()
        for i in range(self.n):
            if self.train_y[i] * h[i] < 0:
                delta+= -1*self.train_y[i]*self.train_x[i]
        return delta

    def train(self,iteration, alpha):
        """use gratitude to move steps
        
        Args:
            iteration
            alpha
        
        Returans:
            a list of loss
        """
        cost_list = []
        for i in range(iteration):
            self.theta -= alpha * self.gratitude()[0]
            cost_list.append(self.loss())
        return cost_list
        
    def predict(self, x):
        """
        Args:
            x:2-dim array to express x, [[x1,x2,...]]
        Returns:
            the predicted y +1,-1
        """
        x = np.array(x, dtype = float)
        if self.need_normalized == 1:
            for i in range(self.k):
                minxi,maxxi = self.normalized_value[i]
                x[:,i] = (x[:,i] - minxi)/(maxxi-minxi)
        x = np.concatenate((np.ones([1,1]), x), axis=1)
        if np.dot(x,self.theta.T) > 0:
            return 1
        return -1
