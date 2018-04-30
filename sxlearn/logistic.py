import numpy as np 

class LogisticRegression2C(object):
    """train_x must be a n*k list
    
    Args:
        train_x, train_y, need_normalized (optional)
    Attributes:
        n: the input_size, 
        k: the dimension of input x
        theta:
        train_x: the dimension here is dim+1
        train_y:
        need_normalzied: 1 xdata need to be normalized, 0 no need
    """
    def __init__(self, train_x, train_y, need_normalized = 0):
        self.n,self.dim = np.shape(train_x)
        self.need_normalized = need_normalized
        self.train_x,self.train_y = train_x,train_y
        if self.need_normalized == 1:
            self.normalized_value = []
            for i in range(self.dim):
                minxi = np.min(self.train_x[:,i])
                maxxi = np.max(self.train_x[:,i])
                self.train_x[:,i] = (self.train_x[:,i] - minxi)/(maxxi-minxi)
                self.normalized_value.append([minxi, maxxi])
        self.train_x = np.concatenate((np.ones([self.n,1]), train_x), axis=1) # there is a theta0
        self.theta = np.array(np.ones(self.dim+1))

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def hypothesis(self):
        z = np.dot(self.train_x, self.theta.T)
        return list(map(self.sigmoid, z))

    def loss(self):
        """
        use -j to represent loss
        """
        j = 0
        h = self.hypothesis()
        for i in range(self.n):
            yi = self.train_y[i]
            if yi == 1:
                j += np.log(h[i])
            elif yi == 0:
                j += np.log(1-h[i])
            else:
                print('error type y not exist')
                j += 0
        return -j
            

    def gradient(self):
        delta = np.array(self.hypothesis() - self.train_y)
        return np.dot(delta.T, self.train_x)

    def train(self,iteration, alpha):
        """use gradient to move steps
        
        Args:
            iteration
            alpha
        
        Returans:
            a list of loss
        """
        cost_list = []
        for _ in range(iteration):
            self.theta -= alpha * self.gradient()
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
            for i in range(self.dim):
                minxi,maxxi = self.normalized_value[i]
                x[:,i] = (x[:,i] - minxi)/(maxxi-minxi)
        x = np.concatenate((np.ones([1,1]), x), axis=1)
        return self.sigmoid(np.dot(x, self.theta.T))

        
class LogisticRegression(object):
    """train_x must be a n*k list
    
    Args:
        train_x, train_y, need_normalized (optional)
    Attributes:
        n: the input_size, 
        k: the dimension of input x
        train_x: the dimension here is k+1
        train_y:
        need_normalzied: 1 xdata need to be normalized, 0 no need

        labels: the class of y
        label_theta: every kind of y related to its theta, only K-1 theta needed for K types of y
    """
    def __init__(self, xdim, ylabes, need_normalized = 0):
        self.dim = xdim
        self.need_normalized = need_normalized
        # self.train_x,self.train_y = train_x,train_y
        self.labels = ylabes

        self.label_theta = {}
        for label in self.labels[:-1]:
            self.label_theta[label] = np.array(np.ones(self.dim+1))

    def hypothesis(self, label, x):
        S = 0
        for l in self.labels[:-1]:
            S += np.exp(np.dot(self.label_theta[l].T, x))
        
        if label in self.labels[:-1]:
            return np.exp(np.dot(self.label_theta[label].T, x)) / (1+S)
        else:
            return 1/(1+S)


    def loss(self, train_x, train_y):
        """
        use 1-j to represent loss
        """
        j = 0
        for i in range(len(train_x)):
            hi = self.hypothesis(train_y[i], train_x[i])
            j += np.log(hi)
        return -j
            

    def feed(self, input_x, input_y, iteration=100, alpha=0.01):
        """use gradient to move steps
        
        Args:
            iteration
            alpha
        
        Returans:
            a list of loss
        """
        if self.need_normalized == 1:
            self.normalized_value = []
            for i in range(self.dim):
                minxi = np.min(input_x[:,i])
                maxxi = np.max(input_x[:,i])
                train_x[:,i] = (input_x[:,i] - minxi)/(maxxi-minxi)
                self.normalized_value.append([minxi, maxxi])
        train_x = np.concatenate((np.ones([self.n,1]), train_x), axis=1) # there is a theta0
        train_y = input_y

        cost_list = []
        for _ in range(iteration):
            for i in range(len(train_x)):
                x,label = train_x[i], train_y[i]
                zlist = []
                for l in self.labels[:-1]:
                    if l == lab .:
                        

    def gradient(self):
        g = {}
        for label in self.labels:
            g[label] = np.zeros(self.dim+1)
        
        for i in range(self.n):
            x,label = self.train_x[i], self.train_y[i]
            delta = 1 - self.hypothesis(label, x)
            g[label] += delta * x
        return g

    #     """
    #     cost_list = []
    #     for _ in range(iteration):
    #         g = self.gradient()
    #         for label in self.labels:
    #             self.label_theta[label] -= alpha * g[label]
    #         cost_list.append(self.loss())
    #     return cost_list
        

            

    def predict(self, x):
        """
        Args:
            x:2-dim array to express x, [[x1,x2,...]]
        Returns:
            the predicted y
        """
        x = np.array(x, dtype = float)
        if self.need_normalized == 1:
            for i in range(self.dim):
                minxi,maxxi = self.normalized_value[i]
                x[:,i] = (x[:,i] - minxi)/(maxxi-minxi)
        x = np.concatenate((np.ones([1,1]), x), axis=1)
        return self.sigmoid(np.dot(x, self.theta.T))
