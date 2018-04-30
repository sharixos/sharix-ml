import numpy as np 

class LogisticRegression2C(object):
    """ 2-classification logistic regression model
    
    Author: [@Wang Xin Gang](https://github.com/sharixos)

    Website: http://www.sharix.site/
    
    Github: https://github.com/sharixos

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
        self.train_x = np.concatenate((np.zeros([self.n,1]), train_x), axis=1) # there is a theta0
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
        theta_list = []
        print(self.theta)
        for _ in range(iteration):
            theta_list.append(self.theta.copy())
            self.theta -= alpha * self.gradient()
            cost_list.append(self.loss())
        return cost_list, theta_list
        
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
    """multi-classification logistic regression model
    
    Author: [@Wang Xin Gang](https://github.com/sharixos)

    Website: http://www.sharix.site/
    
    Github: https://github.com/sharixos

    Args:
        xdim, ylabes, need_normalized (optional)
    Attributes:
        n: the input_size, 
        k: the dimension of input x
        need_normalzied: 1 xdata need to be normalized, 0 no need

        labels: the class of y
        label_theta: every kind of y related to its theta, only K-1 theta needed for K types of y
    """
    def __init__(self, xdim, ylabels, need_normalized = 0):
        self.dim = xdim
        self.need_normalized = need_normalized
        self.labels = ylabels
        self.K = len(ylabels)

        self.label_theta = {}
        for label in self.labels[:-1]:
            self.label_theta[label] = np.array(np.zeros(self.dim+1))

    def hypothesis(self, label, x):
        S = 0
        for l in self.labels[:-1]:
            S += np.exp(np.dot(np.array(self.label_theta[l].T), x))
        
        if label in self.labels[:-1]:
            return np.exp(np.dot(self.label_theta[label].T, x)) / (1+S)
        else:
            return 1/(1+S)


    def loss(self, train_x, train_y):
        """
        use -j to represent loss
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
        Middle Variables:
            train_x: the dimension here is k+1
            train_y: input_y

        Returans:
            cost_list & theta_list
        """
        if self.need_normalized == 1:
            self.normalized_value = []
            for i in range(self.dim):
                minxi = np.min(input_x[:,i])
                maxxi = np.max(input_x[:,i])
                input_x[:,i] = (input_x[:,i] - minxi)/(maxxi-minxi)
                self.normalized_value.append([minxi, maxxi])
        train_x = np.concatenate((np.ones([len(input_x),1]), input_x), axis=1) # there is a theta0
        train_y = input_y

        cost_list,theta_list = [],[]
        for ite in range(iteration):
            real_alpha = (40.0 / (ite+1) + 1) * alpha
            theta_list.append(self.label_theta[self.labels[0]].copy())
            for i in range(len(train_x)):
                x,label = train_x[i], train_y[i]
                S = 0
                expzlist = {}
                for l in self.labels[:-1]:
                    expzlist[l] = np.exp(np.dot(self.label_theta[l].T, x))
                    S += expzlist[l]
                h = self.hypothesis(label, x)
                # h_z_partial = {}
                if label == self.labels[-1]:
                    for l in self.labels[:-1]:
                        delta = -1*expzlist[l] /pow(1+S, 2)
                        self.label_theta[l] += (1/h) * real_alpha * delta * x # update theta
                else:
                    for l in self.labels[:-1]:
                        if label == l:
                            delta = (expzlist[l]*(1+S) - pow(expzlist[l],2)) / pow(1+S, 2)
                        else:
                            delta = -1*(expzlist[l]*expzlist[label]) / pow(1+S, 2)
                        self.label_theta[l] += (1/h) * real_alpha * delta * x # update theta
            cost_list.append(self.loss(train_x,train_y))
        return cost_list,theta_list

            

    def predict(self, x1):
        """
        Args:
            x:2-dim array to express x, [[x1,x2,...]]
        Returns:
            the predicted y
        """
        if self.need_normalized == 1:
            for i in range(self.dim):
                minxi,maxxi = self.normalized_value[i]
                x1[:,i] = (x1[:,i] - minxi)/(maxxi-minxi)
        px = np.concatenate((np.ones([1,1]), x1), axis=1)
        hdict = {}
        for l in self.labels:
            hdict[l] = self.hypothesis(l, px[0])
        return sorted(hdict,key=lambda x:hdict[x])[-1]
