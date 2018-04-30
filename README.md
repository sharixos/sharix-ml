![](http://www.sharix.site/static/img/tiger.svg)

# sharix-ml
=========================

本例实现了一些简单的机器学习算法，以比较通用的形式来实现，很容易移植。

* Author: [@Wang Xin Gang](https://github.com/sharixos)

* Website: http://www.sharix.site/

* Github: https://github.com/sharixos



## 线性回归
对一个问题的结果y与k个因素有关，可以建立多变量线性模型，例如可作假设函数\\(h_\theta(x)\\)，x表示k维向量
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + ... + \theta_k x_k
$$
其对应的损失函数\\(J(\theta)\\) 如下，m为输入训练数据的量，i表示第i个数据
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m}(h_\theta(x_i) - y_i)^2
$$
利用梯度下降的思想损失函数对参数求梯度如下
$$
\frac{\partial J}{\partial \theta} = \frac{1}{2m} \sum_{i=1}^{m} 2(h_\theta(x_i) - y_i)*(1,x_{i1},x_{i2},...x_{ik})
$$
所以对于参数\\(\theta\\)的学习过程为
$$
\theta_new = \theta - \alpha * \frac{\partial J}{\partial \theta}
$$

```python
class LinearRegression(object):
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
        cost_list = []
        for i in range(iteration):
            self.theta -= alpha * self.gratitude()
            cost_list.append(self.loss())
        return cost_list

    def predict(self, x):
        x = np.array(x, dtype = float)
        if self.need_normalized == 1:
            for i in range(self.k):
                minxi,maxxi = self.normalized_value[i]
                print(minxi, maxxi)
                x[:,i] = (x[:,i] - minxi)/(maxxi-minxi)
        x = np.concatenate((np.ones([1,1]), x), axis=1)
        return np.dot(x, self.theta.T)
```
[线性回归完整代码](https://github.com/sharixos/sharix-ml/blob/master/sxlearn/linear_model.py)

## 感知机
输入空间为\\(\chi\subseteq R^n\\)，输出空间为1或-1时，输入的x表示实例的特征向量，对应特征空间的一个点，输出为实例的类别。感知机就是对输入实例的特征向量进行二分类的线性分类模型，这里其实就相当于超平面为 \\(w*x+b\\) ，损失函数就对应误分类点到超平面的总距离。符号函数如下：

$$
f(x) = sign(w*x+b)
$$

感知机损失函数两种表示如下：

$$
L(w,b) = - \sum_{x_i\in M} y_i*(w*x_i+b)
$$

$$
J(\theta) = - \sum_{x_i\in M} y_i(\theta_0 + \theta_1 x_{i1} + ... + \theta_k x_{ik})
$$

利用梯度下降的思想损失函数对参数求梯度如下

$$
\frac{\partial J}{\partial \theta} = - \sum_{x_i\in M} y_i (1,x_{i1},x_{i2},...x_{ik})
$$

感知机学习算法是基于随机梯度下降法的对损失函数的最优化算法，当数据集线性可分时，感知机学习算法收敛。

```python
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
```
[感知机完整代码](https://github.com/sharixos/sharix-ml/blob/master/sxlearn/linear_model.py)

## K近邻法
k近邻法的三个基本要素包括：k值的选择、距离度量及分类决策规则。其输入为特征向量输出为实例的类别，方法主要为根据给定的距离度量在训练集T中找到与x最近的k个点记作\\(N_k(x)\\)，然后根据如下分类决策规则来决定x的类别，如下为预测时y的取值，其中\\(i=1,...,N\\)，和\\(j=1,...,K\\)
$$
y = arg \max_{c_j} \sum_{x_i\in N_k(x)}{I(y_i = c_j)}
$$
其中I为指示函数，当\\(y_i = c_j\\)时I才为1，否则为0。 若要考虑如果快速搜索这k个最近点，可以使用kd树。

## 朴素贝叶斯法
朴素贝叶斯是典型的生成学习方法，由训练数据学习联合概率分布\\(P(X,Y)\\)，然后求得后验概率\\(P(Y|X)\\)，朴素贝叶斯的基本假设是条件独立性，可以使用极大似然估计或贝叶斯估计来获得联合概率分布
$$
y = arg \max_{c_k} P(Y=c_k)\prod_{j=1}^n{P(X_j=x^{(j)}|Y=c_k)}
$$
简单来说就是将输入的x分到后验概率最大的那个y，后验概率最大等价于0~1损失函数时的期望风险最小化

在算法的实现过程中，我主要使用了one-hot的思想，对于每一个可能出现的特征值都给其一个维度，这样以一个0或1的多维向量来表示一个分类样本，朴素贝叶斯的一个特点就是可以多次更新或者说增加训练的样本数据，这里使用feed来进行数据的增加，在predict之前需要进行一些参数的计算，所以需要调用feed_end来更新。这里采用加1平滑，并且利用log求对数来避免概率为0。
```python
def feed(self, train_x, train_y):
    self.num_training += len(train_x)
    for i in range(len(train_x)):
        self.num_y_dict[train_y[i]] += 1
        self.sum_xvec_dict[train_y[i]] += train_x[i]
        self.hot_sum_dict[train_y[i]] += sum(train_x[i])

def feed_end(self):
    self.py_dict = {}
    self.regularized_xvec_dict = {}
    for l in self.labels:
        self.py_dict[l] = np.log(float(self.num_y_dict[l]) / self.num_training)
        self.regularized_xvec_dict[l] = np.log(self.sum_xvec_dict[l] / self.hot_sum_dict[l])

def predict(self, inputx):
    """
    make sure you call feed_end before predict
    """
    x = np.array(inputx)
    result = {}
    for l in self.labels:
        px = sum(self.regularized_xvec_dict[l] * x)
        result[l] = self.py_dict[l] + px # use log so here is +
    return sorted(result, key=lambda x:result[x])[-1] # return max
```

* [朴素贝叶斯完整代码](https://github.com/sharixos/sharix-ml/blob/master/sxlearn/bayes.py)

## 决策树
学习时，利用训练数据，根据损失函数最小化的原则建立决策树模型，预测时，对新的数据利用决策树模型进行分类。决策树学习通常包括：特征选择、决策树生成和决策树修剪。特征选择的目的在于选取对训练数据能够分类的特征，常用的准则有：信息增益和基尼指数（CART），由于生成的决策树存在过拟合问题，所以需要进行剪枝，可以使用CART剪枝算法。

## logistic回归与最大熵模型
关于最大熵模型，吴军在《数学之美》中浅出的解释过，就是要保留全部的不确定性，将风险降到最小，也就是要满足全部已知条件，而对位置情况不作任何假设。logistic回归模型和最大熵模型都属于对数线性模型，若离散随机变量的概率分布为\\(P(X)\\)，则其熵为：
$$
H = -\sum_{x}{P(x)log P(x)}
$$
logistic分布则为：
$$
F(x) = P(X \leq x)=\frac{1}{1+e^{-(x-\mu)/\gamma}}
$$
__逻辑函数其实是一个一层的人工神经网络__，对于NLP领域，需要训练的参数很多，类似于最大熵模型的训练，可以采用 GIS/IIS 等方式训练。logistic回归在广告系二分类问题中，两种类别的概率分别可以表示为：统中得到很好的应用，可以在《数学之美》中了解。

__logistic二分类模型__

首先用\\(z = -(x-\mu)/\gamma\\)来表示线性部分，可以进一步表示为，这里k表示输入x数据的维度，\\(\theta_0\\)则表示bias，也就是b
$$
z = \theta_0 + \theta_1 x_1 + ... + \theta_k x_k
$$
二分类问题中，两种类别的概率，也就是对应的hypothesis分别可以表示为：

$$
P(Y=1|x) = \frac{e^z}{1+e^z} = sigmoid(z)
$$
$$
P(Y=0|x) = \frac{1}{1+e^z} = 1-sigmoid(z)
$$

对于目标函数，表示为样本的概率之间的乘积，对于二分类可以用0和1来表示样本类型，将上述概率合并在一起，我们要学习的目的就是使得对于训练样本下面的object目标都越大越好，将指数问题取对数可以简化计算
$$
object(x,y) = (P(Y=1|x))^y(P(Y=0|x))^{1-y}
$$
$$
object(x,y) = y\log(sigmoid(z)) + (1-y)\log(1-sigmoid(z))
$$
所以对于整个训练样本的目标函数为J，可以用-J来表示损失函数
$$
J(\theta) = \sum_{i=1}^mobject(x_i, y_i)
$$
然后就可以用梯度下降算法更新参数\\(\theta\\)，损失函数对\\(\theta\\)求偏导完后可以得到梯度gradient为
$$
gradient = (sigmoid(z) - y)x
$$
这里可以看出来logistic回归二分类模型与线性回归和感知机有着几乎完全相同的梯度形式，因为他们在本质上是可以等价的，都是用已有的样本来训练一条最优直线
```python
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
```

__logistic多分类模型__

logistci多分类模型在二分类的基础上变得复杂不少，但是核心的思想是相同的，考虑样本的概率连乘为目标函数，目标函数越大越好，主要考虑的是对于多分类问题不同类型的样本的概率的表示是不同的，对于随机变量Y设其取值为\\(k \in {1,2\cdots,K}\\)，对于K种Y只需要对应的K-1个\\(\theta\\)种参数就可以进行分类，也就是对应K-1条分类直线

则当\\(k=1,2,\cdots K-1\\)的时候，其概率表示为：
$$
P(Y=k|x) = \frac{e^{z_k}}{1+\sum_{k=1}^{K-1}e^{z_k}}
$$
当k=K-1的时候，概率为
$$
P(Y=K|x) = \frac{1}{1+\sum_{k=1}^{K-1}e^(z_k)}
$$

由于前k-1个的概率表示与第K种的概率表示是不同的，所以在写hypothesis函数的时候也要进行区分，就跟二分类的时候一个为sigmoid，一个为1-sigmoid一样，另外求偏导的时候也要进行区分。

目标函数可以表示为
$$
J(\theta) = \sum_{i=1}^m\log(h_y(z))
$$

目标函数求偏导，注意这里的z指的是一系列的z，因为有k-1个\\(\theta\\)，所以就有对应的这么多个z
$$
\frac{\partial J}{\partial \theta} = \frac{1}{h_y(z)}  \frac{\partial h_y}{\partial z}  \frac{\partial z}{\partial \theta}
$$
当样本的k==K时，有如下偏导，__注意这里的S表示多个z对应的指数然后求和__
$$
\frac{\partial h}{\partial z_i} = - \frac{1}{(1+S)^2}
$$
当样本的k<K时，其概率表示不同，所以求偏导也不同，而且当对第k个求导与对其它变量求导不同

如下为样本为第k种的目标函数对\\(z_k\\)求偏导
$$
\frac{\partial h}{\partial z_k} = \frac{e^{z_k}(1+S) - e^{2z_k}}{(1+S)^2}
$$
对\\(z_i\\)且\\(i!=k\\)求偏导
$$
\frac{\partial h}{\partial z_k} = \frac{e^{z_k}e^{z_i}}{(1+S)^2}
$$

对于多分类的logistic回归模型，如吴军在《数学之美》中所说，等价于一个一层的人工神经网络，所以神经网络其实与线性模型比较相近，多分类logistic是求各种类对应的权重\\(\theta\\)，而将每一种类的\\(\theta\\)叠加起来就构成了一个简单的神经网络。
```python
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
```
[二分类和多分类logistic回归的完整代码](https://github.com/sharixos/sharix-ml/blob/master/sxlearn/logistic.py)
我使用这个多分类logistic回归模型对部分搜狗的中文数据进行分类，取得了比较好的效果



## 后记
感谢 [@PieHust](https://github.com/PieHust) 在本系列的实现过程中，提供的使用函数式编程的方法