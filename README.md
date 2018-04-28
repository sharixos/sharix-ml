# sharix-ml
simple machine learning library

欢迎访问： www.sharix.site

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



## 后记
感谢 [@PieHust](https://github.com/PieHust) 在本系列的实现过程中，提供的使用函数式编程的方法