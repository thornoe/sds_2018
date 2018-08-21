import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests, random, os

### Supervised learning packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # SVC (support vector classifier)
from sklearn.metrics import accuracy_score

base_url = 'https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch02/'
for filename in ('ch02.py', 'iris.data', 'iris.names.txt'):
    if not os.path.exists(filename):
        response = requests.get(base_url+filename)
        with open(filename,'wb') as f:
            f.write(response.text.encode('utf-8'))
from ch02 import Perceptron, AdalineGD, AdalineSGD, plot_decision_regions

%matplotlib inline
## plot styles
sns.set_style('white')
plt.style.use('seaborn-white')

#############################################
#           The perceptron model            #
#############################################
### The artificial neuron - computation
## 1. initialize the weight with small random number
X = np.random.normal(size=(3, 2))  # feature matrix x_i of input variables
print('X:',X)

y = np.array([1, -1, 1])  # target vector y
print('y:',y)

w = np.random.normal(size=(3))  # weight vector
print('w:',w)

# compute net-input
z = w[0] + X.dot(w[1:])
print('z:\n', z)
# compute errors
positive = z > 0
y_hat = np.where(positive, 1, -1)
e = y - y_hat
SSE = e.T.dot(e)

## 2. for each training observation, i=1,..,n:
## a. compute predicted target, y-hat_i
## b. update weights w-hat
# learning rate
eta = 0.001

# first order derivative
fod = X.T.dot(e) / 2

# update weights
update_vars = eta*X.T.dot(e)  # insert fod
update_bias = eta*e.sum()/2

#############################################
#       Working with the perceptron         #
#############################################
iris = sns.load_dataset('iris').iloc[:100]  # drop virginica

X = iris.iloc[:, [0, 2]].values  # keep petal_length and sepal_length
y = np.where(iris.species=='setosa', 1, -1)  # convert to 1, -1

sns.scatterplot(iris.sepal_length, iris.petal_length, hue=iris.species)
plt.scatter(iris.sepal_length, iris.petal_length)

# initialize the perceptron
clf = Perceptron(n_iter=10)

## fit the perceptron
# runs 10 iterations of updating the model
clf.fit(X, y)
print('Number of errors: %i' % sum(clf.predict(X)!=y))

# we plot the decisions
plot_decision_regions(X,y,clf)

## how does the model performance change?
f,ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(1, len(clf.errors_) + 1), clf.errors_, marker='o')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Number of errors')

### What might we change about the perceptron?
# 1. Change from updating errors that are binary to continuous
# 2. Use more than one observation a time for updating

### Adaptive Linear Neuron (Adaline): No transformation of the net-input

### Activation functions
# - Linear
# - Logistic (Sigmoid)
# - Unit step, sign

## Minimize the sum of squared errors (SSE). The SSE for the Adaline
## Approximate the first order derivative ~ gradient descent (GD)
## Alternative: Approximate both first and second order derivative ~ quasi Newton

#############################################
#   Working with the logistic regression    #
#############################################
# load data
titanic = sns.load_dataset('titanic')

# select and make dummy variables from categorical
cols = ['survived','class', 'sex', 'sibsp', 'age', 'alone']
titanic_sub = pd.get_dummies(titanic[cols].dropna(), drop_first=True).astype(np.int64)
titanic_sub.head(2)

X = titanic_sub.drop('survived', axis=1)
y = titanic_sub.survived

# we split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

# estimate model on train data, evaluate on test data
clf = LogisticRegression()
clf.fit(X_train, y_train) # model training
accuracy = (clf.predict(X_test)==y_test).mean() # model testing
print('Model accuracy is:', np.round(accuracy,3))

#############################################################################
#                  Ex. 11.1: Basic classification models                    #
#############################################################################
# The mathematics and biological reasoning which justifies the perceptron model
# is presented in Raschka, 2017 pp. 18-24

### 11.1.2 load Iris data
iris = sns.load_dataset('iris')
iris = iris.query("species == 'virginica' | species == 'versicolor'").sample(frac = 1, random_state = 3)
X = np.array(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y = np.array(iris['species'].map({'virginica': 1, 'versicolor': -1}))
sns.pairplot(iris, hue="species", palette="husl", diag_kind="kde", diag_kws = {'shade': False})

## General train_test_split
# X.size
# 70/400
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=.18, random_state=0)
# Xtrain.size

## A very simple deterministic test-train split
Xtrain = X[:70]
ytrain = y[:70]

Xtest = X[70:]
ytest = y[70:]

### 11.1.3 Initiate a set of weights w
def random_weights(location = 0.0, scale = 0.01, seed = 1):
    # Init random number generator
    rgen = np.random.RandomState(seed)
    w = rgen.normal(loc=location, scale=scale, size= 1 + X.shape[1])
    return w

### 11.1.4 calculate net-input and predict y
def net_input(X, W):
    return np.dot(X, W[1:]) + W[0]   # Linear product W'X + bias

def predict(X, W):
    linProd = net_input(X, W)
    return np.where(linProd >= 0.0, 1, -1)    # 1(linProd > 0)

# Bonus
def accuracy(y, prediction):
    return np.mean(y == prediction)

accuracy(ytrain, predict(Xtrain, random_weights()))

### 11.1.5 Loop over the training data and update the weights
def weight_update(X, y, W, eta):
    errors = 0
    for xi, yi in zip(X,y):
       update = eta * (yi - predict(xi, W))
       W[1:] = W[1:] + update * xi
       W[0] = W[0] + update
       errors = errors + int(update != 0)
    return W, errors

# Report accuracy
W = random_weights(0.0, 0.01, 1)
eta = 0.1
W, errors = weight_update(Xtrain, ytrain, W, eta)
W
errors
print('Weights:', W, '\nAccuracy:'
    , accuracy(ytrain, predict(Xtrain, random_weights())), ', i.e. no progress yet.')

### 11.1.6 More iterations
def Perceptron(X, y, n_iter):
    eta = 0.1  # sets learning rate
    weights = random_weights()
    errorseq = list()
    accuracyseq = list()

    for i in range(n_iter):
        weights, e = weight_update(X, y, weights, eta)
        a = accuracy(y, predict(X, weights))
        errorseq.append(e)
        accuracyseq.append(a)

    return weights, errorseq, accuracyseq

# Output
n_iter = 1000
weights, errorseq, accuracyseq = Perceptron(Xtrain, ytrain, n_iter)
print('Iteration:', n_iter,
    '\nWeights:', weights,
    '\nErrors:', errorseq[-1],
    '\nAccuracy:', accuracyseq[-1])

# Plot sequence of errors and accuracy
fig, ax1 = plt.subplots(figsize=(12, 5.5))
s1 = errorseq
ax1.plot(range(1, len(errorseq) + 1), errorseq, 'b-')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Errors', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = accuracyseq
ax2.plot(range(1,len(accuracyseq) + 1), accuracyseq, 'g-')
ax2.set_ylabel('Accuracy', color='g')
ax2.tick_params('y', colors='g')

fig.tight_layout()
plt.show()

### 11.1.7 Accuracy of perceptron on the test data
pred = predict(Xtest, weights)
accuracy(ytest, pred)

### 11.1.8 class
# Restructure your code as a class called Perceptron with .fit() and .predict() methods.
# (you) will probably need more helper methods.

#############################################################################
#                     Ex. 11.2: Support Vector Machine                      #
#############################################################################
### 11.2.1 Choose a kernel (e.g. linearm RBF or polynomial):
# http://scikit-learn.org/stable/modules/svm.html

### 11.2.2 Use the imported SVC (support vector classifier)
clf = SVC(random_state=1, kernel='rbf')
fitted_model = clf.fit(Xtrain, ytrain)

train_score = accuracy_score(ytrain, fitted_model.predict(Xtrain))
test_score = accuracy_score(ytest, fitted_model.predict(Xtest))

print(f"On the training data we get a score of {round(train_score, 2)}, while the score on the test data is {round(test_score, 2)}")

#############################################################################
#                     Ex. 11.3: AdaLine                      #
#############################################################################
# Adaline: Adaptive Linear Neuron (no transformation of the net-input)

### 11.3.1 implement
def ada_activation(Z):  # the identify function
    return Z

def ada_predict(X, W):  # a step function
    linProd = net_input(X, W)
    act = ada_activation(linprod)
    return np.where(act >= 0.0, 1, -1)    # 1(linProd > 0)

### 11.3.2 cost function (minimize sum of least squares)
def ada_cost(X, y, W):
    linProd = net_input(X, W)
    errors_sq = (y - ada_activation(linProd))**2
    return errors_sq.sum() / 2.0

# Unlike in undergraduate statistics we will optimize our estimator using gradient descent,
# therefore code up the negative of the derivative of the cost function:
def ada_cost_derivative(X, y, W):
    linProd = net_input(X, W)
    errors = y - ada_activation(linProd)
    return np.array( [errors.sum()] + list(X.T.dot(errors)))

# The derivative should return a list of the same length as the number of weights,
# since there is one derivative for each one.
ada_cost_derivative(Xtrain, ytrain, random_weights())

### 11.3.3 Implement the adaline fitting algorithm using batch gradient descent
# AdaLine treats the entire dataset as a batch, adjusts it's weights and then
# does it all again. Thus you only need to loop over n_iter, not the data rows.
def AdaLine(X, y, n_iter = 10000, eta = 0.00001):
    costseq = []
    W =  random_weights()

    for i in range(n_iter):
        nip = net_input(X, W)
        output = ada_activation(nip)

        W = W + eta * ada_cost_derivative(X, y, W)
        costseq.append(ada_cost(X,y, W))

    return W, costseq

# Use the cost function to track the progress of your algorithm.
w_trained, costs = AdaLine(Xtrain, ytrain)
plt.plot(costs)

### 11.3.4  Write a function that scales each of the variables (including y)
def standardScaler(X, y):
    """ Scales the input. (Horrible code)
    New value = (deviation from the mean of the variable) / (st. deviation)
    """
    X_new = X.copy()

    for i in range(X.shape[1]):
        xj = X[:,i]

        stdev = np.std(xj)
        mean = np.mean(xj)

        X_new[:,i] = (xj - mean)/stdev

    y_stdev = np.std(y)
    y_mean = np.mean(y)

    y_new = (y.copy() - y_mean)/y_stdev

    return X_new, y_new

# rerun the adaline function on the scaled variables.
X_scaled, y_scaled = standardScaler(Xtrain,ytrain)

w_trained, costs = AdaLine(X_scaled, y_scaled)
plt.plot(costs)
