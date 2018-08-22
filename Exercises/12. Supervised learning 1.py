import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests, random, os, warnings

## Supervised learning packages
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC  # SVC (support vector classifier)
# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


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
plt.rcParams['figure.figsize'] = 9, 4  # set default size of plots
plt.rcParams.update({'font.size': 12})

#############################################
#               Modelling data              #
#############################################
### OLS
X = np.random.normal(size=(3,2))
y = np.random.normal(size=(3))
w = np.random.normal(size=(3))

# Gradient descent: Compute errors, multiply with features and update
# like with Adaline, we minimize the sum of squared errors (SSE):
e = y-(w[0]+X.dot(w[1:]))
SSE = e.T.dot(e)

# Updating:
eta = 0.001  # learning rate
fod = X.T.dot(e)  # first-order derivative of SSE wrt. estimated weights
update_vars = eta*fod  # change in estimated weights
update_bias = eta*e.sum()

# Advantages over OLS:
# - Works despite high multicollinarity
# - Speed (with many observations)

### Fitting a polynomial
def true_fct(X):
    return 2+8*X**4

n_samples = 25
n_degrees = 15

np.random.seed(0)

X_train = np.random.normal(size=(n_samples,1))
y_train = true_fct(X_train).reshape(-1) + np.random.randn(n_samples)

X_test = np.random.normal(size=(n_samples,1))
y_test = true_fct(X_test).reshape(-1) + np.random.randn(n_samples)

# Estimate the polynomials
test_mse = []
train_mse = []
parameters = []
degrees = range(n_degrees+1)

for p in degrees:
    X_train_p = PolynomialFeatures(degree=p).fit_transform(X_train)
    X_test_p = PolynomialFeatures(degree=p).fit_transform(X_train)
    reg = LinearRegression().fit(X_train_p, y_train)
    train_mse += [mse(reg.predict(X_train_p),y_train)]
    test_mse += [mse(reg.predict(X_test_p),y_test)]
    parameters.append(reg.coef_)

# Model performance in- and out-of-sample?
degree_index = pd.Index(degrees,name='Polynomial degree ~ model complexity')
ax = pd.DataFrame({'Train set':train_mse, 'Test set':test_mse})\
    .set_index(degree_index)\
    .plot(figsize=(10,4))
ax.set_ylabel('Mean squared error')

# The coefficient size increases
order_idx = pd.Index(range(n_degrees+1),name='Polynomial order')
ax = pd.DataFrame(parameters,index=order_idx)\
.abs().mean(1)\
.plot(logy=True)
ax.set_ylabel('Mean parameter size')

### How might we solve the overfitting problem?
# Make models which are less complex - by reducing
# - the number of variables/coeffients
# - the coefficient size of variables

### Regularization
# Introduction of penalties implies that increased model complexity has to be
# met with high increases in precision of estimates.

## The two most common penalty functions are L1 and L2 regularization.
## L1 regularization (Lasso): $R(\beta)=\sum_{j=1}^{p}|\beta_j|$
#  - Makes coefficients sparse, i.e. selects variables by removing some (if λ is high)
## L2 regularization (Ridge): $R(\beta)=\sum_{j=1}^{p}\beta_j^2$
#  - Reduce coefficient size
#  - Fast due to analytical solution
## The 'Elastic Net' uses a combination of L1 and L2 regularization.

### Rescaling features
# We need to rescale our features:
# - convert to zero mean:
# - standardize to unit std:

# Compute in Python:
# - option 1: StandardScaler in sklearn
# - option 2: (X - np.mean(X)) / np.std(X)

# Fit to the distribution in the training data first, then rescale train and test!
# See more at https://stats.stackexchange.com/questions/174823/how-to-apply-standardization-normalization-to-train-and-testset-if-prediction-i

# The interacted variables need to be gaussian distributed

#############################################################################
#                     Ex. 12.1:                       #
#############################################################################
