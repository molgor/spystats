
import GPflow as gf
import numpy as np
import scipy.spatial.distance as sp
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline

N = 1000
phi = 0.05
sigma2 = 2
nugget = 1
X = np.random.rand(N,4)
# plt.plot(X[:,0], X[:,1], 'kx', mew=2)
# plt.show()

distance = sp.squareform(sp.pdist(X[:, 0:2]))
correlation  = np.exp(- distance / phi)
covariance = correlation * sigma2
# plt.imshow(covariance)
# plt.show()

mu = 10 + 1.5 * X[:, 2] - 1 * X[:, 3]
mu = mu.reshape(N,1)
S = np.random.multivariate_normal(np.zeros(N), correlation) +\
        np.random.normal(size = N) * nugget
S = S.reshape(N,1)
Y = mu + S
plt.scatter(X[:, 0], X[:, 1], c = S)
plt.show()

# ONLY GAUSSIAN PROCESS --------------------------------------------------------

# Defining the model
k = gf.kernels.Matern12(2, lengthscales=1, active_dims = [0,1] )
X1 = X[:, 0:2]
m = gf.gpr.GPR(X1, S, k)
m.likelihood.variance = 0.1
print(m)

# Estimation
m.optimize()
print(m)


# GAUSSIAN PROCESS WITH LINEAR TREND -------------------------------------------

# Defining the model
k = gf.kernels.Matern12(2, lengthscales=1, active_dims = [0,1] )
meanf = gf.mean_functions.Linear(np.ones((4,1)), np.ones(1))
m = gf.gpr.GPR(X, Y, k, meanf)
m.likelihood.variance = 0.1
print(m)

# Estimation
m.optimize()
print(m)

# ------------------------------------------------------------------------------
# Defining the model
k = gf.kernels.Matern12(2, lengthscales=1, active_dims = [0,1])
meanf = gf.mean_functions.LinearG(np.ones((2,1)), np.ones(1))
m = gf.gpr.GPR(X, Y, k, meanf)
m.likelihood.variance = 0.1
print(m)

# Estimation
m.optimize()
print(m)



plot(m)
plt.show()



# PREDICTION -------------------------------------------------------------------
def plot(m):
    xx = np.linspace(-0.1, 1.1, 100)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
    plt.xlim(-0.1, 1.1)
plot(m)
plt.show()

# MEAN FUNCTIONS ---------------------------------------------------------------
k = GPflow.kernels.Matern52(1, lengthscales=0.3)
meanf = GPflow.mean_functions.Constant(1)
m = GPflow.gpr.GPR(X, Y, k, meanf)
m.likelihood.variance = 0.01
print(m)

m.optimize()
plot(m)
plt.show()

print(m)


