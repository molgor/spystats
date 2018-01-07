
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as sp
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline

N = 2000
phi = 0.02
sigma2 = 5
nugget = 0.8
coords = np.random.rand(N,1)
X = np.ones([N, 1])
beta = np.array([5])
# plt.plot(X[:,0], 'kx', mew=2)
# plt.show()

dis = sp.squareform(sp.pdist(coords[:, 0:1]))
cor = np.exp(- dis / phi)
cov = cor * sigma2
# plt.imshow(cov)
# plt.show()

mu = np.sum(X * beta, 1)
S = np.random.multivariate_normal(np.zeros(N), cov) +\
        np.random.normal(size = N) * np.sqrt(nugget)
Y = mu + S
Y = Y.reshape([-1, 1])
# plt.scatter(coords[:, 0], Y)
# plt.show()

# FUNCTIONS TO DEFINE LIKELIHOOD -----------------------------------------------

# def tf_dist(x):
#     normsq = tf.reduce_sum(x * x, 1)
#     normsq = tf.reshape(normsq, [-1, 1])
#     D2 = normsq - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(normsq)
#     return(tf.sqrt(D2))

def tf_cov_exp(d, sigma2, phi, nugget):
    S = sigma2 * tf.exp(- d / phi) + nugget * tf.eye(tf.shape(d)[1])
    return(S)

def reg_mean(x, beta):
    mean = tf.reduce_sum(x * beta, 1)
    mean = tf.reshape(mean, [-1, 1])
    return(mean)

def dmvnorm(y, mean, sigma):
    L = tf.cholesky(sigma)
    kern_sqr = tf.matrix_triangular_solve(L, y - mean, lower = True)
    n = tf.cast(tf.shape(sigma)[1], tf.float32)
    loglike = - 0.5 * n * tf.log( 2.0 * np.pi)
    loglike += - tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    loglike += - 0.5 * tf.reduce_sum(tf.square(kern_sqr))
    return(loglike)
    # return(loglike)

def logistic(x):
    out = 1.0 / (1.0 + tf.exp(-x))
    return(out)

def logistic0(x):
    out = 1.0 / (1.0 + np.exp(-x))
    return(out)


def logit(x):
    out = np.log(x / (1 - x))
    return(out)

# 0.03 0.25 0.14
# Parameters
tf_phi_log = tf.Variable(np.log(phi + 0.5), dtype = tf.float32)
tf_sigma2_log = tf.Variable(np.log(sigma2 + 2), dtype = tf.float32)
tf_nugget_ratio = tf.Variable(logit(nugget / sigma2 + 0.5), dtype = tf.float32)
tf_beta = tf.Variable([5], dtype = tf.float32)
# Transformations
tf_phi = tf.exp(tf_phi_log)
tf_sigma2 = tf.exp(tf_sigma2_log)
tf_nugget = tf_sigma2 * logistic(tf_nugget_ratio)
# Data
tf_dis = tf.placeholder(dtype = tf.float32)
tf_x = tf.placeholder(dtype = tf.float32)
tf_y = tf.placeholder(dtype = tf.float32)
# Computation
tf_mean = reg_mean(tf_x, tf_beta)
tf_sigma = tf_cov_exp(tf_dis, tf_sigma2, tf_phi, tf_nugget)
tf_loss = - dmvnorm(tf_y, tf_mean, tf_sigma)
# Arguments
feed_dict = {
        tf_x: X,
        tf_dis: dis,
        tf_y: Y
        }


train_op = tf.train.AdamOptimizer(0.1).minimize(tf_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
old_loss = 10000
loss = 0
tol = 0.0001
step = 0
while np.abs(old_loss - loss) > tol:
    step += 1
    old_loss = loss
    _, loss = sess.run([train_op, tf_loss], feed_dict)
    print(step, sess.run(tf_loss, feed_dict), sess.run(tf_beta), sess.run(tf_phi),
            sess.run(tf_sigma2), sess.run(tf_nugget))

# for step in range(200):

# # GRADIENT DESCENT OPTIMIZER ---------------------------------------------------
# train_step = tf.train.GradientDescentOptimizer(0.002).minimize(tf_loss)
# sess1 = tf.Session()
# sess1.run(tf.global_variables_initializer())
# # out = sess.run(tf_loss, feed_dict)
# # print(out)
# for step in range(200):
#     sess1.run(train_step, feed_dict)
#     print(step, sess1.run(tf_loss, feed_dict), sess1.run(tf_beta), sess1.run(tf_phi),
#             sess1.run(tf_sigma2), sess1.run(tf_nugget))

# 51 2979.12 [ 5.28443861] 0.0152085 4.81193 0.621492
# 999 367.15 [ 5.26065683] 0.0239065 5.43957 0.711608

# STOCHASTIC GRADIENT DESCENT OPTIMIZER -------------------------------------------
train_step = tf.train.GradientDescentOptimizer(0.002).minimize(tf_loss)
sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())
batch_size = 200
batches = N // batch_size + (N % batch_size != 0)
for step in range(1000):
    i_batch = (step % batches) * batch_size
    feed_dict = {
            tf_x: X[i_batch:i_batch + batch_size],
            tf_dis: dis[i_batch:i_batch + batch_size, i_batch:i_batch + batch_size],
            tf_y: Y[i_batch:i_batch + batch_size]
            }
    sess1.run(train_step, feed_dict)
    print(step, sess1.run(tf_loss, feed_dict), sess1.run(tf_beta), sess1.run(tf_phi),
            sess1.run(tf_sigma2), sess1.run(tf_nugget))

