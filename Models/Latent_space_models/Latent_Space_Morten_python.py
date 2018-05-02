from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import edward as ed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

from edward.models import Normal, Poisson, Bernoulli, InverseGamma
#from observations import celegans
import collections

#Reading data into adjecency and 0'ing last 1% links:
from Feature_extraction.Edgelist import M_onesRemoved, M_fullData, M_zeroing

x_train= M_zeroing[:500,:500]

"""
###################
###################
#Reading by old adjecency matrix:
#Det fulde (samlede datasæt)
G_full = nx.read_edgelist("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Data/network_10k.edgelist",create_using=nx.DiGraph()) # read and parse edgelist to (networkx) graph
A_full = nx.adjacency_matrix(G_full) # make Adjacency matrix
#x_trainn = np.asarray(A.todense()) # convert Adjacency matrix to numpy array

data_full= np.asarray(A_full.todense())

#x_train = celegans("~/data")
G = nx.read_edgelist("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Data/network_excluded.edgelist",create_using=nx.DiGraph()) # read and parse edgelist to (networkx) graph
A = nx.adjacency_matrix(G) # make Adjacency matrix
#x_trainn = np.asarray(A.todense()) # convert Adjacency matrix to numpy array

data= np.asarray(A.todense())
# In[24]:

x_train = data[:500,:500]


#x_test = data[20:40,20:40]
"""

"""
#Drawing graph (heavy!)
G=nx.from_numpy_matrix(x_train)
nx.draw(G, node_size = 5)  # networkx draw()
plt.draw()
plt.show()
"""


N = x_train.shape[0]  # number of data points
K = 5  # latent dimensionality

#scale_z = InverseGamma(tf.ones([1,K])*1e-3, tf.ones([1,K])*1e-3)

#scale_b = InverseGamma(1e-3, 1e-3) #Gelman 2006

z2 = Normal(loc = tf.zeros([N, K]), scale = tf.ones([N, K]))

z1 = Normal(loc = tf.zeros([N, K]), scale = tf.ones([N, K]))

b = Normal(loc = tf.zeros(1), scale = tf.ones(1))


# Calculate N x N distance matrix.
# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2], and tile
# it to create N identical rows.
pi1 = tf.tile(tf.reduce_sum(tf.pow(z1, 2), 1, keep_dims=True), [1, N])
pi2 = tf.tile(tf.reduce_sum(tf.pow(z2, 2), 1, keep_dims=True), [1, N])

# 2. Create a N x N matrix where entry (i, j) is ||z_i||^2 + ||z_j||^2
# - 2 z_i^T z_j.
pi = pi1 + tf.transpose(pi2) - 2 * tf.matmul(z1, z2, transpose_b=True)
# 3. minus pairwise distances and make rate along diagonals to
# be close to zero.
pi = -tf.sqrt(pi + tf.diag(tf.zeros(N) + 1e3))

pi = tf.sigmoid(pi + b) #med bias (mange eller få links)

x = Bernoulli(pi)


# In[11]:
"""
G=nx.from_numpy_matrix(x.eval())
nx.draw(G, node_size = 5)  # networkx draw()
plt.draw()
plt.show()
"""

# In[26]:
#Ed.map er dét Morten har snakket om.
#Kør enten denne eller [51]:
inference = ed.MAP([z1,z2,b], data={x: x_train})


# In[51]:
#Modellen  med KL-divergens:
#tf.reset_default_graph()
qz1 = Normal(loc=tf.get_variable("qz1/loc", [N, K]),
             scale=tf.nn.softplus(tf.get_variable("qz1/scale", [N, K])))
qz2 = Normal(loc=tf.get_variable("qz2/loc", [N, K]),
             scale=tf.nn.softplus(tf.get_variable("qz2/scale", [N, K])))
qb = Normal(loc=tf.get_variable("qb/loc", 1),
             scale=tf.nn.softplus(tf.get_variable("qb/scale", 1)))
inference = ed.KLqp({z1: qz1, z2: qz2, b: qb}, data={x: x_train})


# In[27]:
#Kør denne inferens efter en af de to modeller:
inference.run(n_iter=500)

tf.reduce_max(pi).eval()
tf.reduce_min(pi).eval()

prob=pi.eval()                                  #Defining all probabilities in an array
probs=prob[x_train>0]                           #Taking all probabilities where links (=1) exists in training data
np.mean(probs)                                  #Taking mean of the probabilities

np.max(probs)
np.min(probs)

x_trainTF=tf.convert_to_tensor(x_train, np.float32)

met=tf.metrics.auc(x_trainTF,pi,num_thresholds=200,curve='ROC')
met[0].eval()

# In[29]:

#b_samples = b.sample(10)[:, 0].eval()


# In[30]:

#b_samples


# In[5]:
######
#Kør ikke
inference.initialize(n_iter=2000,logdir='log')

tf.global_variables_initializer().run()

info_loss = np.zeros(2000)


for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  info_loss[_] = info_dict['loss']

inference.finalize()


# In[8]:

plt.plot(info_loss)
plt.show()


# In[ ]:
#Prediction:
x_post = ed.copy(x, {z1: qz1, z2: qz2, b: qb})

x_test = tf.cast(x_test, tf.int32)
ed.evaluate('log_likelihood', data={x_post: x_test})

aa=x_post
