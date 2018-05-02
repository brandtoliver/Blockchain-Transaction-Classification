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

x_train= M_zeroing

M_fullDataA= M_fullData
M_onesRemovedA=M_onesRemoved
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
K = 200  # latent dimensionality

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

"""
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
"""

# In[27]:
#Kør denne inferens efter en af de to modeller:
inference.run(n_iter=500)


#Getting pi for M_onesRemoved:
#type(M_onesRemovedA)
M_onesRemovedtf= tf.convert_to_tensor(M_onesRemovedA, np.float32)
pi_onesRemoved= tf.multiply(M_onesRemovedtf,pi)
pi_onesRemoved_matrix= pi_onesRemoved.eval()
pi_onesRemoved_array=np.asarray(pi_onesRemoved_matrix).reshape(-1)
pi_onesRemoved_array=pi_onesRemoved_array[pi_onesRemoved_array!=0]          #All probabilities for ones_removed
nrOfZeros=len(pi_onesRemoved_array)

#Getting pi for M_fulldata
#type(M_fullDataA)
M_fullDataA= np.asarray(pi_onesRemoved_matrix).reshape(-1)
where_zero=np.where(M_fullDataA==0)[0]
where_zero_index=np.random.choice(where_zero,nrOfZeros)
pi_zeros= pi.eval()
pi_array= np.asarray(pi_zeros).reshape(-1)
pi_originalZeros=pi_array[where_zero_index]                                 #All probabilities for correct zeros

#Creating arrays with zeros and ones:
zeros=np.zeros(nrOfZeros)
ones=np.ones(nrOfZeros)
#Setting together:
y_test= np.concatenate((zeros, ones), axis=0)
p= np.concatenate((pi_originalZeros, pi_onesRemoved_array), axis=0)






##################################
##################################
#Inserting ROC-script (supposed to be imported):
from pylab import *
from scipy.io import loadmat
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
#from toolbox_02450 import rocplot, confmatplot

from sklearn import metrics

def rocplot(p, y):
    '''
    function: AUC, TPR, FPR = rocplot(p, y)
    ROCPLOT Plots the receiver operating characteristic (ROC) curve and
    calculates the area under the curve (AUC).

    Notice that the function assumes values of p are all distinct.


    Usage:
        rocplot(p, y)
        AUC, TPR, FDR = rocplot(p, y)

     Input:
         p: Estimated probability of class 1. (Between 0 and 1.)
         y: True class indices. (Equal to 0 or 1.)

    Output:
        AUC: The area under the ROC curve
        TPR: True positive rate
        FPR: False positive rate
    '''
    #ind = np.argsort(p,0)
    #x = y[ind].A.ravel()
    #FNR = np.mat(np.cumsum(x==1, 0, dtype=float)).T / np.sum(x==1,0)
    #TPR = 1 - FNR
    #TNR = np.mat(np.cumsum(x==0, 0, dtype=float)).T / np.sum(x==0,0)
    #FPR = 1 - TNR
    #onemat = np.mat([1])
    #TPR = np.bmat('onemat; TPR'); FPR = np.mat('onemat; FPR') # Don't get this line.
    #TPR = vstack( (np.ones(1), TPR))
    #FPR = vstack( (np.ones(1), FPR))

    #AUC = -np.diff(FPR,axis=0).T * (TPR[0:-1]+TPR[1:])/2
    #AUC = AUC[0,0]

    #%%
    fpr, tpr, thresholds = metrics.roc_curve(y.ravel(),p.ravel())
    #FPR = fpr
    #TPR = TPR
    #TPR
    AUC = metrics.roc_auc_score(y.ravel(), p.ravel())
    #%%
    plot(fpr, tpr, 'r', [0, 1], [0, 1], 'k')
    grid()
    xlim([-0.01,1.01]); ylim([-0.01,1.01])
    xticks(arange(0,1.1,.1)); yticks(arange(0,1.1,.1))
    xlabel('False positive rate (1-Specificity)')
    ylabel('True positive rate (Sensitivity)')
    title('Receiver operating characteristic (ROC)\n AUC={:.3f}'.format(AUC))


    return AUC, tpr, fpr

rocplot(p, y_test)





#Statistics from Adjacency matrixes:
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



#####################
#####################
#Not used:
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
