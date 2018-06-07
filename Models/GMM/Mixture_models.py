#%matplotlib inline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
import os
import pandas as pd

from edward.models import Categorical

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, RandomVariable, Bernoulli)
from tensorflow.contrib.distributions import Distribution

plt.style.use('ggplot')

"""
Notes til Bernouli: logits: An N-D `Tensor` representing the log-odds of a `1` event. Each
 |          entry in the `Tensor` parametrizes an independent Bernoulli distribution
 |          where the probability of an event is sigmoid(logits). Only one of
 |          `logits` or `probs` should be passed in.


"""

#Reading data:
subset= pd.read_csv('/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Data/user_rec.csv', sep=",", header=0)
x_trainn=subset.values                                                  #Redefine x_train into an array instead of dataframe

#Standardizing:
from sklearn import preprocessing
X= preprocessing.scale(x_trainn)

"""
#Subset:
x_train= X[0:50,]                                                       #Defining training data
x_test= X[50:100,]                                                      #Defining test data
N_testt= len(x_test)                                                    #Defining the length of the test-set
"""

#"""
#Full dataset:
x_train= X[0:5000,]                                                  #Defining training data
x_test= X[5000:10000,]                                                   #Defining test data
N_testt= len(x_test)                                                    #Defining the length of the test-set
#"""
#tf.reset_default_graph()                                                #Necessary for loop


#Next:
#Kør clusters 10,50,250,500,1000,2000,5000
#Noter alle test-likelihoods, og sammenlign.
#Derefter indsnævres intervallet.

N = len(x_train)  # number of data points                               #Setting parameters - N is defined from the number of rows
K = 1000  # number of components                                           #Setting parameters - number of clusters
D = x_train.shape[1]  # dimensionality of data                           #Setting parameters - dimension of data
ed.set_seed(42)



#Model:
pi = Dirichlet(tf.ones(K))                                              #Defining prior: Dirichlet tager nogle X'er som input, der definerer sandsynligheden for hver
                                                                        # af de k clustre. Den defineres her til 1, hviklet angiver samme ssh for alle clustre.
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)                    #Defining mu: definerer normalfordelingen, som bruges til at vise hvor clustrene ligger
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)          #Defining sigma squared: definerer covarians-matricen, bruges til at beskrive clustrene;
                                                                        # hældning + densitet
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},       #Given a certain hidden component, it decides the conditional distribution for the observed
                 MultivariateNormalDiag,                                #veriable
                 sample_shape=N)
z = x.cat                                                               #z is now defined as the component prescribed to the observed variable x.




#Inference:                                                             #a conclusion reached on the basis of evidence and reasoning
T = 1000                                                                #number of MCMC samples
qpi = Empirical(                                                        #Emperical is just a sample of a set, which is good enough representation of the whole set.
    tf.get_variable(                                                    #Gets an existing variable with these parameters or create a new one.
    "qpi/params", [T, K],                                               #Setting shape to be Number of MCMC samples times number of components
    initializer=tf.constant_initializer(1.0 / K)))                      #Initializer that generates tensors with constant values
qmu = Empirical(tf.get_variable(                                        #Defining mu the same way
    "qmu/params", [T, K, D],
    initializer=tf.zeros_initializer()))
qsigmasq = Empirical(tf.get_variable(                                   #Defining sigma the same way
    "qsigmasq/params", [T, K, D],
    initializer=tf.ones_initializer()))
qz = Empirical(tf.get_variable(                                         #Defining z the same way
    "qz/params", [T, N],
    initializer=tf.zeros_initializer(),
    dtype=tf.int32))
"""
We now have the following two measures of (q)pi, (q)mu, (q)sigmasq and (q)z, where:
qpi is the Empirical distribution.
pi is the Dirichlet distribution. 
"""

#Running Gibbs Sampling:
inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},      #Starter Gibbs sampling proceduren, hvor vi har posterior-fordelingen, samt vores emperiske
                     data={x: x_train})                                 # distributions-funktion hvor den bruger x_train som værende vores data.
inference.initialize()                                                   #Initialiserer

sess = ed.get_session()
tf.global_variables_initializer().run()

t_ph = tf.placeholder(tf.int32,[])                                      #Definerer en tom placeholder
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)            #Her defineres en dynamisk reduktion af cluster means.

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("\nInferred cluster means:")
    print(sess.run(running_cluster_means, {t_ph: t - 1}))               #Her printes den opdaterede cluster mean for hver gang der printes.

#Criticism:
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)                                                 #Tager en sample af qmu
sigmasq_sample = qsigmasq.sample(100)                                       #Tager en sample af qsigmasq
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,                      #Ganger disse samples ind på vores normalfordelte posterior-fordeling
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])  #x-broadcasted

#x_post = tf.cast(x_post, tf.float64)                                       #Laves om til float64
x_broadcasted = tf.cast(x_broadcasted, tf.float32)                          #Laves om til float64


# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)                                   #log_prob(value, name='log_prob') method of abc.ParamMixture instance Log probability density/mass function.
log_liks = tf.reduce_sum(log_liks, 3)                                       #Optimization step for sum
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()

#Training log likelihood:
x_neg_log_prob = (-tf.reduce_sum(x_post.log_prob(x_broadcasted)) /
                    tf.cast(tf.shape(x_broadcasted)[0], tf.float32))
x_neg_log_prob.eval()

"""
K=10:
liks: 500406.7
time: 246s

K=50:
liks: 577860.44
time: 1866s

K=250:
liks: 760915.6
time: 6809s

K=500:
time: 16176s

K=1000:
liks: 
time: 33708s 
"""

#Posterior predictive distribution:
x_post = ed.copy(x, {pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz})

#Evaluate the posterior predictive (test-data prediction from model):
x_test = tf.cast(x_test, tf.float32)
ed.evaluate('log_likelihood', data={x_post: x_test})

"""
K=10:
-8517.825

K=50:
-1712.7479

K=250:
-337.85638


"""




#End session from for-loop
#ed.get_session().close()                                                   #Necessary for loop
#tf.Session.reset()


#Plot the clusters:
#plt.scatter(x_train[:, 5], x_train[:, 6], c=clusters, cmap=cm.bwr)
#plt.axis([-5, 100, -5, 100])
#plt.title("Predicted cluster assignments")
#plt.show()
