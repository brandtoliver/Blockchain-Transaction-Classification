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

#Reading subset:
currDir = os.getcwd()                                                   #Defining the current directory
fileSep = os.sep                                                        #Defining filesep
file_path = os.path.join (currDir + fileSep + "Data" + fileSep)         #Defining the folder where the files are located:

sub= os.path.join(file_path, 'user_rec.csv')                            #Path to subset
subset= pd.read_csv(sub, sep=",", header=0)                             #Reading subset
x_trainn=subset.values                                                   #Redefine x_train

#Standardizing:
from sklearn import preprocessing
X= preprocessing.scale(x_trainn)

#x_train= x_train[:,:13]
x_train= X[4900:5000,]
x_test= X[4800:4900,]
N_testt= len(x_test)

N = len(x_train)  # number of data points                               #Setting parameters - N is defined from the number of rows
K = 10  # number of components                                           #Setting parameters - number of clusters
D = x_train.shape[1]  # dimensionality of data                           #Setting parameters - dimension of data
ed.set_seed(42)





#Model:
pi = Dirichlet(tf.ones(K))                                              #Defining prior
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)                    #Defining mu
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)          #Defining sigma squared
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},       #Given a certain hidden component, it decides the conditional distribution for the observed
                 MultivariateNormalDiag,                                #veriable
                 sample_shape=N)
z = x.cat                                                               #z is now defined as the component prescribed to the observed variable x.




#Inference:                                                             #a conclusion reached on the basis of evidence and reasoning
T = 500                                                                #number of MCMC samples
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
inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},
                     data={x: x_train})
inference.initialize()

sess = ed.get_session()
tf.global_variables_initializer().run()

t_ph = tf.placeholder(tf.int32, [])
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("\nInferred cluster means:")
    print(sess.run(running_cluster_means, {t_ph: t - 1}))




#Criticism:
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)
sigmasq_sample = qsigmasq.sample(100)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

#x_post = tf.cast(x_post, tf.float64)                                        #Laves om til float64
x_broadcasted = tf.cast(x_broadcasted, tf.float32)                          #Laves om til float64


# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)                                       #Optimization step for sum
log_liks = tf.reduce_mean(log_liks, 1)


#1
x_neg_log_prob = (-tf.reduce_sum(x_post.log_prob(x_broadcasted)) /
                    tf.cast(tf.shape(x_broadcasted)[0], tf.float32))
x_neg_log_prob.eval()

#2
a=tf.reduce_sum(log_liks)
a.eval()

"""
K=10:
-12720.996 (stand)

K=50:
-70788.31 (stand)

K=500:
-1.1097861e+26

"""


# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()



#Criticism
inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},
                     data={x: x_train})


x_post = ed.copy(x, {pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz})

x_test = tf.cast(x_test, tf.float32)
ed.evaluate('log_likelihood', data={x_post: x_test})
"""
K=10:
-148.55351

K=50:
-29.601395 (stand)

-2.8144713e+20

K=500:
-5.6210855e+18

"""

#Plot the clusters:
#plt.scatter(x_train[:, 5], x_train[:, 6], c=clusters, cmap=cm.bwr)
#plt.axis([-5, 100, -5, 100])
#plt.title("Predicted cluster assignments")
#plt.show()


#Next: Find ud af hvorfor når du plotter andre variable end x_train[:,0] og x_train[:,1] så kommer der ikke farver på plottet
#Måske det bare er fordi du ikke zoomer nok ud?

"""
Finde ud af hvordan vi finder det optimale antal clustre


"""


###Model predictions:

# create local posterior factors for test data, assuming test data
# has N_test many data points
qz_test = Categorical(logits=tf.Variable(tf.zeros([N_testt, K])))

# run local inference conditional on global factors
inference_test = ed.Inference({z: qz_test}, data={x: x_test, pi: qpi, mu: qmu, sigmasq: qsigmasq})
inference_test.run()

# build posterior predictive on test data
x_post = ed.copy(x, {z: qz_test, beta: qbeta}})
ed.evaluate('log_likelihood', data={x_post: x_test})



{pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz}

x_post = ed.copy(x, {pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz})
