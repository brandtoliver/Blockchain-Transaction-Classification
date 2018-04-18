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

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, RandomVariable, Bernoulli)
from tensorflow.contrib.distributions import Distribution

plt.style.use('ggplot')


""" Notes til data: ikke standardiseret, hvilket skal huskes før det giver mening
"""

"""
Notes til Bernouli: logits: An N-D `Tensor` representing the log-odds of a `1` event. Each
 |          entry in the `Tensor` parametrizes an independent Bernoulli distribution
 |          where the probability of an event is sigmoid(logits). Only one of
 |          `logits` or `probs` should be passed in.


"""

"""
#Building dataset:
def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)                                #N*2 matrix all zeros
  for n in range(N):
    k = np.argmax((np.random.multinomial1, pi))                         #np.random.multinomial, gives a vector, of which of the two (0.4 , 0.6) that was chosen in 1 draw: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.multinomial.html
                                                                        #np.argmax gives the index of the biggest number from the output array. In this case, it will
                                                                        #always be 1, because it only makes 1 draw, and therefore none will be chosen 2 or more.

    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))   #Making row n into a random draw from the multivariate normal distribution.
                                                                        #The multivariate normal distribution is a generalisation of the 1-Dimensional normal distribution.
                                                                        #The row is therefore a draw of a two points, taken from the random multivariate ND.
                                                                        #np.random.multivariate_normal is specified by its mean and covariance matrix (mus and stds): https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.multivariate_normal.html
                                                                        #Second argument makes a diagonal matrix from stds[1]

  return x                                                              #x is now a N*2 matrix, composed of random drawn datapoints from the multivariate distribution


N = 500  # number of data points                                        #Setting parameters - N is defined from the number of rows
K = 2  # number of components                                           #Setting parameters - number of clusters
D = 2  # dimensionality of data                                         #Setting parameters - dimension of data
ed.set_seed(42)

x_train = build_toy_dataset(N)                                          #Creating toy-data of type: numpy.ndarray

#Plotting x_train data. Remember x_train holds two positional datapoints, which is why x_train[:,0] and x_train[:,1] gets plotted.
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.title("Simulated dataset")
plt.show()
"""
"""
The plot shows exactly the intention of line 29. Creating seemingly random draws from a multivariate normal distribution with a mean randomly chosen
from mus through pi, but with a bias to mus[1] because this one hold the largest probability c.f. pi. 
"""


#Defining the current directory
currDir = os.getcwd()
#Defining filesep
fileSep = os.sep
#Defining the folder where the files are located:
file_path = os.path.join (currDir + fileSep + "Documents" + fileSep + "blockchain_new_2013" + fileSep)        #Defining the folder where the files are located:

sub= os.path.join(file_path, 'user_rec.csv')                     #Path to subset
subset= pd.read_csv(sub, sep=",")                             #Reading subset
x_train=subset.values                                                   #Redefine x_train
#x_train= x_train[:,:13]
x_train= x_train[4000:5000,]


N = len(x_train)  # number of data points                               #Setting parameters - N is defined from the number of rows
K = 5  # number of components                                           #Setting parameters - number of clusters
D = x_train.shape[1]  # dimensionality of data                           #Setting parameters - dimension of data
ed.set_seed(42)



""" PRIORS """

#Model:
pi = Dirichlet(tf.ones(K))                                              #Defining prior
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)                    #Defining mu
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)          #Defining sigma squared
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},       #Given a certain hidden component, it decides the conditional distribution for the observed
                 MultivariateNormalDiag,                                #veriable
                 sample_shape=N)
z = x.cat                                                               #z is now defined as the component prescribed to the observed variable x.

""" PRIORS """


#Inference:                                                             #a conclusion reached on the basis of evidence and reasoning
T = 10000                                                                #number of MCMC samples
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
"""
Originalt data:
x_post
Out[7]: <ed.RandomVariable 'Normal_1/' shape=(500, 100, 2, 2) dtype=float32>
x_broadcasted
Out[8]: <tf.Tensor 'Tile:0' shape=(500, 100, 2, 2) dtype=float32>


Med subset:
x_post
Out[14]: <ed.RandomVariable 'Normal_1/' shape=(500, 100, 2, 15) dtype=float32>
x_broadcasted
Out[15]: <tf.Tensor 'Tile:0' shape=(500, 100, 2, 15) dtype=float64>
"""

#x_post = tf.cast(x_post, tf.float64)                                        #Laves om til float64
x_broadcasted = tf.cast(x_broadcasted, tf.float32)                          #Laves om til float64

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)                                       #Optimization step for sum
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()

#Plot the clusters:
plt.scatter(x_train[:, 5], x_train[:, 6], c=clusters, cmap=cm.bwr)
plt.axis([-5, 65, -5, 45])
plt.title("Predicted cluster assignments")
plt.show()


#Next: Find ud af hvorfor når du plotter andre variable end x_train[:,0] og x_train[:,1] så kommer der ikke farver på plottet
#Måske det bare er fordi du ikke zoomer nok ud?

"""
Finde ud af hvordan vi finder det optimale antal clustre


"""
