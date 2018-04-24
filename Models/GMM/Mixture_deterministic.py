# exercise 11.1.5
from pylab import *
from scipy.io import loadmat
from sklearn.mixture import GMM
from sklearn import cross_validation
import os
import matplotlib.pyplot as plt

# Load Matlab data file and extract variables of interest
os.chdir("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/3. Semester/Data Mining og Machine Learning/02450Toolbox_R/02450Toolbox_Python/Data")
mat_data = loadmat('synth1.mat')
X = np.matrix(mat_data['X'])
y = np.matrix(mat_data['y'])
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)


# Range of K's to try
KRange = range(1,100,10)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
reps = 3                # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,1))
AIC = np.zeros((T,1))
CVE = np.zeros((T,1))

# K-fold crossvalidation
CV = cross_validation.KFold(N,10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}\n'.format(K))

        # Fit Gaussian mixture model
        gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X)

        # Get BIC and AIC
        BIC[t,0] = gmm.bic(X)
        AIC[t,0] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV:

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score(X_test).sum()


# Plot results
%matplotlib

plt.figure(1); hold(True)
plot(KRange, BIC)
plot(KRange, AIC)
plot(KRange, 2*CVE)
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
plt.show()
GMMGMedsfdsfsdfsdf
