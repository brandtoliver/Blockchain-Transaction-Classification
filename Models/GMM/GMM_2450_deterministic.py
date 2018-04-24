

# exercise 11.1.5
from pylab import *
from scipy.io import loadmat
from sklearn.mixture import GMM
from sklearn import cross_validation
import os
import matplotlib.pyplot as plt
import pandas as pd



#Self-defined data:
currDir = os.getcwd()                                                   #Defining the current directory
fileSep = os.sep                                                        #Defining filesep
file_path = os.path.join (currDir + fileSep + "Data" + fileSep)         #Defining the folder where the files are located:

sub= os.path.join(file_path, 'user_rec.csv')                            #Path to subset
subset= pd.read_csv(sub, sep=",", header=0)                             #Reading subset
x_trainn=subset.values                                                   #Redefine x_train
#x_train= x_train[:,:13]
#x_train= x_trainn[4900:5000,]
#x_test= x_trainn[4800:4900,]
#N_testt= len(x_test)

#N = len(x_train)  # number of data points                               #Setting parameters - N is defined from the number of rows
#K = 50  # number of components                                           #Setting parameters - number of clusters
#D = x_train.shape[1]  # dimensionality of data                           #Setting parameters - dimension of data
#ed.set_seed(42)

X=x_trainn[4000:6000,]
N, M = X.shape


"""
# Load Matlab data file and extract variables of interest
os.chdir("/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/3. Semester/Data Mining og Machine Learning/02450Toolbox_R/02450Toolbox_Python/Data")
mat_data = loadmat('synth1.mat')
X = np.matrix(mat_data['X'])
y = np.matrix(mat_data['y'])
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)
"""

# Range of K's to try
KRange = range(1,300,20)
T = len(KRange)

covar_type = 'diag'     # you can try out 'diag' as well
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
"""
CVE:
array([[  22097.21451067],
       [  17091.70910842],
       [  18070.36720728],
       [  25933.60297168],
       [  40465.9620244 ],
       [  59316.40910018],
       [1801640.14579797],
       [1700389.04319111]])
       
       
KRange = range(1,100,5)       
array([[21933.98036559],
       [24306.40667864],
       [23703.27963108],
       [20683.21398683],
       [21576.32392127],
       [19365.27315322],
       [20638.72247836],
       [18713.44619907],
       [19931.80675331],
       [20572.91415356],
       [20281.21140433],
       [21806.79964106],
       [20531.75629275],
       [24111.22886518],
       [24867.72911918],
       [25316.22376798],
       [31384.37490616],
       [21987.90035339],
       [38472.10499597],
       [24250.58116138]])

"""

# Plot results
%matplotlib

plt.figure(1); hold(True)
plot(KRange, BIC)
plot(KRange, AIC)
plot(KRange, 2*CVE)
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
plt.show()
