
# exercise 2.1.1
import numpy as np
import xlrd
import os
import pandas as pd

#definerer samme tabel med vores data:
doc2= pd.read_csv('/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Data/subset_version1.csv')
doc2= doc2.drop(['Unnamed: 0', 'addrID', 'userID', 'pos_period'], axis=1)
doc2= doc2.iloc[4500:5500]

"""
Datastrukturen fra Mortens data:
# Load xls sheet with data
doc = xlrd.open_workbook('/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/3. Semester/Data Mining og Machine Learning/02450Toolbox_R/02450Toolbox_Python/Data/nanonose.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,3,11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0,2,92)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(5)))

y = np.mat([classDict[value] for value in classLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((90,8)))
for i, col_id in enumerate(range(3,11)):
    X[:,i] = np.mat(doc.col_values(col_id,2,92)).T
"""


#Definerer samme datastruktur med vores data.
attributeNames = list(doc2)

#Samme datastruktur
classLabels = ['BC_tx']*1000
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(5)))


# Extract vector y, convert to NumPy matrix and transpose

y = np.mat([classDict[value] for value in classLabels]).T

#Strukturerer X2 på samme måde som mortens data
X = doc2.as_matrix()
#Standardisering:
from sklearn import preprocessing
X= preprocessing.scale(X)   #http://scikit-learn.org/stable/modules/preprocessing.html

#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)


X = np.asmatrix(X)
#X2 = np.mat(np.empty((5000,14)))
#for i, col_id in enumerate(range(0,14)):
#    X2[:,i] = np.mat(doc2.col_values(col_id,0,5000)).T



# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)






# exercise 2.1.2
#from ex2_1_1 import *
# (requires data structures from ex. 2.1.1)

from pylab import *

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type and need to be cast to array.
figure()
X = array(X)
plot(X[:,i], X[:,j], 'o');

# %%
# Make another more fancy plot that includes legend, class labels,
# attribute names, and a title.
f = figure()
f.hold()
title('NanoNose data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
    plot(X[class_mask,i], X[class_mask,j], 'o')

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()





# exercise 2.1.3
# (requires data structures from ex. 2.2.1)
#from ex2_1_1 import *

from pylab import *
import scipy.linalg as linalg


# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
show()





V = mat(V).T

# Project the centered data onto principal component space
Z = Y * V


# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
f.hold()
title('NanoNose data: PCA')
Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o')
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()
