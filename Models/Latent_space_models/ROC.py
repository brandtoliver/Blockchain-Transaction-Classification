# exercise 9.1.1

from pylab import *
from scipy.io import loadmat
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot

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
    fpr, tpr, thresholds = metrics.roc_curve(y.A.ravel(),p.A.ravel())
    #FPR = fpr
    #TPR = TPR
    #TPR
    AUC = metrics.roc_auc_score(y.A.ravel(), p.A.ravel())
    #%%
    plot(fpr, tpr, 'r', [0, 1], [0, 1], 'k')
    grid()
    xlim([-0.01,1.01]); ylim([-0.01,1.01])
    xticks(arange(0,1.1,.1)); yticks(arange(0,1.1,.1))
    xlabel('False positive rate (1-Specificity)')
    ylabel('True positive rate (Sensitivity)')
    title('Receiver operating characteristic (ROC)\n AUC={:.3f}'.format(AUC))


    return AUC, tpr, fpr


def confmatplot(y_true, y_est):
    '''
    The function plots confusion matrix for classification results.

    Usage:
        confmatplot(y_true, y_estimated)

     Input:
         y_true: Vector of true class labels.
         y_estimated: Vector of estimated class labels.
    '''
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).ravel(); y_est = np.asarray(y_est).ravel()
    C = unique(y_true).shape[0]
    cm = confusion_matrix(y_true, y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    imshow(cm, cmap='binary', interpolation='None');
    colorbar(format='%.2f')
    xticks(range(C)); yticks(range(C));
    xlabel('Predicted class'); ylabel('Actual class');
    title('Confusion matrix (Accuracy: {:}%, Error Rate: {:}%)'.format(accuracy, error_rate));

"""
# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
X = np.matrix(mat_data['X'])
y = np.matrix(mat_data['y'], dtype=int)
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# K-fold crossvalidation
K = 2
CV = cross_validation.StratifiedKFold(y.A.ravel().tolist(),K)

k=0
for train_index, test_index in CV:

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index,:]
    X_test, y_test = X[test_index,:], y[test_index,:]

    logit_classifier = LogisticRegression()
    logit_classifier.fit(X_train, y_train.A.ravel())

    y_test_est = np.mat(logit_classifier.predict(X_test)).T
    p = np.mat(logit_classifier.predict_proba(X_test)[:,1]).T

    figure(k)
    rocplot(p, y_test)

    figure(k+1)
    confmatplot(y_test,y_test_est)

    k+=2
    
show()    
"""
