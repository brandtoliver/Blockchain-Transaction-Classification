import pandas as pd
import numpy as np

netw= pd.read_csv('/Users/MartinJohnsen/Documents/Martin Johnsen/SAS/6. Semester/Bachelorprojekt/PyCharm/Data/network_10k_w_unix.csv', sep=",", header=0)

"""
#Finding x% last made links:
sortt=netw.sort(columns='unixtime', axis=0, ascending=True)
unix=int(np.round(len(sortt)*0.99))
sort=sortt.iloc[unix][1]
"""

#Editing data:
netw_uniq = netw[netw['userID_send'] != netw['userID_recv']]                                            #Removing all users who sent to themselves
netw_uniq= netw_uniq.drop_duplicates(subset=['userID_send','userID_recv'], keep='last', inplace=False)  #Removing identical pairs, keeping the last ones

#Finding x% last made links:
netw_uniq=netw_uniq.sort(columns='unixtime', axis=0, ascending=True)
unix=int(np.round(len(netw_uniq)*0.99))
sort=netw_uniq.iloc[unix][1]

#Creating subset:
data= netw_uniq                                                                                         #Creating a subset

#Editing matrix:
#from: https://stackoverflow.com/questions/49095067/how-to-convert-weighted-edge-list-to-adjacency-matrix-in-python
data.drop(['tx_id'], axis=1)                                                                            #Dropping column w. tx_id
cols= ['userID_send','userID_recv','unixtime']                                                          #Rearraning columns
data=data[cols]                                                                                         #Implementing rearranging
data.rename(index=str, columns={"userID_send": "0", "userID_recv": "1", "unixtime": "2"})               #Renaming columns

nodes = data.iloc[:, 0].tolist() + data.iloc[:, 1].tolist()

nodes = sorted(list(set(nodes)))

nodes = [(i,nodes[i]) for i in range(len(nodes))]

for i in range(len(nodes)):
    data = data.replace(nodes[i][1], nodes[i][0])

from scipy.sparse import coo_matrix
M = coo_matrix((data.iloc[:,2], (data.iloc[:,0],data.iloc[:,1])), shape=(len(nodes), len(nodes)))
M_zeroing = M.todense()
M_fullData = M.todense()
M_originalZero= M.todense()

#M_zeroing = M                                      #Defining adjecency for zeros
#M_fullData = M                                   #Defining adjecency for non-zeros

#Creating correct indexes in adjecency matrix:
#Original dataset, with all links in adjacency matrix.
M_fullData[M_fullData>0]=1                        #Making all non-zeros into ones
OnesBeforeZeroing=(M_fullData>0).sum()                              #Number of 1'nes before sort= 125944

#Before touching:
(M_fullData>sort).sum()  #=1258
(M_fullData==0).sum()    #=99255017
(M_fullData>0).sum()    #= 125944
#Stats are same afterwards



#Adjecency matrix with only 99% of links:
M_zeroing[M_zeroing>sort]= 0                          #Making all the last 1% into zeros
M_zeroing[M_zeroing>0]=1                              #Making all non-zeros into ones
OnesAfterZeroing=(M_zeroing>0).sum()                              #Number of 1'nes after sort =124686

#Before running 61-63:
(M_zeroing>sort).sum()  #=1258
(M_zeroing==0).sum()    #=99255017
(M_zeroing>0).sum()     #=125944
#After:
(M_zeroing>sort).sum()  #=0
(M_zeroing==0).sum()    #=99256275 (=99255017+1258)
(M_zeroing>0).sum()     #=124686 (125944-1258)


percentage =OnesAfterZeroing/OnesBeforeZeroing                                #Checking the number fits with 1%


#Creating dataset consisting of only ones we have removed:
M_onesRemoved= M_fullData-M_zeroing
(M_onesRemoved>0).sum() #=1258


#Creating a dataset consisting of only the correct zero's - same amount as correct ones (=1258)
(M_originalZero==0).sum()

M_originalZero

v = M_originalZero == 0
M_originalZero.iloc[v, 'value'] = np.random.choice((1, 0), v.sum(), p=(.9, .1))



idx = M_originalZero.index[M_originalZero.value==0]
M_originalZero.loc[np.random.choice(idx, size=idx.size/10, replace=False)].value = 1
