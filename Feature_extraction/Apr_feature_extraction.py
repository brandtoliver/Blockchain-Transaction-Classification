#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 12:16:53 2018

@author: AndreasMotzfeldtJensen
"""

import pandas as pd
from pandas import*
import os
import csv
import numpy as np
from numpy import*
import matplotlib.pyplot as plt

# Reading the data
# 1. set the current working directory

#Defining the current directory
currDir = os.getcwd()
#Defining filesep
fileSep = os.sep
#Defining the folder where the files are located:
file_path = os.path.join (currDir + fileSep + "Documents" + fileSep + "blockchain_new_2013" + fileSep)

#Define path to files

txin = os.path.join(file_path, 'txin_new.csv') 
txout = os.path.join(file_path, 'txout_new.csv') 
txdegree = os.path.join(file_path, 'txdegree_new.csv') 
txtime= os.path.join(file_path, 'txtime.csv')
contraction = os.path.join(file_path, 'contraction.csv')
tx= os.path.join(file_path, 'tx_new.csv')

#Reading txin file:
txin = pd.read_csv(txin, sep=",")
#Reading txout file:
txout = pd.read_csv(txout, sep=",")
#Reading txdegree file:
txdegree = pd.read_csv(txdegree, sep=",")
#Readning txtime file:
txtime = pd.read_csv(txtime, sep="\t", header=None)
txtime.columns=['tx_ID','unixtime']
#Reading the tx file
tx= pd.read_csv(tx, sep=",")

# We do the feature extraction aggregating origin variables based on the userID
#Create the feature: number of transactions

# Starting with the txin file
# First we create position period
# Finder max txid for each address ID and attach unixtime
txin2=(txin.groupby('addrID')
    .agg({'tx_id':['max']})
    .reset_index()
    .rename(columns={'addrID':'value'}))
txin2.columns=['addrID','max_txid']
txin3=txin2.join(txtime.set_index('tx_ID')['unixtime'],on='max_txid')
txin3.columns=['addrID','max_txid','unixtime_max']
# Find min txid for each address ID and attach unixtime
txout2=(txout.groupby('addrID')
    .agg({'tx_id':['min']})
    .reset_index()
    .rename(columns={'addrID':'value'}))
txout2.columns=['addrID','min_txid']
txout3=txout2.join(txtime.set_index('tx_ID')['unixtime'],on='min_txid')
txout3.columns=['addrID','min_txid','unixtime_min']
# Join txin3 and txout3 to create the feature: pos_period (minutes), by subtracting min unixtime from max unixtime for each address ID 
txtime1=txout3.join(txin3.set_index('addrID')['unixtime_max'],on='addrID')
txtime1['pos_period']=(txtime1.unixtime_max-txtime1.unixtime_min)

# Then we create active duration
txin4=(txin.groupby('userID')
    .agg({'tx_id':['max']})
    .reset_index()
    .rename(columns={'userID':'value'}))
txin4.columns=['userID','max_txid']
txin5=txin4.join(txtime.set_index('tx_ID')['unixtime'],on='max_txid')
txin5.columns=['userID','max_txid','unixtime_max']
# Find min txid for each address ID and attach unixtime
txout4=(txout.groupby('userID')
    .agg({'tx_id':['min']})
    .reset_index()
    .rename(columns={'userID':'value'}))
txout4.columns=['userID','min_txid']
txout5=txout4.join(txtime.set_index('tx_ID')['unixtime'],on='min_txid')
txout5.columns=['userID','min_txid','unixtime_min']
# Join txin3 and txout3 to create the feature: pos_period (minutes), by subtracting min unixtime from max unixtime for each address ID 
txtime2=txout5.join(txin5.set_index('userID')['unixtime_max'],on='userID')
txtime2['active_duration']=(txtime2.unixtime_max-txtime2.unixtime_min)

#Attach position period to address id
txout = txout.join(txtime1.set_index('addrID')['pos_period'],on='addrID')

#Replace NaN values in position period with real values
txout['pos_period2']=1388256310 - txtime2.unixtime_min
txout['pos_period'].fillna(txout.pos_period2, inplace=True)


# Then create the features: Tot_value_out, Mean_value_out, Tot_unixtime and Mean_unixtime
txin1=txin.groupby(['userID','nr_pseudonyms'],as_index=False).agg({"value":[sum,mean]})
txin1=txin1.reset_index(drop=True)
txin1.columns=['userID','nr_pseudonyms','tot_value_out','mean_value_out']

# Next create the features: Tot_value_in, Mean_value_in, Tot_unixtime and Mean_unixtime
txout1=txout.groupby(['userID','nr_pseudonyms'],as_index=False).agg({"value":[sum,mean],"pos_period":[mean]})
txout1=txout1.reset_index(drop=True)
txout1.columns=['userID','nr_pseudonyms','avg_pos_period','tot_value_in','mean_value_in']

# At last we creat tot and avg for in and out degree
txdegree['nr_pseudonyms']=txdegree.groupby('userID')['addrID'].transform('count')
txdegree=txdegree.groupby('userID',as_index=False).agg({"in_degree":[sum, mean],"out_degree":[sum,mean]})
txdegree=txdegree.reset_index(drop=True)
txdegree.columns=['userID','tot_in_degree','avg_in_degree','tot_out_degree','avg_out_degree']

# Join all features together
user_rec = txout1.join(tx.set_index('userID')['nr_transactions'],on='userID')
user_rec = user_rec.join(txin1.set_index('userID')['tot_value_out'],on='userID')
user_rec = user_rec.join(txin1.set_index('userID')['mean_value_out'],on='userID')

user_rec = user_rec.join(txdegree.set_index('userID')['tot_out_degree'],on='userID')
user_rec = user_rec.join(txdegree.set_index('userID')['avg_out_degree'],on='userID')
user_rec = user_rec.join(txdegree.set_index('userID')['tot_in_degree'],on='userID')
user_rec = user_rec.join(txdegree.set_index('userID')['avg_in_degree'],on='userID')

where_NaN = isnan(user_rec)
user_rec[where_NaN ] = 0

user_rec = user_rec.join(txtime2.set_index('userID')['active_duration'],on='userID')
user_rec = user_rec.join(txtime2.set_index('userID')['unixtime_min'],on='userID')

#Replace NaN values in active duration with there real active duration
user_rec['active_duration2']=1388256310 - user_rec.unixtime_min
user_rec['active_duration'].fillna(user_rec.active_duration2, inplace=True)
user_rec = user_rec.drop('active_duration2',1)
user_rec = user_rec.drop('unixtime_min',1)

#Re-evaluate features by the correlation matrix
normalized_user_rec=(user_rec.mean())/user_rec.std()
corr_ur = user_rec.corr()

#Drop all total" features
user_rec1 = user_rec.drop('tot_value_in',1)
user_rec1 = user_rec1.drop('tot_value_out',1)
user_rec1 = user_rec1.drop('tot_in_degree',1)
user_rec1 = user_rec1.drop('tot_out_degree',1)
user_rec1 = user_rec1.drop('userID',1)

normalized_user_rec=(user_rec1.mean())/user_rec1.std()
corr_ur = user_rec1.corr()

corr_ur.to_csv("corr_matrix1.csv")

df=user_rec1[:12136804]
df.to_csv('user_rec',index=None)

#Filter by number of pseudonyms
df = user_rec[user_rec.nr_pseudonyms > 2]


# DATA EXPLORATION
#First we standardize the dataframe

df = df.drop('userID',1)
df = df.drop('nr_txin',1)
df = df.drop('nr_txout',1)

normalized_user_rec=(df.mean())/user_rec.std()
corr_df = df.corr()

corr_df.to_csv("corr_matrix1.csv")

scatter_matrix(normalized_user_rec,figsize=(16,12),alpha=0.3)
