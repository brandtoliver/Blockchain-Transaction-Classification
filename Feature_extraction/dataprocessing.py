#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:13:35 2018

@author: AndreasMotzfeldtJensen
"""

import pandas as pd
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

txin = os.path.join(file_path, 'txin.csv') 
txout = os.path.join(file_path, 'txout.csv')
txin = os.path.join(file_path, 'txin_new.csv') 
txout = os.path.join(file_path, 'txout_new.csv') 
tx= os.path.join(file_path, 'tx.csv')
contraction = os.path.join(file_path, 'contraction.csv')
unique=os.path.join(file_path,'txedgeunique.csv')
txdegree=os.path.join(file_path, 'degree.csv')
txtime= os.path.join(file_path, 'txtime.csv')
network= os.path.join(file_path, 'network_new.csv')

#Reading contraction file:
contraction = pd.read_csv(contraction, sep="\t", header=None)
contraction.columns=['addrID', 'userID']

#Join userID and unixtime to txin and txout 

#Reading txin file:
txin = pd.read_csv(txin, sep="\t", header=None)
txin.columns=['tx_id', 'addrID', 'value']
#Reading txout file:
txout = pd.read_csv(txout, sep="\t", header=None)
txout.columns=['tx_id', 'addrID', 'value']
#Readning txtime file:
txtime = pd.read_csv(txtime, sep="\t", header=None)
txtime.columns=['tx_ID','unixtime']

#Join functions:
contraction['nr_pseudonyms']=contraction.groupby('userID')['addrID'].transform('count')
txin = txin.join(contraction.set_index('addrID')['userID'],on='addrID')
txin = txin.join(txtime.set_index('tx_ID')['unixtime'],on='tx_id')
txin = txin.join(contraction.set_index('addrID')['nr_pseudonyms'],on='addrID')
txout = txout.join(contraction.set_index('addrID')['userID'],on='addrID')
txout = txout.join(txtime.set_index('tx_ID')['unixtime'],on='tx_id')
txout = txout.join(contraction.set_index('addrID')['nr_pseudonyms'],on='addrID')

#Convert files to csv

dfin=txin[:65714232]
dfin.to_csv('txin_new',index=None)

dfout=txout[:73738345]
dfout.to_csv('txout_new',index=None)

# Join userID to in and out degree

#Reading txdegree file:
txdegree = pd.read_csv(txdegree, sep="\t",header=None)
txdegree.columns=['addrID','in_degree','out_degree']

#Join functions:
txdegree = txdegree.join(contraction.set_index('addrID')['userID'],on='addrID')

#Concert to csv

dfdegree=txdegree[:24575385]
dfdegree.to_csv('txdegree_new',index=None)

#Reading txin file:
txin = pd.read_csv(txin, sep=",")
#Reading txout file:
txout = pd.read_csv(txout, sep=",")
#Reading the tx file
tx= pd.read_csv(tx, sep="\t", header=None)
tx.columns=['tx_ID', 'Block_ID', 'nr_input', 'nr_output']

#Create the feature number of transactions
tx = tx.join(txin.set_index('tx_id')['userID'],on='tx_ID')
tx.columns=['tx_ID', 'Block_ID', 'nr_input', 'nr_output','userID1']
tx = tx.join(txout.set_index('tx_id')['userID'],on='tx_ID')
tx.columns=['tx_ID', 'Block_ID', 'nr_input', 'nr_output','userID','userID2']
tx['userID'].fillna(tx.userID2, inplace=True)
tx = tx.drop('userID2',1)
tx1=(txin.groupby('userID')
    .agg({'tx_id':['count']})
    .reset_index()
    .rename(columns={'userID':'value'}))
tx1.columns=['userID','nr_transactions']

dftx=tx1[:10473595]
dftx.to_csv('tx_new',index=None)