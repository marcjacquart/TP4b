import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv

pathCSV='/home/mjacquar/TP4b/csv'
dfPID = pd.read_csv(f'{pathCSV}/paramSelectionWithPID.csv',sep=' ',names=['nFeatures','Auc','featureList'])
df = pd.read_csv(f'{pathCSV}/paramSelectionWithoutPID.csv',sep=' ',names=['nFeatures','Auc','featureList'])
dfIso = pd.read_csv(f'{pathCSV}/paramSelectionWithoutB0Mass.csv',sep=' ',names=['nFeatures','Auc','featureList'])
#print(df)
plt.figure(figsize=(8,6), dpi=500)
fig, ax = plt.subplots() 
#plt.ylim(0.9893,0.9897)
plt.ylim(0.989,0.99)
ax.plot(dfPID['nFeatures'],dfPID['Auc'],'r.',label='With PID features')
ax.plot(df['nFeatures'],df['Auc'],'b.',label='Without PID features')
ax.plot(dfIso['nFeatures'],dfIso['Auc'],'g.',label='With PID features and 2 separate isolation variables')
#ax.set_yscale('log')
plt.legend()
plt.xlabel("Number of features")
plt.ylabel("Auc")
plt.savefig(f'plots/featuresAuc.pdf')
