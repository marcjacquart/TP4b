import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv

pathCSV='/home/mjacquar/TP4b/csv'
dfWithPID = pd.read_csv(f'{pathCSV}/paramSelectionWithPID.csv',sep=' ',names=['nFeatures','Auc','featureList'])
dfWithoutPID = pd.read_csv(f'{pathCSV}/paramSelectionWithoutPID.csv',sep=' ',names=['nFeatures','Auc','featureList'])
#dfIso = pd.read_csv(f'{pathCSV}/paramSelectionWithoutB0Mass.csv',sep=' ',names=['nFeatures','Auc','featureList'])
#print(df)
#plt.figure(figsize=(6.0,16.0), dpi=500)
fig, ax = plt.subplots(figsize=(4.0,4.0), dpi=500) 
#plt.ylim(0.9893,0.9897)
valYMin=0.976
valYMax=0.99
plt.ylim(valYMin,valYMax)
ax.plot(dfWithPID['nFeatures'],dfWithPID['Auc'],'r.',label='With PID features')
ax.plot(dfWithoutPID['nFeatures'],dfWithoutPID['Auc'],'b.',label='Without PID features')
#ax.plot(dfIso['nFeatures'],dfIso['Auc'],'g.',label='With PID features and 2 separate isolation variables')
#ax.set_yscale('log')
plt.legend()
plt.vlines(	x=8,
			ymin=valYMin,
			ymax=0.9812,
			color='k',
			linewidth=1,
			linestyle='--',
			alpha=0.3) 
plt.vlines(	x=10,
			ymin=valYMin,
			ymax=dfWithPID['Auc'][9],
			color='k',
			linewidth=1,
			linestyle='--',
			alpha=0.3) 
plt.xlabel("Number of features")
plt.ylabel("Auc")
plt.savefig(f'plots/featuresAuc.pdf')
print('Done!')
