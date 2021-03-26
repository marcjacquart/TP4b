import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv

pathCSV='/home/mjacquar/TP4b/csv'
df = pd.read_csv(f'{pathCSV}/paramSelection.csv',sep=' ',names=['nFeatures','Auc',''])
#print(df)
plt.figure(figsize=(8,6), dpi=500)
fig, ax = plt.subplots() 
plt.ylim(0.9995,0.99965)
ax.plot(df['nFeatures'],df['Auc'],'k.')
#ax.set_yscale('log')
plt.xlabel("Number of features")
plt.ylabel("Auc")
plt.savefig(f'plots/featuresAuc.pdf')
