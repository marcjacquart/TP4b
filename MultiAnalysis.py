import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 	# For 3d plots

pathCSV='/home/mjacquar/TP4b/csv'			# Csv path to open
df=pd.read_csv(f'{pathCSV}/paramAuc.csv', sep=' ', names=['algoName','learningRate','maxDepth','nTrees','auc'])

dfR=df[(df['algoName']=='SAMME.R') & (df['auc']>0.98)]		# Divide datafile with algorithm name, first auc cut to clean 3d plot
dfSAMME=df[(df['algoName']=='SAMME')& (df['auc']>0.98)]




# For SAMME.R:
minAuc=min(dfR['auc']) #Min and max values for colour
maxAuc=max(dfR['auc'])
fig = plt.figure()
ax = Axes3D(fig)
colourR= [(dfR['auc']-minAuc)/(maxAuc-minAuc)]

#colourR=np.power(colourR,5) # Accentuate differences


ax.scatter(dfR['learningRate'],dfR['maxDepth'], dfR['nTrees'],cmap=plt.get_cmap('rainbow'), c= colourR)
ax.set_xlabel('learningRate')
ax.set_ylabel('maxDepth')
ax.set_zlabel('nTrees')
plt.show()
plt.savefig(f'plots/multiR.pdf')


# For SAMME:
minAuc=min(dfSAMME['auc']) # Min and max values for colour
maxAuc=max(dfSAMME['auc'])
fig = plt.figure()
ax = Axes3D(fig)
colour= [(dfSAMME['auc']-minAuc)/(maxAuc-minAuc)]

ax.scatter(dfSAMME['learningRate'],dfSAMME['maxDepth'], dfSAMME['nTrees'],cmap=plt.get_cmap('rainbow'), c= colour)
ax.set_xlabel('learningRate')
ax.set_ylabel('maxDepth')
ax.set_zlabel('auc')

plt.savefig(f'plots/multi.pdf')

# Which model is best?

fig, ax = plt.subplots()
ax.hist(dfR['auc'],bins=np.linspace(0.985,0.99,10),label='SAMME.R')
ax.hist(dfSAMME['auc'],bins=np.linspace(0.985,0.99,10),label='SAMME',alpha=0.5)
ax.set_xlabel('auc')
ax.set_ylabel('number of simulations')
plt.legend()
plt.savefig(f'plots/multiCompareHist.pdf')