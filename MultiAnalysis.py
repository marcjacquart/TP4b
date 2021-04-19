# This script:
#	-Analyses the result of the grid auc optimisation with a 3d heatmap
#	-Plot auc on histogram to compare performances of the 2 algorithms


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 			# For 3d plots

plotDistance=False
plotSAMMER=True
plotSAMME=False

pathCSV='/home/mjacquar/TP4b/csv'					# Csv path to open
df=pd.read_csv(f'{pathCSV}/HyperScanFromRFE.csv', sep=' ', names=['algoName','learningRate','maxDepth','nTrees','auc']) # HyperScanFromRFE (backup).csv for distance, add ,'dist'

dfR=df[(df['algoName']=='SAMME.R') & (df['auc']>0.9874)]		# Divide datafile with algorithm name, first auc cut to clean 3d plot
dfSAMME=df[(df['algoName']=='SAMME')& (df['auc']>0.9874)]		# Treshold to clean low values, heatmap is linear on the values range

#df['dist']=pd.to_numeric(df['dist'])
# Distance 3d:
if plotDistance:
	#print(df['dist'])
	minDist=min(df['dist']) 							# Min and max values for colour
	maxDist=max(df['dist'])								# No cut for the dist
	print(f'min:{minDist}, max:{maxDist}')
	fig = plt.figure()
	ax = Axes3D(fig)
	colourR= [(df['dist']-minDist)/(maxDist-minDist)]	# linear color value on the data range

	#colourR=np.power(colourR,5) 						# Accentuate differences

	ax.scatter(df['learningRate'],df['maxDepth'], df['nTrees'],cmap=plt.get_cmap('rainbow'), c= colourR) # 3d scatter plot
	ax.set_xlabel('learningRate')
	ax.set_ylabel('maxDepth')
	ax.set_zlabel('nTrees')

	plt.show()											# show() for 3d interactive plot
	plt.savefig(f'plots/multiDist.pdf')
	plt.close()



	# Plot dist - 2d depth
	plt.figure()
	df.boxplot(column='dist',by='maxDepth')
	plt.title('')
	plt.suptitle('')
	plt.xlabel('Maximal Depth')
	plt.ylabel('Distance between peaks')
	plt.savefig(f'plots/boxplotDist.pdf')
	plt.close()


	fig, ax = plt.subplots()
	ax.scatter(df['maxDepth'],df['dist'])
	plt.xlabel('Max Depth')
	plt.ylabel('Distance between peaks')
	plt.show()
	plt.close()

# For SAMME.R:
if plotSAMMER:
	minAuc=min(dfR['auc']) 								# Min and max values for colour
	maxAuc=max(dfR['auc'])
	print(f'min:{minAuc}, max:{maxAuc}')
	fig = plt.figure()
	ax = Axes3D(fig)
	colourR= [(dfR['auc']-minAuc)/(maxAuc-minAuc)]		# linear color value on the data range

	#colourR=np.power(colourR,5) 						# Accentuate differences


	ax.scatter(dfR['learningRate'],dfR['maxDepth'], dfR['nTrees'],cmap=plt.get_cmap('rainbow'), c= colourR) # 3d scatter plot
	ax.set_xlabel('learningRate')
	ax.set_ylabel('maxDepth')
	ax.set_zlabel('nTrees')

	plt.show()											# show() for 3d interactive plot
	plt.savefig(f'plots/multiR.pdf')


# For SAMME:
if plotSAMME:
	minAuc=min(dfSAMME['auc']) 							# Min and max values for colour
	maxAuc=max(dfSAMME['auc'])

	fig = plt.figure()
	ax = Axes3D(fig)
	colour= [(dfSAMME['auc']-minAuc)/(maxAuc-minAuc)]

	ax.scatter(dfSAMME['learningRate'],dfSAMME['maxDepth'], dfSAMME['nTrees'],cmap=plt.get_cmap('rainbow'), c= colour)
	ax.set_xlabel('learningRate')
	ax.set_ylabel('maxDepth')
	ax.set_zlabel('auc')
	#plt.show()
	plt.savefig(f'plots/multi.pdf')


	#Plot both algorithms auc on histogram to compare performances:
	fig, ax = plt.subplots()
	ax.hist(dfR['auc'],bins=np.linspace(0.985,0.99,10),label='SAMME.R')
	ax.hist(dfSAMME['auc'],bins=np.linspace(0.985,0.99,10),label='SAMME',alpha=0.5) # Alpha transparency to draw on top
	ax.set_xlabel('auc')
	ax.set_ylabel('number of simulations')
	plt.legend()
	plt.savefig(f'plots/multiCompareHist.pdf')

