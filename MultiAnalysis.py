# This script:
#	-Analyses the result of the grid auc optimisation with a 3d heatmap
#	-Plot auc on histogram to compare performances of the 2 algorithms


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 			# For 3d plots


modelType='MLP' # or 'BDT'
#BDT plots:
plotDistance=False
plotSAMMER=False
plotSAMME=False
#MLP plots:
plotRelu=True

pathCSV='/home/mjacquar/TP4b/csv'					# Csv path to open
if modelType=='BDT':
	df=pd.read_csv(f'{pathCSV}/HyperScanFromRFE.csv', sep=' ', names=['algoName','learningRate','maxDepth','nTrees','auc']) # HyperScanFromRFE (backup).csv for distance, add ,'dist'

	dfR=df[(df['algoName']=='SAMME.R') & (df['auc']>0.9874)]		# Divide datafile with algorithm name, first auc cut to clean 3d plot
	dfSAMME=df[(df['algoName']=='SAMME')& (df['auc']>0.9874)]		# Treshold to clean low values, heatmap is linear on the values range
if modelType=='MLP':
	df=pd.read_csv(f'{pathCSV}/HyperScanMLP_RestrictedSignal.csv', sep=' ', names=['activation','alpha','learning_rate_init','beta_1','auc'])
	dfRelu=df[(df['activation']=='relu') & (df['auc']>0.99)]
	dfRelu['alpha']=np.log10(dfRelu['alpha'])
	dfRelu['learning_rate_init']=np.log10(dfRelu['learning_rate_init'])
	#dfRelu['beta_1']=np.log10(1-dfRelu['beta_1'])
	print(df)
#df['dist']=pd.to_numeric(df['dist'])



def plot3D(minTestVar,maxTestVar,df,dfVariables,outputFilename, show,save):
	fig = plt.figure()
	ax = Axes3D(fig)
	colourR= [(df['auc']-minTestVar)/(maxTestVar-minTestVar)]		# linear color value on the data range

	#colourR=np.power(colourR,5) 						# Accentuate differences


	ax.scatter(df[dfVariables[0]],df[dfVariables[1]], df[dfVariables[2]],cmap=plt.get_cmap('rainbow'), c= colourR) # 3d scatter plot
	ax.set_xlabel(f'log10({dfVariables[0]})')
	ax.set_ylabel(f'log10({dfVariables[1]})')
	ax.set_zlabel(f'log10(1-{dfVariables[2]})')

	if show:
		plt.show()											# show() for 3d interactive plot
	if save:
		plt.savefig(f'plots/{outputFilename}.pdf')
	plt.close()



# Distance 3d:
if plotDistance: # Not adapted for MLP yet /!\
	#print(df['dist'])
	minDist=min(df['dist']) 							# Min and max values for colour
	maxDist=max(df['dist'])								# No cut for the dist
	print(f'min:{minDist}, max:{maxDist}')
	plot3D(	minTestVar=minDist,
			maxTestVar=maxDist,
			df=df,
			dfVariables=['learningRate','maxDepth','nTrees'],
			outputFilename='BDT/multiDist', 
			show=True,
			save=True)

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
	plot3D(	minTestVar=minAuc,
			maxTestVar=maxAuc,
			df=dfR,
			dfVariables=['learningRate','maxDepth','nTrees'],
			outputFilename='BDT/multiR', 
			show=True,
			save=False)

# For SAMME:
if plotSAMME:
	minAuc=min(dfSAMME['auc']) 							# Min and max values for colour
	maxAuc=max(dfSAMME['auc'])
	plot3D(	minTestVar=minAuc,
			maxTestVar=maxAuc,
			df=dfSAMME,
			dfVariables=['learningRate','maxDepth','nTrees'],
			outputFilename='BDT/multi', 
			show=True,
			save=False)

	#Plot both algorithms auc on histogram to compare performances:
	fig, ax = plt.subplots()
	ax.hist(dfR['auc'],bins=np.linspace(0.985,0.99,10),label='SAMME.R')
	ax.hist(dfSAMME['auc'],bins=np.linspace(0.985,0.99,10),label='SAMME',alpha=0.5) # Alpha transparency to draw on top
	ax.set_xlabel('auc')
	ax.set_ylabel('number of simulations')
	plt.legend()
	plt.savefig(f'plots/multiCompareHist.pdf')

if plotRelu:
	minAuc=min(dfRelu['auc']) 							# Min and max values for colour
	maxAuc=max(dfRelu['auc'])
	print(f'Auc min: {minAuc}, max: {maxAuc}')
	plot3D(	minTestVar=minAuc,
			maxTestVar=maxAuc,
			df=dfRelu,
			dfVariables=['alpha','learning_rate_init','beta_1'],
			outputFilename='MLP/multiRelu', 
			show=True,
			save=False)