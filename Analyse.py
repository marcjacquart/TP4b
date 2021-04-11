# This script:
#	-Import data from csv file
#	-Import model from given hyperparameters (trained with random seed=0 for data 70/30 split)
#	-Use it on the test sample to plot roc curve and compute auc. 
#	-Do Kolmogorov–Smirnov test to check if distribution between test and test are similar.
#	-Plot histogram of y prediction for BDT response
#	-Plot importance of variables histogram


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc 	# roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp 			# Kolmogorov–Smirnov test to compare 2 distributions
#from scipy import stats

plotRoc=True
plotPrediction_KS=True
modelType='hyperScan' 		# 'featureSelection' or 'hyperScan' /!\ also change between the two sets of features
plotImportance=True


# Import data:
pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']						# 1: signal, 0: background
print("csv loaded")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) # est_size=0.3, train_size=0.7. random state: seed for random assignation of data in the split. Seed given so the test sample is different of training sample even after saving and reopening the bdt

# features = ['B_s0_TAU',				# B lifetime
# 			'MIN_IPCHI2_emu',		# Minimum on the two impact parameters chi2, large if signal (comes from secondary vertex)
# 			'B_s0_IP_OWNPV',		# Impact parameter
# 			'SUM_isolation_emu',	# Isolation: we want nothing in the cone around e or mu
# 			'B_s0_PT',				# Transverse momentum of B0
# 			'LOG1_cosDIRA',			# Direction angle between sum emu and B0 reconstructed -> log(1-cos(DIRA))
# 			'B_s0_CDFiso',			# Different measure of isolation, mathematical def in analysis document
# 			'MAX_PT_emu',			# B0 has high mass -> high p_T for daughter particles 
# 			'B_s0_ENDVERTEX_CHI2',	# Quality of the reconstructed decay vertex.
# 			'DIFF_ETA_emu']			# Absolute difference in pseudorapidity of emu
features = ['B_s0_ENDVERTEX_CHI2', 'B_s0_CDFiso', 'muminus_ProbNNmu', 'eplus_ProbNNe', 'eplus_ETA', 'B_s0_IPCHI2_OWNPV', 'B_s0_minPT', 'MIN_IPCHI2_emu','SUM_isolation_emu', 'LOG1_cosDIRA'] 

#  Retrive model: Select with name from parameters, best .R: 5,130,0.225,// .R first: 6,100,0.15 // best SAMME: 10,1000.1
pathModel='/home/mjacquar/TP4b/modelOpti/HyperScanFromRFE' # Change name of folder for different scans
if modelType == 'hyperScan':
	algoName = 'SAMME.R' 				# Scan parameters
	maxDepth = 3
	nTrees = 80
	learningRate=0.05
	model = joblib.load(f'{pathModel}/bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth.pkl') #Load model to analyse
	# Use it to predict y values on X_test sample
	y_pred_onTest=model.predict_proba(X_test[features])[:,1]	# 2 values, takes first: proba to be signal
	y_pred_onTrain=model.predict_proba(X_train[features])[:,1]

elif modelType == 'featureSelection' :
	nSelect=13
	model = joblib.load(f'{pathModel}/selection_3_50_0.1_{nSelect}variables.pkl') # Load Feature selection model
	
	# Use it to predict y values on X_test sample, need the training features (not from csv, but keep the cleaning code in case it can be useful somewhere)
	# pathCSV='/home/mjacquar/TP4b/csv'
	# csvList = pd.read_csv(f'{pathCSV}/paramSelection.csv',sep=' ',names=['nFeatures','Auc','featureList'])
	# csvList= csvList.loc[csvList['nFeatures'] == nSelect] 					# Select the right values line 
	# csvList= np.array(csvList['featureList'])								# Select the tab with the names, must be cleaned
	# csvList= csvList[0].replace('\']','').replace('[\'','') 				# Only string with the whole tab, delete [' of the begining and '] at the end
	# csvList=csvList.replace('\' \'',',').replace('"','').replace('\n','')	# All values in one string, separated by ,
	# csvList=csvList.replace('\' \'',',').split(',')						# Makes a tab of all the feature names
	# print(csvList)
	trainFeatures=['B_s0_TAU', 'B_s0_DTF_M', 'B_s0_ENDVERTEX_CHI2', 'B_s0_BPVDIRA', 'B_s0_CDFiso', 'B_s0_D1_isolation_Giampi', 'B_s0_D2_isolation_Giampi', 'muminus_ProbNNmu', 'muminus_ProbNNk', 'eplus_ProbNNe', 'muminus_PT', 'eplus_ETA', 'muminus_ETA', 'B_s0_IPCHI2_OWNPV', 'eplus_IPCHI2_OWNPV', 'B_s0_minP', 'B_s0_absdiffP', 'B_s0_minPT', 'B_s0_minIP_OWNPV', 'B_s0_absdiffIPCHI2_OWNPV', 'MIN_IPCHI2_emu', 'LOG1_cosDIRA', 'MAX_PT_emu', 'DIFF_ETA_emu']
	y_pred_onTest=model.predict_proba(X_test[trainFeatures])[:,1]						# 2 values, takes first: proba to be signal. test on the parameter list from csv file
	y_pred_onTrain=model.predict_proba(X_train[trainFeatures])[:,1]

else:																			# If wrong modelType name, display error and quit
	print("Wrong Model Type. Must be 'hyperScan' or 'featureSelection'. Please try again.")
	exit()

fpr, tpr, threshold = roc_curve(y_test, y_pred_onTest) 	# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
aucValue = auc(fpr, tpr) 								# Use built in fct to compute area under curve




# Plot the result: Auc
if plotRoc:
	plt.figure(figsize=(8, 8), dpi=300)
	plt.plot(tpr,1-fpr,linestyle='-',label=f'Auc={aucValue}')
	plt.xlabel('True positive rate')
	plt.ylabel('1-False positive rate')
	plt.legend()
	plt.savefig(f'plots/roc.pdf')
	plt.close()

#				Known result 	 Predicted results			  Want to plot?	 For Multianalysis 	for pdf save
def predictionY_interval(y_train, y_test, y_pred_onTrain,y_pred_onTest, plotResult,   computeInterval,    pdfName='yPred'): #Function form to use in Multianalysis
	yTrainS=y_pred_onTrain[y_train==1]		# Take values at index depending of BDT y output
	yTrainB=y_pred_onTrain[y_train==0]
	yTestS=y_pred_onTest[y_test==1]
	yTestB=y_pred_onTest[y_test==0]

	binsTab=np.linspace(0, 1, num=201) 	# num: number of bins for smooth histogram.
	if plotResult:
		plt.figure(figsize=(8, 8), dpi=300)
		fig, ax = plt.subplots() 																	# Want to stack histograms on top of same axes
		ax.hist(yTrainS,  bins=binsTab, color='r',histtype='bar',density=True, alpha=0.3) 			# Adding transparency for histogram columns
		ax.hist(yTrainB,  bins=binsTab, color='b', histtype='bar',density=True, alpha=0.3) 			# collect the results in varibles for later KS test
		yTrainSHist = ax.hist(yTrainS,  bins=binsTab, color='r', histtype='step',density=True, label='Sig (train)') 	# Histogram values
		yTrainBHist = ax.hist(yTrainB,  bins=binsTab, color='b', histtype='step',density=True, label='Bg (train)') 
		yTestSHist = plt.hist(yTestS,   bins=binsTab, density=True,alpha = 0.0)						# Invisible to collect histogram values
		yTestBHist = plt.hist(yTestB,   bins=binsTab, density=True,alpha = 0.0)
		bin_centers = 0.5*(binsTab[1:] + binsTab[:-1])
		ax.scatter(bin_centers, yTestSHist[0], marker='o', c='r', s=20, alpha=1,label='Sig (test)') 	# Display as dot in histogram
		ax.scatter(bin_centers, yTestBHist[0], marker='o', c='b', s=20, alpha=1,label='Bg (test)') 	# Must take first array for data, second one is bins values

		plt.xlabel('BDT signal response')
		plt.ylabel('Normalized number of events')
		plt.legend()
		plt.savefig(f'plots/{pdfName}.pdf')
		plt.close()	
	if computeInterval:
		numberBins=201
		errorMax=2*(1/(numberBins-1))
		binsTab=np.linspace(0, 1, num=numberBins)
		
		yTestSHist=np.histogram(yTestS,bins=binsTab,density=False) #Density=True to normalise, not necessary here we just want the max
		yTestBHist=np.histogram(yTestB,bins=binsTab,density=False)
		maxS=np.argmax(yTestSHist[0]) #first tab is the histogram values, second is the bins values on axis
		maxB=np.argmax(yTestBHist[0])
		interval=abs(binsTab[maxS]-binsTab[maxB])
		print(f'Interval between Signal and backgroung max in histogram: {interval} pm {errorMax}')
	else: interval=0
	return interval,yTrainSHist,yTrainBHist,yTestSHist,yTestBHist

if plotPrediction_KS:
# Plot the result: y_pred
	interval,yTrainSHist,yTrainBHist,yTestSHist,yTestBHist=predictionY_interval(y_train, y_test,y_pred_onTrain,y_pred_onTest, True, True,'yPred')

# Kolmogorov–Smirnov test

	yTrainS=y_pred_onTrain[y_train==1]		# Take values at index depending of BDT y output. Will be computed twice (also in fct), but better to not add 4 return values...
	yTrainB=y_pred_onTrain[y_train==0]
	yTestS=y_pred_onTest[y_test==1]
	yTestB=y_pred_onTest[y_test==0]

	KSsignal=ks_2samp(yTestSHist[0], yTrainSHist[0], alternative='two-sided', mode='auto')
	KSbackground=ks_2samp(yTrainBHist[0], yTestBHist[0], alternative='two-sided', mode='auto')
	KStest=ks_2samp(yTestSHist[0],yTestBHist[0],alternative='two-sided', mode='auto')
	KStrain=ks_2samp(yTrainSHist[0],yTrainBHist[0],alternative='two-sided', mode='auto')

	print("KS test")
	print(f'Train and Test signal: {KSsignal}')
	print(f'Train and Test background: {KSbackground}')
	print(f'Test: Train and background: {KStest}')
	print(f'Train: Train and background: {KStrain}')

	#Test with the arrays directly:
	#print("KS test Array:")
	#KSsignalArray=ks_2samp(yTrainS,yTestS, alternative='two-sided', mode='auto')
	#KSbackgroundArray=ks_2samp(yTrainB,yTestB, alternative='two-sided', mode='auto')
	#print(f'Train and Test signal (array): {KSsignalArray}')
	#print(f'Train and Test background (array): {KSbackgroundArray}')

	#Visual KS:
	#y=np.linspace(0.0,1.0,101)

	#testSCumul=[1/len(yTestS)*sum(map(lambda yList: yList<=yTresh,yTestS)) for yTresh in y] # NOT EFFICIENT!
	#testBCumul=[1/len(yTestB)*sum(map(lambda yList: yList<=yTresh,yTestB)) for yTresh in y] # Compute the cumulative function

	#trainSCumul=[1/len(yTrainS)*sum(map(lambda yList: yList<=yTresh,yTrainS)) for yTresh in y] # 2 Other distribution to chose which KS test to run in cumulDiff
	#trainBCumul=[1/len(yTrainB)*sum(map(lambda yList: yList<=yTresh,yTrainB)) for yTresh in y] 
	numberBins=201
	binsTab=np.linspace(0, 1, num=numberBins)
	yTestSHist=np.histogram(yTestS,bins=binsTab,density=False) # Need of numpy histograms, not matplotlib Density=Fasle, normalize before cumsum
	yTestBHist=np.histogram(yTestB,bins=binsTab,density=False)
	yTrainSHist=np.histogram(yTrainS,bins=binsTab,density=False) 
	yTrainBHist=np.histogram(yTrainB,bins=binsTab,density=False)

	testSCumul=1/len(yTestS)*np.cumsum(yTestSHist[0])
	testBCumul=1/len(yTestB)*np.cumsum(yTestBHist[0])
	trainSCumul=1/len(yTrainS)*np.cumsum(yTrainSHist[0])
	trainBCumul=1/len(yTrainB)*np.cumsum(yTrainBHist[0])

	def makeKSPlot(dist1,dist2,label1,label2,outFileName):		# Make the KS plot. dist1&2 must be the same size
		cumulDiff=np.abs(np.array(dist1)-np.array(dist2))
		maxSeparation=np.amax(cumulDiff)
		maxSeparationIndex=np.where(cumulDiff==maxSeparation)
		maxSeparationIndex=int(maxSeparationIndex[0]) 			# Transform in float in case multiple same separation
		ksTextResult='{0:.4f}'.format(float(maxSeparation)) 	# 3 decimals format for caption
		print(f'KS test manual result for {label1} and {label2}: {ksTextResult}')
		yAxis=np.linspace(0.0,1.0,len(dist1))
		fig,ax=plt.subplots()
		ax.plot(yAxis,dist1,color='r',label=label1)
		ax.plot(yAxis,dist2,color='b',label=label2)
		valYMin=min(dist1[maxSeparationIndex],dist2[maxSeparationIndex])
		valYMax=max(dist1[maxSeparationIndex],dist2[maxSeparationIndex])
		print(f'valYMin: {valYMin}')
		print(f'valYMax: {valYMax}')
		plt.vlines(	x=yAxis[maxSeparationIndex],
					ymin=valYMin,
					ymax=valYMax,
					color='k',
					linewidth=2,
					linestyle='--',
					label=f'KS: {ksTextResult}') 
		plt.legend()
		plt.xlabel('BDT response')
		plt.ylabel('Cumulative function')
		plt.savefig(f'plots/{outFileName}.pdf')
		plt.close()
	# Make the plots with the above function:
	makeKSPlot(trainSCumul,trainBCumul,'Train signal','Train background','KSplotTrainSB')
	makeKSPlot(testSCumul,testBCumul,'Test signal','Test background','KSplotTestSB')
	makeKSPlot(trainBCumul,testBCumul,'Train background','Test background','KSplotB')
	makeKSPlot(testSCumul,trainSCumul,'Test signal','Train signal','KSplotS')



# Plot the result: Features of importance
if plotImportance:

	importance = model.feature_importances_
	zipValues= list(zip(features,importance))
	orderedValues=sorted(zipValues,key=lambda x: x[1],reverse=True) # Sort by importance
	orderedFeatures,orderedImportance=(zip(*orderedValues))			# Unzip into 2 lists for plot

	plt.figure(figsize=(8, 6), dpi=500)
	plt.bar([x for x in range(len(importance))], orderedImportance)
	plt.xticks(ticks = range(len(importance)) ,labels = orderedFeatures, rotation = 60, fontsize =7 )
	plt.ylabel('Feature importance')
	plt.savefig(f'plots/importance.pdf',bbox_inches='tight')
	plt.close()
