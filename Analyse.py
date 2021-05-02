# This script:
#	-Import data from csv file
#	-Import model from given hyperparameters (trained with random seed=0 for data 70/30 split)
#	-Use it on the test sample to plot roc curve and compute auc. 
#	-Do Kolmogorov–Smirnov test to check if distribution between test and test are similar.
#	-Plot histogram of y prediction for BDT response
#	-Plot importance of variables histogram
#	-Compute Punzi figure of merit to optimize cut value


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc 	# roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp 			# Kolmogorov–Smirnov test to compare 2 distributions
from Functions import predictionY_interval,fitBackground
#from scipy import stats





plotRoc=False
plotPrediction_KS=False
modelType='hyperScan' 		# 'featureSelection' or 'hyperScan' /!\ also change between the two sets of features
plotImportance=False

# /!\ trainModel must be true if anything else than punzi is calculated. False only to make punzi faster
trainModel=False

punzi=True

# Import data:
pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X_bestBDT.csv') # bestBDT file has BDTresponse column filled with response of K-folding of best BDT
y = X['sig']						# 1: signal, 0: background
print("csv loaded")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) # test_size=0.3, train_size=0.7. random state: seed for random assignation of data in the split. Seed given so the test sample is different of training sample even after saving and reopening the bdt

# features = ['B_s0_TAU',			# B lifetime 		#--------/!\ OLD!---------
# 			'MIN_IPCHI2_emu',		# Minimum on the two impact parameters chi2, large if signal (comes from secondary vertex)
# 			'B_s0_IP_OWNPV',		# Impact parameter
# 			'SUM_isolation_emu',	# Isolation: we want nothing in the cone around e or mu
# 			'B_s0_PT',				# Transverse momentum of B0
# 			'LOG1_cosDIRA',			# Direction angle between sum emu and B0 reconstructed -> log(1-cos(DIRA))
# 			'B_s0_CDFiso',			# Different measure of isolation, mathematical def in analysis document
# 			'MAX_PT_emu',			# B0 has high mass -> high p_T for daughter particles 
# 			'B_s0_ENDVERTEX_CHI2',	# Quality of the reconstructed decay vertex.
# 			'DIFF_ETA_emu']			# Absolute difference in pseudorapidity of emu

features=[  'B_s0_ENDVERTEX_CHI2',
			'B_s0_CDFiso',
			'eplus_ProbNNe',
			'eplus_ETA',
			'B_s0_IPCHI2_OWNPV',
			'B_s0_minPT',
			'muminus_ProbNNmuk',
			'MIN_IPCHI2_emu',
			'SUM_isolation_emu',
			'LOG1_cosDIRA']
			

if trainModel:
	#  Retrive model: Select with name from parameters, best .R: 5,130,0.225,// .R first: 6,100,0.15 // best SAMME: 10,1000.1
	pathModel='/home/mjacquar/TP4b/model' # Change name of folder for different scans
	if modelType == 'hyperScan':
		algoName = 'SAMME.R' 				# Scan parameters
		maxDepth = 3
		nTrees = 1000
		learningRate=0.1
		model = joblib.load(f'{pathModel}/bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth.pkl') #Load model to analyse
		# Use it to predict y values on X_test sample
		y_pred_onTest=model.predict_proba(X_test[features])[:,1]	# 2 values, takes first: proba to be signal
		y_pred_onTrain=model.predict_proba(X_train[features])[:,1]
		print('Model trained')

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

if plotPrediction_KS:
# Plot the result: y_pred
	yBinsTab=np.linspace(0.0, 1.0, num=20001)	# num: number of bins for smooth histogram.
	interval,yTrainSHist,yTrainBHist,yTestSHist,yTestBHist=predictionY_interval(yBinsTab, y_train, y_test,y_pred_onTrain,y_pred_onTest, True, True,'yPred')

# Kolmogorov–Smirnov test

	yTrainS=y_pred_onTrain[y_train==1]			# Take values at index depending of BDT y output. Will be computed twice (also in fct), but better to not add 4 return values...
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


if punzi:
	print('Computing Punzi figure of merit...')

	def punziFigureOfMerit(Eps_S,N_background,a=3):
		return Eps_S/(np.sqrt(N_background)+a/2)

	yBinsTab=np.linspace(0.48, 0.53, num=201)
	#interval,yTrainSHist,yTrainBHist,yTestSHist,yTestBHist=predictionY_interval(yBinsTab,y_train, y_test,y_pred_onTrain,y_pred_onTest, False, True)	
	BDTresponse=X['BDTresponse']
	#y Variable tells if the event is signal or background
	# sig variable: 1 for signal (MC), 0 for background (LHC data)
	X=X[((X['B_s0_DTF_M']>5500) & (X['B_s0_DTF_M']<6500) & (X['sig']==1)) |(X['sig']==0) ] # Take all background (for the fit) but only signal in the blinding window
	totalSignalNumber=len(X['sig'][X['sig']==1])


	punzi=[]
	for yBin in yBinsTab:
		#print(f'yBin loop in tab: {yBin}')

		XselectedSig = X[(X['BDTresponse']>=yBin) & (X['sig']==1)] 	# Building again the dataframe at every iteration is a performance tradeoff to be able to scan only a smaller portion of the BDT response
		Eps_S=len(XselectedSig)/totalSignalNumber
		#print(f'Eps_S={Eps_S}')
		XselectedBg = X[(X['BDTresponse']>=yBin) & (X['sig']==0)]
		N_background=fitBackground(XselectedBg, False)
		punzi.append(punziFigureOfMerit(Eps_S,N_background))			# Compute figure of merit to fill tab

	#sigTot=np.sum(yTestSHist[0])
	# for i in reversed(range(len(yBinsTab)-1)): 
	# 	#print(f'i={i}')
	# 	#print(f'len(yBinsTab)={len(yBinsTab)}')	
	# 	#print(f'len(yTestSHist)={len(yTestSHist[0])}')				# Reverse to compute cumulative using previous ones
	# 	Eps_S+=yTestSHist[0][i]/sigTot					# Need to normalise for signal efficiency
	# 	N_background+=yTestBHist[0][i]					# No normalisation for background events
	# 	punzi[i]=punziFigureOfMerit(Eps_S,N_background) # Compute figure of merit
	maxPunzi= np.max(punzi)
	indexMax= np.argmax(punzi)
	yMax=yBinsTab[indexMax]

	fig,ax=plt.subplots(figsize=(4,4),dpi=500)
	plt.vlines(yMax, ymin=0,ymax=maxPunzi,color='k',linewidth=1,linestyle='--')
	ax.plot(yBinsTab,punzi,color='b',label=f'Max: y={"{:.4f}".format(yMax)}')
	plt.xlim(left=0.45,right=0.55)
	plt.xlabel('BDT response')
	plt.ylabel('Punzi figure of merit (arbitrary units)')
	plt.legend()
	plt.savefig(f'plots/punzi.pdf',bbox_inches='tight')
	print(f'Final sum Eps_S to check normalisation: {Eps_S}')
	print(f'Final sum N_background to check normalisation: {N_background}')

# Plot the result: Features of importance
if plotImportance:
	importance = model.feature_importances_
	zipValues= list(zip(features,importance))
	orderedValues=sorted(zipValues,key=lambda x: x[1],reverse=True) # Sort by importance
	orderedFeatures,orderedImportance=(zip(*orderedValues))			# Unzip into 2 lists for plot

	plt.figure(figsize=(6, 4), dpi=500)

	plt.bar([x for x in range(len(importance))], orderedImportance)
	plt.xticks(ticks = range(len(importance)) ,labels = orderedFeatures, rotation = 90, fontsize =10 )
	plt.ylabel('Feature importance')
	plt.savefig(f'plots/importance.pdf',bbox_inches='tight')
	plt.close()
