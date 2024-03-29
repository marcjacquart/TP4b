# This script:
#	-Import or create the csv dataFile with LHC background data and MC events. Adding composition of variables (min/max/mean...)
#	-Plot correlation between variables
#	-Plot histogram of number of correlations to reduce number of variables given as features to the BDT
#	-Plot all variables in histograms with Data and MC to see general shape of input


from root_pandas import read_root
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Settings to chose:
importCSV=True				#If false, will create a new csv file for further use
plotCorrelation=True
plotVar=False

if plotCorrelation:
	import seaborn as sns 	# Takes time to load, so is called only if used 

pathCSV='/home/mjacquar/TP4b/csv'
if importCSV:
	X_data = pd.read_csv(f'{pathCSV}/X_data.csv')
	X_MC =  pd.read_csv(f'{pathCSV}/X_MC.csv')

else:
	#Define useful variables, split into categories to match root variables names (see line 15):
	common = [  'P',                    # Momentum (not used)
	            'PT',                   # Transverse momentum
	            'ETA',                  # Pseudorapidity: Need maximum between e and mu
	            'IP_OWNPV',             # Impact parameter with respect to his own production primary vertex "OWNPV": primary vertex reconstructed with the tracks of other products of collision
	            'IPCHI2_OWNPV',]        # Chi2 of the PV fit: how well is the PV defined. How does it change between e and mu if same vertex?

	Bvars  = [  'TAU',                  # Proper time before decay
	            'DTF_M',                # Reconstructed invarient mass
	            'DOCA',                 # Distance of closest approach, high for background. Meaning for a decay???
	            'ENDVERTEX_CHI2',       # Quality of the reconstructed decay vertex.
	            'BPVDIRA',              # Computes the cosine of the angle between the momentum of the particle and the direction fo flight from the best PV to the decay vertex.
	            'FD_OWNPV',             # Flight distance from primary vertex
	            'CDFiso',               # Isolation (see if other track in a cone arount the selected track. If so probably background)
	            'D1_isolation_Giampi',  # Giampi's isolation tool: number of tracks in a cone of angle 0.27 rad around the reconstructed track.
	            'D2_isolation_Giampi',] # We want the sum of both isolation number. signal will be 0 (no other close trcks in the daughter particles)

	muvars = [  'ProbNNmu',             # Bayesian posteriori probability: The posterior probability distribution is the probability distribution of an unknown quantity, treated as a random variable, conditional on the evidence obtained from an experiment or survey. 
	            'ProbNNk',]             # Same for kaon?

	evars  = [  'ProbNNe',]             # Same for electron?


	#Fill the tab "allvars" with variable names:
	allvars=[]
	for var in Bvars:
	    allvars.append(f'B_s0_{var}')	 # f string: python format: https://realpython.com/python-f-strings/
	for var in muvars:
	    allvars.append(f'muminus_{var}')
	for var in evars:
	    allvars.append(f'eplus_{var}')
	for var in common:
	    for p in ('B_s0', 'eplus', 'muminus'):
	        allvars.append(f'{p}_{var}')
	        
	allvars.append('eventNumber')


	# Data path & import with variable names:
	clusterPath="/panfs/tully/B2emu"
	X_data_2016_MU = read_root(clusterPath+"/data/Selected_B2emu2016MU.root", columns=allvars)
	X_data_2016_MD = read_root(clusterPath+"/data/Selected_B2emu2016MD.root", columns=allvars)
	X_data_2017_MU = read_root(clusterPath+"/data/Selected_B2emu2017MU.root", columns=allvars)
	X_data_2017_MD = read_root(clusterPath+"/data/Selected_B2emu2017MD.root", columns=allvars)
	X_data_2018_MU = read_root(clusterPath+"/data/Selected_B2emu2018MU.root", columns=allvars)
	X_data_2018_MD = read_root(clusterPath+"/data/Selected_B2emu2018MD.root", columns=allvars)
	print("Data loaded")
	X_MC_2016_MU = read_root(clusterPath+"/MC/Selected_B2emu2016MU.root", columns=allvars)
	X_MC_2016_MD = read_root(clusterPath+"/MC/Selected_B2emu2016MD.root", columns=allvars)
	X_MC_2017_MU = read_root(clusterPath+"/MC/Selected_B2emu2017MU.root", columns=allvars)
	X_MC_2017_MD = read_root(clusterPath+"/MC/Selected_B2emu2017MD.root", columns=allvars)
	X_MC_2018_MU = read_root(clusterPath+"/MC/Selected_B2emu2018MU.root", columns=allvars)
	X_MC_2018_MD = read_root(clusterPath+"/MC/Selected_B2emu2018MD.root", columns=allvars)
	print("MC loaded")

	#Putting togeter up and down polarity from every year for larger data sample:
	X_data = pd.concat([X_data_2016_MU, X_data_2016_MD,X_data_2017_MU, X_data_2017_MD,X_data_2018_MU, X_data_2018_MD]) 
	X_MC = pd.concat([X_MC_2016_MU, X_MC_2016_MD,X_MC_2017_MU, X_MC_2017_MD,X_MC_2018_MU, X_MC_2018_MD])
	print("Data concatenated")

	#Train above the target to get only noise: No need for MC, only signal
	X_data = X_data.loc[(X_data['B_s0_DTF_M'] > 5500) & (X_data['B_s0_DTF_M'] < 6500)] 



	#Adding more variables:
	print("Adding new variables:")
	for df in (X_MC,X_data):
		for var in common:
			df[f'B_s0_max{var}']=df[[f'eplus_{var}',f'muminus_{var}']].max(axis=1)
			df[f'B_s0_min{var}']=df[[f'eplus_{var}',f'muminus_{var}']].min(axis=1)
			df[f'B_s0_sum{var}']=df[[f'eplus_{var}',f'muminus_{var}']].sum(axis=1)
			df[f'B_s0_absdiff{var}']=(df[f'eplus_{var}']-df[f'muminus_{var}']).abs()
			allvars.append(f'B_s0_max{var}')
			allvars.append(f'B_s0_min{var}')
			allvars.append(f'B_s0_sum{var}')
			allvars.append(f'B_s0_absdiff{var}')

		df['muminus_ProbNNmuk'] =df['muminus_ProbNNmu']*(1-df['muminus_ProbNNk'])
		allvars.append('muminus_ProbNNmuk')

		df['MIN_IPCHI2_emu']=np.minimum(df['eplus_IPCHI2_OWNPV'],df['muminus_IPCHI2_OWNPV'])  # Minimum impact parameter between the two tracks
		allvars.append('MIN_IPCHI2_emu')

		df['SUM_isolation_emu']=df['B_s0_D1_isolation_Giampi']+df['B_s0_D2_isolation_Giampi'] # Sum of isolation of electron and muon (expect 0 in signal)
		allvars.append('SUM_isolation_emu')

		df['LOG1_cosDIRA']=np.log(1-df['B_s0_BPVDIRA'])										  # log(1-cos(DIRA))
		allvars.append('LOG1_cosDIRA')

		#df['MAX_PT_emu']=np.maximum(df['eplus_PT'],df['muminus_PT'])						  # Max of daughter P_T. Already describe by B_0_maxPT
		#allvars.append('MAX_PT_emu')

		#df['DIFF_ETA_emu']=abs(df['eplus_ETA']-df['muminus_ETA'])							  # Difference in pseudorapidity, absolute value, already described by B_s0_absdiffETA
		#allvars.append('DIFF_ETA_emu')

	X_MC['sig']= 1					# sig variable: 1 for signal (MC), 0 for background (LHC data)
	X_data['sig']=0
	print("Dataset complete")
	
	# Saving datasets to CSV for further use
	X_MC.to_csv(f'{pathCSV}/X_MC.csv')
	X_data.to_csv(f'{pathCSV}/X_data.csv')

	X =pd.concat([X_MC,X_data])		# Merge everything in 1 csv file for training
	X.to_csv(f'{pathCSV}/X.csv')
	print("Dataset saved")


#Plot correlation matrix between variables: 						(from: https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
if plotCorrelation:
	print(X_data.columns)
	X_data=X_data.drop(columns=['Unnamed: 0','sig','eventNumber']) 	# Drop uninteresting variables column. axis: drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
	#print (X_data)
	corrData = X_data.corr() 	
	#print(corrData)									# Compute the correlation matrix
	mask = np.triu(np.ones_like(corrData, dtype=bool))				# Generate a mask for the upper triangle
	f, ax = plt.subplots(figsize=(16, 13),dpi=500) 					# Set up the matplotlib figure
	cmap = sns.diverging_palette(230, 20, as_cmap=True)				# Generate a custom diverging colormap
	sns.heatmap(corrData, mask=mask, cmap=cmap, center=0, 			# Draw the heatmap with the mask and correct aspect ratio
	            square=True, linewidths=.5, cbar_kws={"shrink": .5})
	plt.savefig('plots/correlation_X_data.pdf',bbox_inches='tight')	# Save result to PDF
	plt.close()

	#Same for MC
	X_MC=X_MC.drop(['Unnamed: 0','sig','eventNumber'],axis=1) 		# Drop uninteresting variables column.
	corrMC = X_MC.corr()											# Compute the correlation matrix
	mask = np.triu(np.ones_like(corrMC, dtype=bool))				# Generate a mask for the upper triangle
	f, ax = plt.subplots(figsize=(16, 13),dpi=500)					# Set up the matplotlib figure
	cmap = sns.diverging_palette(230, 20, as_cmap=True)				# Generate a custom diverging colormap
	sns.heatmap(corrMC, mask=mask, cmap=cmap, center=0,				# Draw the heatmap with the mask and correct aspect ratio 
	            square=True, linewidths=.5, cbar_kws={"shrink": .5})
	plt.savefig('plots/correlation_X_MC.pdf',bbox_inches='tight')	# Save result to PDF
	plt.close()


	tooCorrelated =['B_s0_DTF_M','muminus_ProbNNmu','B_s0_D2_isolation_Giampi','B_s0_D1_isolation_Giampi', 'muminus_ProbNNk', 'eplus_ProbNNe','muminus_ProbNNmuk'] 	# Must delete too corelated ones, we don't want two variables describing the same thing for training. Other cuts: 'B_s0_D1_isolation_Giampi','B_s0_D2_isolation_Giampi','muminus_ProbNNmu', 'muminus_ProbNNk', 'eplus_ProbNNe','muminus_ProbNNmuk'
	#					  ^------------------------------------------------------------------ We don't want to train the model on the mass since the artificial cut on the B0 mass to have only background data introduce a selection bias. We delete it first. Then both isolation will try to be replaced by the sum.
	
	# Detect high correlation: Do it in a function to use it again with the reducted set of variables
	def computeCorr(corrData,corrMC,tooCorrelated,featuresList, outputFilename, plot):
		#print(f'Test length: corrData={len(corrData)}, corrMC={len(corrMC)}, featuresList={len(featuresList)}. ') # Length ok
		indexCorrData=np.zeros(len(featuresList))
		indexCorrMC=np.zeros(len(featuresList))
		corrTreshold=0.5

		for i in range(len(featuresList)):
			for j in range(len(featuresList)):
				if (i<j) & (abs(corrData.iloc[i,j])>corrTreshold): # iloc: return value in pandas data frame
					indexCorrData[i]+=1
					indexCorrData[j]+=1
				if (i<j) & (abs(corrMC.iloc[i,j])>corrTreshold):
					indexCorrMC[i]+=1
					indexCorrMC[j]+=1
		corrDf=pd.DataFrame([featuresList,indexCorrData,indexCorrMC],index=['varName','corrData','corrMC']) # Put results in  dataframe
		corrDf=corrDf.transpose()
		if plot:
			# Plot result on histogram:
			plt.figure(figsize=(10, 5), dpi=500)
			plt.bar([x for x in range(corrDf.shape[0])], corrDf['corrData'],color='b', label='Data')			# Plot Data and MC on the same histogram using alpha trensparency for the one on top
			plt.bar([x for x in range(corrDf.shape[0])], corrDf['corrMC'],  color='r', label='MC', alpha=0.7) 	# df.shape[0]: number of df rows
			plt.xticks(ticks = range(corrDf.shape[0]) ,labels = corrDf['varName'], rotation = 90, fontsize =8 )
			plt.legend()
			plt.ylabel('Number of correlation with other variables')
			plt.savefig(f'plots/{outputFilename}.pdf',bbox_inches='tight')
			plt.close()
		return indexCorrData, indexCorrMC

	allVarHeaders=list(X_data.columns.values) # Same as in MC (verified with ==)


	#print(allVarHeaders)
	computeCorr(corrData,corrMC,tooCorrelated,allVarHeaders, 'correlationVariables',True) 	# Computation with all variables

	# Getting rid of too correlated variables:
	
	maxCorr=1 			# Max value of correlation for both MC and data. Continue to eliminate varibles while > 0
	nStop=100
	n=0
	while (maxCorr>0) and (n<nStop):
		#print(tooCorrelated)
		#print(n)
		n+=1														# Conter to avoid infinite loop
		newCorrData=corrData 										# Reduce correlation matrix
		newCorrData=newCorrData.drop(columns=tooCorrelated)
		newCorrData=newCorrData.drop(index=tooCorrelated)
		newCorrMC=corrMC 											# Same for MC
		newCorrMC=newCorrMC.drop(columns=tooCorrelated)
		newCorrMC=newCorrMC.drop(index=tooCorrelated)
		features = [x for x in allVarHeaders if x not in tooCorrelated]
		#print(f'features: {features}')
		indexCorrData, indexCorrMC = computeCorr(newCorrData,newCorrMC,tooCorrelated,features,'correlationVariablesSelection',False) 	# Computation with less variables
		maxCorr=0
		maxCorrMC=0
		indexMax=0
		for i in range(len(indexCorrData)):
			if min(indexCorrData[i],indexCorrMC[i])>maxCorr: 		# We want to delete variables correlated in both sets (Data & MC)
				maxCorr=min(indexCorrData[i],indexCorrMC[i])
				maxCorrMC=indexCorrMC[i]
				indexMax=i
			if (min(indexCorrData[i],indexCorrMC[i])==maxCorr) and (indexCorrMC[i]>maxCorrMC): # In caase of a tie, MC is more important: we throw the one with most corelated MC part
				maxCorr=min(indexCorrData[i],indexCorrMC[i])
				maxCorrMC=indexCorrMC[i]
				indexMax=i
		#print(f'i={indexMax}')
		if maxCorr !=0:
			print(f'Nmax corr for both: {maxCorr}')
			print(f'Delete variable  {features[indexMax]}')
			tooCorrelated.append(features[indexMax])

	computeCorr(newCorrData,newCorrMC,tooCorrelated,features,'correlationVariablesSelection',True)		# Save final plot
	finalFeatures = [x for x in allVarHeaders if x not in tooCorrelated]
	print(f'Final features: {finalFeatures}')
	
#Plot variables in histograms and save to pdf:
if plotVar:
	for var in allvars:
	    range_low  = min(X_data[var].min(), X_MC[var].min())
	    range_high = max(X_data[var].max(), X_MC[var].max())

	    plt.hist(X_data[var], bins=100, label='data', density=True, alpha=0.7, range=(range_low, range_high))
	    plt.hist(X_MC[var],   bins=100, label='MC',   density=True, alpha=0.7, range=(range_low, range_high))

	    plt.xlabel(var)
	    plt.ylabel('Arbitrary units')
	    
	    plt.legend(frameon=False)
	    plt.savefig(f'plots/data_MC_{var}.pdf')
	    plt.close()
