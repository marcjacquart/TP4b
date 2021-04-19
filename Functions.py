import numpy as np


#							Known result 	 Predicted results			  Want to plot?	 For Multianalysis 	for pdf save
def predictionY_interval(y_train, y_test, y_pred_onTrain,y_pred_onTest, plotResult,   computeInterval,    pdfName='yPred'): #Function form to use in Multianalysis
	interval=yTrainSHist=yTrainBHist=yTestSHist=yTestBHist=0										# For the return if not computed
	yTestS=y_pred_onTest[y_test==1]
	yTestB=y_pred_onTest[y_test==0]

	binsTab=np.linspace(0, 1, num=201) 																# num: number of bins for smooth histogram.
	if plotResult:
		yTrainS=y_pred_onTrain[y_train==1]															# Take values at index depending of BDT y output
		yTrainB=y_pred_onTrain[y_train==0]
		plt.figure(figsize=(8, 8), dpi=300)
		fig, ax = plt.subplots() 																	# Want to stack histograms on top of same axes
		ax.hist(yTrainS,  bins=binsTab, color='r',histtype='bar',density=True, alpha=0.3) 			# Adding transparency for histogram columns
		ax.hist(yTrainB,  bins=binsTab, color='b', histtype='bar',density=True, alpha=0.3) 			# collect the results in varibles for later KS test
		yTrainSHist = ax.hist(yTrainS,  bins=binsTab, color='r', histtype='step',density=True, label='Sig (train)') 	# Histogram values
		yTrainBHist = ax.hist(yTrainB,  bins=binsTab, color='b', histtype='step',density=True, label='Bg (train)') 
		yTestSHist = plt.hist(yTestS,   bins=binsTab, density=True,alpha = 0.0)						# Invisible to collect histogram values
		yTestBHist = plt.hist(yTestB,   bins=binsTab, density=True,alpha = 0.0)
		bin_centers = 0.5*(binsTab[1:] + binsTab[:-1])
		ax.scatter(bin_centers, yTestSHist[0], marker='o', c='r', s=20, alpha=1,label='Sig (test)') # Display as dot in histogram
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
		#print(f'Interval between Signal and backgroung max in histogram: {interval} pm {errorMax}')
	return interval,yTrainSHist,yTrainBHist,yTestSHist,yTestBHist

