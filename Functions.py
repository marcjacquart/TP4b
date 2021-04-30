import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#							Known result 	 Predicted results			  Want to plot?	 For Multianalysis 	for pdf save
def predictionY_interval(binsTab, y_train, y_test, y_pred_onTrain, y_pred_onTest, plotResult, computeInterval, pdfName='yPred'): #Function form to use in Multianalysis
	
	interval=yTrainSHist=yTrainBHist=yTestSHist=yTestBHist=0											# For the return if not computed
	yTestS=y_pred_onTest[y_test==1]
	yTestB=y_pred_onTest[y_test==0]

	if plotResult:
		yTrainS=y_pred_onTrain[y_train==1]																# Take values at index depending of BDT y output
		yTrainB=y_pred_onTrain[y_train==0]
		
		fig, ax = plt.subplots(figsize=(8, 4), dpi=300) 												# Want to stack histograms on top of same axes
		ax.hist(yTrainS,  bins=binsTab, color='r',histtype='bar',density=True, alpha=0.3) 				# Adding transparency for histogram columns
		ax.hist(yTrainB,  bins=binsTab, color='b', histtype='bar',density=True, alpha=0.3) 				# collect the results in varibles for later KS test
		yTrainSHist = ax.hist(yTrainS,  bins=binsTab, color='r', histtype='step',density=True, label='Signal (train)') 	# Histogram values
		yTrainBHist = ax.hist(yTrainB,  bins=binsTab, color='b', histtype='step',density=True, label='Background (train)') 
		yTestSHist = plt.hist(yTestS,   bins=binsTab, density=True,alpha = 0.0)							# Invisible to collect histogram values
		yTestBHist = plt.hist(yTestB,   bins=binsTab, density=True,alpha = 0.0)
		bin_centers = 0.5*(binsTab[1:] + binsTab[:-1])
		ax.scatter(bin_centers, yTestSHist[0], marker='o', c='r', s=20, alpha=1,label='Signal (test)') 	# Display as dot in histogram
		ax.scatter(bin_centers, yTestBHist[0], marker='o', c='b', s=20, alpha=1,label='Background (test)') 	# Must take first array for data, second one is bins values

		plt.xlabel('BDT signal response')
		plt.ylabel('Normalized number of events')
		plt.legend()
		plt.savefig(f'plots/{pdfName}.pdf')
		plt.close()	
	if computeInterval:
		numberBins=len(binsTab)
		errorMax=2*(1/(numberBins-1))
		binsTab=np.linspace(0, 1, num=numberBins)
		
		yTestSHist=np.histogram(yTestS,bins=binsTab,density=False) #Density=True to normalise, not necessary here we just want the max
		yTestBHist=np.histogram(yTestB,bins=binsTab,density=False)
		maxS=np.argmax(yTestSHist[0]) #first tab is the histogram values, second is the bins values on axis
		maxB=np.argmax(yTestBHist[0])
		interval=abs(binsTab[maxS]-binsTab[maxB])
		#print(f'Interval between Signal and backgroung max in histogram: {interval} pm {errorMax}')
	return interval,yTrainSHist,yTrainBHist,yTestSHist,yTestBHist


def monoExp(x, a, b):
	
	return (a * np.exp(-b * x) )

def fitBackground(Xselected,printPlotFit):
	bgMin=5500
	bgMax=6500
	steps=21
	massTab=np.linspace(bgMin,bgMax,steps)
	binCenters=[0.5*(massTab[i]+massTab[i+1]) for i in range (len(massTab)-1)]
	massB0=Xselected['B_s0_DTF_M']
	massHist,binEdges=np.histogram(massB0,bins=massTab,density=False)
	#print(f'massHist:{massHist}')
	params, paramCov = curve_fit(monoExp, binCenters, massHist,bounds=(0, [100000, 0.001]))
	a, b=params
	a=float(a)
	b=float(b)
	
	#print(f'a: {a}, b: {b}, c:{c}')
	#print(f'paramCov: {paramCov}')
	#dA, dB, dC=paramCov # Estimated covarience on parameters
	if printPlotFit:
		fig, ax = plt.subplots(figsize=(8, 4), dpi=300) 
		ax.plot(binCenters,massHist,'k.',label=f'Mass Histogram')
		ax.plot(binCenters,monoExp(np.array(binCenters),*params),'b--', # * in call to expands the tuple into separate elements
									label=f'fit: N={"{0:.4f}".format(a)}*exp(-{"{0:.4f}".format(b)}*M)')
		plt.xlabel('B_0 mass')
		plt.ylabel('Number of events')
		plt.legend()
		plt.show()
		plt.close()	

	blindingMin=5100
	blindingMax=5500
	return (steps-1)/(bgMax-bgMin)*( (a/b)*(np.exp(-b*blindingMin)-np.exp(-b*blindingMax)) )#+ c*(blindingMax-blindingMin) )