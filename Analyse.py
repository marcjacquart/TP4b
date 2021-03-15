import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc 	# roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp 			# Kolmogorov–Smirnov test to compare 2 distributions


pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']						# 1: signal, 0: background
print("csv loaded")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) # est_size=0.3, train_size=0.7. random state: seed for random assignation of data in the split. Seed given so the test sample is different of training sample even after saving and reopening the bdt

features = ['B_s0_TAU',				# B lifetime
			'MIN_IPCHI2_emu',		# Minimum on the two impact parameters chi2, large if signal (comes from secondary vertex)
			'B_s0_IP_OWNPV',		# Impact parameter
			'SUM_isolation_emu',	# Isolation: we want nothing in the cone around e or mu
			'B_s0_PT',				# Transverse momentum of B0
			'LOG1_cosDIRA',			# Direction angle between sum emu and B0 reconstructed -> log(1-cos(DIRA))
			'B_s0_CDFiso',			# Different measure of isolation, mathematical def in analysis document
			'MAX_PT_emu',			# B0 has high mass -> high p_T for daughter particles 
			'B_s0_ENDVERTEX_CHI2',	# Quality of the reconstructed decay vertex.
			'DIFF_ETA_emu']			# Absolute difference in pseudorapidity of emu


#  Retrive model
pathModel = '/home/mjacquar/TP4b/model'
depth = 3
model = joblib.load(f'{pathModel}/bdt{depth}_model.pkl')

# Use it to predict y values on X_test sample
y_pred=model.predict_proba(X_test[features])[:,1]	# 2 values, takes first: proba to be signal
y_pred_on_train=model.predict_proba(X_train[features])[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred) 	# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
aucValue = auc(fpr, tpr) 							# Use built in fct to compute area under curve

# Plot the result: Auc
# plt.figure(figsize=(8, 8), dpi=300)
# plt.plot(tpr,1-fpr,linestyle='-',label=f'Auc={auc}')
# plt.xlabel('True positive rate')
# plt.ylabel('1-False positive rate')
# plt.legend()
# plt.savefig(f'plots/rocAnaBiased.pdf')
# plt.close()



yTrainS=y_pred_on_train[y_train==1]
yTrainB=y_pred_on_train[y_train==0]
yTestS=y_pred[y_test==1]
yTestB=y_pred[y_test==0]
# lenyTrainS=len(yTrainS)
# lenyTrainB=len(yTrainB)
# lenyTestS=len(yTestS)
# lenyTestB=len(yTestB)
# Plot the result: y_pred
binsTab=np.linspace(0.1, 0.9, num=41)
plt.figure(figsize=(8, 8), dpi=300)
fig, ax = plt.subplots() # Want to stack histograms on top of same axes
ax.hist(yTrainS,  bins=binsTab, color='r',histtype='bar',density=True, alpha=0.3) 
ax.hist(yTrainB,  bins=binsTab, color='b', histtype='bar',density=True, alpha=0.3) 
ax.hist(yTrainS,  bins=binsTab, color='r',histtype='step',density=True,label='S (train)') 
ax.hist(yTrainB,  bins=binsTab, color='b', histtype='step',density=True, label='B (train)') 
yTestSHist = plt.hist(yTestS,  bins=binsTab,density=True,alpha = 0.0)
#print(yTestSHist)
yTestBHist = plt.hist(yTestB,  bins=binsTab,density=True,alpha = 0.0)
bin_centers = 0.5*(binsTab[1:] + binsTab[:-1])
#ax.hist(yTestS,  bins=binsTab, color='r', histtype='step',density=True, label='yTestS') 
#ax.hist(yTestB,  bins=binsTab, color='b', histtype='step',density=True, label='yTestB')
ax.scatter(bin_centers, yTestSHist[0], marker='o', c='r', s=20, alpha=1,label='S (test)') 
ax.scatter(bin_centers, yTestBHist[0], marker='o', c='b', s=20, alpha=1,label='S (test)') 

plt.xlabel('BDT signal response')
plt.ylabel('Normalized number of events')
plt.legend()
plt.savefig(f'plots/yPred.pdf')
plt.close()


# Bias difference:

# valuesDepth=[1,2,3,4,5,6,7,8,9,10]
# aucDepth=[]
# for depth in valuesDepth:
# 	model = joblib.load(f'{pathModel}/bdt{depth}_model.pkl')

# 	# Use it to predict y values on X_test sample
# 	y_pred=model.predict_proba(X_test[features])[:,1]	# 2 values, takes first: proba to be signal


# 	fpr, tpr, threshold = roc_curve(y_test, y_pred) 	# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
# 	aucValue = auc(fpr, tpr) 							# Use built in fct to compute area under curve
# 	print(f'Depth= {depth}, Auc={aucValue}')
# 	aucDepth.append(aucValue)

# nonBiased=[0.9694694238382793,0.9752754910380111,0.978709794734407,0.9826026459068448,0.9844838818924959,0.9860500966700275,0.9869476553904142,0.987378132074234,0.9866472542841878,0.985312914806173]
# plt.figure(figsize=(8, 8), dpi=300)
# plt.plot(valuesDepth,aucDepth,'.',label=f'Biased')
# plt.plot(valuesDepth,nonBiased,'.',label=f'Non biased')
# plt.xlabel('Tree depth')
# plt.ylabel('Auc')
# plt.legend()
# plt.savefig(f'plots/aucDepthBIASED.pdf')
# plt.close()

