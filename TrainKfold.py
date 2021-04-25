# Part of the code taken from example https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split 	# To use method with same name
from sklearn.metrics import roc_curve, auc 				# Roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import KFold				# K-fold data selection for training
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

existingKfold=True # Have to train the model if not existing


# Import Data:
pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']
#print("csv loaded")
#print(X.columns)

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

# Run classifier with cross-validation and plot ROC curves
cv = KFold(n_splits=5, random_state=0, shuffle=True) # K-fold is a Cross-validation method

algoName = 'SAMME.R' 				# Define Model
maxDepth = 3
nTrees = 1000
learningRate=0.1
dt    = DecisionTreeClassifier(max_depth=maxDepth)														# Define the decision tree
model = AdaBoostClassifier(dt, algorithm=algoName, n_estimators=nTrees, learning_rate=learningRate)		# Define the model using the decision tree

fprs = []
tprs = []
aucs = []
thresholds = []
mean_fpr = np.linspace(0, 1, 100)
pathModel='/home/mjacquar/TP4b/model/Kfold'
fig, ax = plt.subplots()
X=X[features] # Only selects features from list
#print(X)
for i, (train, test) in enumerate(cv.split(X, y)):
	print(i)
	if existingKfold:
		model=joblib.load(f'{pathModel}/bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth_Kfold{i}.pkl')
	else:
		model.fit(X.iloc[train], y.iloc[train])
		joblib.dump(model,f'{pathModel}/bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth_Kfold{i}.pkl')		# Save trained model for later
	
	y_pred_onTest=model.predict_proba(X.iloc[test])[:,1]						# 2 values, takes first: proba to be signal. test on the parameter list from csv file
	fpr, tpr, threshold = roc_curve(y.iloc[test], y_pred_onTest) 	# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
	tprs.append(tpr)
	fprs.append(fpr)
	thresholds.append(threshold)
	
	aucValue = auc(fpr, tpr) 								# Use built in fct to compute area under curve
	aucs.append(aucValue)
	
fig, ax = plt.subplots(figsize=(4.0,4.0), dpi=500) 
ax.plot([1, 0], [0, 1], linestyle='--', lw=1, color='r',
        label='Chance', alpha=.8)
for i in range(len(fprs)):
	ax.plot(tpr,1-fpr,linestyle='-',label=f'Kfold{i+1}: Auc={"{0:.4f}".format(aucs[i])}')
mean_auc = np.mean(aucs)#auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot([0], [0], linestyle='--', lw=1, color='k',
        label=f'Mean Auc: {"{0:.4f}".format(mean_auc)}±{"{0:.4f}".format(std_auc)}', alpha=0.0)
plt.xlabel('True positive rate')
plt.ylabel('1-False positive rate')
plt.legend(loc='lower left')
plt.savefig(f'plots/NewKfold_bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth.pdf')
plt.close()