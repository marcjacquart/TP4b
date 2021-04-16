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

existingKfold=False # Have to train the model if not existing


# Import Data:
pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']
print("csv loaded")

features = ['B_s0_ENDVERTEX_CHI2', 
			'B_s0_CDFiso', 
			'muminus_ProbNNmu', 
			'eplus_ProbNNe', 
			'eplus_ETA', 
			'B_s0_IPCHI2_OWNPV',
			'B_s0_minPT', 
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


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
pathModel='/home/mjacquar/TP4b/model/Kfold'
fig, ax = plt.subplots()
X=X[features] # Only selects features from list
#print(X)
for i, (train, test) in enumerate(cv.split(X, y)):
	#print(train)
	#print(test)
	#if not existingKfold:
	model.fit(X.iloc[train], y.iloc[train])
	joblib.dump(model,f'{pathModel}/bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth_Kfold{i}.pkl')		# Save trained model for later
	viz = plot_roc_curve(model, X.iloc[test], y.iloc[test],
							name='ROC fold {}'.format(i),
							alpha=0.3, lw=1, ax=ax)
			
	interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
	interp_tpr[0] = 0.0
	tprs.append(interp_tpr)
	aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
		label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
		title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.savefig(f'plots/Kfold_bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth.pdf')
plt.close()