# This script:
#	-Tries to reduce the number of features for he BDT by recursive feature elimination


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split 	# To use method with same name
from sklearn.metrics import roc_curve, auc 				# Roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import KFold				# K-fold data selection for training
from sklearn.feature_selection import RFE				# Recursive feature elimination

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib 											# To save model
import csv 												# To write result in CSV file
from sys import argv									# To pass variables



pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']
print("csv loaded")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, train_size=0.3, random_state=0) #random state: seed for random assignation of data in the split, done wih kFold

# From the selection in Example.py
features = ['B_s0_TAU', 'B_s0_DTF_M', 'B_s0_ENDVERTEX_CHI2', 'B_s0_BPVDIRA', 'B_s0_CDFiso', 'B_s0_D1_isolation_Giampi', 'B_s0_D2_isolation_Giampi', 'muminus_ProbNNmu', 'muminus_ProbNNk', 'eplus_ProbNNe', 'muminus_PT', 'eplus_ETA', 'muminus_ETA', 'B_s0_IPCHI2_OWNPV', 'eplus_IPCHI2_OWNPV', 'B_s0_minP', 'B_s0_absdiffP', 'B_s0_minPT', 'B_s0_minIP_OWNPV', 'B_s0_absdiffIPCHI2_OWNPV', 'MIN_IPCHI2_emu', 'LOG1_cosDIRA', 'MAX_PT_emu', 'DIFF_ETA_emu']
#nFeaturesTot=len(features)
#print (nFeaturesTot)
nSelect = int(argv[1]) 									# For the scan. Need to convert the string to int
#print(nSelect)
dt    = DecisionTreeClassifier(max_depth=3)
model = AdaBoostClassifier(dt, algorithm='SAMME.R', n_estimators=50, learning_rate=0.1)

sel = RFE(estimator=model,n_features_to_select=nSelect,step=1, verbose=0) 	# Recursive feature elimination. step=1: eject one feature at each step

sel.fit(X_train[features], y_train)											# Train the model


pathModel='/home/mjacquar/TP4b/modelOpti'									# https://medium.com/@harsz89/persist-reuse-trained-machine-learning-models-using-joblib-or-pickle-in-python-76f7e4fd707
joblib.dump(sel,f'{pathModel}/selection_3_50_0.1_{nSelect}variables.pkl')	# Save trained model for later

#print(sel.get_support())
featureArray=np.array(features)
#print(featureArray[sel.get_support()])					# Support is boolean array with selected index=True
finalFeatures=featureArray[sel.get_support()]

# Compute roc curve to measure performances:
y_pred=sel.predict_proba(X_test[features])[:,1]			# 2 values, takes first: proba to be signal
fpr, tpr, threshold = roc_curve(y_test, y_pred) 		# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
aucValue = auc(fpr, tpr) 								# Use built in fct to compute area under curve
print(f'Auc value: {aucValue}')



pathCSV='/home/mjacquar/TP4b/csv'
with open(f'{pathCSV}/paramSelection.csv', 'a', newline='') as csvfile:		# 'a': append mode to not overwrite
	spamwriter = csv.writer(csvfile, delimiter=' ')
	spamwriter.writerow([nSelect,aucValue,finalFeatures])

#selectedFeatures = features[sel.get_support()] # support is a boolean array. true = feature is selected
#print (selectedFeatures)

# Tuning hyperparameters: https://scikit-learn.org/stable/modules/grid_search.html#grid-search
# Feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
