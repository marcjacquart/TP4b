from sys import argv									# To pass variables
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split 	# To use method with same name
from sklearn.metrics import roc_curve, auc 				# Roc: receiver operating characteristic, auc: Area under the ROC Curve
import pandas as pd
import numpy as np
import joblib
import csv 												# Write aoc result in file


pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']
#csv loaded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) #random state: seed for random assignation of data in the split, done wih kFold

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

# Load variables:
algoName = argv[1]
learningRate=float(argv[2])			# Must convert because argv pass the arguments as strings
maxDepth=int(argv[3])
nTrees=int(argv[4])

dt    = DecisionTreeClassifier(max_depth=maxDepth)													# Define the decision tree
model = AdaBoostClassifier(dt, algorithm=algoName, n_estimators=nTrees, learning_rate=learningRate)	# Define the model using the decision tree
print("Model defined")

model.fit(X_train[features], y_train)
print("model trained")


pathModel='/home/mjacquar/TP4b/modelOpti'			# https://medium.com/@harsz89/persist-reuse-trained-machine-learning-models-using-joblib-or-pickle-in-python-76f7e4fd707
joblib.dump(model,f'{pathModel}/bdt_{algoName}_lr{learningRate}_{nTrees}Trees_{maxDepth}Depth.pkl')		# Save trained model for later
#Model saved

y_pred=model.predict_proba(X_test[features])[:,1]
print("Predictions done")
fpr, tpr, threshold = roc_curve(y_test, y_pred) 	# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
auc = auc(fpr, tpr) 								# Use built in fct to compute area under curve
print(f'Auc={auc}')
with open(f'{pathCSV}/paramAuc.csv', 'a', newline='') as csvfile: #'a': append mode to not overwrite
	spamwriter = csv.writer(csvfile, delimiter=' ')
	spamwriter.writerow([algoName,learningRate,maxDepth,nTrees,auc])