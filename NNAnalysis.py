# This script:
#	-Does a simple performance analysis of a neural network (Train + AUC roc computation)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc 	# roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

# Import data:
pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']						# 1: signal, 0: background
print("csv loaded")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) # test_size=0.3, train_size=0.7. random state: seed for random assignation of data in the split. Seed given so the test sample is different of training sample even after saving and reopening the bdt

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
X=X[features]


solver='adam', # Good for datastet of >1000
alpha=1e-5,
hidden_layer_sizes=(5, 2)

pathModel='/home/mjacquar/TP4b/model/NeuralNetwork'	
model= joblib.load(f'{pathModel}/NN_testFast.pkl')


#Use it to predict the test sample:
y_pred=model.predict_proba(X_test[features])[:,1]
print("Predictions done")
fpr, tpr, threshold = roc_curve(y_test, y_pred) 	# Use built in fct to compute:  false/true positive read, using the answer and predictions of the test sample
auc = auc(fpr, tpr) 								# Use built in fct to compute area under curve
print(f'Auc={auc}')

# Plot the result:
plt.figure(figsize=(8, 8), dpi=300)
plt.plot(tpr,1-fpr,linestyle='-',label=f'Auc={auc}')
plt.xlabel('True positive rate')
plt.ylabel('1-False positive rate')
plt.legend()
plt.savefig(f'plots/rocMLPC.pdf')
plt.close()
