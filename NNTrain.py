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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) # test_size=0.3, train_size=0.7. random state: seed for random assignation of data in the split. Seed given so the test sample is different of training sample even after saving and reopening the bdt


solver='adam' # Good for datastet of >1000
alpha=1e-5
n_layers_=4
max_iter=100
hidden_layer_sizes=(15,10,5)

model= MLPClassifier(	solver=solver, 
						alpha=alpha,
						max_iter=max_iter,
						hidden_layer_sizes=hidden_layer_sizes, 
						random_state=0)

model.fit(X_train, y_train)


pathModel='/home/mjacquar/TP4b/model/NeuralNetwork'				# https://medium.com/@harsz89/persist-reuse-trained-machine-learning-models-using-joblib-or-pickle-in-python-76f7e4fd707
joblib.dump(model,f'{pathModel}/NN_testFast.pkl')					# Save trained model for later# Save trained model for later
print("Model saved")
