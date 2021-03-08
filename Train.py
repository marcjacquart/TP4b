from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) #random state: seed for random assignation of data in the split

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


dt    = DecisionTreeClassifier(max_depth=3)												# Define the decision tree
model = AdaBoostClassifier(dt, algorithm='SAMME.R', n_estimators=50, learning_rate=0.1)	# Define the model using the decision tree

model.fit(X_train[features], y_train)

# https://medium.com/@harsz89/persist-reuse-trained-machine-learning-models-using-joblib-or-pickle-in-python-76f7e4fd707