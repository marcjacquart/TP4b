# This script:
#	-Tries to reduce the number of features for he BDT by recursive feature elimination


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split 	# To use method with same name
from sklearn.metrics import roc_curve, auc 				# Roc: receiver operating characteristic, auc: Area under the ROC Curve
from sklearn.model_selection import KFold				# K-fold data selection for training
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib


pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']
print("csv loaded")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) #random state: seed for random assignation of data in the split, done wih kFold

features = list(X)
features= features[1:-1] # First line and last line aren't features, we get rid of them
tooCorrelated =[] # Must delete too corelated ones, we don't want two variables describing the same thing for training.
features = [x for x in features if x not in tooCorrelated]
print(features)

# Tuning hyperparameters: https://scikit-learn.org/stable/modules/grid_search.html#grid-search
# Feature selection: https://scikit-learn.org/stable/modules/feature_selection.html
