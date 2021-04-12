import pandas as pd
import numpy as np
import joblib
from Functions import predictionY_interval				# Import function, will run the script
from sklearn.model_selection import train_test_split 	# To use method with same name

# Import data:
pathCSV='/home/mjacquar/TP4b/csv'
X = pd.read_csv(f'{pathCSV}/X.csv')
y = X['sig']						# 1: signal, 0: background
print("csv loaded")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0) # test_size=0.3, train_size=0.7. random state: seed for random assignation of data in the split. Seed given so the test sample is different of training sample even after saving and reopening the bdt

# Import Results file
dfOutput=pd.read_csv(f'{pathCSV}/HyperScanFromRFE.csv', sep=' ', names=['algoName','learningRate','maxDepth','nTrees','auc'])
#print(dfOutput)


features =['B_s0_ENDVERTEX_CHI2', 'B_s0_CDFiso', 'muminus_ProbNNmu', 'eplus_ProbNNe', 'eplus_ETA', 'B_s0_IPCHI2_OWNPV', 'B_s0_minPT', 'MIN_IPCHI2_emu','SUM_isolation_emu', 'LOG1_cosDIRA'] # List from the recursive feature elimination

# Results we want to add while the model is tested:
dist=[]
dfOutput=dfOutput[::-1]# Reverse DF to take last
pathModel='/home/mjacquar/TP4b/modelOpti/HyperScanFromRFE'
for index,row in dfOutput.iterrows():
	model=joblib.load(f'{pathModel}/bdt_{row[0]}_lr{row[1]}_{row[3]}Trees_{row[2]}Depth.pkl')	# Load trained model
	#print(f'{pathModel}/bdt_{row[0]}_lr{row[1]}_{row[3]}Trees_{row[2]}Depth.pkl')
	y_pred_onTest=model.predict_proba(X_test[features])[:,1]
	#print('test')
	computedDist=predictionY_interval(y_train, y_test, [],y_pred_onTest, False,   True)[0]
	#print('fin test')
	dist.append(computedDist) 	# Dist is the first output of the fct, no need prediction on train so put []
	print(computedDist)
dfOutput['dist']=dist


# Save in csv:
dfOutput.to_csv(f'{pathCSV}/HyperScanFromRFEdist.csv',sep=' ',index=False)
