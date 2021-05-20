import os
import time
from sys import argv
import pandas as pd
import numpy as np
import csv

def allButOne(list,excludeName):
	result=[]
	for element in list:
		if element != excludeName:
			result.append(element)

	return result

def writeStringFromList(list):
	result=""
	for element in list:
		result+=element
		result+=','
	result = result[:-1] # Delete lst ','
	return result

# Initial list: B_s0_TAU,B_s0_ENDVERTEX_CHI2,B_s0_BPVDIRA,B_s0_CDFiso,eplus_ProbNNe,eplus_PT,muminus_PT,eplus_ETA,muminus_ETA,eplus_IP_OWNPV,B_s0_IPCHI2_OWNPV,B_s0_minP,B_s0_absdiffP,B_s0_minPT,B_s0_absdiffPT,B_s0_absdiffETA,B_s0_minIP_OWNPV,B_s0_absdiffIP_OWNPV,muminus_ProbNNmuk,MIN_IPCHI2_emu,SUM_isolation_emu,LOG1_cosDIRA
# scancel -u mjacquar

mainPath='/home/mjacquar/TP4b/'

# Import variables
features=[element for element in argv[1].split(',')] # Features to test at this stage
lenFeatures=len(features)
N=lenFeatures-1 # Number of features trained in each model: number of features -1 for the permutation
print(N)
if N==0:
	exit() # End of scan if everything goes well, script is called with only the last best variable and nothing is left to do.

# Launch N models:
for element in features:
	featuresToTest=allButOne(list=features,excludeName=element)
	toTestString=writeStringFromList(list=featuresToTest)
	#print(f'python3 submit.py -d test{N} "python3 {mainPath}MultiTrain.py {toTestString}"')

	os.system(f'python3 {mainPath}submit.py -d test{N} "python3 {mainPath}MultiTrain.py {toTestString} {element}"')



pathCSV='/home/mjacquar/TP4b/csv/permutationImportance/'
waitingResult=True
while waitingResult:
	time.sleep(60) # Wait 10 minutes
	csvList = pd.read_csv(f'{pathCSV}/N{N}.csv',sep=',',names=['features','testedFeature','lenFeatures','auc'])
	row_count = sum(1 for row in csvList['features'])
	#print(csvList)
	#print(f'row_count:{row_count}')
	#print(f'N+2:{N+2}')
	if row_count == N+1: # The csv file is full = all the models have been trained, +1 for number of total features
		maxAuc= csvList['auc'].max()
		#print(f'maxAuc:{maxAuc}')
		argMaxAuc= csvList['auc'].idxmax()
		worstFeature=csvList['testedFeature'][argMaxAuc]

		# Write result for final analysis
		with open(f'{pathCSV}result.csv', 'a', newline='') as csvfile: # 'a': append mode to not overwrite
			spamwriter = csv.writer(csvfile, delimiter=',')
			spamwriter.writerow([N, worstFeature, maxAuc])

		# Launch recursively new round:
		listToContinue=allButOne(list=features,excludeName=worstFeature)
		stringToContinue=writeStringFromList(list=listToContinue)
		#print(f'python3 submit.py -d main{N} "python3 {mainPath}permutationImportance.py {stringToContinue}"')
		os.system(f'python3 {mainPath}submit.py -d main{N} "python3 {mainPath}permutationImportance.py {stringToContinue}"')
	
		# Exit loop:
		waitingResult=False


exit()
