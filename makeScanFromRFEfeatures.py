# This script:
#	-Makes the shell script to launch grid auc optimization on lphe cluster using the submit.py file
#	-Create new file for new series of scans from RFE optimized features 


algoName = ['SAMME.R']#,'SAMME']
learningRate=[0.05,0.2,0.4,0.6,0.8]						# First scan
maxDepth=[3,6,9,12,20]
nTrees=[20,80,200,500,1000]

def notInOld(lr,learningRate,maxD,maxDepth,nT,nTrees): # Skip already computed points from past scans
	return not((lr in learningRate) & (maxD in maxDepth) & (nT in nTrees))
# To use:
# if notInOld(lr,learningRate,maxD,maxDepth,nT,nTrees)
#else:
#						print(f'learningRate: {lr}, maxDepth: {maxD}, nTrees:{nT}')										# Print the skipped measurement (already exist from previous run) 

i=1
with open ('runScan.sh', 'w') as rsh:
	for algo in algoName:
		for lr in learningRate:
			for maxD in maxDepth:
				for nT in nTrees:
					 
					rsh.write(f'python3 submit.py -d {i}_scan "python3 MultiTrain.py \'{algo}\' {lr} {maxD} {nT}"\n') 	# Just he number of line in description to easily execute again if fail, write the command in the bash file
					i=i+1
					