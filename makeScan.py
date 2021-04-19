# This script:
#	-Makes the shell script to launch grid auc optimization on lphe cluster using the submit.py file


algoName = ['SAMME.R']#,'SAMME']
learningRate=[0.05,0.1,0.15,0.2]						# First scan
maxDepth=[2,4,6,8,10]
nTrees=[20,40,60,80,100]

learningRateNew=[0.125,0.15,0.175,0.2,0.225,0.25]		# Second scan
maxDepthNew=[4,5,6,7,8]
nTreesNew=[60,70,80,90,100,110,120,130]

learningRateNew2=[0.15,0.2,0.25,0.3]					# Third scan
maxDepthNew2=[2,4,6,8]
nTreesNew2=[100,200,500]


def notInOld(lr,learningRate,maxD,maxDepth,nT,nTrees): # Skip already computed points from past scans
	return not((lr in learningRate) & (maxD in maxDepth) & (nT in nTrees))

i=1
with open ('runScan.sh', 'w') as rsh:
	for algo in algoName:
		for lr in learningRateNew2:
			for maxD in maxDepthNew2:
				for nT in nTreesNew2:
					if notInOld(lr,learningRate,maxD,maxDepth,nT,nTrees) or notInOld(lr,learningRateNew,maxD,maxDepthNew,nT,nTreesNew):
						rsh.write(f'python3 submit.py -d {i} "python3 MultiTrain.py \'{algo}\' {lr} {maxD} {nT}"\n') 	# Just he number of line in description to easily execute again if fail, write the command in the bash file
						i=i+1
					else:
						print(f'learningRate: {lr}, maxDepth: {maxD}, nTrees:{nT}')										# Print the skipped measurement (already exist from previous run) 
