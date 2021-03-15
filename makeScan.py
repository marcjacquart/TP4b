algoName = ['SAMME.R','SAMME']
learningRate=[0.05,0.1,0.15,0.2]			
maxDepth=[2,4,6,8,10]
nTrees=[20,40,60,80,100]

with open ('runScan.sh', 'w') as rsh:
	for algo in algoName:
		for lr in learningRate:
			for maxD in maxDepth:
				for nT in nTrees:
					rsh.write(f'python3 submit.py -d MultiparamScan "python3 MultiTrain.py \'{algo}\' {lr} {maxD} {nT}"\n')
								
