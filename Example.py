#Import libraries:
from root_pandas import read_root
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Define useful variables, split into categories to match root variables names (see line 15):
common = [  'P',                    # Momentum (not used)
            'PT',                   # Transverse momentum
            'ETA',                  # Pseudorapidity: Need maximum between e and mu
            'IP_OWNPV',             # Impact parameter with respect to his own production primary vertex "OWNPV": primary vertex reconstructed with the tracks of other products of collision
            'IPCHI2_OWNPV',]        # Chi2 of the PV fit: how well is the PV defined. How does it change between e and mu if same vertex?

Bvars  = [  'TAU',                  # Proper time before decay
            'DTF_M',                # Reconstructed invarient mass
            'DOCA',                 # Distance of closest approach, high for background. Meaning for a decay???
            'ENDVERTEX_CHI2',       # Quality of the reconstructed decay vertex.
            'BPVDIRA',              # Computes the cosine of the angle between the momentum of the particle and the direction fo flight from the best PV to the decay vertex.
            'FD_OWNPV',             # Flight distance from primary vertex
            'CDFiso',               # Isolation (see if other track in a cone arount the selected track. If so probably background)
            'D1_isolation_Giampi',  # Giampi's isolation tool: number of tracks in a cone of angle 0.27 rad around the reconstructed track.
            'D2_isolation_Giampi',] # We want the sum of both isolation number. signal will be 0 (no other close trcks in the daughter particles)

muvars = [  'ProbNNmu',             # Bayesian posteriori probability: The posterior probability distribution is the probability distribution of an unknown quantity, treated as a random variable, conditional on the evidence obtained from an experiment or survey. 
            'ProbNNk',]             # --- same for kaon?

evars  = [  'ProbNNe',]             # --- same for electron?


#Fill the tab "allvars" with variable names:
allvars=[]
for var in Bvars:
    allvars.append(f'B_s0_{var}') #f string: python format: https://realpython.com/python-f-strings/
for var in muvars:
    allvars.append(f'muminus_{var}')
for var in evars:
    allvars.append(f'eplus_{var}')
for var in common:
    for p in ('B_s0', 'eplus', 'muminus'):
        allvars.append(f'{p}_{var}')
        
allvars.append('eventNumber')

#Display chosen variables:
#print(allvars) 


# Data path & import with variable names:
clusterPath="/panfs/tully/B2emu"
X_data_2016_MU = read_root(clusterPath+"/data/Selected_B2emu2016MU.root", columns=allvars)
X_data_2016_MD = read_root(clusterPath+"/data/Selected_B2emu2016MD.root", columns=allvars)
X_data_2017_MU = read_root(clusterPath+"/data/Selected_B2emu2017MU.root", columns=allvars)
X_data_2017_MD = read_root(clusterPath+"/data/Selected_B2emu2017MD.root", columns=allvars)
X_data_2018_MU = read_root(clusterPath+"/data/Selected_B2emu2018MU.root", columns=allvars)
X_data_2018_MD = read_root(clusterPath+"/data/Selected_B2emu2018MD.root", columns=allvars)
print("Data loaded")
X_MC_2016_MU = read_root(clusterPath+"/MC/Selected_B2emu2016MU.root", columns=allvars)
X_MC_2016_MD = read_root(clusterPath+"/MC/Selected_B2emu2016MD.root", columns=allvars)
X_MC_2017_MU = read_root(clusterPath+"/MC/Selected_B2emu2017MU.root", columns=allvars)
X_MC_2017_MD = read_root(clusterPath+"/MC/Selected_B2emu2017MD.root", columns=allvars)
X_MC_2018_MU = read_root(clusterPath+"/MC/Selected_B2emu2018MU.root", columns=allvars)
X_MC_2018_MD = read_root(clusterPath+"/MC/Selected_B2emu2018MD.root", columns=allvars)
print("MC loaded")

#Putting togeter up and down polarity from every year for larger data sample:
X_data = pd.concat([X_data_2016_MU, X_data_2016_MD,X_data_2017_MU, X_data_2017_MD,X_data_2018_MU, X_data_2018_MD]) 
X_MC = pd.concat([X_MC_2016_MU, X_MC_2016_MD,X_MC_2017_MU, X_MC_2017_MD,X_MC_2018_MU, X_MC_2018_MD])

print("Data concatenated")
#Train above the target to get only noise:
#X_MC = X_MC.loc[(X_MC['B_s0_DTF_M'] > 5500) & (X_MC['B_s0_DTF_M'] < 6500)] # No need for MC, only signal
X_data = X_data.loc[(X_data['B_s0_DTF_M'] > 5500) & (X_data['B_s0_DTF_M'] < 6500)] 
print("Adding new variables:")


#Adding more variables:
for df in (X_MC,X_data):
	for var in common:
		df[f'B_s0_max{var}']=df[[f'eplus_{var}',f'muminus_{var}']].max(axis=1)
		df[f'B_s0_min{var}']=df[[f'eplus_{var}',f'muminus_{var}']].min(axis=1)
		df[f'B_s0_sum{var}']=df[[f'eplus_{var}',f'muminus_{var}']].sum(axis=1)
		df[f'B_s0_absdiff{var}']=(df[f'eplus_{var}']-df[f'muminus_{var}']).abs()
		allvars.append(f'B_s0_max{var}')
		allvars.append(f'B_s0_min{var}')
		allvars.append(f'B_s0_sum{var}')
		allvars.append(f'B_s0_absdiff{var}')

	df['muminus_ProbNNmuk'] =df['muminus_ProbNNmu']*(1-df['muminus_ProbNNk'])
	allvars.append('muminus_ProbNNmuk')

	df['MIN_IPCHI2_emu']=np.minimum(df['eplus_IPCHI2_OWNPV'],df['muminus_IPCHI2_OWNPV'])  # Minimum impact parameter between the two tracks
	allvars.append('MIN_IPCHI2_emu')

	df['SUM_isolation_emu']=df['B_s0_D1_isolation_Giampi']+df['B_s0_D2_isolation_Giampi'] # Sum of isolation of electron and muon (expect 0 in signal)
	allvars.append('SUM_isolation_emu')

	df['LOG1_cosDIRA']=np.log(1-df['B_s0_BPVDIRA'])										  # log(1-cos(DIRA))
	allvars.append('LOG1_cosDIRA')

	df['MAX_PT_emu']=np.maximum(df['eplus_PT'],df['muminus_PT'])						  # Max of daughter P_T
	allvars.append('MAX_PT_emu')

	df['DIFF_ETA_emu']=abs(df['eplus_ETA']-df['muminus_ETA'])							  # Difference in pseudorapidity, absolute value
	allvars.append('DIFF_ETA_emu')

print("Dataset complete")


# #Plot correlation matrix between variables:
# plt.figure(figsize=(20,10),dpi=300,)
# corr=X_MC.corr()
# corr.style.background_gradient(cmap='coolwarm', axis=None)
# plt.matshow(corr)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.savefig('plots/correlation_X_MC.pdf')
# plt.close()

# plt.figure(figsize=(20,10),dpi=300,)
# corr=X_data.corr()
# corr.style.background_gradient(cmap='RdBu_r', axis=None)
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );
# plt.matshow(corr)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.savefig('plots/correlation_X_data.pdf')
# plt.close()




# Compute the correlation matrix
corr = X_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 13))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('plots/correlation_X_data.pdf')
plt.close()

#Same for MC
# Compute the correlation matrix
corr = X_MC.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 13))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('plots/correlation_X_MC.pdf')
plt.close()






# corr = X_MC.corr()
# ax = sns.heatmap(
#     corr, 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );
# plt.savefig('plots/correlation_X_dataHeatTest.pdf')
# plt.close()


#trainingVar=['B_s0_TAU','MIN_IPCHI2_emu','B_s0_IP_OWNPV','SUM_isolation_emu','B_s0_PT','LOG1_cosDIRA','B_s0_CDFiso','MAX_PT_emu','B_s0_ENDVERTEX_CHI2','DIFF_ETA_emu']                 # Important variables from the overleaf analysis document to plot, in order
#Plot variables in histograms and save to pdf:
# for var in allvars:
#     range_low  = min(X_data[var].min(), X_MC[var].min())
#     range_high = max(X_data[var].max(), X_MC[var].max())

#     plt.hist(X_data[var], bins=100, label='data', density=True, alpha=0.7, range=(range_low, range_high))
#     plt.hist(X_MC[var],   bins=100, label='MC',   density=True, alpha=0.7, range=(range_low, range_high))

#     plt.xlabel(var)
#     plt.ylabel('Arbitrary units')
    
#     plt.legend(frameon=False)
#     plt.savefig(f'plots/data_MC_{var}.pdf')
#     plt.close()
