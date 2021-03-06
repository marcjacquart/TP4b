#Import libraries:
from root_pandas import read_root
import pandas as pd
import matplotlib.pyplot as plt

#Define useful variables, split into categories to match root variables names (see line 15):
common = [  'P',                    # Momentum (not used)
            'PT',                   # Transverse momentum, We will use only B_(P_T)
            'ETA',                  # Pseudorapidity
            'IP_OWNPV',             # Impact parameter with respect to his own production primary vertex "OWNPV": primary vertex reconstructed with the tracks of other products of collision
            'IPCHI2_OWNPV',]        # Chi2 of the PV fit: how well is the PV defined

Bvars  = [  'DTF_M',                # Reconstructed invarient mass
            'DOCA',                 # Distance of closest approach, high for background. Meaning for a decay???
            'ENDVERTEX_CHI2',       # Quality of the reconstructed decay vertex.
            'BPVDIRA',              # Computes the cosine of the angle between the momentum of the particle and the direction fo flight from the best PV to the decay vertex.
            'TAU',                  # Proper time before decay
            'FD_OWNPV',             # Flight distance from primary vertex
            'CDFiso',               # Isolation (see if other track in a cone arount the selected track. If so probably background)
            'D1_isolation_Giampi',  # Giampi's isolation tool: number of tracks in a cone of angle 0.27 rad around the reconstructed track.
            'D2_isolation_Giampi',] # We want the sum of both isolation number. signal will be 0 (no other close trcks in the daughter particles)

muvars = [  'ProbNNmu',             # Bayesian posteriori probability: The posterior probability distribution is the probability distribution of an unknown quantity, treated as a random variable, conditional on the evidence obtained from an experiment or survey. 
            'ProbNNk',]             # --- same for kaon?

evars  = [  'ProbNNe',]             # --- same for electron?


#Fill tab with variable names:
allvars=[]
for var in Bvars:
    allvars.append(f'B_s0_{var}')
for var in muvars:
    allvars.append(f'muminus_{var}')
for var in evars:
    allvars.append(f'eplus_{var}')
for var in common:
    for p in ('B_s0', 'eplus', 'muminus'):
        allvars.append(f'{p}_{var}')
        
allvars.append('eventNumber')

#Display chosen variables:
print(allvars) 

# Data path & import with variable names:
clusterPath="/panfs/tully/B2emu"
X_data_2016_MU = read_root(clusterPath+"/data/Selected_B2emu2016MU.root", columns=allvars)
X_data_2016_MD = read_root(clusterPath+"/data/Selected_B2emu2016MD.root", columns=allvars)
X_MC_2016_MU = read_root(clusterPath+"/MC/Selected_B2emu2016MU.root", columns=allvars)
X_MC_2016_MD = read_root(clusterPath+"/MC/Selected_B2emu2016MD.root", columns=allvars)

#Putting togeter up and down polarity for larger data sample:
X_data_2016 = pd.concat([X_data_2016_MU, X_data_2016_MD]) 
X_MC_2016 = pd.concat([X_MC_2016_MU, X_MC_2016_MD])

#Train above the target to get only noise:
X_MC_2016 = X_MC_2016.loc[(X_MC_2016['B_s0_DTF_M'] > 5500) & (X_MC_2016['B_s0_DTF_M'] < 6500)]
X_data_2016 = X_data_2016.loc[(X_data_2016['B_s0_DTF_M'] > 5500) & (X_data_2016['B_s0_DTF_M'] < 6500)] 

#Plot variables in histograms and save to pdf:
for var in allvars:
    range_low  = min(X_data_2016[var].min(), X_MC_2016[var].min())
    range_high = max(X_data_2016[var].max(), X_MC_2016[var].max())

    plt.hist(X_data_2016[var], bins=100, label='data', density=True, alpha=0.7, range=(range_low, range_high))
    plt.hist(X_MC_2016[var],   bins=100, label='MC',   density=True, alpha=0.7, range=(range_low, range_high))

    plt.xlabel(var)
    plt.ylabel('Arbitrary units')
    
    plt.legend(frameon=False)
    plt.savefig(f'plots/data_MC_{var}.pdf')
    plt.close()
