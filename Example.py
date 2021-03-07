#Import libraries:
from root_pandas import read_root
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

print(allvars) # Display chosen variables:


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
X_MC = X_MC.loc[(X_MC['B_s0_DTF_M'] > 5500) & (X_MC['B_s0_DTF_M'] < 6500)]
X_data = X_data.loc[(X_data['B_s0_DTF_M'] > 5500) & (X_data['B_s0_DTF_M'] < 6500)] 
print("Adding new variables:")


#Adding variables from analysis doc
print("MIN_IPCHI2_emu")
X_MC['MIN_IPCHI2_emu']=np.minimum(X_MC['eplus_IPCHI2_OWNPV'],X_MC['muminus_IPCHI2_OWNPV'])  # Minimum impact parameter between the two tracks
X_data['MIN_IPCHI2_emu']=np.minimum(X_data['eplus_IPCHI2_OWNPV'],X_data['muminus_IPCHI2_OWNPV'])

print("SUM_isolation_emu")
X_MC['SUM_isolation_emu']=X_MC['B_s0_D1_isolation_Giampi']+X_MC['B_s0_D2_isolation_Giampi'] # Sum of isolation of electron and muon (expect 0 in signal)
X_data['SUM_isolation_emu']=X_data['B_s0_D1_isolation_Giampi']+X_data['B_s0_D2_isolation_Giampi']

print("LOG1_cosDIRA")
X_MC['LOG1_cosDIRA']=np.log(1-X_MC['B_s0_BPVDIRA'])                                         # log(1-cos(DIRA))
X_data['LOG1_cosDIRA']=np.log(1-X_data['B_s0_BPVDIRA'])

print("MAX_PT_emu")
X_MC['MAX_PT_emu']=np.maximum(X_MC['eplus_PT'],X_MC['muminus_PT'])                          # Max of daughter P_T
X_data['MAX_PT_emu']=np.maximum(X_data['eplus_PT'],X_data['muminus_PT'])

print("DIFF_ETA_emu")
X_MC['DIFF_ETA_emu']=abs(X_MC['eplus_ETA']-X_MC['muminus_ETA'])                             # Difference in pseudorapidity, absolute value
X_data['DIFF_ETA_emu']=abs(X_data['eplus_ETA']-X_data['muminus_ETA'])
print("Dataset complete")


#Plot variables in histograms and save to pdf:
varToPlot=['B_s0_TAU','MIN_IPCHI2_emu','B_s0_IP_OWNPV','SUM_isolation_emu','B_s0_PT','LOG1_cosDIRA','B_s0_CDFiso','MAX_PT_emu','B_s0_ENDVERTEX_CHI2','DIFF_ETA_emu']  # Important variables from the overleaf analysis document to plot, in order
for var in varToPlot:
    range_low  = min(X_data[var].min(), X_MC[var].min())
    range_high = max(X_data[var].max(), X_MC[var].max())

    plt.hist(X_data[var], bins=100, label='data', density=True, alpha=0.7, range=(range_low, range_high))
    plt.hist(X_MC[var],   bins=100, label='MC',   density=True, alpha=0.7, range=(range_low, range_high))

    plt.xlabel(var)
    plt.ylabel('Arbitrary units')
    
    plt.legend(frameon=False)
    plt.savefig(f'plots/data_MC_{var}.pdf')
    plt.close()
