#==============================================================================
# Imports
#==============================================================================
import matplotlib.pyplot as plt
import numpy as np
from physics.constants import q

#%%
#==============================================================================
# Define general functions for creating plots
#==============================================================================
def Plot_time_dependent_quantity(figsize,quantity,ylabel,filename,filetype,
                                 UnitConversionFactor,MatPropDict,
                                 Time,SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=True,Semilog=False,SavePlot=False):
    
    fig,ax = plt.subplots(figsize=figsize)
    
    SelectTime = Time.squeeze()*10**15
    SelectQuantity = (UnitConversionFactor*
                      quantity[:,SelectWavelengthIndex,SelectIntensityIndex])
    
    if Total: #For plotting the total contribution from all VBs
        ax.plot(SelectTime,np.sum(SelectQuantity,axis=1),color='gray')

    else: #For plotting the quantity for each VB
        LineObjects = ax.plot(SelectTime,SelectQuantity)
        labels = [MatPropDict['VB name'][i] for i in range(MatPropDict['VB name'].size)]
        ax.legend(LineObjects, labels)
    
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel(ylabel)
    
    if Semilog:
        plt.semilogy()
    
    if SavePlot:
        plt.savefig(filename + filetype)


def EnergyDistributionHistogram(ax, EnergyDistribution, nCB_contrib, Time,
                                SelectWavelengthIndex, SelectIntensityIndex,
                                SelectTime=0,NumBins=20,Total=True):
    
    SelectTimeIndex = min(range(len(Time)),key=lambda i: abs(Time[i]-SelectTime)) #Closest index
    SelectTime = Time[SelectTimeIndex] #Closest value
    
    arr = EnergyDistribution[SelectTimeIndex,0:SelectTimeIndex,
                             SelectWavelengthIndex,SelectIntensityIndex,:]/q
    weights = nCB_contrib[0:SelectTimeIndex,SelectWavelengthIndex,
                          SelectIntensityIndex,:]*10**-6

    if Total: #For plotting the total energy distribution
        hist,bins = np.histogram(arr, weights=weights, bins=NumBins)
        width = 0.7*(bins[1] - bins[0])
        center = (bins[:-1] + bins[1:])/2
        ax.bar(center,hist,width=width,color='gray',zorder=0,alpha=0.8)

    else: #For plotting the energy distribution for each VB
        for idx in range(arr.shape[1]):
            hist,bins = np.histogram(arr[:,idx], weights=weights[:,idx], bins=NumBins)
            width = 0.7*(bins[1] - bins[0])
            center = (bins[:-1] + bins[1:])/2
            ax.bar(center,hist,width=width)

    return center, width, hist


def EnergyDistributionCDF(ax, EnergyDistribution, nCB_contrib, Time,
                                SelectWavelengthIndex, SelectIntensityIndex,
                                SelectTime=0,NumBins=20):
    
    SelectTimeIndex = min(range(len(Time)),key=lambda i: abs(Time[i]-SelectTime)) #Closest index
    SelectTime = Time[SelectTimeIndex] #Closest value
    
    arr = EnergyDistribution[SelectTimeIndex,0:SelectTimeIndex,
                             SelectWavelengthIndex,SelectIntensityIndex]/q
    weights = nCB_contrib[0:SelectTimeIndex,SelectWavelengthIndex,
                          SelectIntensityIndex]*10**-6
                          
    ax.hist(arr, weights=weights, cumulative=True, density=True, bins=NumBins,
            histtype='step', linewidth=1.5)