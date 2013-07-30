from __future__ import division
import numpy as np
#Function to calculate the Spike-Triggered-Average
def STA(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened = False
    meanout = np.mean(outputs)
    meanstim = np.mean(stimuli,axis=0)
    hzero = meanout
    hzeroold = hzero+1
    numiteration = 0
    hone = np.mean(np.array([stimulus*outputs[i] for i,stimulus in enumerate(stimuli)]),axis=0)-hzero*meanstim
    while np.absolute(hzero-hzeroold)>10**-10:
        numiteration +=1
        hone = np.mean(np.array([stimulus*outputs[i] for i,stimulus in enumerate(stimuli)]),axis=0)-hzero*meanstim
        hzeroold = hzero
        hzero = meanout-np.dot(hone,meanstim)
        if whitened:
            #pseudoinverse of autocorrelation matrix
            aci = np.linalg.pinv(np.mean(np.array([np.outer(stim,stim) for stim in stimuli]),axis=0))
            hone = np.dot(aci,hone)
        if numiteration > 100:
            print 'Kernels not converging'
            break
    return np.concatenate((np.array([hzero]),hone))
