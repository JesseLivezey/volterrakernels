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

def STASys(stimuli,outputs,whitened=None):
    nSTRFs = outputs.shape[1]
    hzeros = np.zeros(nSTRFs)
    hones = np.zeros((nSTRFs,stimuli.shape[1]))
    for ii in xrange(nSTRFs):
        temp = STA(stimuli,outputs[:,ii],whitened)
        hzeros[ii] = temp[0]
        hones[ii] = temp[1:]
    return (hzeros,hones)

def STC(stimuli,outputs):
    stcM = np.mean(np.array([np.outer(stimuli[ii],stimuli[ii])*outputs[ii] for ii in xrange(output.shape[0])]),axis=0)
    stcM = stcM-np.mean(np.array([np.outer(stimuli[ii],stimuli[ii]) for ii in xrange(output.shape[0])]),axis=0)
    return stcM

def STCSys(stimuli,outputs):
    stcs = np.array([STC(stimuli,outputs[:,ii]) for ii in xrange(outputs.shape[1])])
    return stcs

def MaxRelDimSTC(stcs):
    strfs = []
    for stc in stcs:
        eVals,eVecs = np.linalg.eigh(stc)
        idx = np.argsort(np.absolute(eVals))
        strfs.append((eVals[idx][-1],eVecs[idx][-1]))
    return strfs
