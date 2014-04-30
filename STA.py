from __future__ import division
import numpy as np
import scipy as sp
#Function to calculate the Spike-Triggered-Average
def STASys(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened=False
    strfs = np.dot(stimuli.T,outputs)/float(stimuli.shape[0])
    if whitened:
        strfs = np.dot(np.linalg.pinv(np.dot(stimuli.T,stimuli)),strfs)
    return strfs.T

def sparseSTASys(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened = False
    strfs = np.dot(stimuli.T,outputs)/stimuli.shape[0]
    if whitened:
        strfs = np.dot(strfs,np.linalg.pinv(np.dot(outputs.T,outputs)))/stimuli.shape[0]
    return strfs.T

def STC(stimuli,outputs,meancov):
    #stcM = np.sum(np.array([np.outer(stimuli[ii],stimuli[ii])*outputs[ii] for ii in xrange(outputs.shape[0])]),axis=0)/np.mean(outputs)
    stcM = np.dot(stimuli.T,np.multiply(np.array([outputs]).T,stimuli))/np.mean(outputs)
    stcM = stcM-meancov
    return stcM

def STCSys(stimuli,outputs):
    meancov = np.dot(stimuli.T,stimuli)/stimuli.shape[0]
    stcs = np.array([STC(stimuli,outputs[:,ii],meancov) for ii in xrange(outputs.shape[1])])
    return stcs

def STC2(stimuli,outputs):
    #stcM = np.mean(np.array([np.outer(stimuli[ii],stimuli[ii])*outputs[ii] for ii in xrange(outputs.shape[0])]),axis=0)
    stcM = np.dot(stimuli.T,np.multiply(np.array([outputs]).T,stimuli))/outputs.shape[0]
    return stcM

def STCSys2(stimuli,outputs):
    outputs = np.absolute(outputs)
    stcs = np.array([STC2(stimuli,outputs[:,ii]) for ii in xrange(outputs.shape[1])])
    return stcs

def MaxRelDimSTC(stcs):
    vals = []
    strfs = []
    for stc in stcs:
        eVals,eVecs = sp.linalg.eigh(stc,turbo=True)
        idx = np.argsort(np.absolute(eVals))
        vals.append(eVals[idx][-1])
        strfs.append(eVecs[:idx][:,-1])
    return (np.array(vals),np.array(strfs))

def MaxStimDimSTC(stcs):
    vals = []
    strfs = []
    for stc in stcs:
        length = stcs[0].shape[0]
        eVal,eVec = sp.linalg.eigh(stc,eigvals=(length-1,length-1))
        vals.append(eVal)
        strfs.append(eVec)
    return (np.array(vals),np.array(strfs))
