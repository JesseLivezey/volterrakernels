from __future__ import division
import numpy as np
#Function to calculate the Spike-Triggered-Average
def STA(stimuli,outputs,meanstim,whitened=None):
    if whitened is None:
        whitened = False
    meanout = np.mean(outputs)
    hzero = meanout
    hzeroold = hzero+1
    numiteration = 0
    staConst = np.zeros(stimuli.shape[1])
    for ii in xrange(stimuli.shape[0]):
        staConst += stimuli[ii]*outputs[ii]
    staConst = staConst/stimuli.shape[0]
    hone = staConst-hzero*meanstim
    if whitened:
        ac = np.zeros((stimuli.shape[1],stimuli.shape[1]))
        for ii in xrange(stimuli.shape[0]):
            ac += np.outer(stimuli[ii],stimuli[ii])
        ac = ac/stimuli.shape[0]
        aci = np.linalg.pinv(ac)
    while np.absolute(hzero-hzeroold)>10**-6:
        numiteration +=1
        hone = staConst-hzero*meanstim
        hzeroold = hzero
        if whitened:
            #pseudoinverse of autocorrelation matrix
            hone = np.dot(aci,hone)
        hzero = meanout-np.dot(hone,meanstim)
        if numiteration > 100:
            print 'Kernels not converging'
            break
    return np.concatenate((np.array([hzero]),hone))

def STA2(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened = False
    meanout = np.mean(outputs)
    hzero = 0.
    hone = np.zeros(stimuli.shape[1])
    for ii in xrange(stimuli.shape[0]):
        hone += stimuli[ii]*outputs[ii]
    hone = hone/stimuli.shape[0]
    if whitened:
        ac = np.zeros((stimuli.shape[1],stimuli.shape[1]))
        for ii in xrange(stimuli.shape[0]):
            ac += np.outer(stimuli[ii],stimuli[ii])
        ac = ac/stimuli.shape[0]
        aci = np.linalg.pinv(ac)
        #pseudoinverse of autocorrelation matrix
        hone = np.dot(aci,hone)
    return np.concatenate((np.array([hzero]),hone))

def STASys(stimuli,outputs,whitened=None):
    nSTRFs = outputs.shape[1]
    hzeros = np.zeros(nSTRFs)
    hones = np.zeros((nSTRFs,stimuli.shape[1]))
    meanstim = np.mean(stimuli,axis=0)
    for ii in xrange(nSTRFs):
        temp = STA(stimuli,outputs[:,ii],meanstim,whitened)
        hzeros[ii] = temp[0]
        hones[ii] = temp[1:]
    return (hzeros,hones)

def STASys2(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened=False
    strfs = np.dot(stimuli.T,outputs)/stimuli.shape[0]
    if whitened:
        strfs = np.dot(np.linalg.pinv(np.dot(stimuli.T,stimuli)),strfs)/stimuli.shape[0]
    return strfs.T

def sparseSTASys(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened = False
    strfs = np.dot(stimuli.T,outputs)/stimuli.shape[0]
    if whitened:
        strfs = np.dot(strfs,np.linalg.pinv(np.dot(outputs.T,outputs)))/stimuli.shape[0]
    return strfs.T

def STC(stimuli,outputs,meancov):
    stcM = np.sum(np.array([np.outer(stimuli[ii],stimuli[ii])*outputs[ii] for ii in xrange(outputs.shape[0])]),axis=0)/np.mean(outputs)
    stcM = stcM-meancov
    return stcM

def STCSys(stimuli,outputs):
    meancov = np.mean(np.array([np.outer(stim,stim) for stim in stimuli]),axis=0)
    stcs = np.array([STC(stimuli,outputs[:,ii],meancov) for ii in xrange(outputs.shape[1])])
    return stcs

def STC2(stimuli,outputs):
    stcM = np.mean(np.array([np.outer(stimuli[ii],stimuli[ii])*outputs[ii] for ii in xrange(outputs.shape[0])]),axis=0)
    return stcM

def STCSys2(stimuli,outputs):
    outputs = np.absolute(outputs)
    stcs = np.array([STC2(stimuli,outputs[:,ii]) for ii in xrange(outputs.shape[1])])
    return stcs

def MaxRelDimSTC(stcs):
    vals = []
    strfs = []
    for stc in stcs:
        eVals,eVecs = np.linalg.eigh(stc)
        idx = np.argsort(np.absolute(eVals))
        vals.append(eVals[idx][-1])
        strfs.append(eVecs[idx][-1])
    return (np.array(vals),np.array(strfs))

def MaxStimDimSTC(stcs):
    vals = []
    strfs = []
    for stc in stcs:
        eVals,eVecs = np.linalg.eigh(stc)
        idx = np.argsort(eVals)
        vals.append(eVals[idx][-1])
        strfs.append(eVecs[idx][-1])
    return (np.array(vals),np.array(strfs))
