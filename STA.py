import numpy as np
from __future__ import division
import numpy.linalg
#Function to calculate the Spike-Triggered-Average
def STA(stimuli,outputs,whitened=None):
    if whitened is None:
        whitened = False
    meanout = np.mean(outputs)
    meanstim = np.mean(stimuli,axis=0)
    hzero = meanout
    hone = numpy.mean(np.array([stimulus*output[i] for i,stimulus in enumerate(stimuli)]),axis=0)/outputs.shape[0]-meanout*meanstim
    if whitened:
        #pseudoinverse of autocorrelation matrix
        aci = np.linalg.pinv(np.array([[np.dot(stimI,stimJ) for stimJ in stimuli] for stimI in stimuli]))
        hone = np.dot(aci,htwo)
    return np.concatenate((hzero,hone))
