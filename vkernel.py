import numpy as np
from scipy import optimize as opt
#Function that calculates output from 0th order Volterra model
def vzero(stimulus,kernels):
    return kernels[0]
#Returns vzero function to be minimized based on given stimuli and measured response 
def vzerof(stimuli,outputs):
    func = lambda kernels: (.5/outputs.shape[0])*np.sum(np.array([(outputs[i]-vzero(stimuli[i],kernels))**2
                                                                   for i in xrange(outputs.shape[0])]))
    return func
#Returns gradient of vzero function to be minimized
def vzerof_grad(stimuli,outputs):
    func = lambda kernels: (-1./outputs.shape[0])*np.sum(np.array([(outputs[i]-vzero(stimuli[i],kernels))
                                                                    for i in xrange(outputs.shape[0])]))
    return func
#Function that does minimization to find vzero.
def get_vzero(stimuli,outputs,guess=None,method=None):
    if guess is None:
        guess = np.zeros(1)
    return opt.minimize(vzerof(stimuli,outputs),guess,method = method,jac=vzerof_grad(stimuli,outputs)).x

#Function that calculates output from 1st order Volterra model
def vone(stimulus,kernels):
    return kernels[0]+np.dot(stimulus,kernels[1:])
#Returns vone function to be minimized based on given stimuli and measured response 
def vonef(stimuli,outputs):
    func = lambda kernels: (.5/outputs.shape[0])*np.sum(np.array([(outputs[i]-vone(stimuli[i],kernels))**2
                                                                   for i in xrange(outputs.shape[0])]))
    return func
#Returns gradient of vone function
def vonef_grad(stimuli,outputs):
    gradhzero = lambda kernels: (-1./outputs.shape[0])*np.sum(np.array([(outputs[i]-vone(stimuli[i],kernels))
                                                                         for i in xrange(outputs.shape[0])]))
    gradhone = lambda k,kernels: (-1./outputs.shape[0])*np.sum(np.array([(stimuli[i][k]*(outputs[i]-vone(stimuli[i],kernels)))
                                                                          for i in xrange(outputs.shape[0])]))
    func = lambda kernels: np.concatenate((np.array([gradhzero(kernels)]),np.array([gradhone(k,kernels) for k in xrange(stimuli[0].shape[0])])))
    return func 
#Function that does minimization to find vone.
def get_vone(stimuli,outputs,guess=None,method=None):
    if guess is None:
        guess = np.zeros(1+stimuli[0].shape[0])
    return opt.minimize(vonef(stimuli,outputs),guess,method = method,jac=vonef_grad(stimuli,outputs)).x

#Function that calculates response from 2nd order Volterra model
def vtwo(stimulus,kernels):
    lengths = stimulus.shape[0]
    one = kernels[1:1+lengths]
    two = np.reshape(kernels[1+lengths:],(lengths,lengths))
    return kernels[0]+np.dot(stimulus,kernels[1:1+lengths])+np.dot(stimulus,np.dot(two,stimulus))
#Returns vtwo function to be minimized based on given stimuli and measured response
def vtwof(stimuli,outputs):
    func = lambda kernels: (.5/outputs.shape[0])*np.sum(np.array([(outputs[i]-vtwo(stimuli[i],kernels))**2
                                                                  for i in xrange(outputs.shape[0])]))
    return func
#Returns gradient of vtwo function
def vtwof_grad(stimuli,outputs):
    gradhzero = lambda kernels: (-1./outputs.shape[0])*np.sum(np.array([(outputs[i]-vtwo(stimuli[i],kernels))
                                                                         for i in xrange(outputs.shape[0])]))
    gradhone = lambda k,kernels: (-1./outputs.shape[0])*np.sum(np.array([(stimuli[i][k]*(outputs[i]-vtwo(stimuli[i],kernels)))
                                                                          for i in xrange(outputs.shape[0])]))
    gradhtwo = lambda n,m,kernels:(-1./outputs.shape[0])*np.sum(np.array([(stimuli[i][n]*stimuli[i][m]*(outputs[i]-vtwo(stimuli[i],kernels)))
                                                                           for i in xrange(outputs.shape[0])]))
    func = lambda kernels: np.concatenate((np.array([gradhzero(kernels)]),
                                           np.array([gradhone(k,kernels) for k in xrange(stimuli[0].shape[0])]),
                                           np.array([[gradhtwo(n,m,kernels) for m in xrange(stimuli[0].shape[0])] for n in xrange(stimuli[0].shape[0])]).flatten()))
    return func 
#Function that does minimization to find vtwo.
def get_vtwo(stimuli,outputs,guess=None,method=None):
    if guess is None:
        guess = np.zeros(1+stimuli[0].shape[0]+stimuli[0].shape[0]**2)
    return opt.minimize(vtwof(stimuli,outputs),guess,method = method,jac=vtwof_grad(stimuli,outputs)).x
    
#Takes stimuli and kernels as input and generates expected response. Calculates correct order based on length of kernels
def vresponse(stimuli,kernels):
    lengthk = kernels.shape[0]
    lengths = stimuli[0].shape[0]
    response = np.zeros(stimuli.shape[0])
    if lengthk == 1:
        vker = vzero
    elif lengthk == 1+lengths:
        vker = vone
    elif lengthk == 1+lengths+lengths**2:
        vker = vtwo
    for i,stimulus in enumerate(stimuli):
        response[i] = vker(stimulus,kernels)
    return response
    
