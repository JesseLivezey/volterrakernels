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
def get_vzero(stimuli,outputs,guess=None,meth=None):
    if guess is None:
        guess = np.zeros(1)
    if meth is None:
        meth = 'L-BFGS-B'
    output = opt.minimize(vzerof(stimuli,outputs),guess,method = meth,jac=vzerof_grad(stimuli,outputs)).x
    return output

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
def get_vone(stimuli,outputs,guess=None,meth=None):
    if guess is None:
        guess = np.zeros(1+stimuli[0].shape[0])
    if meth is None:
        meth = 'L-BFGS-B'
    output = opt.minimize(vonef(stimuli,outputs),guess,method = meth,jac=vonef_grad(stimuli,outputs)).x
    return output

#Function to take matrix and return array of symmetrix elements
def symm(matrix):
    length = matrix.shape[0]
    output = np.zeroes((length**2+length)/2)
    outii = 0
    for ii in xrange(length):
        for jj in xrange(ii+1):
            output[outii] = matrix[ii,jj]
            outii += 1
    return output
#Function to take an array of symmetric elements and return a symmetrix matrix
def unsymm(array):
    length = array.shape[0]
    lengthMat = (int(round(np.sqrt(8*length+1)))-1)/2
    output = np.zeros((lengthMat,lengthMat))
    outii = 0
    for ii in xrange(lengthMat):
        for jj in xrange(ii+1):
            output[ii,jj] = array[outii]
            output[jj,ii] = array[outii]
            outii += 1
    return output
#Function that calculates response from 2nd order Volterra model
def vtwo(stimulus,kernels):
    lengths = stimulus.shape[0]
    one = kernels[1:1+lengths]
    two = np.reshape(kernels[1+lengths:],(lengths,lengths))
    return kernels[0]+np.dot(stimulus,kernels[1:1+lengths])+np.dot(stimulus,np.dot(two,stimulus))
#Function that calculates response from 2nd order Volterra model, given symmetric kernel
def vtwoS(stimulus,kernels):
    lengths = stimulus.shape[0]
    one = kernels[1:1+lengths]
    two = np.reshape(unsymm(kernels[1+lengths:]),(lengths,lengths))
    return kernels[0]+np.dot(stimulus,kernels[1:1+lengths])+np.dot(stimulus,np.dot(two,stimulus))
#Returns vtwo function to be minimized based on given stimuli and measured response
def vtwof(stimuli,outputs):
    func = lambda kernels: (.5/outputs.shape[0])*np.sum(np.array([(outputs[i]-vtwo(stimuli[i],kernels))**2
                                                                  for i in xrange(outputs.shape[0])]))
    return func
#Returns vtwo function to be minimized based on given stimuli and measured response
def vtwofS(stimuli,outputs):
    func = lambda kernels: (.5/outputs.shape[0])*np.sum(np.array([(outputs[i]-vtwoS(stimuli[i],kernels))**2
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
#Returns coefficient for diagonal or off-diagonal terms
def oneOrTwo(n,m):
    if n == m:
        output = 1
    else:
        output = 2
    return output
#Returns gradient of vtwo function
def vtwof_gradS(stimuli,outputs):
    gradhzero = lambda kernels: (-1./outputs.shape[0])*np.sum(np.array([(outputs[i]-vtwoS(stimuli[i],kernels))
                                                                         for i in xrange(outputs.shape[0])]))
    gradhone = lambda k,kernels: (-1./outputs.shape[0])*np.sum(np.array([(stimuli[i][k]*(outputs[i]-vtwoS(stimuli[i],kernels)))
                                                                          for i in xrange(outputs.shape[0])]))
    gradhtwo = lambda n,m,kernels:(-1./outputs.shape[0])*oneOrTwo(n,m)*np.sum(np.array([(stimuli[i][n]*stimuli[i][m]*(outputs[i]-vtwoS(stimuli[i],kernels)))
                                                                           for i in xrange(outputs.shape[0])]))
    func = lambda kernels: np.concatenate((np.array([gradhzero(kernels)]),)+
                                           (np.array([gradhone(k,kernels) for k in xrange(stimuli[0].shape[0])]),)+
                                           tuple((np.array([gradhtwo(n,m,kernels) for m in xrange(n+1)]) for n in xrange(stimuli[0].shape[0]))))
    return func 
#Function that does minimization to find vtwo.
def get_vtwo(stimuli,outputs,guess=None,meth=None):
    if guess is None:
        guess = np.zeros(1+stimuli[0].shape[0]+stimuli[0].shape[0]**2)
    if meth is None:
        meth = 'L-BFGS-B'
    output = opt.minimize(vtwof(stimuli,outputs),guess,method = meth,jac=vtwof_grad(stimuli,outputs)).x
    return output
#Function that does minimization to find vtwo.
def get_vtwoS(stimuli,outputs,guess=None,meth=None):
    length = stimuli[0].shape[0]
    if guess is None:
        guess = np.zeros(1+length+(length**2+length)/2)
    if meth is None:
        meth = 'L-BFGS-B'
    output = opt.minimize(vtwofS(stimuli,outputs),guess,method = meth,jac=vtwof_gradS(stimuli,outputs)).x
    return output
    
#Takes stimuli and kernels as input and generates expected response. Calculates correct order based on length of kernels
def vresponse(stimuli,kernels):
    lengthk = kernels.shape[0]
    lengths = stimuli[0].shape[0]
    response = np.zeros(stimuli.shape[0])
    if lengthk == 1:
        vker = vzero
    elif lengthk == 1+lengths:
        vker = vone
    elif lengthk > 1+lengths:
        vker = vtwo
    for i,stimulus in enumerate(stimuli):
        response[i] = vker(stimulus,kernels)
    return response
    
