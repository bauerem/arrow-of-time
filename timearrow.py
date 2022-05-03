import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt

## Define Algorithm from paper, using the HSIC independence test implementation
## of github user Seth or amber0309 (soubo.sub@gmail.com): https://github.com/amber0309

from HSIC.HSIC import hsic_gam, rbf_dot

def algorithm_1(f, b=None, sig1=0.1, sig2=0.05, v='a', p=2, q=0):  # b=f.T , how is p determined?
    """
    TODO: complete docstring
    Implementation of pseudocode of Algorithm 1 in
    "The Arrow of Time in Multivariate Time Series" Bauer et al. (2016)
    https://arxiv.org/pdf/1603.00784.pdf
    f : forward
    b : backward....
    sig1 : These are the defaults from the paper...
    sig2 :
    """
    if b is None:
        b = f[::-1]
    if v=='a':
        fct = lambda x,y: - hsic_gam(x,y)[1]
    elif v=='b':
        fct = lambda x,y: p - hsic_gam(x,y)[1]
    else:
        raise ValueError("Algorithm version not correctly specified.")

    model_f = sm.tsa.VAR(f).fit(p)
    resid_f = model_f.resid
    fw = fct(f[1:,:],resid_f)
    model_b = sm.tsa.VAR(b).fit(p); resid_b = model_b.resid; bw = fct(b[1:,:],resid_b)
    print(fw)
    print(bw)
    if max(fw,bw) > sig1 and max(fw,bw) < sig2:
        if fw > bw:
            return "Time series in correct direction."
        else:
            return "Time series in reverse direction."
    else:
        return "I don't know."


## Apply algorithm to simulated VAR model

## Define sample hyperparameters
T = 2000

## Define model hyperparameters as they are in the simulation section of the paper
r = 3
#alpha = 0.4 # or np.random.rand(1)
p = 1
lamb = 2.5
#R = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]]) / 4
R = np.random.rand(2,2)
# Change following line if i != 1
Phi = lamb**(-1)*R-(2*lamb)**(-1)*np.ones(R.shape)

## Initialize time series
X = [np.array([0,0])]

## Simulate VAR model
for i in range(T-1):
    noise = np.random.rand(2)
    noise = noise / np.linalg.norm(noise) * np.abs(noise)**r
    ## One innovation
    X.append( np.dot(Phi, X[-1])+ noise)
X = np.vstack(X)

## Fit VAR model using statsmodels package
model = sm.tsa.VAR(X)
result = model.fit(p)
print(result.summary())

## Forcast 10 further innovations of approximated VAR model
result.forecast(X[-p:],10)

## Plot sample and 10 forecasted innovations using statsmodels
## and matplotlib.pyplot packages
result.plot_forecast(10)
plt.show()

## Apply Algorithm 1 from https://arxiv.org/pdf/1603.00784.pdf
print(algorithm_1(X,p=1,v='b'))

"""
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(T),X[0,:])
plt.subplot(2,1,2)
plt.plot(np.arange(T),X[1,:])
plt.show()
"""

"""
videodata = skvideo.io.vread("test.mp4")
videodata = videodata.astype(np.float32)
#videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny()) #np.random.random(size=(800, 200, 200, 3)) * 255 #
#print(videodata.shape)
#skvideo.io.vwrite("outputvideo.mp4", videodata)

#videodata = np.random.random(size=(400, 4, 4, 1)) * 255
videodata = videodata.reshape((videodata.shape[0],-1))
t = np.arange(videodata.shape[0])

exog = videodata[:,0]
endog = videodata[:,1:4]
#model = sm.tsa.VARMAX(endog, order=(2,0), trend='n') #, exog=exog)
#res = model.fit(maxiter=100, disp=False)

model = sm.tsa.VAR(endog) #, exog=exog)
res = model.fit(2)

print(res.summary())

plt.figure()
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(t,videodata[:,i])
#plt.plot(t,videodata[:,1])
plt.show()
"""
#print(model.L1.y1)
#print(model.params)
