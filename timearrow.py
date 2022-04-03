import skvideo.io
import skvideo.datasets
import numpy as np

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from HSIC.HSIC import hsic_gam, rbf_dot

def algorithm_1(f, b=None, sig1=0.1, sig2=0.05, v='a', p=2, q=0):  # b=f.T , how is p determined?
    """
    f : forward
    b : backward....
    sig1 : These are the defaults from the paper...
    sig2 :
    """
    if b is None:
        b = f[::-1]
    if v=='a':
        fct = lambda x,y: - hsic_gam(x,y)
    elif v=='b':
        fct = lambda x,y: p - value_hsic_gam(x,y)
    else:
        raise ValueError("Algorithm version not correctly specified.")

    model_f = sm.tsa.VAR(f).fit(p); resid_f = model_f.resid; fw = fct(f,resid_f)
    model_b = sm.tsa.VAR(b).fit(p); resid_b = model_b.resid; bw = fct(b,resid_b)
    if np.max(fw,bw) > sig1 and np.max(fw,bw) < sig2:
        if fw > bw:
            return "Time series in correct direction."
        else:
            return "Time series in reverse direction."
    else:
        return "I don't know."

T = 200
alpha = 0.4
p = 1
Phi = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]]) / 4
X = [np.array([0,0])]

for i in range(T-1):
    X.append( np.dot(Phi,X[-1])+np.random.rand(2) )
X = np.vstack(X)

model = sm.tsa.VAR(X)
result = model.fit(p)
print(result.summary())
result.forecast(X[-p:],10)
result.plot_forecast(10)
#plt.show()

print(algorithm_1(X,p=1))

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
